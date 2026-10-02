import Foundation

/// adb 单条逻辑流（本端 open 的 shell/sync/reverse，或 reverse 隧道回连的 scrcpy 流）。
/// 语义对齐 muntashirakon/adb AdbStream：载荷队列 + 条件变量，read 阻塞取包。
public final class AdbStream {
    enum StreamError: Error, LocalizedError {
        case closed
        var errorDescription: String? { "adb stream closed" }
    }

    public let localId: UInt32
    private var remoteId: UInt32 = 0
    private unowned let connection: AdbConnection

    private let condition = NSCondition()
    private var payloadQueue: [Data] = []
    private var readable = false
    private var isClosed = false
    /// close 前已收到的载荷（close 后仍需消费完，模拟半关闭）。
    private var drained = false

    init(localId: UInt32, connection: AdbConnection) {
        self.localId = localId
        self.connection = connection
    }

    fileprivate func setRemoteId(_ id: UInt32) {
        condition.lock()
        remoteId = id
        condition.unlock()
    }

    fileprivate func markReadable() {
        condition.lock()
        readable = true
        condition.signal()
        condition.unlock()
    }

    fileprivate func enqueue(_ payload: Data) {
        condition.lock()
        payloadQueue.append(payload)
        condition.signal()
        condition.unlock()
    }

    fileprivate func close(drain: Bool) {
        condition.lock()
        isClosed = true
        drained = drain
        condition.broadcast()
        condition.unlock()
        connection.removeStream(localId: localId)
    }

    fileprivate var remote: UInt32 {
        condition.lock(); defer { condition.unlock() }
        return remoteId
    }

    var isOpenInternal: Bool {
        condition.lock(); defer { condition.unlock() }
        return !isClosed || !payloadQueue.isEmpty
    }

    /// 读一块载荷；流结束返回 nil。writer 侧 close 时把剩余载荷排干后返回 nil。
    public func read() throws -> Data? {
        condition.lock()
        defer { condition.unlock() }
        while payloadQueue.isEmpty {
            if isClosed { return drained ? nil : nil }
            condition.wait()
        }
        return payloadQueue.removeFirst()
    }

    /// 读满 count 字节（跨包拼接），EOF 抛错。
    public func readExact(_ count: Int) throws -> Data {
        var out = Data()
        while out.count < count {
            guard let chunk = try read() else { throw StreamError.closed }
            out.append(chunk)
        }
        if out.count > count { pushBack(Data(out[count...])) ; out = out.prefix(count) }
        return out
    }

    private func pushBack(_ data: Data) {
        condition.lock()
        payloadQueue.insert(data, at: 0)
        condition.unlock()
    }

    /// 读一行（到 \n 或 EOF），用于 shell / 服务应答。
    public func readLine(timeout: TimeInterval = 15) throws -> String? {
        var line = Data()
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            condition.lock()
            while payloadQueue.isEmpty {
                if isClosed {
                    condition.unlock()
                    return line.isEmpty ? nil : String(decoding: line, as: UTF8.self)
                }
                if !condition.wait(until: deadline) {
                    condition.unlock()
                    return String(decoding: line, as: UTF8.self)
                }
            }
            let chunk = payloadQueue.removeFirst()
            condition.unlock()
            if let nl = chunk.firstIndex(of: 0x0a) {
                line.append(chunk[..<nl])
                if nl + 1 < chunk.count { pushBack(Data(chunk[(nl + 1)...])) }
                return String(decoding: line, as: UTF8.self)
            }
            line.append(chunk)
        }
    }

    /// 写载荷（自动按 maxData 分片；每片 WRTE 后等待 adbd 的 OKAY 背压）。
    public func write(_ data: Data) throws {
        try connection.writeOnStream(self, data)
    }

    public func closeAndWriteEOF() {
        connection.closeStream(self)
    }
}

/// adb TCP 客户端连接：CNXN/AUTH 握手、消息收发、流管理。
/// reverse 隧道（adbd 主动 OPEN 回连）通过 [deviceStreamHandler] 分发给 scrcpy 会话。
public final class AdbConnection {
    public enum AdbError: Error, LocalizedError {
        case unauthorised
        case connectFailed(String)
        case notConnected
        case openFailed(String)
        case serviceFailed(String)

        public var errorDescription: String? {
            switch self {
            case .unauthorised: return "adbd 未授权本机密钥"
            case .connectFailed(let m): return "adb 握手失败: \(m)"
            case .notConnected: return "adb 未连接"
            case .openFailed(let m): return "adb open 失败: \(m)"
            case .serviceFailed(let m): return "adb 服务应答失败: \(m)"
            }
        }
    }

    private let io = SocketIO()
    private let auth: AdbAuth
    private let writeLock = NSLock()
    private var maxData: UInt32 = AdbMessage.maxPayload

    private var streamsLock = NSLock()
    private var streamsById: [UInt32: AdbStream] = [:]
    private var nextLocalId: UInt32 = 0

    private(set) public var deviceName: String = ""
    private(set) public var connected = false
    /// reverse 隧道回连（adbd 发 OPEN）时的分发回调：参数为回连序号（0 起）。
    public var deviceStreamHandler: ((AdbStream, UInt32) -> Void)?

    public init(auth: AdbAuth) {
        self.auth = auth
    }

    // MARK: - 握手

    public func connect(host: String, port: UInt16, timeoutMs: Int = 8000) throws {
        try io.connect(host: host, port: port, timeoutMs: timeoutMs)

        // CNXN
        try send(AdbMessage(command: AdbCommand.cnxn, arg0: AdbMessage.protocolVersion, arg1: AdbMessage.maxPayload))

        let reader = Thread { [weak self] in self?.readerLoop() }
        reader.name = "gscp-adb-reader"
        reader.stackSize = 1 << 20
        reader.start()

        // 等待连接建立（readerLoop 处理 CNXN/AUTH 往返）
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
        while true {
            if connected { return }
            if !io.isOpen { throw AdbError.connectFailed("socket closed during handshake") }
            if Date() > deadline { throw AdbError.connectFailed("timeout") }
            Thread.sleep(forTimeInterval: 0.02)
        }
    }

    private func readerLoop() {
        do {
            while io.isOpen {
                let msg = try AdbMessageParser.parse(from: io)
                switch msg.command {
                case AdbCommand.cnxn:
                    maxData = max(4096, msg.arg1)
                    connected = true

                case AdbCommand.auth:
                    if msg.arg0 == AdbAuthType.token {
                        if sentSignature {
                            // 已签过仍被拒 → 回公钥请求授权
                            try send(AdbMessage(command: AdbCommand.auth,
                                                arg0: AdbAuthType.rsaPublicKey,
                                                arg1: 0,
                                                payload: auth.publicKeyPayload()))
                        } else {
                            let sig = try auth.sign(token: msg.payload)
                            sentSignature = true
                            try send(AdbMessage(command: AdbCommand.auth,
                                                arg0: AdbAuthType.signature,
                                                arg1: 0, payload: sig))
                        }
                    }

                case AdbCommand.open:
                    // adbd 主动回连（reverse 隧道）：OKAY 后交给分发回调
                    streamsLock.lock()
                    nextLocalId += 1
                    let localId = nextLocalId
                    let stream = AdbStream(localId: localId, connection: self)
                    stream.setRemoteId(msg.arg0)
                    streamsById[localId] = stream
                    streamsLock.unlock()
                    try send(AdbMessage(command: AdbCommand.okay, arg0: localId, arg1: msg.arg0))
                    deviceStreamHandler?(stream, msg.arg1)

                case AdbCommand.okay:
                    if let stream = streamBy(localId: msg.arg1) {
                        stream.setRemoteId(msg.arg0)
                        stream.markReadable()
                    }

                case AdbCommand.wrte:
                    if let stream = streamBy(localId: msg.arg1) {
                        stream.enqueue(msg.payload)
                        try send(AdbMessage(command: AdbCommand.okay, arg0: msg.arg1, arg1: msg.arg0))
                    }

                case AdbCommand.clse:
                    if let stream = streamBy(localId: msg.arg1) {
                        stream.close(drain: true)
                    }

                default:
                    break
                }
            }
        } catch {
            // 连接断开：通知所有流
            streamsLock.lock()
            let all = Array(streamsById.values)
            streamsById.removeAll()
            streamsLock.unlock()
            connected = false
            for s in all { s.close(drain: true) }
        }
    }

    // MARK: - 流管理

    private func streamBy(localId: UInt32) -> AdbStream? {
        streamsLock.lock(); defer { streamsLock.unlock() }
        return streamsById[localId]
    }

    fileprivate func removeStream(localId: UInt32) {
        streamsLock.lock(); defer { streamsLock.unlock() }
        streamsById.removeValue(forKey: localId)
    }

    /// 打开一条到 adbd 的服务流（shell:/sync:/reverse:...）。
    public func openStream(_ destination: String, timeout: TimeInterval = 15) throws -> AdbStream {
        guard connected else { throw AdbError.notConnected }
        streamsLock.lock()
        nextLocalId += 1
        let localId = nextLocalId
        let stream = AdbStream(localId: localId, connection: self)
        streamsById[localId] = stream
        streamsLock.unlock()
        try send(AdbMessage(command: AdbCommand.open, arg0: localId, arg1: 0,
                            payload: Data(destination.utf8)))
        // 等 adbd 的 OKAY（markReadable）
        let deadline = Date().addingTimeInterval(timeout)
        stream.waitReadable(until: deadline)
        if !stream.isOpenInternal {
            throw AdbError.openFailed("stream closed on open")
        }
        return stream
    }

    // MARK: - 发送

    private var sentSignature = false

    fileprivate func send(_ msg: AdbMessage) throws {
        var data = msg.encoded()
        writeLock.lock()
        defer { writeLock.unlock() }
        try io.write(data)
        data.removeAll()
    }

    fileprivate func writeOnStream(_ stream: AdbStream, _ data: Data) throws {
        var offset = 0
        let chunkMax = Int(maxData)
        let bytes = [UInt8](data)
        while offset < bytes.count {
            let end = min(offset + chunkMax, bytes.count)
            try send(AdbMessage(command: AdbCommand.wrte, arg0: stream.localId,
                                arg1: stream.remote, payload: Data(bytes[offset..<end])))
            offset = end
        }
    }

    fileprivate func closeStream(_ stream: AdbStream) {
        try? send(AdbMessage(command: AdbCommand.clse, arg0: stream.localId, arg1: stream.remote))
        stream.close(drain: true)
    }

    /// 断开连接（等价 adb disconnect：关 socket 即可）。
    public func disconnect() {
        io.closeSocket()
    }
}

extension AdbStream {
    fileprivate func waitReadable(until deadline: Date) {
        condition.lock()
        while !readable && !isClosed {
            if !condition.wait(until: deadline) { break }
        }
        condition.unlock()
    }
}
