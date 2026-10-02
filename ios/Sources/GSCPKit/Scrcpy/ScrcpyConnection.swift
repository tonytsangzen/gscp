import Foundation

/// scrcpy 会话事件（线程：各自流线程；与 Android 端 ScrcpyConnection.EventCallback 对齐）。
public protocol ScrcpyEventDelegate: AnyObject {
    func scrcpyOnConnect()
    /// 视频流就绪：76B 头解析出 codec/宽高（overlay-only 模式不触发）。
    func scrcpyOnVideoPrepare(codec: String, width: Int, height: Int)
    /// 视频帧包（config=true 为 SPS/PPS 配置包，必须先于普通帧喂解码器）。
    func scrcpyOnVideoFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64)
    func scrcpyOnAudioPrepare(codec: String)
    /// 音频首帧 = 配置包（OpusHead），不可丢弃。
    func scrcpyOnAudioConfig(_ csd: Data)
    func scrcpyOnAudioFrame(_ data: Data)
    func scrcpyOnOverlayPrepare(codec: String, width: Int, height: Int)
    func scrcpyOnOverlayFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64)
    func scrcpyOnDisconnect()
    func scrcpyOnError(_ message: String)
}

/// scrcpy 连接会话（iOS 端，能力对齐 Android 端 ScrcpyConnection.kt）：
/// 1. adb TCP 连眼镜 adbd（或模拟器/真机的 adbd）；
/// 2. 推送 bundle 内 scrcpy-server 并注册 reverse 隧道后拉起；
/// 3. 服务器按 [video, audio, control, overlay] 顺序回连，各路独立线程分流。
public final class ScrcpyConnection {
    public struct Options {
        /// 完整投屏（camera+audio+overlay）；AR 唯一模式下固定 false（与 Android HEAD 一致）。
        public var fullScreenMirroring: Bool
        /// 用户设置：是否要音频。
        public var audioEnabled: Bool
        public init(audioEnabled: Bool = true, fullScreenMirroring: Bool = false) {
            self.audioEnabled = audioEnabled
            self.fullScreenMirroring = fullScreenMirroring
        }
    }

    public private(set) var connected = false
    public weak var delegate: ScrcpyEventDelegate?

    private let options: Options
    private var adb: AdbConnection?
    private let stateLock = NSLock()
    private var streamIndex: UInt32 = 0
    /// scid（%08x 十六进制，须 < 0x80000000）
    private var scid: String = ""

    public init(options: Options = Options()) {
        self.options = options
    }

    // MARK: - 连接

    public func connect(ip: String, port: UInt16 = 5555) throws {
        let auth = try AdbAuth(
            keyBase64: Self.loadResource("ca.key"),
            deviceName: "gscp",
            pubKeyString: Self.adbPublicKeyString
        )
        let conn = AdbConnection(auth: auth)
        self.adb = conn

        conn.deviceStreamHandler = { [weak self] stream, _ in
            guard let self else { return }
            let idx = self.nextStreamIndex()
            DispatchQueue.global(qos: .userInitiated).async {
                self.dispatch(stream: stream, index: idx)
            }
        }
        try conn.connect(host: ip, port: port)
        connected = true
        delegate?.scrcpyOnConnect()

        // 查询 server 端（眼镜）SDK：决定 audio_source=output 是否可用（API 29+）。
        // 查询失败按 31（真机眼镜为 Android 12+）处理。
        serverSdkCache = Int((try? AdbServices.getProperty(conn, "ro.build.version.sdk")) ?? "") ?? 31

        // scid：合法 hex 且 < 0x80000000（server 按 Int32.parseInt(…,16) 解析）
        scid = String(format: "%08x", UInt32.random(in: 0x1000_0000...0x7fff_ffff))

        let serverData = try Self.loadResourceData("scrcpy-server")
        try AdbServices.push(conn, data: serverData, to: "/data/local/tmp/scrcpy-server.jar")
        try AdbServices.reverse(conn, deviceSocket: "scrcpy_\(scid)", clientPort: 27813)

        let param = serverParam()
        // 拉起 server：读输出到 EOF（EOF = server 退出 → 会话结束）
        let shell = try conn.openStream(
            "shell:CLASSPATH=/data/local/tmp/scrcpy-server.jar app_process / " +
                "com.genymobile.scrcpy.Server 3.3.1 scid=\(scid) \(param)")
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var log = ""
            while let chunk = try? shell.read() {
                log.append(String(decoding: chunk, as: UTF8.self))
                if log.count > 8192 { log.removeFirst(log.count - 8192) }
            }
            guard let self else { return }
            let wasConnected = self.connected
            self.connected = false
            if wasConnected {
                self.delegate?.scrcpyOnDisconnect()
            }
        }
    }

    public func disconnect() {
        guard let conn = adb, connected else { return }
        // 与 Android 端一致：杀掉设备侧 server 进程再断开
        _ = try? AdbServices.shell(conn, "killall app_process")
        connected = false
        conn.disconnect()
    }

    // MARK: - 参数与流分发（槽位对齐是硬约束：param 与 handleList 必须同源）

    private var serverSdkCache = 31

    private func serverParam() -> String {
        if options.fullScreenMirroring {
            // 完整投屏：camera 源（server 侧要求 Android 12+，真机眼镜满足）
            return "log_level=info video_source=camera audio_source=output " +
                "max_size=1024 video=true audio=\(options.audioEnabled) overlay=true"
        }
        // AR 唯一模式（overlay-only）：
        //  - video_source=display：video 关闭时无实际作用，但 server 缺省按 camera 走
        //    版本检查（Android < 12 直接拒绝整个会话），显式给 display 绕开；
        //  - audio_source=output 依赖 server 端（眼镜）API 29+，按 server SDK 判定
        //    （比 Android 端按手机 API 判定更准确）。
        let audio = options.audioEnabled && serverSdkCache >= 29
        return "log_level=info video=false video_source=display audio=\(audio) " +
            "max_size=640 overlay=true audio_source=output"
    }

    /// 流槽位表（必须与 serverParam 的开关一致，否则错位）：
    /// 完整模式: [video, audio?, control, overlay]；overlay-only+audio: [audio, control, overlay]；
    /// overlay-only 无 audio: [control, overlay]（control 顶到 0 槽 → 映射时 +1）。
    private func dispatch(stream: AdbStream, index: UInt32) {
        let audioOn = options.audioEnabled && serverSdkCache >= 29
        var slot = Int(index)
        if !options.fullScreenMirroring && !audioOn { slot += 1 }
        switch slot {
        case 0: options.fullScreenMirroring ? handleVideo(stream) : handleAudio(stream)
        case 1: handleControl(stream)
        case 2: handleOverlay(stream)
        default: drain(stream)
        }
    }

    private func nextStreamIndex() -> UInt32 {
        stateLock.lock(); defer { stateLock.unlock() }
        defer { streamIndex += 1 }
        return streamIndex
    }

    private func drain(_ stream: AdbStream) {
        while (try? stream.read()) != nil {}
    }

    // MARK: - 各路处理器（帧格式与 Android 端逐字节一致）

    /// control：只持有通道，读 server 输出（基本无数据）。
    private func handleControl(_ stream: AdbStream) {
        while connected, let _ = (try? stream.read()) ?? nil {}
    }

    /// overlay 流：12B 头 [codec(4) width(4BE) height(4BE)]，随后 12B 帧头帧序列。
    private func handleOverlay(_ stream: AdbStream) {
        do {
            let header = try stream.readExact(12)
            let codec = String(decoding: header[0..<4], as: UTF8.self)
            let width = Int(header.loadBE32(at: 4))
            let height = Int(header.loadBE32(at: 8))
            delegate?.scrcpyOnOverlayPrepare(codec: codec, width: width, height: height)
            while connected {
                guard let (payload, config, pts) = try readFrame(stream, bufferLimit: 512 * 1024) else { break }
                delegate?.scrcpyOnOverlayFrame(payload, config: config, ptsAndFlags: pts)
            }
        } catch {
            // 流结束
        }
    }

    /// video 流（完整投屏）：76B 头 = 64B 设备名 + codec(4) + w(4BE) + h(4BE)。
    private func handleVideo(_ stream: AdbStream) {
        do {
            let header = try stream.readExact(76)
            let codec = String(decoding: header[64..<68], as: UTF8.self)
            let width = Int(header.loadBE32(at: 68))
            let height = Int(header.loadBE32(at: 72))
            delegate?.scrcpyOnVideoPrepare(codec: codec, width: width, height: height)
            while connected {
                guard let (payload, config, pts) = try readFrame(stream, bufferLimit: 1024 * 1024) else { break }
                delegate?.scrcpyOnVideoFrame(payload, config: config, ptsAndFlags: pts)
            }
        } catch {}
    }

    /// audio 流：首 socket 可能带 64B 设备名前缀（与 Android 端相同的容错解析）。
    private func handleAudio(_ stream: AdbStream) {
        do {
            let buffered = BufferedStream(stream)
            // 先按带前缀（64B name + 4B codec）解析；codec 字节非法则回退无前缀
            _ = try buffered.readExact(64)
            let codecBytes = try buffered.readExact(4)
            var codec = String(decoding: codecBytes, as: UTF8.self)
            if !codec.allSatisfy({ $0.isLetter || $0.isNumber }) {
                buffered.rewind(68)
                let raw = try buffered.readExact(4)
                codec = String(decoding: raw, as: UTF8.self)
            }
            delegate?.scrcpyOnAudioPrepare(codec: codec)
            while connected {
                let header = try buffered.readExact(12)
                let pts = header.loadBE64(at: 0)
                let isConfig = (header[header.startIndex] & 0x80) != 0
                let size = Int(header.loadBE32(at: 8))
                guard size > 0, size <= 256 * 1024 else { break }
                let payload = try buffered.readExact(size)
                if isConfig {
                    delegate?.scrcpyOnAudioConfig(payload)
                } else {
                    delegate?.scrcpyOnAudioFrame(payload)
                }
            }
        } catch {}
    }

    /// scrcpy 帧头：ptsAndFlags(8, BE；最高位=配置包) + size(4, BE)。
    private func readFrame(_ stream: AdbStream, bufferLimit: Int) throws -> (Data, Bool, UInt64)? {
        let header = try stream.readExact(12)
        let pts = header.loadBE64(at: 0)
        let isConfig = (header[header.startIndex] & 0x80) != 0
        let size = Int(header.loadBE32(at: 8))
        guard size > 0, size <= bufferLimit else { return nil }
        let payload = try stream.readExact(size)
        return (payload, isConfig, pts)
    }

    // MARK: - 资源

    public static func loadResource(_ name: String) -> String {
        let url = Bundle.module.url(forResource: name, withExtension: nil)!
        return try! String(contentsOf: url, encoding: .utf8)
    }

    public static func loadResourceData(_ name: String) throws -> Data {
        let url = Bundle.module.url(forResource: name, withExtension: nil)!
        return try Data(contentsOf: url)
    }

    /// 与 Android 端同一密钥对的 adb 公钥串（android_pubkey 结构 base64 + " gscp\0"）。
    /// 由 tools 脚本从 ca.key 生成；两端一致才能被已配对的眼镜/设备信任。
    public static let adbPublicKeyString =
        "QAAAADsQsycBAAEAy2hT6E+yWUGjNdAefMUtwi7BsbXV2BYE05aq+r/H9WsqMnxrieHfYWRO7lH97Pwtt5F+m+qgt6QSg3KKDuSxF1mDV+kVURi9CoqtwwhYeuIu/3gLAw8t/mpL4AACkln8hD41XMDL3ldRtHJFsM+AdEJrffUnzZXYO03htxXe8jRPafjTh2qQoV3dORUCtg1zr+U4WSnnlA9ZvFGdjPUXPF4R+ZOspwS0FJ0jspCzj5HaEgM9SXVjCYPnkhnt4RqcpcPhtQyJ1AaAPBiGXZ8exaXlfiqn4o1zoZcfQrlS0zOnBy6OhfApCUNNcdMWvJKmWaPLdbqxvt1cGxvKewq3DQ== gscp\0"
}

// MARK: - 字节序工具

public extension Data {
    func loadBE32(at offset: Int) -> UInt32 {
        let i = startIndex + offset
        return (UInt32(self[i]) << 24) | (UInt32(self[i + 1]) << 16)
            | (UInt32(self[i + 2]) << 8) | UInt32(self[i + 3])
    }

    func loadBE64(at offset: Int) -> UInt64 {
        var v: UInt64 = 0
        for k in 0..<8 { v = (v << 8) | UInt64(self[startIndex + offset + k]) }
        return v
    }
}

/// 支持回退（rewind）的流封装：audio 首包两种格式容错解析用。
final class BufferedStream {
    private let base: AdbStream
    private var buffer = Data()
    private var history = Data()

    init(_ base: AdbStream) { self.base = base }

    func readExact(_ count: Int) throws -> Data {
        while buffer.count < count {
            guard let chunk = try base.read() else { throw AdbStream.StreamError.closed }
            buffer.append(chunk)
        }
        let out = Data(buffer.prefix(count))
        history.append(out)
        if history.count > 256 { history.removeFirst(history.count - 256) }
        buffer.removeFirst(count)
        return out
    }

    /// 回退最近 count 字节（audio 容错只需回退一次 ≤68B）。
    func rewind(_ count: Int) {
        let n = min(count, history.count)
        buffer = history.suffix(n) + buffer
        history.removeLast(n)
    }
}
