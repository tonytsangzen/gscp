import Foundation

/// adb 线协议消息帧（对齐 AOSP adb 协议 / muntashirakon/adb AdbProtocol）：
/// 24 字节头 = command(4) arg0(4) arg1(4) dataLen(4) crc32(4) magic(4)，均小端。
public enum AdbCommand {
    public static let sync: UInt32 = 0x434e5953   // "SYNC"
    public static let cnxn: UInt32 = 0x4e584e43   // "CNXN"
    public static let auth:  UInt32 = 0x48545541  // "AUTH"
    public static let open:  UInt32 = 0x4e45504f  // "OPEN"
    public static let okay:  UInt32 = 0x59414b4f  // "OKAY"
    public static let clse:  UInt32 = 0x45534c43  // "CLSE"
    public static let wrte:  UInt32 = 0x45545257  // "WRTE"
}

enum AdbAuthType {
    static let token: UInt32 = 1
    static let signature: UInt32 = 2
    static let rsaPublicKey: UInt32 = 3
}

public struct AdbMessage {
    public var command: UInt32
    public var arg0: UInt32
    public var arg1: UInt32
    public var payload: Data

    /// 协议版本 0x01000001（CHECKSUM 免校验起始于该版本，crc 字段恒 0 也可）。
    public static let protocolVersion: UInt32 = 0x01000001
    /// v2 最大载荷 256KB。
    public static let maxPayload: UInt32 = 262144
    public static let headerLength = 24

    public init(command: UInt32, arg0: UInt32, arg1: UInt32, payload: Data = Data()) {
        self.command = command
        self.arg0 = arg0
        self.arg1 = arg1
        self.payload = payload
    }

    /// magic = command ^ 0xffffffff
    var magic: UInt32 { command ^ 0xffffffff }

    public func encoded() -> Data {
        var out = Data(capacity: AdbMessage.headerLength + payload.count)
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        le32(command); le32(arg0); le32(arg1)
        le32(UInt32(payload.count))
        le32(AdbMessage.crc32(payload))   // 新版本 adbd 不校验，保持语义正确
        le32(magic)
        out.append(payload)
        return out
    }

    public static func crc32(_ data: Data) -> UInt32 {
        // 标准 CRC-32/IEEE（adb 沿用 zlib crc32）。
        if table.isEmpty { buildTable() }
        var crc: UInt32 = 0xffffffff
        for byte in data {
            crc = table[Int((crc ^ UInt32(byte)) & 0xff)] ^ (crc >> 8)
        }
        return crc ^ 0xffffffff
    }

    private static var table: [UInt32] = []
    private static func buildTable() {
        var t = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (0xedb88320 ^ (c >> 1)) : (c >> 1)
            }
            t[i] = c
        }
        table = t
    }
}

/// 从 socket 字节流解析一条消息（阻塞读满 24B 头 + dataLen 载荷）。
enum AdbMessageParser {
    enum ParseError: Error {
        case connectionClosed
        case badMagic(UInt32)
    }

    static func parse(from io: SocketIO) throws -> AdbMessage {
        let header = try io.readExact(AdbMessage.headerLength)
        func le32(_ off: Int) -> UInt32 {
            header.loadLE32(at: off)
        }
        let command = le32(0)
        let dataLen = Int(le32(12))
        guard header.loadLE32(at: 20) == (command ^ 0xffffffff) else {
            throw ParseError.badMagic(le32(20))
        }
        let payload = dataLen > 0 ? try io.readExact(dataLen) : Data()
        return AdbMessage(command: command, arg0: le32(4), arg1: le32(8), payload: payload)
    }
}

public extension Data {
    func loadLE16(at offset: Int) -> UInt16 {
        let i = startIndex + offset
        return UInt16(self[i]) | (UInt16(self[i + 1]) << 8)
    }

    func loadLE32(at offset: Int) -> UInt32 {
        let i = startIndex + offset
        return UInt32(self[i])
            | (UInt32(self[i + 1]) << 8)
            | (UInt32(self[i + 2]) << 16)
            | (UInt32(self[i + 3]) << 24)
    }
}
