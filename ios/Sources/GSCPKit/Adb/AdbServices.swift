import Foundation

/// adb 高层服务（对齐 Android 端 Adb.kt 的 run/push/reverse 语义）。
public enum AdbServices {
    /// 执行 shell 命令，收集输出（到流关闭为止）。
    public static func shell(_ conn: AdbConnection, _ command: String) throws -> String {
        let stream = try conn.openStream("shell:\(command)")
        var out = Data()
        while let chunk = try stream.read() {
            out.append(chunk)
        }
        return String(decoding: out, as: UTF8.self)
    }

    /// adb push（sync 协议）：SEND/DATA/DONE，读 OKAY/FAIL 应答。
    /// 帧格式对齐 Android 端 Adb.kt：cmd(4) + u32LE len + payload。
    public static func push(_ conn: AdbConnection, data: Data, to remotePath: String) throws {
        let stream = try conn.openStream("sync:")
        defer { stream.closeAndWriteEOF() }

        func frame(_ cmd: String, param: UInt32? = nil, payload: Data = Data()) -> Data {
            var f = Data()
            f.append(cmd.data(using: .ascii)!)
            var le = param ?? UInt32(payload.count)
            withUnsafeBytes(of: le.littleEndian) { f.append(contentsOf: $0) }
            f.append(payload)
            return f
        }

        // SEND: "path,mode"（33188 = 0100644）
        let sendArg = "\(remotePath),33188"
        try stream.write(frame("SEND", payload: Data(sendArg.utf8)))
        // DATA 分片（≤60KB，与 Android 端一致）
        var offset = 0
        let bytes = [UInt8](data)
        while offset < bytes.count {
            let end = min(offset + 65500, bytes.count)
            try stream.write(frame("DATA", payload: Data(bytes[offset..<end])))
            offset = end
        }
        // DONE: u32 mtime
        try stream.write(frame("DONE", param: UInt32(Date().timeIntervalSince1970)))

        // 应答：OKAY(4B) 或 FAIL(4B)+u32 len+msg
        let status = try stream.readExact(4)
        if status == Data("OKAY".utf8) { return }
        if status == Data("FAIL".utf8) {
            let lenData = try stream.readExact(4)
            let len = Int(lenData.loadLE32(at: 0))
            let msg = len > 0 ? try stream.readExact(len) : Data()
            throw AdbConnection.AdbError.serviceFailed("push: \(String(decoding: msg, as: UTF8.self))")
        }
        throw AdbConnection.AdbError.serviceFailed("push: bad status \(String(decoding: status, as: UTF8.self))")
    }

    /// 注册 reverse 隧道：设备侧 localabstract:<sock> → 客户端侧 tcp:<port>。
    /// adbd 对每条设备侧连接主动 OPEN 回客户端，由 AdbConnection.deviceStreamHandler 分发。
    public static func reverse(_ conn: AdbConnection, deviceSocket: String, clientPort: UInt16) throws {
        let stream = try conn.openStream("reverse:forward:localabstract:\(deviceSocket);tcp:\(clientPort)")
        defer { stream.closeAndWriteEOF() }
        let resp = try stream.readExact(4)
        guard resp == Data("OKAY".utf8) else {
            throw AdbConnection.AdbError.serviceFailed(
                "reverse: \(String(decoding: resp, as: UTF8.self))")
        }
    }

    /// 读取设备属性（getprop）。
    public static func getProperty(_ conn: AdbConnection, _ name: String) throws -> String {
        let out = try shell(conn, "getprop \(name)")
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
