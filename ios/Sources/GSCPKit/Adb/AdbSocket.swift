import Foundation

/// 阻塞式 TCP 字节流（POSIX socket）。读写在专用线程发生，线程安全由调用方保证：
/// 连接对象串行化写（NSLock），读仅发生在 reader 线程。
final class SocketIO {
    enum SocketError: Error, LocalizedError {
        case connectFailed(Int32)
        case readFailed(Int32)
        case writeFailed(Int32)
        case closed

        var errorDescription: String? {
            switch self {
            case .connectFailed(let e): return "adb TCP 连接失败 errno=\(e)"
            case .readFailed(let e): return "adb 读失败 errno=\(e)"
            case .writeFailed(let e): return "adb 写失败 errno=\(e)"
            case .closed: return "adb 连接已关闭"
            }
        }
    }

    private var fd: Int32 = -1
    private let fdLock = NSLock()

    func connect(host: String, port: UInt16, timeoutMs: Int) throws {
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else {
            throw SocketError.connectFailed(EINVAL)
        }

        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { throw SocketError.connectFailed(errno) }

        // 非阻塞 connect + select 超时
        let flags = fcntl(sock, F_GETFL, 0)
        _ = fcntl(sock, F_SETFL, flags | O_NONBLOCK)
        let connectResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(sock, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if connectResult != 0 {
            if errno == EINPROGRESS {
                var pfd = pollfd(fd: sock, events: Int16(POLLOUT), revents: 0)
                let rc = poll(&pfd, 1, Int32(timeoutMs))
                guard rc > 0 else {
                    Darwin.close(sock)
                    throw SocketError.connectFailed(ETIMEDOUT)
                }
                var err: Int32 = 0
                var len = socklen_t(MemoryLayout<Int32>.size)
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &len)
                guard err == 0 else {
                    Darwin.close(sock)
                    throw SocketError.connectFailed(err)
                }
            } else {
                Darwin.close(sock)
                throw SocketError.connectFailed(errno)
            }
        }
        _ = fcntl(sock, F_SETFL, flags)   // 恢复阻塞模式

        var noDelay: Int32 = 1
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &noDelay, socklen_t(MemoryLayout<Int32>.size))

        fdLock.lock()
        fd = sock
        fdLock.unlock()
    }

    func readExact(_ count: Int) throws -> Data {
        var out = Data(capacity: count)
        var remaining = count
        var buf = [UInt8](repeating: 0, count: max(count, 4096))
        while remaining > 0 {
            let want = min(remaining, buf.count)
            let n = buf.withUnsafeMutableBytes { raw -> Int in
                recv(fd, raw.baseAddress, want, 0)
            }
            if n > 0 {
                out.append(contentsOf: buf[0..<n])
                remaining -= n
            } else if n == 0 {
                throw SocketError.closed
            } else if errno == EINTR {
                continue
            } else {
                throw SocketError.readFailed(errno)
            }
        }
        return out
    }

    func write(_ data: Data) throws {
        fdLock.lock()
        let sock = fd
        fdLock.unlock()
        guard sock >= 0 else { throw SocketError.closed }
        var offset = 0
        let bytes = [UInt8](data)
        while offset < bytes.count {
            let n = bytes.withUnsafeBufferPointer { raw -> Int in
                send(sock, raw.baseAddress! + offset, bytes.count - offset, 0)
            }
            if n > 0 {
                offset += n
            } else if errno == EINTR {
                continue
            } else {
                throw SocketError.writeFailed(errno)
            }
        }
    }

    var isOpen: Bool {
        fdLock.lock(); defer { fdLock.unlock() }
        return fd >= 0
    }

    func closeSocket() {
        fdLock.lock()
        if fd >= 0 {
            Darwin.close(fd)
            fd = -1
        }
        fdLock.unlock()
    }
}
