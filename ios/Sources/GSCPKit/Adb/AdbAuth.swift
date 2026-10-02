import Foundation
import Security

/// adb 认证：加载 PKCS8 RSA 私钥（与 Android 端 assets/ca.key 同一份）→
/// SecKey 签 AUTH TOKEN；被要求公钥时回预生成的 adb 公钥串。
/// 公钥串按 AOSP android_pubkey 结构（u32 len/n0inv/exponent + 大端 256B 模数）
/// base64 + " gscp\0" 生成，与 Android 端 libadb AndroidPubkey.encodeWithName 一致。
public final class AdbAuth {
    public enum AuthError: Error, LocalizedError {
        case invalidKey
        case signingFailed

        public var errorDescription: String? {
            switch self {
            case .invalidKey: return "ca.key 解析失败（期望 base64 PKCS#8 RSA）"
            case .signingFailed: return "AUTH TOKEN 签名失败"
            }
        }
    }

    public let deviceName: String
    private let privateKey: SecKey

    /// ca.key（base64 PKCS#8）+ ca.crt 内容；pubKeyString 为预生成的 adb 公钥串。
    public init(keyBase64: String, deviceName: String = "gscp", pubKeyString: String) throws {
        self.deviceName = deviceName
        guard let der = Data(base64Encoded: keyBase64.filter { !$0.isWhitespace }) else {
            throw AuthError.invalidKey
        }
        // ca.key 实测为 PKCS#1 RSAPrivateKey DER（Android 端 KeyFactory 宽容解析）；
        // 兼容 PKCS#8 输入（PrivateKeyInfo）：探测 version 后的标签区分两者。
        let pkcs1 = try AdbAuth.toPkcs1(der)
        var err: Unmanaged<CFError>?
        guard let key = SecKeyCreateWithData(pkcs1 as CFData, [
            kSecAttrKeyType: kSecAttrKeyTypeRSA,
            kSecAttrKeyClass: kSecAttrKeyClassPrivate,
        ] as CFDictionary, &err) else {
            throw err?.takeRetainedValue() ?? AuthError.invalidKey as Error
        }
        self.privateKey = key
        Self.pubKey = pubKeyString
    }

    private static var pubKey: String = ""

    /// 对 20B TOKEN 做 RSASSA-PKCS1-v1_5(SHA-1) 签名（adb key.cpp RSA_sign 语义）。
    public func sign(token: Data) throws -> Data {
        var err: Unmanaged<CFError>?
        guard let sig = SecKeyCreateSignature(
            privateKey,
            .rsaSignatureMessagePKCS1v15SHA1,
            token as CFData, &err
        ) else {
            throw err?.takeRetainedValue() ?? AuthError.signingFailed as Error
        }
        return sig as Data
    }

    /// RSAPUBLICKEY 应答负载：base64(结构体) + " " + 设备名 + "\0"。
    public func publicKeyPayload() -> Data {
        Data((Self.pubKey).utf8)
    }

    // MARK: - PKCS#8 → PKCS#1

    /// PKCS#1 / PKCS#8 自动识别 → 统一返回 PKCS#1 RSAPrivateKey DER（SecKey 原生格式）。
    public static func toPkcs1(_ der: Data) throws -> Data {
        var reader = DerReader(der)
        let outer = try reader.readSequence()          // PKCS#8 PrivateKeyInfo / PKCS#1 内容
        var inner = DerReader(outer)
        _ = try inner.readInteger()                    // version
        if try inner.peekTag() == 0x30 {               // PKCS#8：AlgorithmIdentifier SEQUENCE
            _ = try inner.readSequence()
            return try inner.readOctetString()         // OCTET STRING 内即 PKCS#1
        }
        return der                                     // PKCS#1：完整 SEQUENCE 即 SecKey 所需格式
    }

    struct DerReader {
        let bytes: [UInt8]
        var pos = 0

        init(_ data: Data) { bytes = [UInt8](data) }
        init(_ slice: ArraySlice<UInt8>) { bytes = Array(slice) }

        mutating func readByte() throws -> UInt8 {
            guard pos < bytes.count else { throw AuthError.invalidKey }
            defer { pos += 1 }
            return bytes[pos]
        }

        mutating func readLength() throws -> Int {
            let first = try readByte()
            if first & 0x80 == 0 { return Int(first) }
            let count = Int(first & 0x7f)
            var len = 0
            for _ in 0..<count { len = (len << 8) | Int(try readByte()) }
            return len
        }

        mutating func readTLV() throws -> (tag: UInt8, content: ArraySlice<UInt8>) {
            let tag = try readByte()
            let len = try readLength()
            guard pos + len <= bytes.count else { throw AuthError.invalidKey }
            let content = bytes[pos..<pos + len]
            pos += len
            return (tag, content)
        }

        mutating func readSequence() throws -> ArraySlice<UInt8> {
            let tag = try readByte()
            guard tag == 0x30 else { throw AuthError.invalidKey }
            let len = try readLength()
            guard pos + len <= bytes.count else { throw AuthError.invalidKey }
            let content = bytes[pos..<pos + len]
            pos += len
            return content
        }

        mutating func readOctetString() throws -> Data {
            let tag = try readByte()
            guard tag == 0x04 else { throw AuthError.invalidKey }
            let len = try readLength()
            guard pos + len <= bytes.count else { throw AuthError.invalidKey }
            let content = Data(bytes[pos..<pos + len])
            pos += len
            return content
        }

        mutating func readInteger() throws -> Int {
            let tag = try readByte()
            guard tag == 0x02 else { throw AuthError.invalidKey }
            return try readLength()
        }

        mutating func peekTag() throws -> UInt8 {
            defer { pos -= 1 }
            return try readByte()
        }
    }
}
