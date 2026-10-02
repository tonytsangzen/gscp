import Foundation
import opus

/// Opus 解码（能力对齐 Android 端 AudioNative/nativeOpusDecode，底层同为 libopus）：
/// 解析 scrcpy 音频首帧的 OpusHead 配置 → 建 48kHz 解码器 → 包 → 交错的 16bit PCM。
public final class OpusAudioDecoder {
    public enum OpusError: Error, LocalizedError {
        case badOpusHead
        case createFailed(Int)
        case decodeFailed(Int)

        public var errorDescription: String? {
            switch self {
            case .badOpusHead: return "OpusHead 配置包非法"
            case .createFailed(let c): return "opus_decoder_create 失败 code=\(c)"
            case .decodeFailed(let c): return "opus_decode 失败 code=\(c)"
            }
        }
    }

    private var decoder: OpaquePointer?
    public private(set) var channels: Int32 = 2
    /// OpusHead 里的预跳过样本数（首帧输出应丢弃）。
    public private(set) var preSkip: Int32 = 0
    public private(set) var inputSampleRate: Int32 = 48000
    private var skipped = false

    public init() {}

    deinit {
        if let decoder { opus_decoder_destroy(decoder) }
    }

    /// 解析 scrcpy audio config 帧（OpusHead，19B）并创建解码器。
    /// OpusHead: "OpusHead"(8) version(1) ch(1) preskip(2 LE) rate(4 LE) gain(2 LE) mapping(1)
    public func configure(withOpusHead head: Data) throws {
        guard head.count >= 19,
              head.prefix(8) == Data("OpusHead".utf8) else {
            throw OpusError.badOpusHead
        }
        channels = Int32(head[head.startIndex + 9])
        preSkip = Int32(head.loadLE16(at: 10))
        inputSampleRate = Int32(head.loadLE32(at: 12))

        var err: Int32 = 0
        let dec = opus_decoder_create(48000, channels, &err)
        guard err == 0, let dec else { throw OpusError.createFailed(Int(err)) }
        if let decoder { opus_decoder_destroy(decoder) }
        decoder = dec
        skipped = false
    }

    /// 解码一包 → 交错 PCM16（48kHz）。
    /// - Returns: PCM 数据（空 Data = 未初始化/跳过）。
    public func decode(_ packet: Data) throws -> Data {
        guard let decoder else { return Data() }
        let frameCapacity = 5760 * Int(channels)   // 120ms 上限
        var out = [Int16](repeating: 0, count: frameCapacity)
        let n = packet.withUnsafeBytes { raw -> Int32 in
            opus_decode(decoder,
                        raw.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        Int32(packet.count),
                        &out, Int32(frameCapacity / Int(channels)), 0)
        }
        guard n >= 0 else { throw OpusError.decodeFailed(Int(n)) }
        var samples = Int(n) * Int(channels)

        // 丢弃 OpusHead preskip（仅在会话开头一次）
        if !skipped, preSkip > 0 {
            let skip = min(Int(preSkip) * Int(channels), samples)
            out.removeFirst(skip)
            samples -= skip
            skipped = true
        }
        return out.withUnsafeBytes { Data($0.prefix(samples * 2)) }
    }
}
