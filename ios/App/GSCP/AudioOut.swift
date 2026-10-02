import AVFoundation

/// 眼镜音频播放：Opus 解码出的 48kHz 交错 PCM16 → AVAudioEngine 播放
/// （能力对齐 Android AudioPlayer；iOS 侧解码用 GSCPKit.OpusAudioDecoder/libopus）。
final class AudioOut {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat?
    private var pending = 0
    private(set) var started = false
    private(set) var playedFrames = 0

    init(channels: Int) {
        format = AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: 48000,
            channels: AVAudioChannelCount(max(1, channels)), interleaved: true)
        engine.attach(player)
        if let format {
            engine.connect(player, to: engine.mainMixerNode, format: format)
        }
        do {
            try engine.start()
            player.play()
            started = true
        } catch {
            started = false
        }
    }

    /// 入队一帧解码后的交错 PCM16 数据（自动限流，防积压延迟）。
    func enqueue(pcm: Data) {
        guard started, let format, pcm.count > 0 else { return }
        guard pending < 40 else { return }   // >约0.8s 待播：丢弃，保实时
        let ch = Int(format.channelCount)
        let frames = pcm.count / (2 * ch)
        guard frames > 0,
              let buf = AVAudioPCMBuffer(pcmFormat: format,
                                         frameCapacity: AVAudioFrameCount(frames)) else { return }
        buf.frameLength = AVAudioFrameCount(frames)
        pcm.withUnsafeBytes { raw in
            memcpy(buf.int16ChannelData![0], raw.baseAddress, frames * 2 * ch)
        }
        pending += 1
        player.scheduleBuffer(buf) { [weak self] in
            DispatchQueue.main.async { self?.pending -= 1 }
        }
    }

    func stop() {
        player.stop()
        engine.stop()
        started = false
    }
}
