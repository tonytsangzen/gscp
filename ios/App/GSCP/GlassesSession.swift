import Foundation
import Combine

/// 会话状态（对齐 Android ArActivity 的 recOverlay 状态文案语义）。
enum GlassesState: Equatable {
    case idle
    case connecting
    case streaming(codec: String, width: Int, height: Int)
    case error(String)
}

/// 眼镜会话：包装 GSCPKit.ScrcpyConnection，把事件汇总成主线程可观察状态。
/// AR 唯一连接模式（overlay-only），与 Android HEAD 行为一致。
@MainActor
final class GlassesSession: ObservableObject {
    @Published private(set) var state: GlassesState = .idle
    @Published private(set) var overlayFrames = 0
    @Published private(set) var audioPackets = 0
    /// v1 说明：iOS 无系统 Opus 解码器，音频流已接收但暂不播放（见 ios/README.md）。
    @Published private(set) var audioNote: String = ""

    let overlayPlayer = OverlayPlayer()
    private var connection: ScrcpyConnection?
    private var settings: AppSettings?
    /// Opus 解码 + 播放（按需创建：拿到 OpusHead 后才知道声道数）
    private let audioDecoder = OpusAudioDecoder()
    private var audioOut: AudioOut?

    func connect(ip: String, port: UInt16 = 5555, settings: AppSettings) {
        guard state != .connecting, stateIsNotStreaming else { return }
        self.settings = settings
        state = .connecting
        overlayPlayer.reset()
        audioOut = AudioOut(channels: 2)   // OpusHead 到达后如声道不同由解码侧容错
        if let audioOut, !audioOut.started {
            audioOut = nil                 // 引擎启动失败（如模拟器/无音频设备）→ 静默降级
        }
        audioNote = audioOut == nil ? "" : ""

        let options = ScrcpyConnection.Options(
            audioEnabled: settings.audioEnabled,
            fullScreenMirroring: false)   // AR 唯一模式
        let conn = ScrcpyConnection(options: options)
        connection = conn
        let player = overlayPlayer
        let session = self
        let opus = audioDecoder
        let audioOutBox = audioOut

        // ScrcpyEventDelegate 的回调在流线程；这里手动切主线程更新 UI 状态
        final class DelegateBox: ScrcpyEventDelegate {
            weak var session: GlassesSession?
            let player: OverlayPlayer
            let opus: OpusAudioDecoder
            let audioOutBox: AudioOut?
            init(session: GlassesSession?, player: OverlayPlayer,
                 opus: OpusAudioDecoder, audioOutBox: AudioOut?) {
                self.session = session
                self.player = player
                self.opus = opus
                self.audioOutBox = audioOutBox
            }

            func scrcpyOnConnect() {
                Task { @MainActor in session?.state = .connecting }
            }

            func scrcpyOnVideoPrepare(codec: String, width: Int, height: Int) {}
            func scrcpyOnVideoFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64) {}

            func scrcpyOnAudioPrepare(codec: String) {
                Task { @MainActor in
                    session?.audioNote = "\(codec)"
                }
            }

            func scrcpyOnAudioConfig(_ csd: Data) {
                try? opus.configure(withOpusHead: csd)
            }

            func scrcpyOnAudioFrame(_ data: Data) {
                let pcm = try? opus.decode(data)
                if let pcm, let out = audioOutBox {
                    out.enqueue(pcm: pcm)
                }
                Task { @MainActor in session?.audioPackets += 1 }
            }

            func scrcpyOnOverlayPrepare(codec: String, width: Int, height: Int) {
                player.setStreamInfo(codec: codec, width: width, height: height)
                Task { @MainActor in
                    session?.state = .streaming(codec: codec, width: width, height: height)
                }
            }

            func scrcpyOnOverlayFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64) {
                player.handleFrame(data, config: config)
                Task { @MainActor in session?.overlayFrames += 1 }
            }

            func scrcpyOnDisconnect() {
                audioOutBox?.stop()
                Task { @MainActor in
                    guard let s = session, s.state != .idle else { return }
                    s.state = .idle
                }
            }

            func scrcpyOnError(_ message: String) {
                Task { @MainActor in
                    session?.state = .error("连接失败：\(message)")
                }
            }
        }
        conn.delegate = DelegateBox(session: session, player: player,
                                    opus: opus, audioOutBox: audioOutBox)
        overlayPlayer.onFirstFrame = { [weak self] in
            // 首帧即视为流 established（与 Android 看门狗语义一致：8s 无帧 = 超时）
        }

        let ip = ip.trimmingCharacters(in: .whitespaces)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                try conn.connect(ip: ip, port: port)
            } catch {
                Task { @MainActor [weak self] in
                    self?.state = .error("连接失败：\(error.localizedDescription)")
                }
            }
        }
    }

    private var stateIsNotStreaming: Bool {
        if case .streaming = state { return false }
        return true
    }

    func disconnect() {
        connection?.disconnect()
        connection = nil
        audioOut?.stop()
        audioOut = nil
        state = .idle
    }
}
