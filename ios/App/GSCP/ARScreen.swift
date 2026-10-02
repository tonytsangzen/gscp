import SwiftUI
import AVFoundation
import UIKit

/// AR 会话屏（能力对齐 Android ArActivity 的前摄 AR 模式）：
/// 相机实景 + 眼镜 overlay 流（人脸锚点定位）+ 录像 + 参数设置。
struct ARScreen: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var session: GlassesSession
    @Environment(\.dismiss) private var dismiss

    @State private var showSettings = false
    @State private var recording = false
    @State private var statusText = ""

    var body: some View {
        ZStack {
            ARContainerView(session: session, settings: settings,
                            recording: $recording, statusText: $statusText)
                .ignoresSafeArea()

            VStack {
                Spacer()
                HStack(spacing: 48) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 26))
                            .foregroundColor(.white.opacity(0.85))
                            .padding(16)
                            .background(Circle().fill(.black.opacity(0.35)))
                    }
                    Button {
                        recording.toggle()
                    } label: {
                        Image(systemName: recording ? "stop.circle.fill" : "record.circle")
                            .font(.system(size: 44))
                            .foregroundColor(recording ? .red : .white)
                            .padding(12)
                            .background(Circle().fill(.black.opacity(0.35)))
                    }
                    Button {
                        session.disconnect()
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 26))
                            .foregroundColor(.white.opacity(0.85))
                            .padding(16)
                            .background(Circle().fill(.black.opacity(0.35)))
                    }
                }
                .padding(.bottom, 28)
            }
        }
        .statusBarHidden(true)
        .sheet(isPresented: $showSettings) {
            SettingsSheet()
                .environmentObject(settings)
        }
        .onAppear {
            // Android 端看门狗语义对齐：8s 无 overlay 帧 = 连接超时
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                if session.overlayFrames == 0, case .streaming = session.state {} else { return }
                statusText = "连接超时，未收到眼镜画面"
                session.disconnect()
                dismiss()
            }
        }
    }
}

/// 相机预览 + overlay 层的 UIKit 宿主（合成与锚点变换在此层完成）。
struct ARContainerView: UIViewRepresentable {
    let session: GlassesSession
    let settings: AppSettings
    @Binding var recording: Bool
    @Binding var statusText: String

    func makeUIView(context: Context) -> ARContainerUIView {
        let v = ARContainerUIView()
        v.attach(session: session, settings: settings)
        return v
    }

    func updateUIView(_ uiView: ARContainerUIView, context: Context) {
        uiView.applySettings(settings)
        uiView.setRecording(recording)
    }
}

/// 合成视图：底层相机预览，上层 overlay DisplayLayer 随人脸锚点变换。
final class ARContainerUIView: UIView {
    private weak var session: GlassesSession?
    private var tracker = CameraFaceTracker()
    private var recorder: SessionRecorder?
    private var settings: AppSettings?
    private var recording = false
    private var latestAnchor = FaceAnchor(center: .zero, size: 0, roll: 0, yaw: 0, valid: false)

    func attach(session: GlassesSession, settings: AppSettings) {
        self.session = session
        self.settings = settings

        // 相机预览层
        tracker.session.sessionPreset = .high
        let preview = AVCaptureVideoPreviewLayer(session: tracker.session)
        preview.videoGravity = .resizeAspectFill
        preview.frame = bounds
        preview.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        layer.addSublayer(preview)

        // overlay 层（置顶）
        let overlay = session.overlayPlayer.displayLayer
        overlay.videoGravity = .resizeAspect
        overlay.frame = bounds
        overlay.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        layer.addSublayer(overlay)

        // 人脸锚点 → overlay 变换
        tracker.onAnchor = { [weak self] anchor in
            DispatchQueue.main.async { self?.apply(anchor: anchor) }
        }
        // 相机帧 → 录像器
        tracker.onCameraBuffer = { [weak self] pb in
            self?.recorder?.appendCamera(pixelBuffer: pb)
        }
        tracker.start()
    }

    func applySettings(_ settings: AppSettings) {
        let overlay = session?.overlayPlayer.displayLayer
        overlay?.opacity = Float(settings.overlayAlphaPct) / 100
        apply(anchor: latestAnchor)   // 缩放参数变化立即生效
    }

    func setRecording(_ on: Bool) {
        guard recording != on else { return }
        recording = on
        if on {
            let rec = SessionRecorder()
            rec.start(size: bounds.size)
            recorder = rec
        } else {
            recorder?.finish()
            recorder = nil
        }
    }

    private func apply(anchor: FaceAnchor) {
        latestAnchor = anchor
        guard let player = session?.overlayPlayer else { return }
        let overlay = player.displayLayer
        let scalePct = CGFloat(settings?.overlayScalePct ?? 100)
        // 流宽高比（CALayer 无 width/height，用 player 记录的流信息）
        let streamAspect: CGFloat = {
            guard player.width > 0, player.height > 0 else { return 1.6 }
            return CGFloat(player.height) / CGFloat(player.width)
        }()
        if anchor.valid {
            // overlay 画在人脸上方（对齐 Android：overlay 画在人脸正前方，
            // 尺寸随人脸宽度 ≈ 距离）
            let faceW = anchor.size * bounds.width
            let targetW = faceW * 1.35 * (scalePct / 100)
            let targetH = targetW * streamAspect
            let center = CGPoint(x: anchor.center.x * bounds.width,
                                 y: anchor.center.y * bounds.height - targetH * 0.1)
            var t = CGAffineTransform.identity
            t = t.translatedBy(x: center.x, y: center.y)
            t = t.rotated(by: anchor.roll)
            t = t.translatedBy(x: -targetW / 2, y: -targetH / 2)
            overlay.bounds = CGRect(x: 0, y: 0, width: targetW, height: targetH)
            overlay.setAffineTransform(t)
            overlay.isHidden = false
        } else {
            // 无脸：居中全幅（与 Android 无脸时的兜底一致）
            overlay.bounds = bounds
            overlay.setAffineTransform(.identity)
            overlay.isHidden = false
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        apply(anchor: latestAnchor)
    }
}
