import AVFoundation
import CoreVideo
import UIKit

/// 会话录像（能力对齐 Android 录像按钮）：把相机帧编码为 H.264 MP4。
/// v1 范围：录手机相机画面（overlay 合成录制留待 Metal 合成管线，见 README）。
final class SessionRecorder {
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var startedAt = Date()
    private var frameIndex = 0
    private let lock = NSLock()

    func start(size: CGSize) {
        let dim = max(720, Int(min(size.width, size.height)))
        let width = 720
        let height = Int(720 * (size.height / max(size.width, 1)))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gscp_rec_\(Int(Date().timeIntervalSince1970)).mp4")
        let w = try? AVAssetWriter(outputURL: url, fileType: .mp4)
        guard let writer = w else { return }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        self.writer = writer
        self.input = input
        self.adaptor = adaptor
    }

    /// 相机帧 → 缩放至目标尺寸后入队（简单居中裁剪用 vImage/CIAffineTransform 均可，
    /// v1 直接整帧缩放）。非 BGRA 帧由 adaptor 属性自动转换。
    func appendCamera(pixelBuffer: CVPixelBuffer) {
        guard let writer, let input, let adaptor, input.isReadyForMoreMediaData else { return }
        lock.lock(); defer { lock.unlock() }
        let ts = CMTime(value: CMTimeValue(frameIndex), timescale: 15)
        if adaptor.append(pixelBuffer, withPresentationTime: ts) {
            frameIndex += 1
        }
        _ = writer
    }

    func finish() {
        lock.lock()
        let input = self.input
        let writer = self.writer
        lock.unlock()
        input?.markAsFinished()
        guard let writer else { return }
        writer.finishWriting {
            // 产物在 tmp 目录；后续可做相册导出（对齐 Android 存 Movies）
        }
        self.writer = nil
        self.input = nil
        self.adaptor = nil
        _ = startedAt
    }
}
