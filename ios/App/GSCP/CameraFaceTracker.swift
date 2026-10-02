import Foundation
import AVFoundation
import Vision
import CoreGraphics

/// 人脸锚点（语义对齐 Android 端 ncnn 管线的 anchor 用途：驱动 overlay 的
/// 位置/缩放/旋转。Android 用 det_10g+facemesh+hopenet 自研管线；iOS 端 v1
/// 直接采用系统 Vision（人脸框 + roll/yaw），追踪器不同、渲染语义一致。）
struct FaceAnchor {
    var center: CGPoint      // 归一化 [0,1]，view 坐标（y 向下）
    var size: CGFloat        // 归一化人脸宽（≈距离的倒数）
    var roll: CGFloat        // 弧度
    var yaw: CGFloat         // 弧度
    var valid: Bool
}

/// 前摄采集 + Vision 人脸追踪（对齐 Android ArActivity 前摄 AR 模式的输入侧）。
final class CameraFaceTracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "gscp-cam")
    private var lastProcess = TimeInterval(0)

    /// 人脸锚点更新（相机线程节流 ~15fps）。
    var onAnchor: ((FaceAnchor) -> Void)?
    /// 相机帧回调（录像合成用，主队列外）。
    var onCameraBuffer: ((CVPixelBuffer) -> Void)?

    private(set) var started = false

    func start() {
        guard !started else { return }
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)
        session.commitConfiguration()
        queue.async { [weak self] in
            self?.session.startRunning()
            self?.started = true
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.started else { return }
            self.session.stopRunning()
            self.started = false
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onCameraBuffer?(pb)

        // 节流 ~15fps 送 Vision
        let now = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard now - lastProcess >= 1.0 / 15.0 else { return }
        lastProcess = now

        let request = VNDetectFaceLandmarksRequest { [weak self] req, _ in
            guard let self else { return }
            guard let face = (req.results as? [VNFaceObservation])?.first else {
                self.onAnchor?(FaceAnchor(center: .zero, size: 0, roll: 0, yaw: 0, valid: false))
                return
            }
            let box = face.boundingBox          // 归一化，原点左下
            let center = CGPoint(x: box.midX, y: 1 - box.midY)   // 翻成 view 坐标
            let roll = CGFloat(face.roll?.doubleValue ?? 0)
            let yaw = CGFloat(face.yaw?.doubleValue ?? 0)
            self.onAnchor?(FaceAnchor(center: center, size: box.width,
                                      roll: roll, yaw: yaw, valid: true))
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .leftMirrored, options: [:])
        try? handler.perform([request])
    }
}
