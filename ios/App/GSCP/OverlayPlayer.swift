import Foundation
import CoreMedia
import AVFoundation

/// overlay 流播放器：scrcpy overlay 帧 → AVSampleBufferDisplayLayer 硬解上屏。
/// （能力对齐 Android GlassesPlayer 的 overlay 解码渲染路径；iOS 用系统
///  AVSampleBufferDisplayLayer 免自管解码器生命周期。）
final class OverlayPlayer: NSObject {
    let displayLayer = AVSampleBufferDisplayLayer()
    private var format: CMVideoFormatDescription?
    private(set) var codec = ""
    private(set) var width = 0
    private(set) var height = 0
    private(set) var frameCount = 0
    var onFirstFrame: (() -> Void)?
    private var notifiedFirst = false

    override init() {
        super.init()
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.black.cgColor
    }

    /// scrcpy 帧事件（任意线程调用）。
    func handleFrame(_ data: Data, config: Bool) {
        if config {
            let (sps, pps) = SampleBufferFactory.parameterSets(from: data)
            if let sps, let pps, let f = try? SampleBufferFactory.makeFormat(sps: sps, pps: pps) {
                format = f
            }
            return
        }
        guard let format else { return }
        if let sb = try? SampleBufferFactory.makeSample(data, format: format) {
            frameCount += 1
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.displayLayer.enqueue(sb)
                if !self.notifiedFirst {
                    self.notifiedFirst = true
                    self.onFirstFrame?()
                }
            }
        }
    }

    func setStreamInfo(codec: String, width: Int, height: Int) {
        self.codec = codec
        self.width = width
        self.height = height
    }

    func reset() {
        format = nil
        frameCount = 0
        notifiedFirst = false
        DispatchQueue.main.async { [weak self] in
            self?.displayLayer.flush()
        }
    }
}
