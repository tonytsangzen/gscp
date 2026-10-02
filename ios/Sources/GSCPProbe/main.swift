import Foundation
import CoreImage
import CoreMedia
import GSCPKit

// gscp-probe：iOS 端核心能力的 macOS 闭环探针。
// 跑通 App 将要执行的完整链路：adb 连 Android 设备（模拟器 adbd / 真机眼镜）→
// 推 scrcpy-server → reverse 隧道 → overlay/视频/音频流解析 → VideoToolbox 硬解。
// 用法: gscp-probe <host> [port] [seconds] [--full]
//   默认 overlay-only（AR 唯一模式，与 Android App HEAD 行为一致）；--full 走完整投屏。

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("用法: gscp-probe <host> [port] [seconds] [--full]")
    exit(2)
}
let host = args[1]
let port = UInt16(args.count > 2 ? Int(args[2]) ?? 5555 : 5555)
let seconds = args.count > 3 ? (Double(args[3]) ?? 15) : 15
let full = args.contains("--full")

final class ProbeDelegate: ScrcpyEventDelegate {
    let decoder = H264Decoder()
    let opus = OpusAudioDecoder()
    var pcmSamples = 0
    // App 上屏路径同款：SampleBufferFactory 构造 CMSampleBuffer（供 AVSampleBufferDisplayLayer）
    var sampleFormat: CMVideoFormatDescription?
    var sampleOk = 0
    var sampleFail = 0
    var overlayPackets = 0
    var overlayBytes = 0
    var videoPackets = 0
    var audioPackets = 0
    var overlayInfo: String = "-"
    var audioInfo: String = "-"
    var dumped = false
    let ciContext = CIContext()

    func scrcpyOnConnect() {
        print("[probe] adb connected")
    }

    func scrcpyOnVideoPrepare(codec: String, width: Int, height: Int) {
        print("[probe] video: \(codec) \(width)x\(height)")
    }

    func scrcpyOnVideoFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64) {
        videoPackets += 1
        try? decoder.decode(data, config: config)
    }

    func scrcpyOnAudioPrepare(codec: String) {
        audioInfo = codec
        print("[probe] audio codec: \(codec)")
    }

    func scrcpyOnAudioConfig(_ csd: Data) {
        audioInfo += " opusHead=\(csd.prefix(8).map { String(format: "%02x", $0) }.joined())"
        try? opus.configure(withOpusHead: csd)
    }

    func scrcpyOnAudioFrame(_ data: Data) {
        audioPackets += 1
        if let pcm = try? opus.decode(data) {
            pcmSamples += pcm.count / 2
        }
    }

    func scrcpyOnOverlayPrepare(codec: String, width: Int, height: Int) {
        overlayInfo = "\(codec) \(width)x\(height)"
        print("[probe] overlay: \(overlayInfo)")
    }

    func scrcpyOnOverlayFrame(_ data: Data, config: Bool, ptsAndFlags: UInt64) {
        overlayPackets += 1
        overlayBytes += data.count
        try? decoder.decode(data, config: config)
        // 验证 App 的 layer 上屏路径
        if config {
            let (sps, pps) = SampleBufferFactory.parameterSets(from: data)
            if let sps, let pps {
                sampleFormat = try? SampleBufferFactory.makeFormat(sps: sps, pps: pps)
            }
        } else if let format = sampleFormat {
            if (try? SampleBufferFactory.makeSample(data, format: format)) != nil {
                sampleOk += 1
            } else {
                sampleFail += 1
            }
        }
    }

    func scrcpyOnDisconnect() {
        print("[probe] server exited (session end)")
    }

    func scrcpyOnError(_ message: String) {
        print("[probe] error: \(message)")
    }

    func attachDecoder() {
        decoder.pixelBufferHandler = { [weak self] pb in
            guard let self else { return }
            if !self.dumped {
                self.dumped = true
                let w = CVPixelBufferGetWidth(pb)
                let h = CVPixelBufferGetHeight(pb)
                let ci = CIImage(cvPixelBuffer: pb)
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
                let out = URL(fileURLWithPath: "/tmp/gscp_ios_probe_frame.png")
                try? self.ciContext.writePNGRepresentation(
                    of: ci, to: out, format: .RGBA8, colorSpace: colorSpace)
                print("[probe] decoded frame \(w)x\(h) → \(out.path)")
            }
        }
    }
}

print("[probe] gscp iOS 核心闭环探针 — \(full ? "完整投屏" : "overlay-only") → \(host):\(port), \(seconds)s")

let delegate = ProbeDelegate()
delegate.attachDecoder()
let conn = ScrcpyConnection(options: .init(audioEnabled: true, fullScreenMirroring: full))
conn.delegate = delegate

do {
    try conn.connect(ip: host, port: port)
} catch {
    print("[probe] 连接失败: \(error)")
    exit(1)
}

// 周期统计
let start = Date()
var lastFrames = 0
var lastDecoded = 0
let ticker = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
    let f = delegate.overlayPackets + delegate.videoPackets
    let d = delegate.decoder.decodedFrames
    print(String(format: "[probe] %4.1fs | 流包 %d (+%d, %@) | 硬解 %d (+%d) | audio %d",
                 -start.timeIntervalSinceNow, f, f - lastFrames,
                 delegate.overlayInfo, d, d - lastDecoded, delegate.audioPackets))
    lastFrames = f
    lastDecoded = d
}
RunLoop.main.run(until: start.addingTimeInterval(seconds))
ticker.invalidate()

conn.disconnect()
decoderDeinitWait()

let decoded = delegate.decoder.decodedFrames
print("""
[probe] ===== 结果 =====
  overlay 流:   \(delegate.overlayInfo), \(delegate.overlayPackets) 包 / \(delegate.overlayBytes) B
  video 流:     \(delegate.videoPackets) 包
  audio 流:     \(delegate.audioInfo), \(delegate.audioPackets) 包
  VideoToolbox: \(decoded) 帧硬解
  Layer 路径:   \(delegate.sampleOk) 个 CMSampleBuffer 构造成功（失败 \(delegate.sampleFail)）
  Opus 解码:    \(delegate.pcmSamples) 个 PCM 采样（48kHz 交错 PCM16）
""")
exit(decoded > 0 ? 0 : 3)

func decoderDeinitWait() {
    Thread.sleep(forTimeInterval: 0.3)
}
