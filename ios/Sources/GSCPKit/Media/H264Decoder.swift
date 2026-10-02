import Foundation
import VideoToolbox
import CoreMedia

/// H.264 硬解（VideoToolbox），能力对齐 Android 端 VideoDecoder（MediaCodec 硬解）：
///  - 输入：scrcpy 帧负载（annexB NAL 流；config 帧 = SPS/PPS，普通帧 = 视频数据）；
///  - 输出：解码回调给 CVPixelBuffer（probe 统计像素，App 侧直接上屏/合成）。
public final class H264Decoder {
    public enum DecodeError: Error, LocalizedError {
        case sessionCreateFailed(OSStatus)
        case formatCreateFailed(OSStatus)
        case blockBufferCreateFailed(OSStatus)

        public var errorDescription: String? {
            switch self {
            case .sessionCreateFailed(let s): return "VTDecompressionSession 创建失败 status=\(s)"
            case .formatCreateFailed(let s): return "CMVideoFormatDescription 创建失败 status=\(s)"
            case .blockBufferCreateFailed(let s): return "CMBlockBuffer 创建失败 status=\(s)"
            }
        }
    }

    private var session: VTDecompressionSession?
    private var format: CMVideoFormatDescription?
    private(set) public var width = 0
    private(set) public var height = 0
    private(set) public var decodedFrames = 0

    /// 解码输出回调（VT 解码线程；回调返回后 pixelBuffer 即失效）。
    public var pixelBufferHandler: ((CVPixelBuffer) -> Void)?

    private var sps: Data?
    private var pps: Data?

    public init() {}

    deinit {
        teardownSession()
    }

    /// 显式停止：销毁解码 session 并解除闭包对 self 的引用。
    public func stop() {
        teardownSession()
    }

    private func teardownSession() {
        if let session {
            VTDecompressionSessionInvalidate(session)
            self.session = nil
        }
    }

    private func frameDecoded(_ pb: CVPixelBuffer) {
        decodedFrames += 1
        pixelBufferHandler?(pb)
    }

    /// 喂一包 scrcpy 帧（config 帧或普通帧，annexB）。
    public func decode(_ data: Data, config: Bool) throws {
        let nals = H264AnnexB.split(data)
        if config {
            for nal in nals {
                switch nal.type {
                case 7: sps = nal.data
                case 8: pps = nal.data
                default: break
                }
            }
            try rebuildSessionIfParameterSetsChanged()
            return
        }
        guard session != nil, format != nil else { return }   // 等首个 config 帧

        var parameterChange = false
        var payloadNals: [H264AnnexB.NAL] = []
        for nal in nals {
            switch nal.type {
            case 7:
                if nal.data != sps { sps = nal.data; parameterChange = true }
            case 8:
                if nal.data != pps { pps = nal.data; parameterChange = true }
            default:
                payloadNals.append(nal)
            }
        }
        if parameterChange {
            try rebuildSessionIfParameterSetsChanged()
        }
        guard let format, let session, !payloadNals.isEmpty else { return }

        // annexB → AVCC（4B 大端长度前缀），VT 要求的封装
        var avcc = Data()
        for nal in payloadNals {
            var len = UInt32(nal.data.count).bigEndian
            withUnsafeBytes(of: &len) { avcc.append(contentsOf: $0) }
            avcc.append(nal.data)
        }

        var block: CMBlockBuffer?
        let createStatus = CMBlockBufferCreateWithMemoryBlock(
            allocator: nil, memoryBlock: nil, blockLength: avcc.count,
            blockAllocator: nil, customBlockSource: nil,
            offsetToData: 0, dataLength: avcc.count, flags: 0, blockBufferOut: &block)
        guard createStatus == kCMBlockBufferNoErr, let block else {
            throw DecodeError.blockBufferCreateFailed(createStatus)
        }
        let replaceStatus = avcc.withUnsafeBytes { raw -> OSStatus in
            CMBlockBufferReplaceDataBytes(
                with: raw.baseAddress!, blockBuffer: block,
                offsetIntoDestination: 0, dataLength: avcc.count)
        }
        guard replaceStatus == kCMBlockBufferNoErr else {
            throw DecodeError.blockBufferCreateFailed(replaceStatus)
        }

        var sampleSize = avcc.count
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReady(
            allocator: nil, dataBuffer: block, formatDescription: format,
            sampleCount: 1, sampleTimingEntryCount: 0, sampleTimingArray: nil,
            sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize, sampleBufferOut: &sampleBuffer)
        guard let sb = sampleBuffer else { return }

        // kVTDecodeFrame_EnableAsynchronousDecompression == 1 << 0
        let flags = VTDecodeFrameFlags(rawValue: 1 << 0)
        var infoFlags = VTDecodeInfoFlags()
        VTDecompressionSessionDecodeFrame(
            session, sampleBuffer: sb, flags: flags, infoFlagsOut: &infoFlags) { _, _, imageBuffer, _, _ in
                guard let imageBuffer else { return }
                self.frameDecoded(imageBuffer)
            }
    }

    // MARK: - session 管理

    private func rebuildSessionIfParameterSetsChanged() throws {
        guard let sps, let pps else { return }
        let (status, newFormat) = H264Decoder.createFormat(sps: sps, pps: pps)
        guard status == 0, let newFormat else {
            throw DecodeError.formatCreateFailed(status)
        }
        if let old = format, CMFormatDescriptionEqual(old, otherFormatDescription: newFormat),
           session != nil {
            return   // 参数未变
        }
        teardownSession()
        format = newFormat

        var sessionOut: VTDecompressionSession?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
        ]
        // 新 SDK：session 创建无回调参数，帧输出改由
        // VTDecompressionSessionDecodeFrameWithOutputHandler 的逐帧 handler 交付。
        let sstatus = VTDecompressionSessionCreate(
            allocator: nil,
            formatDescription: newFormat,
            decoderSpecification: nil,
            imageBufferAttributes: attrs as CFDictionary,
            decompressionSessionOut: &sessionOut)
        guard sstatus == 0, let s = sessionOut else {
            throw DecodeError.sessionCreateFailed(sstatus)
        }
        session = s
        let dims = CMVideoFormatDescriptionGetDimensions(newFormat)
        width = Int(dims.width)
        height = Int(dims.height)
    }

    private static func createFormat(sps: Data, pps: Data) -> (OSStatus, CMVideoFormatDescription?) {
        var formatOut: CMVideoFormatDescription?
        var status: OSStatus = -1
        sps.withUnsafeBytes { sp in
            pps.withUnsafeBytes { pp in
                var ptrs: [UnsafePointer<UInt8>] = [
                    sp.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pp.baseAddress!.assumingMemoryBound(to: UInt8.self),
                ]
                var sizes: [Int] = [sps.count, pps.count]
                status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    allocator: nil,
                    parameterSetCount: 2,
                    parameterSetPointers: &ptrs,
                    parameterSetSizes: &sizes,
                    nalUnitHeaderLength: 4,
                    formatDescriptionOut: &formatOut)
            }
        }
        return (status, formatOut)
    }
}

/// annexB NAL 解析（00 00 01 / 00 00 00 01 起始码）。
public enum H264AnnexB {
    public struct NAL {
        public let type: UInt8
        public let data: Data   // 含 NAL header 字节
    }

    public static func split(_ data: Data) -> [NAL] {
        let bytes = [UInt8](data)
        var marks: [Int] = []   // 每个起始码后的 NAL 数据起点
        var i = 0
        while i + 2 < bytes.count {
            if bytes[i] == 0, bytes[i + 1] == 0, bytes[i + 2] == 1 {
                marks.append(i + 3)
                i += 3
            } else {
                i += 1
            }
        }
        guard !marks.isEmpty else { return [] }
        var nals: [NAL] = []
        for (k, s) in marks.enumerated() {
            let e: Int
            if k + 1 < marks.count {
                // 下一 NAL 的边界 = 下一起始码模式起点（先退 00 00 01，再退其前导零）
                var end = marks[k + 1] - 3
                while end > s, bytes[end - 1] == 0 { end -= 1 }
                e = end
            } else {
                e = bytes.count
            }
            guard s < e else { continue }
            let payload = Data(bytes[s..<e])
            if !payload.isEmpty {
                nals.append(NAL(type: payload[0] & 0x1f, data: payload))
            }
        }
        return nals
    }
}
