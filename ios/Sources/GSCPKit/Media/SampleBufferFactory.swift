import Foundation
import CoreMedia

/// annexB H.264 → 可直接 enqueue 给 AVSampleBufferDisplayLayer 的 CMSampleBuffer。
/// iOS App 的 overlay 上屏路径（GSCPKit 内共享，macOS 探针同样可编译验证）。
public enum SampleBufferFactory {
    public enum FactoryError: Error, LocalizedError {
        case missingParameterSets
        case createFailed(OSStatus)

        public var errorDescription: String? {
            switch self {
            case .missingParameterSets: return "缺少 SPS/PPS（尚未收到 config 帧）"
            case .createFailed(let s): return "CMSampleBuffer 创建失败 status=\(s)"
            }
        }
    }

    /// 从帧里提取参数集（config 帧返回 SPS/PPS；普通帧的参数集变化也会带出）。
    public static func parameterSets(
        from annexb: Data
    ) -> (sps: Data?, pps: Data?) {
        var sps: Data?
        var pps: Data?
        for nal in H264AnnexB.split(annexb) {
            switch nal.type {
            case 7: sps = nal.data
            case 8: pps = nal.data
            default: break
            }
        }
        return (sps, pps)
    }

    /// config 帧（SPS/PPS）→ CMVideoFormatDescription。
    public static func makeFormat(sps: Data, pps: Data) throws -> CMVideoFormatDescription {
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
                    allocator: nil, parameterSetCount: 2,
                    parameterSetPointers: &ptrs, parameterSetSizes: &sizes,
                    nalUnitHeaderLength: 4, formatDescriptionOut: &formatOut)
            }
        }
        guard status == 0, let f = formatOut else { throw FactoryError.createFailed(status) }
        return f
    }

    /// 普通帧（annexB）→ AVCC 封装 CMSampleBuffer（带 format，可入 AVSampleBufferDisplayLayer）。
    public static func makeSample(
        _ annexb: Data, format: CMVideoFormatDescription
    ) throws -> CMSampleBuffer {
        var payload: [H264AnnexB.NAL] = []
        for nal in H264AnnexB.split(annexb) {
            switch nal.type {
            case 7, 8: break          // 参数集由 format 承载
            default: payload.append(nal)
            }
        }
        guard !payload.isEmpty else { throw FactoryError.missingParameterSets }

        var avcc = Data()
        for nal in payload {
            var len = UInt32(nal.data.count).bigEndian
            withUnsafeBytes(of: &len) { avcc.append(contentsOf: $0) }
            avcc.append(nal.data)
        }

        var block: CMBlockBuffer?
        let cs = CMBlockBufferCreateWithMemoryBlock(
            allocator: nil, memoryBlock: nil, blockLength: avcc.count,
            blockAllocator: nil, customBlockSource: nil,
            offsetToData: 0, dataLength: avcc.count, flags: 0, blockBufferOut: &block)
        guard cs == kCMBlockBufferNoErr, let block else { throw FactoryError.createFailed(cs) }
        avcc.withUnsafeBytes { raw in
            _ = CMBlockBufferReplaceDataBytes(
                with: raw.baseAddress!, blockBuffer: block,
                offsetIntoDestination: 0, dataLength: avcc.count)
        }

        var sampleSize = avcc.count
        var sample: CMSampleBuffer?
        let ss = CMSampleBufferCreateReady(
            allocator: nil, dataBuffer: block, formatDescription: format,
            sampleCount: 1, sampleTimingEntryCount: 0, sampleTimingArray: nil,
            sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize, sampleBufferOut: &sample)
        guard ss == noErr, let sb = sample else { throw FactoryError.createFailed(ss) }
        return sb
    }
}
