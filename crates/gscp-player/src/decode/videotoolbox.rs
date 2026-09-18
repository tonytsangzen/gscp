//! macOS VideoToolbox 系统硬件解码。
//!
//! 直接 FFI 链接系统框架（CoreFoundation/CoreMedia/CoreVideo/VideoToolbox），
//! 无第三方依赖。输入：scrcpy config 包（annex-B）→ AVCC/HVCC extradata +
//! 长度前缀封包；输出：BGRA CVPixelBuffer → RGBA。

#![allow(non_snake_case, clippy::too_many_arguments)]

use anyhow::{Context, Result, bail};
use gscp_core::codec::{
    DecodedFrame, VideoDecoder, annexb_to_length_prefixed, build_extradata, rgba_frame,
};
use gscp_core::logbus;
use gscp_core::scrcpy::VideoCodec;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

type OSStatus = i32;
type CFAllocatorRef = *const std::ffi::c_void;
type CFStringRef = *const std::ffi::c_void;
type CFDictionaryRef = *const std::ffi::c_void;
type CFDataRef = *const std::ffi::c_void;
type CFNumberRef = *const std::ffi::c_void;
type CMFormatDescriptionRef = *const std::ffi::c_void;
type CMBlockBufferRef = *mut std::ffi::c_void;
type CMSampleBufferRef = *mut std::ffi::c_void;
type CVImageBufferRef = *mut std::ffi::c_void;
type VTDecompressionSessionRef = *mut std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

type VTDecompressionOutputCallback = extern "C" fn(
    decompressionOutputRefCon: *mut std::ffi::c_void,
    sourceFrameRefCon: *mut std::ffi::c_void,
    status: OSStatus,
    infoFlags: u32,
    imageBuffer: CVImageBufferRef,
    presentationTimeStamp: CMTime,
    presentationDuration: CMTime,
);

#[repr(C)]
struct VTDecompressionOutputCallbackRecord {
    decompressionOutputCallback: VTDecompressionOutputCallback,
    decompressionOutputRefCon: *mut std::ffi::c_void,
}

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT32_TYPE: isize = 3;
const K_CV_PIXEL_FORMAT_32_BGRA: u32 = 0x4247_5241; // 'BGRA'
const NO_ERR: OSStatus = 0;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFAllocatorGetDefault() -> CFAllocatorRef;
    fn CFStringCreateWithCString(
        alloc: CFAllocatorRef,
        c_str: *const std::os::raw::c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFDataCreate(alloc: CFAllocatorRef, bytes: *const u8, numBytes: isize) -> CFDataRef;
    fn CFNumberCreate(
        alloc: CFAllocatorRef,
        allocatorType: isize,
        valuePtr: *const std::ffi::c_void,
    ) -> CFNumberRef;
    fn CFDictionaryCreate(
        alloc: CFAllocatorRef,
        keys: *const *const std::ffi::c_void,
        values: *const *const std::ffi::c_void,
        numValues: isize,
        keyCallBacks: *const std::ffi::c_void,
        valueCallBacks: *const std::ffi::c_void,
    ) -> CFDictionaryRef;
    fn CFRelease(cf: *const std::ffi::c_void);
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    // 等价于 CMVideoFormatDescriptionCreateFromCMVideoCodecType
    // （后者在新 SDK 中是它的 inline 别名）
    fn CMVideoFormatDescriptionCreate(
        allocator: CFAllocatorRef,
        codecType: u32,
        width: i32,
        height: i32,
        extensions: CFDictionaryRef,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    fn CMBlockBufferCreateWithMemoryBlock(
        structureAllocator: CFAllocatorRef,
        memoryBlock: *mut std::ffi::c_void,
        blockLength: usize,
        blockAllocator: CFAllocatorRef,
        customBlockSource: *const std::ffi::c_void,
        offsetToData: usize,
        dataLength: usize,
        flags: u32,
        newCMBlockBufferOut: *mut CMBlockBufferRef,
    ) -> OSStatus;
    fn CMBlockBufferReplaceDataBytes(
        sourceBytes: *const std::ffi::c_void,
        destinationCMBlockBuffer: CMBlockBufferRef,
        offsetIntoDestination: usize,
        dataLength: usize,
    ) -> OSStatus;
    fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        dataBuffer: CMBlockBufferRef,
        formatDescription: CMFormatDescriptionRef,
        numSamples: isize,
        numSampleTimingEntries: isize,
        sampleTimingArray: *const std::ffi::c_void,
        numSampleSizeEntries: isize,
        sampleSizeArray: *const u32,
        sBufOut: *mut CMSampleBufferRef,
    ) -> OSStatus;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferLockBaseAddress(pixelBuffer: CVImageBufferRef, options: u32) -> OSStatus;
    fn CVPixelBufferUnlockBaseAddress(pixelBuffer: CVImageBufferRef, options: u32) -> OSStatus;
    fn CVPixelBufferGetBaseAddress(pixelBuffer: CVImageBufferRef) -> *mut std::ffi::c_void;
    fn CVPixelBufferGetBytesPerRow(pixelBuffer: CVImageBufferRef) -> usize;
    fn CVPixelBufferGetWidth(pixelBuffer: CVImageBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pixelBuffer: CVImageBufferRef) -> usize;
    fn CVPixelBufferRetain(pixelBuffer: CVImageBufferRef) -> CVImageBufferRef;
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        formatDescription: CMFormatDescriptionRef,
        videoDecoderSpecification: CFDictionaryRef,
        destinationImageBufferAttributes: CFDictionaryRef,
        outputCallback: *const VTDecompressionOutputCallbackRecord,
        decompressionSessionOut: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sampleBuffer: CMSampleBufferRef,
        decodeFlags: u32,
        infoFlagsOut: *mut u32,
    ) -> OSStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

// ── 回调状态 ──

struct CallbackState {
    /// (status, retained CVPixelBuffer)
    frames: Mutex<VecDeque<(OSStatus, CVImageBufferRef)>>,
}

extern "C" fn output_callback(
    ref_con: *mut std::ffi::c_void,
    _source_frame_ref_con: *mut std::ffi::c_void,
    status: OSStatus,
    _info_flags: u32,
    image_buffer: CVImageBufferRef,
    _pts: CMTime,
    _dts: CMTime,
) {
    if ref_con.is_null() {
        return;
    }
    let state = unsafe { &*(ref_con as *const CallbackState) };
    if status != NO_ERR || image_buffer.is_null() {
        state
            .frames
            .lock()
            .unwrap()
            .push_back((status, std::ptr::null_mut()));
        return;
    }
    let retained = unsafe { CVPixelBufferRetain(image_buffer) };
    state.frames.lock().unwrap().push_back((NO_ERR, retained));
}

unsafe fn cf_string(s: &str) -> CFStringRef {
    let c = std::ffi::CString::new(s).expect("cf string");
    CFStringCreateWithCString(CFAllocatorGetDefault(), c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
}

unsafe fn cf_dict(
    keys: &[CFStringRef],
    values: &[*const std::ffi::c_void],
) -> CFDictionaryRef {
    CFDictionaryCreate(
        CFAllocatorGetDefault(),
        keys.as_ptr() as *const *const std::ffi::c_void,
        values.as_ptr(),
        keys.len() as isize,
        std::ptr::null(),
        std::ptr::null(),
    )
}

fn cm_video_codec_type(codec: VideoCodec) -> u32 {
    match codec {
        VideoCodec::H264 => 0x6176_6331, // 'avc1'
        VideoCodec::H265 => 0x6876_6331, // 'hvc1'
        VideoCodec::Av1 => 0x6176_3031,  // 'av01'
    }
}

fn extradata_key(codec: VideoCodec) -> Option<&'static str> {
    match codec {
        VideoCodec::H264 => Some("avcC"),
        VideoCodec::H265 => Some("hvcC"),
        VideoCodec::Av1 => None,
    }
}

pub struct VideoToolboxDecoder {
    format_desc: CMFormatDescriptionRef,
    session: VTDecompressionSessionRef,
    callback_state: Arc<CallbackState>,
    /// Arc::into_raw 出来的 refCon，会话销毁时归还引用计数。
    ref_con: *mut std::ffi::c_void,
    seq: u64,
}

// 会话与回调只在解码线程上驱动；VT session 自身线程安全。
unsafe impl Send for VideoToolboxDecoder {}

impl VideoToolboxDecoder {
    /// 可用性探测（真正的会话在 configure() 建立）。
    pub fn new() -> Result<Self> {
        let _ = unsafe { CFAllocatorGetDefault() };
        Ok(Self {
            format_desc: std::ptr::null(),
            session: std::ptr::null_mut(),
            callback_state: Arc::new(CallbackState {
                frames: Mutex::new(VecDeque::new()),
            }),
            ref_con: std::ptr::null_mut(),
            seq: 0,
        })
    }

    fn teardown(&mut self) {
        unsafe {
            if !self.session.is_null() {
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session);
                self.session = std::ptr::null_mut();
            }
            if !self.format_desc.is_null() {
                CFRelease(self.format_desc);
                self.format_desc = std::ptr::null();
            }
        }
        self.callback_state.frames.lock().unwrap().clear();
    }

    fn create_session(
        &mut self,
        codec: VideoCodec,
        width: u32,
        height: u32,
        extradata: &[u8],
    ) -> Result<()> {
        self.teardown();
        unsafe {
            let codec_type = cm_video_codec_type(codec);

            // extensions: { "SampleDescriptionExtensionAtoms": { "avcC"/"hvcC": data } }
            let extensions: CFDictionaryRef = match extradata_key(codec) {
                Some(key) if !extradata.is_empty() => {
                    let data = CFDataCreate(
                        CFAllocatorGetDefault(),
                        extradata.as_ptr(),
                        extradata.len() as isize,
                    );
                    let atom_key = cf_string(key);
                    let atoms = cf_dict(&[atom_key], &[data as *const std::ffi::c_void]);
                    let container_key = cf_string("SampleDescriptionExtensionAtoms");
                    cf_dict(&[container_key], &[atoms as *const std::ffi::c_void])
                }
                _ => std::ptr::null(),
            };

            let mut format_desc: CMFormatDescriptionRef = std::ptr::null();
            let status = CMVideoFormatDescriptionCreate(
                CFAllocatorGetDefault(),
                codec_type,
                width as i32,
                height as i32,
                extensions,
                &mut format_desc,
            );
            if status != NO_ERR {
                bail!("创建 CMVideoFormatDescription 失败: {status}");
            }
            self.format_desc = format_desc;

            // decoderSpecification: NULL —— 由系统自动选择（Apple Silicon 默认硬解）
            let spec: CFDictionaryRef = std::ptr::null();

            // destination: BGRA 像素缓冲
            let pixel_format: u32 = K_CV_PIXEL_FORMAT_32_BGRA;
            let number = CFNumberCreate(
                CFAllocatorGetDefault(),
                K_CF_NUMBER_SINT32_TYPE,
                &pixel_format as *const u32 as *const std::ffi::c_void,
            );
            let format_key = cf_string("PixelFormatType");
            let dest_attrs = cf_dict(&[format_key], &[number as *const std::ffi::c_void]);

            // refCon 持有一份 Arc 引用（into_raw），Drop 时归还
            self.ref_con = Arc::into_raw(self.callback_state.clone()) as *mut std::ffi::c_void;
            let record = VTDecompressionOutputCallbackRecord {
                decompressionOutputCallback: output_callback,
                decompressionOutputRefCon: self.ref_con,
            };

            let mut session: VTDecompressionSessionRef = std::ptr::null_mut();
            let status = VTDecompressionSessionCreate(
                CFAllocatorGetDefault(),
                self.format_desc,
                spec,
                dest_attrs,
                &record,
                &mut session,
            );
            if status != NO_ERR {
                self.teardown();
                bail!("创建 VTDecompressionSession 失败: {status}（codec 可能不受支持）");
            }
            self.session = session;
        }
        Ok(())
    }

    /// 取回所有已解出的帧（回调队列 → RGBA 帧列表）。
    fn drain_output(&mut self, out: &mut Vec<Result<DecodedFrame>>) {
        let frames: VecDeque<(OSStatus, CVImageBufferRef)> =
            std::mem::take(&mut *self.callback_state.frames.lock().unwrap());
        for (status, pixel_buffer) in frames {
            if status != NO_ERR {
                out.push(Err(anyhow::anyhow!("VideoToolbox 解码错误: {status}")));
                continue;
            }
            if pixel_buffer.is_null() {
                continue;
            }
            match unsafe { copy_bgra_to_rgba(pixel_buffer) } {
                Ok((width, height, rgba)) => {
                    self.seq += 1;
                    out.push(rgba_frame(width, height, self.seq, rgba));
                }
                Err(err) => out.push(Err(err)),
            }
            unsafe { CFRelease(pixel_buffer) };
        }
    }
}

unsafe fn copy_bgra_to_rgba(pixel_buffer: CVImageBufferRef) -> Result<(u32, u32, Vec<u8>)> {
    if CVPixelBufferLockBaseAddress(pixel_buffer, 0) != NO_ERR {
        bail!("CVPixelBufferLockBaseAddress 失败");
    }
    let width = CVPixelBufferGetWidth(pixel_buffer);
    let height = CVPixelBufferGetHeight(pixel_buffer);
    let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
    let base = CVPixelBufferGetBaseAddress(pixel_buffer);
    if base.is_null() {
        CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
        bail!("CVPixelBufferGetBaseAddress 返回空");
    }

    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..height {
        let src = base.add(row * bytes_per_row) as *const u8;
        let dst = &mut rgba[row * width * 4..(row + 1) * width * 4];
        for px in 0..width {
            let b = *src.add(px * 4);
            let g = *src.add(px * 4 + 1);
            let r = *src.add(px * 4 + 2);
            let a = *src.add(px * 4 + 3);
            dst[px * 4] = r;
            dst[px * 4 + 1] = g;
            dst[px * 4 + 2] = b;
            dst[px * 4 + 3] = a;
        }
    }
    CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
    Ok((width as u32, height as u32, rgba))
}

impl VideoDecoder for VideoToolboxDecoder {
    fn configure(
        &mut self,
        codec: VideoCodec,
        config_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let extradata = build_extradata(codec, config_data);
        self.create_session(codec, width, height, &extradata)
            .context("VideoToolbox 会话创建失败")?;
        Ok(())
    }

    fn decode(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>> {
        if self.session.is_null() {
            bail!("VideoToolbox 会话未初始化（未收到 config 包）");
        }
        unsafe {
            let avcc = annexb_to_length_prefixed(packet);
            if avcc.is_empty() {
                return Ok(None);
            }

            let mut block_buffer: CMBlockBufferRef = std::ptr::null_mut();
            let status = CMBlockBufferCreateWithMemoryBlock(
                CFAllocatorGetDefault(),
                std::ptr::null_mut(),
                avcc.len(),
                CFAllocatorGetDefault(),
                std::ptr::null(),
                0,
                avcc.len(),
                0,
                &mut block_buffer,
            );
            if status != NO_ERR {
                bail!("创建 CMBlockBuffer 失败: {status}");
            }
            let status = CMBlockBufferReplaceDataBytes(
                avcc.as_ptr() as *const std::ffi::c_void,
                block_buffer,
                0,
                avcc.len(),
            );
            if status != NO_ERR {
                CFRelease(block_buffer);
                bail!("填充 CMBlockBuffer 失败: {status}");
            }

            let mut sample_buffer: CMSampleBufferRef = std::ptr::null_mut();
            let status = CMSampleBufferCreateReady(
                CFAllocatorGetDefault(),
                block_buffer,
                self.format_desc,
                1,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                &mut sample_buffer,
            );
            if status != NO_ERR {
                CFRelease(block_buffer);
                bail!("创建 CMSampleBuffer 失败: {status}");
            }

            let status = VTDecompressionSessionDecodeFrame(
                self.session,
                sample_buffer,
                0,
                std::ptr::null_mut(),
            );
            CFRelease(sample_buffer);
            CFRelease(block_buffer);
            if status != NO_ERR {
                // 个别包失败不中断会话
                logbus::emit(&format!(
                    "[decoder] VTDecompressionSessionDecodeFrame: {status}"
                ));
            }

            let mut results = Vec::new();
            self.drain_output(&mut results);
            results.into_iter().next().transpose()
        }
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        if self.session.is_null() {
            return Ok(vec![]);
        }
        unsafe {
            VTDecompressionSessionWaitForAsynchronousFrames(self.session);
        }
        let mut results = Vec::new();
        self.drain_output(&mut results);
        results.into_iter().collect()
    }

    fn name(&self) -> &'static str {
        "videotoolbox"
    }
}

impl Drop for VideoToolboxDecoder {
    fn drop(&mut self) {
        self.teardown();
        if !self.ref_con.is_null() {
            let shared = unsafe { Arc::from_raw(self.ref_con as *const CallbackState) };
            drop(shared);
            self.ref_con = std::ptr::null_mut();
        }
    }
}
