//! Windows 硬件解码：Media Foundation H.264 MFT（系统内置，无第三方依赖）。
//!
//! - `new()` 完成 COM 初始化、MFT 枚举与输入类型设置（失败即回退软解）；
//! - `configure()` 收到流尺寸后设置 NV12 输出与流消息；
//! - 优先挂 D3D11 设备管理器走 DXVA 硬解，输出视频内存 NV12，经 staging 纹理
//!   拷回；D3D 不可用时 MFT 按系统内存 NV12 输出（OS 自带优化解码路径）。
//!
//! NV12 → RGBA 转换口径与 [`crate::decode::software::yuv420_to_rgba`] 一致
//! （BT.601 limited range + 色度双线性），保证跨平台渲染效果一致。

use anyhow::{anyhow, bail, Context, Result};
use gscp_core::codec::{rgba_frame, DecodedFrame, VideoCodec, VideoDecoder};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D,
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_FLAG, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED};

/// RPC_E_CHANGED_MODE：线程已是其他 COM 单元，可继续使用。
const RPC_E_CHANGED_MODE: windows::core::HRESULT =
    windows::core::HRESULT(0x8001_0106_u32 as i32);

pub struct MediaFoundationDecoder {
    mft: Option<IMFTransform>,
    session: Option<Session>,
    pending: std::collections::VecDeque<DecodedFrame>,
    com_inited: bool,
    com_thread: Option<std::thread::ThreadId>,
    seq: u64,
    pts_tick: i64,
}

struct Session {
    mft: IMFTransform,
    out_stream_id: u32,
    out_info: MFT_OUTPUT_STREAM_INFO,
    /// true：MFT 不自备输出样本，必须每次 ProcessOutput 前分配。
    provide_output: bool,
    out_type: IMFMediaType,
    width: u32,
    height: u32,
    d3d: Option<D3dCtx>,
}

struct D3dCtx {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    _manager: IMFDXGIDeviceManager,
    staging: Option<(ID3D11Texture2D, u32, u32, u32)>,
}

impl MediaFoundationDecoder {
    pub fn new() -> Result<Self> {
        // 解码线程 MTA；线程已初始化为其他单元时忽略 RPC_E_CHANGED_MODE
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        let (com_inited, com_thread) = if hr.is_ok() {
            (true, Some(std::thread::current().id()))
        } else if hr == RPC_E_CHANGED_MODE {
            (false, None)
        } else {
            anyhow::bail!("CoInitializeEx 失败: {hr:?}");
        };

        let mft = create_h264_mft().context("枚举 H.264 解码 MFT 失败")?;
        let in_type = unsafe { MFCreateMediaType()? };
        unsafe {
            in_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            in_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)?;
            mft.SetInputType(0, &in_type, 0)?;
        }
        gscp_core::logbus::emit("[decoder] Windows Media Foundation 解码器已就绪");
        Ok(Self {
            mft: Some(mft),
            session: None,
            pending: Default::default(),
            com_inited,
            com_thread,
            seq: 0,
            pts_tick: 0,
        })
    }

    fn feed_and_drain(&mut self, packet: &[u8]) -> Result<()> {
        let sample = make_sample(packet, self.pts_tick * 1_000)?;
        self.pts_tick += 1;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("Media Foundation 会话未初始化"))?;
        if let Err(err) = unsafe { session.mft.ProcessInput(0, &sample, 0) } {
            if err.code() == MF_E_NOTACCEPTING {
                // 先排空已产出帧再重试一次
                let mut drained = Vec::new();
                let mut seq = self.seq;
                Self::drain_into(session, &mut seq, &mut drained)?;
                self.seq = seq;
                self.pending.extend(drained);
                unsafe {
                    session
                        .mft
                        .ProcessInput(0, &sample, 0)
                        .context("ProcessInput 重试失败")?;
                }
            } else {
                return Err(err).context("ProcessInput 失败");
            }
        }
        let mut drained = Vec::new();
        let mut seq = self.seq;
        Self::drain_into(session, &mut seq, &mut drained)?;
        self.seq = seq;
        self.pending.extend(drained);
        Ok(())
    }

    fn drain_into(session: &mut Session, seq: &mut u64, out: &mut Vec<DecodedFrame>) -> Result<()> {
        let mut idle = 0_u32;
        loop {
            let mut buffer = MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: session.out_stream_id,
                pSample: std::mem::ManuallyDrop::new(None),
                dwStatus: 0,
                pEvents: std::mem::ManuallyDrop::new(None),
            };
            if session.provide_output {
                buffer.pSample =
                    std::mem::ManuallyDrop::new(Some(unsafe { make_output_sample(session)? }));
            }
            let mut status = 0_u32;
            match unsafe {
                session
                    .mft
                    .ProcessOutput(0, std::slice::from_mut(&mut buffer), &mut status)
            } {
                Ok(()) => {
                    if buffer.dwStatus & MFT_OUTPUT_DATA_BUFFER_FORMAT_CHANGE.0 as u32 != 0 {
                        unsafe { session.mft.SetOutputType(0, &session.out_type, 0)? };
                        continue;
                    }
                    let sample =
                        unsafe { std::mem::ManuallyDrop::take(&mut buffer.pSample) };
                    match sample {
                        Some(sample) => match extract_frame(session, &sample)? {
                            Some((width, height, rgba)) => {
                                idle = 0;
                                out.push(rgba_frame(width, height, *seq, rgba)?);
                                *seq += 1;
                            }
                            None => idle += 1,
                        },
                        None => idle += 1,
                    }
                    if idle > 64 {
                        bail!("Media Foundation 输出停滞（连续无帧）");
                    }
                }
                Err(err) => {
                    let code = err.code();
                    if code == MF_E_TRANSFORM_NEED_MORE_INPUT {
                        return Ok(());
                    }
                    if code == MF_E_TRANSFORM_STREAM_CHANGE {
                        unsafe { session.mft.SetOutputType(0, &session.out_type, 0)? };
                        continue;
                    }
                    return Err(err).context("ProcessOutput 失败");
                }
            }
        }
    }
}

impl VideoDecoder for MediaFoundationDecoder {
    fn configure(
        &mut self,
        codec: VideoCodec,
        config_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        if codec != VideoCodec::H264 {
            bail!(
                "Media Foundation 解码器仅支持 H264，远端下发的是 {}",
                codec.as_str()
            );
        }
        if self.session.is_none() {
            let mft = self.mft.take().ok_or_else(|| anyhow!("解码会话已关闭"))?;
            let session = Session::new(mft, width, height)
                .context("配置 Media Foundation H.264 解码会话失败")?;
            gscp_core::logbus::emit(&format!(
                "[decoder] Media Foundation 解码会话就绪 ({}×{}，{})",
                width,
                height,
                if session.d3d.is_some() {
                    "DXVA 硬解"
                } else {
                    "系统内存模式"
                }
            ));
            self.session = Some(session);
        }
        // SPS/PPS 作为首个输入喂入
        if !config_data.is_empty() {
            self.decode(config_data)?;
        }
        Ok(())
    }

    fn decode(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>> {
        if self.pending.is_empty() {
            if packet.is_empty() {
                return Ok(None);
            }
            self.feed_and_drain(packet)?;
        }
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        let mut out: Vec<_> = self.pending.drain(..).collect();
        if let Some(session) = self.session.as_mut() {
            unsafe {
                let _ = session.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
            }
            let mut seq = self.seq;
            Self::drain_into(session, &mut seq, &mut out)?;
            self.seq = seq;
            // 复位为可继续接收输入的状态
            unsafe {
                let _ = session.mft.ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0);
                let _ = session.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
                let _ = session.mft.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
            }
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        match self.session.as_ref() {
            Some(s) if s.d3d.is_some() => "MediaFoundation(DXVA)",
            Some(_) => "MediaFoundation",
            None => "MediaFoundation(待配置)",
        }
    }
}

/// COM 对象仅在创建它的解码线程上使用（MTA 单元）；MFT/D3D 对象均为
/// 自由线程聚合对象，跨线程 Release 安全。VideoDecoder::Send 要求由此保证。
unsafe impl Send for MediaFoundationDecoder {}

impl Drop for MediaFoundationDecoder {
    fn drop(&mut self) {
        // CoUninitialize 必须与初始化在同一线程；跨线程 drop 时交由进程退出清理
        if self.com_inited && self.com_thread == Some(std::thread::current().id()) {
            unsafe { CoUninitialize() };
        }
    }
}

impl Session {
    fn new(mft: IMFTransform, width: u32, height: u32) -> Result<Self> {
        // D3D11/DXVA：MFT 声明 D3D11 感知时挂设备管理器启用硬解
        let mut d3d = None;
        let aware = unsafe {
            mft.GetAttributes()
                .ok()
                .and_then(|attrs| attrs.GetUINT32(&MF_SA_D3D11_AWARE).ok())
                .unwrap_or(0)
        };
        if aware != 0 {
            match D3dCtx::new() {
                Ok(ctx) => {
                    let manager_param = ctx.manager_raw();
                    match unsafe { mft.ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, manager_param) }
                    {
                        Ok(()) => {
                            gscp_core::logbus::emit("[decoder] 已启用 D3D11/DXVA 硬件解码");
                            d3d = Some(ctx);
                        }
                        Err(err) => {
                            gscp_core::logbus::emit(&format!(
                                "[decoder] 挂接 D3D11 管理器失败，MFT 走系统内存模式: {err:#}"
                            ));
                        }
                    }
                }
                Err(err) => {
                    gscp_core::logbus::emit(&format!(
                        "[decoder] D3D11 设备不可用，MFT 走系统内存模式: {err:#}"
                    ));
                }
            }
        }

        let out_type = unsafe { MFCreateMediaType()? };
        unsafe {
            out_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            out_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
            out_type.SetUINT64(&MF_MT_FRAME_SIZE, ((width as u64) << 32) | (height as u64))?;
            mft.SetOutputType(0, &out_type, 0)?;
        }

        let mut in_ids = [0_u32; 1];
        let mut out_ids = [0_u32; 1];
        unsafe {
            // 简单 MFT 单流；GetStreamIDs 失败时沿用 0
            let _ = mft.GetStreamIDs(&mut in_ids, &mut out_ids);
        }
        let out_stream_id = out_ids[0];
        let out_info =
            unsafe { mft.GetOutputStreamInfo(out_stream_id)? };
        let provide_output = out_info.dwFlags
            & ((MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0)
                as u32)
            == 0;

        unsafe {
            mft.ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)?;
            mft.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
        }

        Ok(Self {
            mft,
            out_stream_id,
            out_info,
            provide_output,
            out_type,
            width,
            height,
            d3d,
        })
    }
}

impl D3dCtx {
    fn new() -> Result<Self> {
        unsafe {
            let mut device: Option<ID3D11Device> = None;
            let mut context: Option<ID3D11DeviceContext> = None;
            // 优先硬件 GPU；WARP 兜底（虚机/无独显场景）
            for driver in [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP] {
                let flags = D3D11_CREATE_DEVICE_FLAG(D3D11_CREATE_DEVICE_VIDEO_SUPPORT.0);
                if D3D11CreateDevice(
                    None,
                    driver,
                    None,
                    flags,
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut context),
                )
                .is_ok()
                {
                    break;
                }
            }
            let device = device.ok_or_else(|| anyhow!("D3D11 设备创建失败"))?;
            let context = context.ok_or_else(|| anyhow!("D3D11 上下文创建失败"))?;
            // MF 要求设备并发安全
            let mt: ID3D11Multithread = context.cast()?;
            let _ = mt.SetMultithreadProtected(true);
            let mut reset_token = 0_u32;
            let mut manager_opt: Option<IMFDXGIDeviceManager> = None;
            MFCreateDXGIDeviceManager(&mut reset_token, &mut manager_opt)?;
            let manager = manager_opt.ok_or_else(|| anyhow!("MFCreateDXGIDeviceManager 返回空"))?;
            manager.ResetDevice(&device, 0)?;
            Ok(Self {
                device,
                context,
                _manager: manager,
                staging: None,
            })
        }
    }

    /// ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER) 的 lParam 为 IUnknown 指针。
    fn manager_raw(&self) -> usize {
        self._manager.as_raw() as usize
    }

    /// DXVA 输出：NV12 纹理 → staging 全量拷贝 → Map 读回。
    /// 返回 (width, height, RGBA)。
    fn read_nv12(&mut self, dxgi: &IMFDXGIBuffer) -> Result<Option<(u32, u32, Vec<u8>)>> {
        unsafe {
            let tex: ID3D11Texture2D = {
                let mut tex_opt: Option<ID3D11Texture2D> = None;
                dxgi.GetResource(
                    &ID3D11Texture2D::IID,
                    &mut tex_opt as *mut _ as *mut *mut std::ffi::c_void,
                )?;
                tex_opt.ok_or_else(|| anyhow!("DXVA 缓冲未携带纹理"))?
            };
            let sub = dxgi.GetSubresourceIndex()?;
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            tex.GetDesc(&mut desc);
            if desc.Format != DXGI_FORMAT_NV12 {
                bail!("DXVA 输出格式非 NV12: {:?}", desc.Format);
            }
            let need_new = match &self.staging {
                Some((_, w, h, a)) => {
                    *w != desc.Width || *h != desc.Height || *a != desc.ArraySize
                }
                None => true,
            };
            if need_new {
                let mut sd = desc;
                sd.Usage = D3D11_USAGE_STAGING;
                sd.BindFlags = 0;
                sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
                sd.MiscFlags = 0;
                let mut staging_opt: Option<ID3D11Texture2D> = None;
                self.device
                    .CreateTexture2D(&sd, None, Some(&mut staging_opt))?;
                let staging =
                    staging_opt.ok_or_else(|| anyhow!("staging 纹理创建失败"))?;
                self.staging = Some((staging, desc.Width, desc.Height, desc.ArraySize));
            }
            let (staging, _, _, array_size) = self.staging.as_ref().unwrap();
            self.context.CopyResource(staging, &tex);
            // D3D11 视频平面子资源：Y 为基索引，UV 追加在全部 Y 之后
            let (y_sub, uv_sub) = if desc.ArraySize > 1 {
                (sub, sub + *array_size)
            } else {
                (sub * 2, sub * 2 + 1)
            };
            let (width, height) = (desc.Width as usize, desc.Height as usize);
            let y_plane = map_plane(&self.context, staging, y_sub, width, height)?;
            let uv_h = height.div_ceil(2);
            let uv_plane = map_plane(&self.context, staging, uv_sub, width, uv_h)?;
            let rgba = crate::decode::software::nv12_to_rgba(
                &y_plane,
                &uv_plane,
                (width, width),
                width,
                height,
            );
            Ok(Some((width as u32, height as u32, rgba)))
        }
    }
}

fn map_plane(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    subresource: u32,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    unsafe {
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context.Map(
            staging,
            subresource,
            D3D11_MAP_READ,
            0,
            Some(&mut mapped),
        )?;
        let pitch = mapped.RowPitch as usize;
        let mut out = vec![0_u8; width * height];
        let src = std::slice::from_raw_parts(mapped.pData as *const u8, pitch * height);
        for row in 0..height {
            out[row * width..(row + 1) * width]
                .copy_from_slice(&src[row * pitch..row * pitch + width]);
        }
        context.Unmap(staging, subresource);
        Ok(out)
    }
}

/// 系统内存 NV12 输出：优先 IMF2DBuffer::Lock2D（真实 stride），回退连续 Lock。
/// 返回 (width, height, RGBA)。
fn read_nv12_from_buffer(
    buffer: &IMFMediaBuffer,
    width: u32,
    height: u32,
) -> Result<Option<(u32, u32, Vec<u8>)>> {
    unsafe {
        let rgba;
        if let Ok(b2d) = buffer.cast::<IMF2DBuffer>() {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut pitch: i32 = 0;
            b2d.Lock2D(&mut ptr, &mut pitch).context("Lock2D 失败")?;
            let pitch = pitch.unsigned_abs() as usize;
            let rows = height as usize + height.div_ceil(2) as usize;
            let base = std::slice::from_raw_parts(ptr, pitch * rows);
            let (y, uv) = base.split_at(pitch * height as usize);
            rgba = crate::decode::software::nv12_to_rgba(
                y,
                uv,
                (pitch, pitch),
                width as usize,
                height as usize,
            );
            let _ = b2d.Unlock2D();
        } else {
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut len = 0_u32;
            buffer
                .Lock(&mut ptr, Some(&mut len), None)
                .context("MF buffer Lock 失败")?;
            let total = width as usize * height as usize * 3 / 2;
            let base = std::slice::from_raw_parts(ptr, total.min(len as usize));
            let (y, uv) = base.split_at(width as usize * height as usize);
            rgba = crate::decode::software::nv12_to_rgba(
                y,
                uv,
                (width as usize, width as usize),
                width as usize,
                height as usize,
            );
            let _ = buffer.Unlock();
        }
        Ok(Some((width, height, rgba)))
    }
}

fn extract_frame(
    session: &mut Session,
    sample: &IMFSample,
) -> Result<Option<(u32, u32, Vec<u8>)>> {
    let buffer = unsafe { sample.GetBufferByIndex(0)? };
    if let Ok(dxgi) = buffer.cast::<IMFDXGIBuffer>() {
        if let Some(d3d) = session.d3d.as_mut() {
            return d3d.read_nv12(&dxgi);
        }
    }
    read_nv12_from_buffer(&buffer, session.width, session.height)
}

fn make_sample(data: &[u8], time_100ns: i64) -> Result<IMFSample> {
    unsafe {
        let sample = MFCreateSample()?;
        let buffer = MFCreateMemoryBuffer(data.len() as u32)?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        buffer.Lock(&mut ptr, None, None)?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        buffer.SetCurrentLength(data.len() as u32)?;
        buffer.Unlock()?;
        sample.AddBuffer(&buffer)?;
        sample.SetSampleTime(time_100ns)?;
        sample.SetSampleDuration(0)?;
        Ok(sample)
    }
}

unsafe fn make_output_sample(session: &Session) -> Result<IMFSample> {
    let size = session.out_info.cbSize.max(1);
    let alignment = session.out_info.cbAlignment;
    let buffer = if alignment > 1 {
        MFCreateAlignedMemoryBuffer(size, alignment)?
    } else {
        MFCreateMemoryBuffer(size)?
    };
    let sample = MFCreateSample()?;
    sample.AddBuffer(&buffer)?;
    Ok(sample)
}

/// 枚举同步 H.264 → NV12 解码 MFT（内置于 Windows，DXVA 感知）。
fn create_h264_mft() -> Result<IMFTransform> {
    unsafe {
        let input = MFT_REGISTER_TYPE_INFO {
            guidMajorType: MFMediaType_Video,
            guidSubtype: MFVideoFormat_H264,
        };
        let output = MFT_REGISTER_TYPE_INFO {
            guidMajorType: MFMediaType_Video,
            guidSubtype: MFVideoFormat_NV12,
        };
        let mut activates: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0_u32;
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_DECODER,
            MFT_ENUM_FLAG_SYNCMFT,
            Some(&input),
            Some(&output),
            &mut activates,
            &mut count,
        )?;
        for i in 0..count as usize {
            let activate = (*activates.add(i))
                .take()
                .ok_or_else(|| anyhow!("MFT 激活对象为空"))?;
            if let Ok(mft) = activate.cast::<IMFTransform>() {
                return Ok(mft);
            }
        }
        CoTaskMemFree(Some(activates as *const std::ffi::c_void));
        bail!("未找到支持 H264→NV12 的同步解码 MFT")
    }
}
