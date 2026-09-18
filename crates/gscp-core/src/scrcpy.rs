//! scrcpy 协议实现。
//!
//! 覆盖：流头解析（dummy byte / 64B 设备名 / codec meta / frame meta）、
//! config 包判定（ptsAndFlags 最高位）、bootstrap 模式的三路 socket 嗅探
//! 与 camera/overlay 面积分路、直连模式握手、TCP socket 工具。
//! 移植自 mixplayer 并补充单元测试。

use anyhow::{anyhow, bail, Context, Result};
use std::collections::VecDeque;
use std::io::{ErrorKind, Read};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

pub const DEVICE_NAME_BYTES: usize = 64;
pub const SCRCPY_FLAG_CONFIG: u64 = 1 << 63;
pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(15);
pub const CONNECT_TIMEOUT: Duration = SOCKET_TIMEOUT;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

impl FrameSize {
    pub fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
}

impl VideoCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "h265",
            Self::Av1 => "av1",
        }
    }

    pub fn from_header(id: u32) -> Result<Self> {
        match id {
            0x68_32_36_34 => Ok(Self::H264),
            0x68_32_36_35 => Ok(Self::H265),
            0x00_61_76_31 => Ok(Self::Av1),
            _ => bail!("未知的 scrcpy codec id: 0x{id:08x}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Opus,
    Aac,
    Flac,
    Raw,
}

impl AudioCodec {
    pub fn from_header(id: u32) -> Result<Self> {
        match id {
            0x6f_70_75_73 => Ok(Self::Opus),
            0x00_61_61_63 => Ok(Self::Aac),
            0x66_6c_61_63 => Ok(Self::Flac),
            0x00_72_61_77 => Ok(Self::Raw),
            0 => bail!("远端音频流被禁用"),
            1 => bail!("远端音频配置错误"),
            _ => bail!("未知的 scrcpy audio codec id: 0x{id:08x}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Raw => "raw",
        }
    }
}

/// 帧头（12 字节）解析：pts_and_flags + packet_size。
pub fn parse_frame_header(header: &[u8; 12]) -> (u64, usize) {
    let pts_and_flags = u64::from_be_bytes(header[0..8].try_into().expect("pts+flags"));
    let packet_size = u32::from_be_bytes(header[8..12].try_into().expect("packet size")) as usize;
    (pts_and_flags, packet_size)
}

pub fn is_config_packet(pts_and_flags: u64) -> bool {
    pts_and_flags & SCRCPY_FLAG_CONFIG != 0
}

// ── 带缓冲回放的流 ──

pub struct BufferedStream {
    pub stream: TcpStream,
    pub prefix: VecDeque<u8>,
}

impl BufferedStream {
    pub fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            prefix: VecDeque::new(),
        }
    }

    pub fn from_parts(stream: TcpStream, consumed: &[u8]) -> Self {
        Self {
            stream,
            prefix: consumed.iter().copied().collect(),
        }
    }

    pub fn read_exact_buffered(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        let mut filled = 0;
        while filled < buf.len() {
            if let Some(byte) = self.prefix.pop_front() {
                buf[filled] = byte;
                filled += 1;
                continue;
            }
            match self.stream.read(&mut buf[filled..])? {
                0 => {
                    return Err(std::io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                n => filled += n,
            }
        }
        Ok(())
    }

    pub fn read_buffered(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut filled = 0;
        while filled < buf.len() {
            if let Some(byte) = self.prefix.pop_front() {
                buf[filled] = byte;
                filled += 1;
            } else {
                break;
            }
        }
        if filled > 0 {
            return Ok(filled);
        }
        self.stream.read(buf)
    }

    pub fn stream_ref(&self) -> &TcpStream {
        &self.stream
    }
}

// ── 嗅探 ──

#[derive(Debug, Clone, PartialEq)]
pub enum SniffedStream {
    Video {
        codec: VideoCodec,
        size: FrameSize,
        device_name: Option<String>,
        consumed: usize,
    },
    Audio {
        codec: AudioCodec,
        device_name: Option<String>,
        consumed: usize,
    },
}

pub fn parse_device_name(raw_name: &[u8]) -> Option<String> {
    let trimmed: Vec<u8> = raw_name
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .collect();
    if trimmed.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&trimmed).to_string())
    }
}

fn plausible_device_name(raw_name: &[u8]) -> bool {
    let mut seen_zero = false;
    for &byte in raw_name {
        if byte == 0 {
            seen_zero = true;
            continue;
        }
        if seen_zero {
            return false;
        }
        if !(byte.is_ascii_graphic() || byte == b' ') {
            return false;
        }
    }
    true
}

/// 无 meta 的裸流：12 字节视频头或 4 字节音频头。
pub fn sniff_plain_stream(prefix: &[u8]) -> Option<SniffedStream> {
    if prefix.len() >= 12 {
        let codec_id = u32::from_be_bytes(prefix[0..4].try_into().expect("video codec id"));
        if let Ok(codec) = VideoCodec::from_header(codec_id) {
            let width = u32::from_be_bytes(prefix[4..8].try_into().expect("width"));
            let height = u32::from_be_bytes(prefix[8..12].try_into().expect("height"));
            return Some(SniffedStream::Video {
                codec,
                size: FrameSize { width, height },
                device_name: None,
                consumed: 12,
            });
        }
    }
    if prefix.len() >= 4 {
        let codec_id = u32::from_be_bytes(prefix[0..4].try_into().expect("audio codec id"));
        if let Ok(codec) = AudioCodec::from_header(codec_id) {
            return Some(SniffedStream::Audio {
                codec,
                device_name: None,
                consumed: 4,
            });
        }
    }
    None
}

/// 首端口语义：1 dummy + 64 设备名 + codec meta(4/12)。
pub fn sniff_first_socket_stream(prefix: &[u8]) -> Option<SniffedStream> {
    let meta_offset = 1 + DEVICE_NAME_BYTES;
    if prefix.len() < meta_offset + 4 {
        return None;
    }
    let raw_name = &prefix[1..meta_offset];
    if !plausible_device_name(raw_name) {
        return None;
    }
    let device_name = parse_device_name(raw_name);
    let codec_id = u32::from_be_bytes(
        prefix[meta_offset..meta_offset + 4]
            .try_into()
            .expect("first socket codec id"),
    );
    if prefix.len() >= meta_offset + 12 {
        if let Ok(codec) = VideoCodec::from_header(codec_id) {
            let width = u32::from_be_bytes(
                prefix[meta_offset + 4..meta_offset + 8]
                    .try_into()
                    .expect("width"),
            );
            let height = u32::from_be_bytes(
                prefix[meta_offset + 8..meta_offset + 12]
                    .try_into()
                    .expect("height"),
            );
            return Some(SniffedStream::Video {
                codec,
                size: FrameSize { width, height },
                device_name,
                consumed: meta_offset + 12,
            });
        }
    }
    if let Ok(codec) = AudioCodec::from_header(codec_id) {
        return Some(SniffedStream::Audio {
            codec,
            device_name,
            consumed: meta_offset + 4,
        });
    }
    None
}

pub fn sniff_bootstrap_stream(prefix: &[u8]) -> Result<SniffedStream> {
    sniff_first_socket_stream(prefix)
        .or_else(|| sniff_plain_stream(prefix))
        .ok_or_else(|| anyhow!("bootstrap 未识别的流头部"))
}

/// 逐步读前缀直到能识别或长度用尽。
pub fn sniff_read_prefix(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut prefix = Vec::with_capacity(1 + DEVICE_NAME_BYTES + 12);
    let probe_lengths = [
        4_u64,
        12,
        1 + DEVICE_NAME_BYTES as u64 + 4,
        1 + DEVICE_NAME_BYTES as u64 + 12,
    ];
    for target_len in probe_lengths {
        let target_len = target_len as usize;
        let completed = sniff_read_until(stream, &mut prefix, target_len)?;
        if sniff_bootstrap_stream(&prefix).is_ok() {
            return Ok(prefix);
        }
        if !completed {
            break;
        }
    }
    if prefix.is_empty() {
        bail!("bootstrap 流在发送头部前已关闭");
    }
    Err(anyhow!(
        "bootstrap 未识别的流头部，已读取 {} 字节",
        prefix.len()
    ))
}

fn sniff_read_until(stream: &mut TcpStream, prefix: &mut Vec<u8>, target_len: usize) -> Result<bool> {
    let mut chunk = [0_u8; 64];
    while prefix.len() < target_len {
        let want = (target_len - prefix.len()).min(chunk.len());
        match stream.read(&mut chunk[..want]) {
            Ok(0) => return Ok(false),
            Ok(read) => prefix.extend_from_slice(&chunk[..read]),
            Err(err) => return Err(err).context("读取 bootstrap sniff 前缀失败"),
        }
    }
    Ok(true)
}

// ── camera/overlay 面积分路 ──

pub fn closer_to_camera(current: FrameSize, camera: FrameSize, overlay: FrameSize) -> bool {
    let current_area = current.area();
    let camera_gap = current_area.abs_diff(camera.area());
    let overlay_gap = current_area.abs_diff(overlay.area());
    camera_gap <= overlay_gap
}

// ── socket 工具 ──

pub fn connect_socket(address: &str, kind: &str) -> Result<TcpStream> {
    let addrs = address
        .to_socket_addrs()
        .with_context(|| format!("解析 {kind} socket 地址失败: {address}"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        bail!("未解析到可用的 {kind} socket 地址: {address}");
    }

    let mut last_error = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => return configure_socket(stream),
            Err(err) => last_error = Some(err),
        }
    }

    let err = last_error.unwrap_or_else(|| std::io::Error::other("unknown connect error"));
    Err(err).with_context(|| format!("连接 {kind} socket 失败: {address}"))
}

pub fn configure_socket(stream: TcpStream) -> Result<TcpStream> {
    stream.set_nodelay(true).context("设置 TCP_NODELAY 失败")?;
    stream
        .set_read_timeout(Some(SOCKET_TIMEOUT))
        .context("设置 socket 读超时失败")?;
    stream
        .set_write_timeout(Some(SOCKET_TIMEOUT))
        .context("设置 socket 写超时失败")?;
    Ok(stream)
}

/// camera 码流的断联判定阈值（秒）：拍照等场景会短暂占用码流，
/// 静默超过该时长仍无数据才判定连接失效。
///
/// 注意：只有 camera 适合作为断联判定源——
/// overlay 抓取的是渲染输出，渲染停止后没有数据属正常现象；
/// audio 在无声音输出时也可能长时间静默。二者均不应触发重连。
pub const CAMERA_STALL_SECS: u64 = 6;

/// 读超时错误判定：socket 静默超过阈值后，阻塞读会以 WouldBlock/TimedOut 返回。
pub fn is_stall_error(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// 流式阶段的卡死保护：读/写超时 = 静默阈值；并启用 TCP keepalive，
/// 让半开连接（对端不可达）在 TCP 层尽快暴露为读错误，而不是永久阻塞。
///
/// 之前沿袭 mixplayer 的做法是握手后彻底取消超时、只等 EOF——
/// 但 WiFi 长连接的静默断链（省电、隧道老化）不会产生 EOF，
/// 读线程会永久阻塞，画面从此卡死且不触发重连。
pub fn enable_stream_mode(stream: &TcpStream, stall: Option<Duration>) -> Result<()> {
    stream.set_nodelay(true).context("设置 TCP_NODELAY 失败")?;
    match stall {
        Some(t) => {
            stream
                .set_read_timeout(Some(t))
                .context("设置 socket 读超时失败")?;
            stream
                .set_write_timeout(Some(t))
                .context("设置 socket 写超时失败")?;
        }
        // overlay/audio 的静默（渲染停止、无声音）可持续任意久，
        // 必须显式清除握手阶段遗留的读超时，否则正常静默会被误判为断联。
        None => {
            stream
                .set_read_timeout(None)
                .context("清除 socket 读超时失败")?;
            stream
                .set_write_timeout(None)
                .context("清除 socket 写超时失败")?;
        }
    }
    // keepalive 对所有流启用：静默的 overlay/audio 连接若出现半开
    // （路径失效而非数据静默），TCP 层约 35s 内暴露为读错误并重连。
    #[cfg(unix)]
    tcp_keepalive_unix(stream);
    Ok(())
}

#[cfg(unix)]
fn tcp_keepalive_unix(stream: &TcpStream) {
    use std::os::unix::io::AsRawFd;
    let fd = stream.as_raw_fd();
    let on: libc::c_int = 1;
    let idle: libc::c_int = 15;
    let intvl: libc::c_int = 5;
    let cnt: libc::c_int = 4;
    unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_KEEPALIVE,
            &on as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
        #[cfg(target_os = "macos")]
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_KEEPALIVE,
            &idle as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
        #[cfg(target_os = "linux")]
        {
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_KEEPIDLE,
                &idle as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_KEEPINTVL,
                &intvl as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_KEEPCNT,
                &cnt as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
        let _ = (intvl, cnt);
    }
}

/// 直连模式头部：dummy byte + 64B 设备名。
pub fn read_dummy_and_device_meta(
    stream: &mut TcpStream,
    send_dummy_byte: bool,
    send_device_meta: bool,
) -> Result<Option<String>> {
    if send_dummy_byte {
        let mut dummy = [0_u8; 1];
        stream
            .read_exact(&mut dummy)
            .context("读取 scrcpy dummy byte 失败")?;
    }
    if !send_device_meta {
        return Ok(None);
    }
    let mut raw_name = [0_u8; DEVICE_NAME_BYTES];
    stream
        .read_exact(&mut raw_name)
        .context("读取 scrcpy device meta 失败")?;
    Ok(parse_device_name(&raw_name))
}

// ── 源定义 ──

#[derive(Clone, Debug)]
pub struct VideoSource {
    pub host: String,
    pub port: u16,
    pub send_device_meta: bool,
    pub send_frame_meta: bool,
    pub send_dummy_byte: bool,
    pub send_codec_meta: bool,
    pub codec: Option<VideoCodec>,
    pub size_hint: Option<FrameSize>,
    pub raw_stream: bool,
    pub debug: bool,
    /// 流式阶段静默超时；None = 默认（或 GSCP_STREAM_STALL_SECS 环境变量）。
    pub stall_timeout: Option<Duration>,
}

impl VideoSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: String,
        port: u16,
        send_device_meta: bool,
        send_frame_meta: bool,
        send_dummy_byte: bool,
        send_codec_meta: bool,
        codec: Option<VideoCodec>,
        size_hint: Option<FrameSize>,
        raw_stream: bool,
        debug: bool,
    ) -> Result<Self> {
        let source = Self {
            host,
            port,
            send_device_meta,
            send_frame_meta,
            send_dummy_byte,
            send_codec_meta,
            codec,
            size_hint,
            raw_stream,
            debug,
            stall_timeout: None,
        };
        source.validate()?;
        Ok(source)
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    fn validate(&self) -> Result<()> {
        if self.raw_stream {
            if self.codec.is_none() {
                bail!("raw_stream 视频模式必须显式提供 codec");
            }
            if self.size_hint.is_none() {
                bail!("raw_stream 视频模式必须显式提供 size");
            }
        } else if !self.send_codec_meta {
            if self.codec.is_none() {
                bail!("socket 源关闭 codec_meta 时必须显式传 codec=h264|h265|av1");
            }
            if self.size_hint.is_none() {
                bail!("socket 源关闭 codec_meta 时必须显式传 size=宽x高");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AudioSource {
    pub host: String,
    pub port: u16,
    pub send_device_meta: bool,
    pub send_frame_meta: bool,
    pub send_dummy_byte: bool,
    pub send_codec_meta: bool,
    pub codec: Option<AudioCodec>,
    pub raw_stream: bool,
    pub debug: bool,
    /// 流式阶段静默超时；None = 默认（或 GSCP_STREAM_STALL_SECS 环境变量）。
    pub stall_timeout: Option<Duration>,
}

impl AudioSource {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: String,
        port: u16,
        send_device_meta: bool,
        send_frame_meta: bool,
        send_dummy_byte: bool,
        send_codec_meta: bool,
        codec: Option<AudioCodec>,
        raw_stream: bool,
        debug: bool,
    ) -> Result<Self> {
        let source = Self {
            host,
            port,
            send_device_meta,
            send_frame_meta,
            send_dummy_byte,
            send_codec_meta,
            codec,
            raw_stream,
            debug,
            stall_timeout: None,
        };
        source.validate()?;
        Ok(source)
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    fn validate(&self) -> Result<()> {
        if self.raw_stream && self.codec.is_none() {
            bail!("raw_stream 音频模式必须显式提供 audio codec");
        }
        if !self.raw_stream && !self.send_codec_meta && self.codec.is_none() {
            bail!("音频关闭 codec_meta 时必须显式提供 audio codec");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_meta_block(name: &str) -> Vec<u8> {
        let mut block = vec![0_u8]; // dummy byte
        let mut name_bytes = name.as_bytes().to_vec();
        name_bytes.resize(DEVICE_NAME_BYTES, 0);
        block.extend_from_slice(&name_bytes);
        block
    }

    #[test]
    fn parse_frame_header_and_config_flag() {
        let mut header = [0_u8; 12];
        header[0..8].copy_from_slice(&(SCRCPY_FLAG_CONFIG | 42).to_be_bytes());
        header[8..12].copy_from_slice(&77_u32.to_be_bytes());
        let (pts, size) = parse_frame_header(&header);
        assert!(is_config_packet(pts));
        assert_eq!(pts & !SCRCPY_FLAG_CONFIG, 42);
        assert_eq!(size, 77);

        header[0..8].copy_from_slice(&1234_u64.to_be_bytes());
        let (pts, _) = parse_frame_header(&header);
        assert!(!is_config_packet(pts));
    }

    #[test]
    fn codec_headers() {
        assert_eq!(VideoCodec::from_header(0x68323634).unwrap(), VideoCodec::H264);
        assert_eq!(VideoCodec::from_header(0x68323635).unwrap(), VideoCodec::H265);
        assert_eq!(VideoCodec::from_header(0x00617631).unwrap(), VideoCodec::Av1);
        assert!(VideoCodec::from_header(0x12345678).is_err());
        assert_eq!(AudioCodec::from_header(0x6f707573).unwrap(), AudioCodec::Opus);
        assert_eq!(AudioCodec::from_header(0x00726177).unwrap(), AudioCodec::Raw);
        assert_eq!(
            AudioCodec::from_header(0).unwrap_err().to_string(),
            "远端音频流被禁用"
        );
        assert_eq!(
            AudioCodec::from_header(1).unwrap_err().to_string(),
            "远端音频配置错误"
        );
    }

    #[test]
    fn sniff_first_socket_video() {
        let mut prefix = device_meta_block("RG-glass");
        prefix.extend_from_slice(&0x68323634_u32.to_be_bytes());
        prefix.extend_from_slice(&1920_u32.to_be_bytes());
        prefix.extend_from_slice(&1080_u32.to_be_bytes());
        let sniffed = sniff_bootstrap_stream(&prefix).unwrap();
        match sniffed {
            SniffedStream::Video {
                codec,
                size,
                device_name,
                consumed,
            } => {
                assert_eq!(codec, VideoCodec::H264);
                assert_eq!(size, FrameSize { width: 1920, height: 1080 });
                assert_eq!(device_name.as_deref(), Some("RG-glass"));
                assert_eq!(consumed, 1 + DEVICE_NAME_BYTES + 12);
            }
            other => panic!("期望视频流，实际 {other:?}"),
        }
    }

    #[test]
    fn sniff_first_socket_audio() {
        let mut prefix = device_meta_block("RG-glass");
        prefix.extend_from_slice(&0x00726177_u32.to_be_bytes());
        let sniffed = sniff_bootstrap_stream(&prefix).unwrap();
        match sniffed {
            SniffedStream::Audio { codec, consumed, .. } => {
                assert_eq!(codec, AudioCodec::Raw);
                assert_eq!(consumed, 1 + DEVICE_NAME_BYTES + 4);
            }
            other => panic!("期望音频流，实际 {other:?}"),
        }
    }

    #[test]
    fn sniff_plain_video_falls_back() {
        // 12 字节裸视频头（设备名位置不可信）
        let mut prefix = vec![0x68, 0x32, 0x36, 0x34];
        prefix.extend_from_slice(&640_u32.to_be_bytes());
        prefix.extend_from_slice(&480_u32.to_be_bytes());
        let sniffed = sniff_bootstrap_stream(&prefix).unwrap();
        match sniffed {
            SniffedStream::Video { size, device_name, consumed, .. } => {
                assert_eq!(size, FrameSize { width: 640, height: 480 });
                assert!(device_name.is_none());
                assert_eq!(consumed, 12);
            }
            other => panic!("期望裸视频流，实际 {other:?}"),
        }
    }

    #[test]
    fn sniff_plain_audio_4bytes() {
        let prefix = 0x6f707573_u32.to_be_bytes().to_vec();
        let sniffed = sniff_bootstrap_stream(&prefix).unwrap();
        assert!(matches!(sniffed, SniffedStream::Audio { codec: AudioCodec::Opus, .. }));
    }

    #[test]
    fn sniff_rejects_garbage() {
        let prefix = vec![0xff_u8; 77];
        assert!(sniff_bootstrap_stream(&prefix).is_err());
    }

    #[test]
    fn closer_to_camera_area_heuristic() {
        let camera = FrameSize { width: 1920, height: 1080 };
        let overlay = FrameSize { width: 640, height: 480 };
        assert!(closer_to_camera(camera, camera, overlay));
        assert!(!closer_to_camera(overlay, camera, overlay));
    }

    #[test]
    fn source_validation() {
        assert!(VideoSource::new(
            "127.0.0.1".into(), 1, true, true, true, true, None, None, false, false
        )
        .is_ok());
        // raw_stream 必须提供 codec+size
        assert!(VideoSource::new(
            "127.0.0.1".into(), 1, true, true, true, false, None, None, true, false
        )
        .is_err());
        assert!(VideoSource::new(
            "127.0.0.1".into(), 1, true, true, true, false, Some(VideoCodec::H264),
            Some(FrameSize { width: 640, height: 480 }), true, false
        )
        .is_ok());
        // 关闭 codec_meta 必须提供 codec+size
        assert!(VideoSource::new(
            "127.0.0.1".into(), 1, true, true, true, false, None, None, false, false
        )
        .is_err());
        // 音频关闭 codec_meta 必须提供 codec
        assert!(AudioSource::new(
            "127.0.0.1".into(), 1, true, true, true, false, None, false, false
        )
        .is_err());
    }
}
