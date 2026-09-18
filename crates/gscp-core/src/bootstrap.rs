//! 远端 scrcpy server 拉起与会话守护。
//!
//! 职责（移植 mixplayer）：
//! - `bootstrap_remote_guard`：adb connect → push jar → forward →
//!   `app_process` 启动 server → renice；
//! - [`RemoteBootstrapManager`]：generation 看门狗（server 退出自动重建）、
//!   三路 socket 的预连接缓存与 camera/overlay 面积分路分发；
//! - 直连模式（不经 bootstrap）的流握手。

use crate::adb::AdbDevice;
use crate::logbus;
use crate::scrcpy::{
    self, AudioCodec, AudioSource, BufferedStream, FrameSize, VideoCodec, VideoSource,
};
use crate::settings::ResolvedConfig;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::VecDeque;
use std::io::Read;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const BOOTSTRAP_HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(1);
pub const RECONNECT_DELAY: Duration = Duration::from_secs(1);
pub const RESTART_RETRY_DELAY: Duration = Duration::from_secs(15);
/// 无协议级就绪信号，用保守的启动宽限期。
pub const REMOTE_SERVER_READY_DELAY: Duration = Duration::from_secs(4);

// ── 预处理后的流 ──

pub struct PreparedVideoStream {
    pub stream: BufferedStream,
    pub device_name: Option<String>,
    pub codec: VideoCodec,
    pub size: FrameSize,
}

pub struct PreparedAudioStream {
    pub stream: BufferedStream,
    pub device_name: Option<String>,
    pub codec: AudioCodec,
}

pub enum PreparedStream {
    Video(PreparedVideoStream),
    Audio(PreparedAudioStream),
}

// ── 直连模式握手 ──

pub fn prepare_direct_video_stream(source: &VideoSource) -> Result<PreparedVideoStream> {
    let address = source.address();
    let mut stream = scrcpy::connect_socket(&address, "video")?;
    let device_name = scrcpy::read_dummy_and_device_meta(
        &mut stream,
        source.send_dummy_byte,
        source.send_device_meta,
    )?;
    let (codec, size) = if source.send_codec_meta {
        let mut codec_header = [0_u8; 4];
        stream
            .read_exact(&mut codec_header)
            .context("读取 scrcpy codec header 失败")?;
        let codec = VideoCodec::from_header(u32::from_be_bytes(codec_header))?;
        let mut size_bytes = [0_u8; 8];
        stream
            .read_exact(&mut size_bytes)
            .context("读取 scrcpy codec size 失败")?;
        let width = u32::from_be_bytes(size_bytes[0..4].try_into().expect("width"));
        let height = u32::from_be_bytes(size_bytes[4..8].try_into().expect("height"));
        (codec, FrameSize { width, height })
    } else {
        (
            source.codec.context("缺少 socket codec 配置")?,
            source.size_hint.context("缺少 socket size 配置")?,
        )
    };
    Ok(PreparedVideoStream {
        stream: BufferedStream::new(stream),
        device_name,
        codec,
        size,
    })
}

pub fn prepare_direct_audio_stream(source: &AudioSource) -> Result<PreparedAudioStream> {
    let address = source.address();
    let mut stream = scrcpy::connect_socket(&address, "audio")?;
    let device_name = scrcpy::read_dummy_and_device_meta(
        &mut stream,
        source.send_dummy_byte,
        source.send_device_meta,
    )?;
    let codec = if source.send_codec_meta {
        let mut header = [0_u8; 4];
        stream
            .read_exact(&mut header)
            .context("读取 audio codec header 失败")?;
        AudioCodec::from_header(u32::from_be_bytes(header))?
    } else {
        source.codec.context("缺少 audio codec 配置")?
    };
    Ok(PreparedAudioStream {
        stream: BufferedStream::new(stream),
        device_name,
        codec,
    })
}

// ── bootstrap 模式 ──

fn prepare_bootstrap_stream(address: &str, debug: bool) -> Result<PreparedStream> {
    let mut stream = scrcpy::connect_socket(address, "bootstrap")?;
    let prefix = scrcpy::sniff_read_prefix(&mut stream)?;
    let sniffed = scrcpy::sniff_bootstrap_stream(&prefix)?;
    match sniffed {
        scrcpy::SniffedStream::Audio {
            codec,
            device_name,
            consumed,
        } => {
            if debug {
                logbus::emit(&format!(
                    "[bootstrap] routed audio codec={} first_socket_meta={}",
                    codec.as_str(),
                    if device_name.is_some() { "yes" } else { "no" }
                ));
            }
            Ok(PreparedStream::Audio(PreparedAudioStream {
                stream: BufferedStream::from_parts(stream, &prefix[consumed..]),
                device_name,
                codec,
            }))
        }
        scrcpy::SniffedStream::Video {
            codec,
            size,
            device_name,
            consumed,
        } => {
            if debug {
                logbus::emit(&format!(
                    "[bootstrap] routed video codec={} width={} height={} first_socket_meta={}",
                    codec.as_str(),
                    size.width,
                    size.height,
                    if device_name.is_some() { "yes" } else { "no" }
                ));
            }
            Ok(PreparedStream::Video(PreparedVideoStream {
                stream: BufferedStream::from_parts(stream, &prefix[consumed..]),
                device_name,
                codec,
                size,
            }))
        }
    }
}

pub fn is_bootstrap_generation_changed_error(err: &anyhow::Error) -> bool {
    err.to_string().contains("bootstrap generation 已变化")
}

pub fn is_retryable_bootstrap_acquire_error(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("bootstrap 流在发送头部前已关闭")
        || message.contains("读取 bootstrap sniff 前缀失败")
        || message.contains("连接 bootstrap socket 失败")
}

pub fn generate_socket_id() -> u32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    nanos ^ std::process::id()
}

pub fn parse_host_port(target: &str) -> Result<(&str, u16)> {
    let (host, port_str) = target
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("adb connect 目标格式无效 (应为 host:port): {target}"))?;
    let port: u16 = port_str
        .parse()
        .with_context(|| format!("解析端口失败: {port_str}"))?;
    Ok((host, port))
}

// ── Guard ──

pub struct RemoteBootstrapGuard {
    shell_handle: Option<crate::adb::AdbShellHandle>,
    device_serial: Option<String>,
    forward_port: u16,
    server_addr: String,
}

impl RemoteBootstrapGuard {
    fn try_wait(&mut self) -> Result<bool> {
        let Some(handle) = self.shell_handle.as_mut() else {
            return Ok(true);
        };
        if handle
            .try_wait()
            .context("检查远端 scrcpy server 进程状态失败")?
        {
            self.shell_handle = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Drop for RemoteBootstrapGuard {
    fn drop(&mut self) {
        if let Some(mut handle) = self.shell_handle.take() {
            handle.kill();
        }
        if let Some(ref serial) = self.device_serial {
            let mut device = AdbDevice::new(&self.server_addr, serial);
            device.kill_forward(self.forward_port);
        }
    }
}

/// 执行一次完整的远端拉起序列（[1/4]..[4/4]）。
pub fn bootstrap_remote_guard(config: &ResolvedConfig) -> Result<RemoteBootstrapGuard> {
    let server_addr = config.adb_server_address.as_str();

    if let Some(target) = config.adb_connect.as_ref() {
        let (host, port) = parse_host_port(target)?;
        logbus::emit(&format!("[adb] connect {host}:{port}"));
        AdbDevice::connect(server_addr, host, port)?;
    }

    let forward_port = config.forward_port;
    let server_remote_path = config.server_remote_path.as_str();
    let server_version = config.scrcpy_server_version.as_str();
    let socket_id = generate_socket_id();
    let socket_name = format!("scrcpy_{socket_id:08x}");

    let mut device = AdbDevice::new(
        server_addr,
        config
            .device_serial
            .as_deref()
            .ok_or_else(|| anyhow!("缺少 device_serial"))?,
    );

    logbus::emit(&format!(
        "[1/4] 推送 scrcpy-server ({:.1} KB)",
        crate::SCRCPY_SERVER_BYTES.len() as f64 / 1024.0
    ));
    device
        .push_bytes(crate::SCRCPY_SERVER_BYTES, server_remote_path)
        .context("推送 scrcpy-server 失败")?;

    logbus::emit(&format!(
        "[2/4] 建立端口转发: tcp:{} -> localabstract:{}",
        forward_port, socket_name
    ));
    device.kill_forward(forward_port);
    device
        .forward(forward_port, &socket_name)
        .context("建立端口转发失败")?;

    logbus::emit("[3/4] 启动远端 scrcpy server");
    let server_args = vec![
        format!("CLASSPATH={server_remote_path}"),
        "app_process".to_string(),
        "/".to_string(),
        "com.genymobile.scrcpy.Server".to_string(),
        server_version.to_string(),
        format!("scid={socket_id:08x}"),
        "log_level=debug".to_string(),
        "tunnel_forward=true".to_string(),
        format!("video={}", config.enable_camera),
        format!("audio={}", config.enable_audio),
        "control=false".to_string(),
        "send_device_meta=true".to_string(),
        "send_dummy_byte=true".to_string(),
        format!("send_codec_meta={}", config.send_codec_meta),
        format!("send_frame_meta={}", config.send_frame_meta),
        format!("raw_stream={}", config.remote_raw_stream),
        "video_source=camera".to_string(),
        format!("camera_size={}", config.server_camera_size),
        format!("max_size={}", config.server_max_size),
        format!("video_codec_options={}", config.server_video_codec_options),
        format!("overlay={}", config.enable_overlay),
        "overlay_source=display".to_string(),
        format!(
            "overlay_codec_options={}",
            config.server_overlay_codec_options
        ),
        "audio_source=output".to_string(),
        format!("audio_codec={}", config.server_audio_codec),
    ];
    let shell_cmd = server_args.join(" ");
    let mut shell_handle = device
        .shell_spawn(&shell_cmd)
        .context("启动远端 scrcpy server 失败")?;

    std::thread::sleep(REMOTE_SERVER_READY_DELAY);
    if shell_handle
        .try_wait()
        .context("检查远端 scrcpy server 状态失败")?
    {
        shell_handle.kill();
        device.kill_forward(forward_port);
        bail!("远端 scrcpy server 提前退出");
    }

    // 提升 scrcpy server 进程优先级，防止画面卡顿
    match device.shell_exec("renice -20 $(pidof app_process) 2>/dev/null || true") {
        Ok(output) if !output.is_empty() => {
            logbus::emit(&format!("[4/4] scrcpy server 进程优先级已提升: {output}"));
        }
        Ok(_) => logbus::emit("[4/4] scrcpy server 进程优先级提升完成"),
        Err(err) => logbus::emit(&format!(
            "[4/4] 提升 scrcpy server 进程优先级失败（非关键错误）: {err:#}"
        )),
    }

    Ok(RemoteBootstrapGuard {
        shell_handle: Some(shell_handle),
        device_serial: config.device_serial.clone(),
        forward_port,
        server_addr: server_addr.to_string(),
    })
}

/// 带超时的 bootstrap：adb_client 在 server 半挂死时可能无限阻塞，
/// 超时后放弃本次尝试（稍后由重试循环再次拉起），避免重连线程永久卡死。
/// 超时后才完成的拉起线程会自行 Drop guard，清理 shell 与转发。
pub fn bootstrap_remote_guard_with_timeout(
    config: &ResolvedConfig,
    timeout: Duration,
) -> Result<RemoteBootstrapGuard> {
    let config = config.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(bootstrap_remote_guard(&config));
    });
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => Err(anyhow!("bootstrap 超时（{}s），稍后自动重试", timeout.as_secs())),
    }
}

// ── Manager ──

struct RemoteBootstrapState {
    generation: u64,
    guard: Option<RemoteBootstrapGuard>,
    reconnect_token: u64,
    restarting: bool,
    shutting_down: bool,
    pending_camera_streams: VecDeque<PreparedVideoStream>,
    pending_overlay_streams: VecDeque<PreparedVideoStream>,
    pending_audio_streams: VecDeque<PreparedAudioStream>,
    pending_unknown_video: Option<PreparedVideoStream>,
    last_camera_size: Option<FrameSize>,
    last_overlay_size: Option<FrameSize>,
}

pub enum RestartPlan {
    Immediate(u64),
    Scheduled { generation: u64, token: u64 },
}

/// 三路 socket 分发 + 看门狗。多线程共享。
pub struct RemoteBootstrapManager {
    config: ResolvedConfig,
    state: Mutex<RemoteBootstrapState>,
    condvar: Condvar,
}

impl RemoteBootstrapManager {
    pub fn new(config: ResolvedConfig) -> Arc<Self> {
        let restarting = config.bootstrap_remote;
        Arc::new(Self {
            config,
            state: Mutex::new(RemoteBootstrapState {
                generation: 0,
                guard: None,
                reconnect_token: 0,
                restarting,
                shutting_down: false,
                pending_camera_streams: VecDeque::new(),
                pending_overlay_streams: VecDeque::new(),
                pending_audio_streams: VecDeque::new(),
                pending_unknown_video: None,
                last_camera_size: None,
                last_overlay_size: None,
            }),
            condvar: Condvar::new(),
        })
    }

    pub fn config(&self) -> &ResolvedConfig {
        &self.config
    }

    pub fn spawn_initial_bootstrap(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let _ = std::thread::Builder::new()
            .name("bootstrap-initial".to_string())
            .spawn(move || manager.initial_bootstrap_loop());
    }

    pub fn spawn_guard_monitor(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let _ = std::thread::Builder::new()
            .name("bootstrap-guard-monitor".to_string())
            .spawn(move || manager.guard_monitor_loop());
    }

    pub fn wait_until_ready(&self) -> u64 {
        let mut state = self.state.lock().expect("bootstrap state poisoned");
        loop {
            if state.guard.is_some() && !state.restarting {
                return state.generation;
            }
            if state.shutting_down {
                return state.generation;
            }
            state = self.condvar.wait(state).expect("bootstrap state poisoned");
        }
    }

    fn reset_stream_dispatch_state(state: &mut RemoteBootstrapState) {
        state.pending_camera_streams.clear();
        state.pending_overlay_streams.clear();
        state.pending_audio_streams.clear();
        state.pending_unknown_video = None;
        state.last_camera_size = None;
        state.last_overlay_size = None;
    }

    pub fn acquire_video_stream(
        &self,
        role: &str,
        address: &str,
        generation: u64,
        debug: bool,
    ) -> Result<PreparedVideoStream> {
        loop {
            {
                let mut state = self.state.lock().expect("bootstrap state poisoned");
                if state.shutting_down {
                    bail!("bootstrap 已关闭");
                }
                if state.generation != generation || state.restarting {
                    bail!("bootstrap generation 已变化，放弃当前视频连接");
                }
                let ready = match role {
                    "camera" => state.pending_camera_streams.pop_front(),
                    "overlay" => state.pending_overlay_streams.pop_front(),
                    _ => None,
                };
                if let Some(stream) = ready {
                    return Ok(stream);
                }
            }

            let prepared = match prepare_bootstrap_stream(address, debug) {
                Ok(prepared) => prepared,
                Err(err) if is_retryable_bootstrap_acquire_error(&err) => {
                    std::thread::sleep(RECONNECT_DELAY);
                    continue;
                }
                Err(err) => return Err(err),
            };
            let mut state = self.state.lock().expect("bootstrap state poisoned");
            if state.shutting_down {
                bail!("bootstrap 已关闭");
            }
            if state.generation != generation || state.restarting {
                continue;
            }
            self.store_prepared_stream(&mut state, prepared)?;
            self.condvar.notify_all();
        }
    }

    pub fn acquire_audio_stream(
        &self,
        address: &str,
        generation: u64,
        debug: bool,
    ) -> Result<PreparedAudioStream> {
        loop {
            {
                let mut state = self.state.lock().expect("bootstrap state poisoned");
                if state.shutting_down {
                    bail!("bootstrap 已关闭");
                }
                if state.generation != generation || state.restarting {
                    bail!("bootstrap generation 已变化，放弃当前音频连接");
                }
                if let Some(stream) = state.pending_audio_streams.pop_front() {
                    return Ok(stream);
                }
            }

            let prepared = match prepare_bootstrap_stream(address, debug) {
                Ok(prepared) => prepared,
                Err(err) if is_retryable_bootstrap_acquire_error(&err) => {
                    std::thread::sleep(RECONNECT_DELAY);
                    continue;
                }
                Err(err) => return Err(err),
            };
            let mut state = self.state.lock().expect("bootstrap state poisoned");
            if state.shutting_down {
                bail!("bootstrap 已关闭");
            }
            if state.generation != generation || state.restarting {
                continue;
            }
            self.store_prepared_stream(&mut state, prepared)?;
            self.condvar.notify_all();
        }
    }

    fn store_prepared_stream(
        &self,
        state: &mut RemoteBootstrapState,
        prepared: PreparedStream,
    ) -> Result<()> {
        match prepared {
            PreparedStream::Audio(stream) => {
                state.pending_audio_streams.push_back(stream);
            }
            PreparedStream::Video(stream) => {
                self.store_video_stream(state, stream)?;
            }
        }
        Ok(())
    }

    /// camera/overlay 路由：面积启发式（移植 mixplayer store_video_stream）。
    fn store_video_stream(
        &self,
        state: &mut RemoteBootstrapState,
        stream: PreparedVideoStream,
    ) -> Result<()> {
        let camera_enabled = self.config.enable_camera;
        let overlay_enabled = self.config.enable_overlay;
        if camera_enabled && !overlay_enabled {
            state.last_camera_size = Some(stream.size);
            state.pending_camera_streams.push_back(stream);
            return Ok(());
        }
        if overlay_enabled && !camera_enabled {
            state.last_overlay_size = Some(stream.size);
            state.pending_overlay_streams.push_back(stream);
            return Ok(());
        }
        if !camera_enabled && !overlay_enabled {
            bail!("未启用任何视频流，却收到视频连接");
        }

        if let (Some(camera_size), Some(overlay_size)) =
            (state.last_camera_size, state.last_overlay_size)
        {
            if scrcpy::closer_to_camera(stream.size, camera_size, overlay_size) {
                state.last_camera_size = Some(stream.size);
                state.pending_camera_streams.push_back(stream);
            } else {
                state.last_overlay_size = Some(stream.size);
                state.pending_overlay_streams.push_back(stream);
            }
            return Ok(());
        }

        if let Some(camera_size) = state.last_camera_size {
            if stream.size.area() < camera_size.area() {
                state.last_overlay_size = Some(stream.size);
                state.pending_overlay_streams.push_back(stream);
            } else {
                state.last_camera_size = Some(stream.size);
                state.pending_camera_streams.push_back(stream);
            }
            return Ok(());
        }

        if let Some(overlay_size) = state.last_overlay_size {
            if stream.size.area() > overlay_size.area() {
                state.last_camera_size = Some(stream.size);
                state.pending_camera_streams.push_back(stream);
            } else {
                state.last_overlay_size = Some(stream.size);
                state.pending_overlay_streams.push_back(stream);
            }
            return Ok(());
        }

        if let Some(other) = state.pending_unknown_video.take() {
            let current_area = stream.size.area();
            let other_area = other.size.area();
            if current_area == other_area {
                bail!(
                    "无法仅根据视频分辨率区分 camera/overlay: {}x{} vs {}x{}",
                    stream.size.width,
                    stream.size.height,
                    other.size.width,
                    other.size.height
                );
            }
            if current_area > other_area {
                state.last_camera_size = Some(stream.size);
                state.last_overlay_size = Some(other.size);
                state.pending_camera_streams.push_back(stream);
                state.pending_overlay_streams.push_back(other);
            } else {
                state.last_camera_size = Some(other.size);
                state.last_overlay_size = Some(stream.size);
                state.pending_camera_streams.push_back(other);
                state.pending_overlay_streams.push_back(stream);
            }
            return Ok(());
        }

        state.pending_unknown_video = Some(stream);
        Ok(())
    }

    fn poll_guard_exit_generation(&self) -> Result<Option<u64>> {
        let mut state = self.state.lock().expect("bootstrap state poisoned");
        if state.shutting_down || state.restarting {
            return Ok(None);
        }
        let generation = state.generation;
        let Some(guard) = state.guard.as_mut() else {
            return Ok(None);
        };
        if guard.try_wait()? {
            Ok(Some(generation))
        } else {
            Ok(None)
        }
    }

    fn guard_monitor_loop(self: Arc<Self>) {
        loop {
            std::thread::sleep(BOOTSTRAP_HEALTH_POLL_INTERVAL);

            let exited_generation = match self.poll_guard_exit_generation() {
                Ok(generation) => generation,
                Err(err) => {
                    logbus::emit(&format!("[watchdog] 检查远端 scrcpy server 状态失败: {err:#}"));
                    continue;
                }
            };

            let Some(generation) = exited_generation else {
                let state = self.state.lock().expect("bootstrap state poisoned");
                if state.shutting_down {
                    break;
                }
                continue;
            };

            logbus::emit("[watchdog] 远端 scrcpy server 已退出，准备触发重连");
            if let Err(err) = self.execute_restart_plan(self.schedule_restart(generation)) {
                logbus::emit(&format!("[watchdog] 触发重连失败: {err:#}"));
                std::thread::sleep(RESTART_RETRY_DELAY);
            }
        }
    }

    fn initial_bootstrap_loop(self: Arc<Self>) {
        loop {
            {
                let state = self.state.lock().expect("bootstrap state poisoned");
                if state.shutting_down {
                    break;
                }
            }
            match bootstrap_remote_guard_with_timeout(&self.config, Duration::from_secs(120)) {
                Ok(guard) => {
                    let mut state = self.state.lock().expect("bootstrap state poisoned");
                    if state.shutting_down {
                        drop(state);
                        drop(guard);
                        break;
                    }
                    state.guard = Some(guard);
                    state.restarting = false;
                    Self::reset_stream_dispatch_state(&mut state);
                    state.reconnect_token = state.reconnect_token.wrapping_add(1);
                    self.condvar.notify_all();
                    logbus::emit("[bootstrap] 远端 scrcpy server 就绪");
                    break;
                }
                Err(err) => {
                    {
                        let mut state = self.state.lock().expect("bootstrap state poisoned");
                        if state.shutting_down {
                            break;
                        }
                        state.restarting = false;
                        state.reconnect_token = state.reconnect_token.wrapping_add(1);
                        self.condvar.notify_all();
                    }
                    logbus::emit(&format!(
                        "[bootstrap] 初始化远端失败，{} 秒后重试: {err:#}",
                        RESTART_RETRY_DELAY.as_secs()
                    ));
                    std::thread::sleep(RESTART_RETRY_DELAY);
                    let mut state = self.state.lock().expect("bootstrap state poisoned");
                    if state.shutting_down {
                        break;
                    }
                    if state.guard.is_none() {
                        state.restarting = true;
                    }
                }
            }
        }
    }

    pub fn shutdown(&self) {
        let old_guard = {
            let mut state = self.state.lock().expect("bootstrap state poisoned");
            state.shutting_down = true;
            state.restarting = false;
            Self::reset_stream_dispatch_state(&mut state);
            state.reconnect_token = state.reconnect_token.wrapping_add(1);
            self.condvar.notify_all();
            state.guard.take()
        };
        drop(old_guard);
    }

    pub fn schedule_restart(&self, observed_generation: u64) -> RestartPlan {
        let mut state = self.state.lock().expect("bootstrap state poisoned");
        if state.generation != observed_generation || state.restarting {
            return RestartPlan::Immediate(state.generation);
        }
        state.reconnect_token = state.reconnect_token.wrapping_add(1);
        RestartPlan::Scheduled {
            generation: observed_generation,
            token: state.reconnect_token,
        }
    }

    pub fn execute_restart_plan(&self, plan: RestartPlan) -> Result<u64> {
        match plan {
            RestartPlan::Immediate(generation) => Ok(generation),
            RestartPlan::Scheduled { generation, token } => {
                let old_guard = {
                    let mut state = self.state.lock().expect("bootstrap state poisoned");
                    loop {
                        if state.generation != generation || state.reconnect_token != token {
                            return Ok(state.generation);
                        }
                        let (next_state, wait_result) = self
                            .condvar
                            .wait_timeout(state, RECONNECT_DELAY)
                            .expect("bootstrap state poisoned");
                        state = next_state;
                        if state.generation != generation || state.reconnect_token != token {
                            return Ok(state.generation);
                        }
                        if wait_result.timed_out() {
                            break;
                        }
                    }
                    if state.generation != generation || state.reconnect_token != token {
                        return Ok(state.generation);
                    }
                    if state.restarting {
                        return Ok(state.generation);
                    }
                    state.restarting = true;
                    state.guard.take()
                };

                drop(old_guard);
                let restart_result = bootstrap_remote_guard_with_timeout(&self.config, Duration::from_secs(60));

                let mut state = self.state.lock().expect("bootstrap state poisoned");
                state.restarting = false;
                match restart_result {
                    Ok(new_guard) => {
                        state.guard = Some(new_guard);
                        state.generation += 1;
                        Self::reset_stream_dispatch_state(&mut state);
                        state.reconnect_token = state.reconnect_token.wrapping_add(1);
                        let next_generation = state.generation;
                        self.condvar.notify_all();
                        Ok(next_generation)
                    }
                    Err(err) => {
                        state.reconnect_token = state.reconnect_token.wrapping_add(1);
                        self.condvar.notify_all();
                        Err(err)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::PlayerOptions;
    use std::net::TcpStream;

    fn test_config() -> ResolvedConfig {
        let mut opts = PlayerOptions::default();
        opts.bootstrap_remote = Some(true);
        opts.device_serial = Some("test-serial".into());
        opts.enable_audio = Some(false);
        opts.resolve().unwrap()
    }

    fn camera_only_config() -> ResolvedConfig {
        let mut opts = PlayerOptions::default();
        opts.bootstrap_remote = Some(true);
        opts.device_serial = Some("test-serial".into());
        opts.enable_audio = Some(false);
        opts.enable_overlay = Some(false);
        opts.resolve().unwrap()
    }

    #[test]
    fn routing_single_camera() {
        let manager = RemoteBootstrapManager::new(camera_only_config());
        let mut state = manager.state.lock().unwrap();
        let stream = fake_video(1920, 1080);
        manager.store_video_stream(&mut state, stream).unwrap();
        assert_eq!(state.pending_camera_streams.len(), 1);
        assert_eq!(state.last_camera_size.unwrap().width, 1920);
    }

    #[test]
    fn routing_unknown_then_pair_by_area() {
        let manager = RemoteBootstrapManager::new(test_config());
        let mut state = manager.state.lock().unwrap();
        let big = fake_video(1920, 1080);
        manager.store_video_stream(&mut state, big).unwrap();
        // 第一路未知，暂存
        assert!(state.pending_unknown_video.is_some());

        let small = fake_video(640, 480);
        manager.store_video_stream(&mut state, small).unwrap();
        assert_eq!(state.pending_camera_streams.len(), 1);
        assert_eq!(state.pending_overlay_streams.len(), 1);
        assert!(state.pending_unknown_video.is_none());
        assert_eq!(state.last_camera_size.unwrap().width, 1920);
        assert_eq!(state.last_overlay_size.unwrap().width, 640);
    }

    #[test]
    fn routing_equal_area_rejected() {
        let manager = RemoteBootstrapManager::new(test_config());
        let mut state = manager.state.lock().unwrap();
        let a = fake_video(640, 480);
        manager.store_video_stream(&mut state, a).unwrap();
        let b = fake_video(480, 640); // 相同面积
        let err = manager.store_video_stream(&mut state, b).unwrap_err();
        assert!(err.to_string().contains("无法仅根据视频分辨率区分"));
    }

    #[test]
    fn routing_with_known_sizes_uses_nearest() {
        let manager = RemoteBootstrapManager::new(test_config());
        let mut state = manager.state.lock().unwrap();
        state.last_camera_size = Some(FrameSize { width: 1920, height: 1080 });
        state.last_overlay_size = Some(FrameSize { width: 640, height: 480 });
        // 面积 1280x720 介于两者之间，更接近 640x480
        manager
            .store_video_stream(&mut state, fake_video(1280, 720))
            .unwrap();
        assert_eq!(state.pending_overlay_streams.len(), 1);
    }

    #[test]
    fn socket_id_is_stable_format() {
        let id = generate_socket_id();
        assert_eq!(format!("scrcpy_{id:08x}").len(), "scrcpy_".len() + 8);
    }

    #[test]
    fn parse_host_port_ok_and_err() {
        assert_eq!(parse_host_port("192.168.1.2:5555").unwrap(), ("192.168.1.2", 5555));
        assert!(parse_host_port("no-port").is_err());
        assert!(parse_host_port("host:port").is_err());
    }

    fn fake_video(width: u32, height: u32) -> PreparedVideoStream {
        PreparedVideoStream {
            stream: BufferedStream::new(fake_stream()),
            device_name: None,
            codec: VideoCodec::H264,
            size: FrameSize { width, height },
        }
    }

    fn fake_stream() -> TcpStream {
        // 未连接的 TcpStream 仅用于占位（路由测试不触达 IO）
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .ok()
            .and_then(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok())
            .expect("loopback stream")
    }
}
