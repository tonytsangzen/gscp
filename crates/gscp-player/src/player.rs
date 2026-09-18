//! 播放器主循环：窗口/渲染/自适应节拍/标题状态/效果热更新。
//! 移植自 mixplayer run_player，新增效果参数文件监视（运行中热调）。

use crate::render;
use crate::session;
use crate::slot::{self, SharedSlot};
use anyhow::{anyhow, Result};
use gscp_core::bootstrap::RemoteBootstrapManager;
use gscp_core::logbus;
use gscp_core::scrcpy::{AudioSource, VideoSource, CAMERA_STALL_SECS};
use gscp_core::settings::{self, EffectParams, PlayerOptions, ResolvedConfig};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowAttributes};

/// 播放器入口：解析选项 → bootstrap → 三路源 → 渲染循环。
pub fn run_player(opts: PlayerOptions) -> Result<()> {
    let config = opts.resolve()?;
    logbus::emit(&format!(
        "[startup] gscp-player v{} | streams c:{} a:{} o:{} | decoder={:?}",
        gscp_core::VERSION,
        config.enable_camera,
        config.enable_audio,
        config.enable_overlay,
        config.decoder_backend
    ));

    let effects = Arc::new(Mutex::new(config.effects));
    let settings_path = opts
        .config
        .clone()
        .unwrap_or_else(gscp_core::paths::settings_path);

    let bootstrap_mgr = prepare_sources(&config)?;

    let base_slot = SharedSlot::default();
    let overlay_slot = SharedSlot::default();
    if config.enable_camera {
        let port = config
            .camera_port
            .ok_or_else(|| anyhow!("远端模式缺少 camera 端口"))?;
        let mut camera = VideoSource::new(
            config.remote_host.clone(),
            port,
            config.camera_send_device_meta,
            config.send_frame_meta,
            config.camera_send_dummy_byte,
            config.send_codec_meta,
            None,
            None,
            config.remote_raw_stream,
            config.remote_debug,
        )?;
        // 断联判定仅以 camera 码流为准：拍照占用会短暂停流，静默 6s 才判定
        camera.stall_timeout = Some(Duration::from_secs(CAMERA_STALL_SECS));
        session::spawn_video_source(
            "camera",
            camera,
            base_slot.clone(),
            config.decoder_backend,
            bootstrap_mgr.clone(),
        )?;
    } else {
        base_slot.set_info("disabled".to_string());
    }

    if config.enable_overlay {
        if let Some(port) = config.remote_overlay_port {
            overlay_slot.set_info("connecting".to_string());
            let overlay = VideoSource::new(
                config.remote_host.clone(),
                port,
                config.overlay_send_device_meta,
                config.send_frame_meta,
                config.overlay_send_dummy_byte,
                config.send_codec_meta,
                None,
                None,
                config.remote_raw_stream,
                config.remote_debug,
            )?;
            session::spawn_video_source(
                "overlay",
                overlay,
                overlay_slot.clone(),
                config.decoder_backend,
                bootstrap_mgr.clone(),
            )?;
        }
    } else {
        overlay_slot.set_info("disabled".to_string());
    }

    if config.enable_audio {
        if let Some(port) = config.audio_port {
            let audio = AudioSource::new(
                config.remote_host.clone(),
                port,
                config.audio_send_device_meta,
                config.send_frame_meta,
                config.audio_send_dummy_byte,
                config.send_codec_meta,
                None,
                config.remote_raw_stream,
                config.remote_debug,
            )?;
            session::spawn_audio_source("audio", audio, bootstrap_mgr.clone())?;
        }
    }

    let result = run_window_loop(&config, effects, settings_path, base_slot, overlay_slot);
    if let Some(manager) = bootstrap_mgr.as_ref() {
        manager.shutdown();
    }
    result
}

/// bootstrap 模式下确保 adb 可用（按需下载）并创建会话管理器。
fn prepare_sources(config: &ResolvedConfig) -> Result<Option<Arc<RemoteBootstrapManager>>> {
    if !config.bootstrap_remote {
        return Ok(None);
    }
    let adb_path = gscp_core::adb::ensure_adb_available()?;
    gscp_core::logbus::emit(&format!("[adb] 使用工具: {adb_path}"));
    let manager = RemoteBootstrapManager::new(config.clone());
    manager.spawn_initial_bootstrap();
    manager.spawn_guard_monitor();
    Ok(Some(manager))
}

/// 效果参数热更新：监视 settings 文件 mtime，变化即重载效果子集。
fn spawn_effects_watcher(
    settings_path: std::path::PathBuf,
    effects: Arc<Mutex<EffectParams>>,
) {
    std::thread::Builder::new()
        .name("effects-watcher".to_string())
        .spawn(move || {
            let mut last_modified = file_mtime(&settings_path);
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let mtime = file_mtime(&settings_path);
                if mtime != last_modified {
                    last_modified = mtime;
                    match settings::load_file(Some(&settings_path)) {
                        Ok(opts) => {
                            let new_effects = opts.effects();
                            *effects.lock().unwrap() = new_effects;
                            logbus::emit(&format!(
                                "[effects] 效果参数已热更新 alpha={:.2} scale={:.2}",
                                new_effects.overlay_alpha, new_effects.overlay_scale
                            ));
                        }
                        Err(err) => {
                            logbus::emit(&format!("[effects] 重载设置失败: {err:#}"));
                        }
                    }
                }
            }
        })
        .ok();
}

fn file_mtime(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn run_window_loop(
    config: &ResolvedConfig,
    effects: Arc<Mutex<EffectParams>>,
    settings_path: std::path::PathBuf,
    base: SharedSlot,
    overlay: SharedSlot,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = PlayerApp {
        config: config.clone(),
        effects,
        settings_path,
        base,
        overlay,
        window: None,
        renderer: None,
        startup_at: Instant::now(),
        frame_interval: Duration::from_secs_f64(1.0 / config.fps.max(1.0)),
        next_frame: Instant::now(),
        first_base_frame_logged: false,
        last_left_click_at: None,
        last_buffer_log: Instant::now(),
        render_frames_since_log: 0,
        last_bytes_received: 0,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// winit 0.30 ApplicationHandler：窗口生命周期 + 渲染节拍。
struct PlayerApp {
    config: ResolvedConfig,
    effects: Arc<Mutex<EffectParams>>,
    settings_path: std::path::PathBuf,
    base: SharedSlot,
    overlay: SharedSlot,
    window: Option<Arc<Window>>,
    renderer: Option<Box<dyn render::Renderer>>,
    startup_at: Instant,
    frame_interval: Duration,
    next_frame: Instant,
    first_base_frame_logged: bool,
    last_left_click_at: Option<Instant>,
    last_buffer_log: Instant,
    render_frames_since_log: u64,
    last_bytes_received: u64,
}

impl ApplicationHandler for PlayerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
        let requested = render::FrameSize {
            width: self.config.canvas_width,
            height: self.config.canvas_height,
        };
        let attrs: WindowAttributes = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(LogicalSize::new(
                requested.width as f64,
                requested.height as f64,
            ));
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                logbus::emit(&format!("创建原生窗口失败: {err}"));
                event_loop.exit();
                return;
            }
        };
        let startup_canvas_size = window
            .current_monitor()
            .map(|monitor| fit_size_to_aspect(requested, monitor.size()))
            .unwrap_or(requested);
        let _ = window.request_inner_size(LogicalSize::new(
            startup_canvas_size.width as f64,
            startup_canvas_size.height as f64,
        ));
        if self.config.fullscreen {
            window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
        }
        self.config.canvas_width = startup_canvas_size.width;
        self.config.canvas_height = startup_canvas_size.height;

        match render::create_renderer(window.clone(), self.effects.clone()) {
            Ok(renderer) => {
                spawn_effects_watcher(self.settings_path.clone(), self.effects.clone());
                self.renderer = Some(renderer);
                self.window = Some(window);
            }
            Err(err) => {
                logbus::emit(&format!("初始化渲染器失败: {err:#}"));
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                let now = Instant::now();
                if let Some(last_click_at) = self.last_left_click_at {
                    if now.duration_since(last_click_at) <= self.double_click_window() {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        } else {
                            window.set_fullscreen(Some(Fullscreen::Borderless(
                                window.current_monitor(),
                            )));
                        }
                        self.last_left_click_at = None;
                    } else {
                        self.last_left_click_at = Some(now);
                    }
                } else {
                    self.last_left_click_at = Some(now);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Err(err) = renderer.resize(size) {
                        logbus::emit(&format!("更新渲染尺寸失败: {err:#}"));
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                let base_snapshot = self.base.snapshot_for_render();
                let overlay_snapshot = self.overlay.snapshot_for_render();
                if !self.first_base_frame_logged && base_snapshot.frame.is_some() {
                    self.first_base_frame_logged = true;
                    logbus::emit(&format!(
                        "[player] first base frame after {} ms",
                        self.startup_at.elapsed().as_millis()
                    ));
                }
                self.render_frames_since_log += 1;
                if self.last_buffer_log.elapsed() >= Duration::from_secs(1) {
                    let elapsed = self.last_buffer_log.elapsed().as_secs_f64();
                    let render_fps = self.render_frames_since_log as f64 / elapsed;
                    let bytes_delta = base_snapshot
                        .bytes_received
                        .saturating_sub(self.last_bytes_received);
                    let kbps = bytes_delta as f64 / elapsed / 1024.0;
                    self.last_buffer_log = Instant::now();
                    self.render_frames_since_log = 0;
                    self.last_bytes_received = base_snapshot.bytes_received;
                    let camera_fps = if base_snapshot.avg_frame_interval_secs > 0.0 {
                        1.0 / base_snapshot.avg_frame_interval_secs
                    } else {
                        0.0
                    };
                    logbus::emit(&format!(
                        "[buffer] camera={}/{} (src={:.1}fps rdr={:.0}fps net={:.0}KB/s) overlay={}/{}",
                        base_snapshot.buffer_depth,
                        slot::VIDEO_BUFFER_CAPACITY,
                        camera_fps,
                        render_fps,
                        kbps,
                        overlay_snapshot.buffer_depth,
                        slot::VIDEO_BUFFER_CAPACITY,
                    ));
                }
                if let Err(err) = renderer.render(&base_snapshot, &overlay_snapshot) {
                    logbus::emit(&format!("渲染失败: {err:#}"));
                    event_loop.exit();
                    return;
                }
                window.set_title(&build_title(
                    &self.config.title,
                    self.config.enable_camera,
                    self.config.enable_audio,
                    self.config.enable_overlay,
                    &base_snapshot,
                    &overlay_snapshot,
                    renderer.backend_name(),
                ));
                if base_snapshot.ended && overlay_snapshot.ended {
                    event_loop.exit();
                    return;
                }
                let playback_buffer_depth = if self.config.enable_camera {
                    base_snapshot.buffer_depth
                } else if self.config.enable_overlay {
                    overlay_snapshot.buffer_depth
                } else {
                    0
                };
                // 相机实测帧间隔优先，否则回退 --fps
                let camera_frame_interval = if base_snapshot.avg_frame_interval_secs > 0.0 {
                    Duration::from_secs_f64(base_snapshot.avg_frame_interval_secs)
                } else {
                    self.frame_interval
                };
                self.next_frame = Instant::now()
                    + slot::playback_interval_for_buffer_depth(
                        camera_frame_interval,
                        playback_buffer_depth,
                    );
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // 仅在下一帧到期时请求重绘，避免烧 CPU 的紧循环
        if Instant::now() >= self.next_frame {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

impl PlayerApp {
    fn double_click_window(&self) -> Duration {
        Duration::from_millis(300)
    }
}

fn fit_size_to_aspect(
    requested: render::FrameSize,
    target: winit::dpi::PhysicalSize<u32>,
) -> render::FrameSize {
    let requested_width = requested.width.max(1);
    let requested_height = requested.height.max(1);
    let target_width = target.width.max(1);
    let target_height = target.height.max(1);
    let target_aspect = target_width as f64 / target_height as f64;

    let mut width = requested_width;
    let mut height = ((width as f64) / target_aspect).round().max(1.0) as u32;
    if height > requested_height {
        height = requested_height;
        width = ((height as f64) * target_aspect).round().max(1.0) as u32;
    }

    render::FrameSize { width, height }
}

fn build_title(
    title: &str,
    enable_camera: bool,
    enable_audio: bool,
    enable_overlay: bool,
    base: &slot::SlotSnapshot,
    overlay: &slot::SlotSnapshot,
    gpu_backend: &str,
) -> String {
    let mut parts = vec![
        title.to_string(),
        format!(
            "streams=c:{} a:{} o:{}",
            stream_flag(enable_camera),
            stream_flag(enable_audio),
            stream_flag(enable_overlay)
        ),
    ];
    if let Some(info) = base.info.as_ref() {
        parts.push(format!("base={}", compact_title_text(info, 32)));
    }
    if let Some(info) = overlay.info.as_ref() {
        parts.push(format!("overlay={}", compact_title_text(info, 28)));
    }
    if let Some(err) = base.error.as_ref() {
        parts.push(format!("base_err={}", compact_title_text(err, 32)));
    }
    if let Some(err) = overlay.error.as_ref() {
        parts.push(format!("overlay_err={}", compact_title_text(err, 28)));
    }
    parts.push(format!("gpu={gpu_backend}"));
    parts.join(" | ")
}

fn stream_flag(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn compact_title_text(text: &str, max_chars: usize) -> String {
    let mut compact = text.replace('\n', " ").replace('\r', " ");
    compact = compact.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let truncated = compact
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_compacts_and_truncates() {
        assert_eq!(compact_title_text("a  b\nc", 10), "a b c");
        let long = compact_title_text("0123456789abcdef", 8);
        assert_eq!(long, "01234...");
    }

    #[test]
    fn fit_size_keeps_aspect() {
        let fit = fit_size_to_aspect(
            render::FrameSize { width: 1280, height: 720 },
            winit::dpi::PhysicalSize::new(2560, 1440),
        );
        assert_eq!(fit.width, 1280);
        assert_eq!(fit.height, 720);

        let fit = fit_size_to_aspect(
            render::FrameSize { width: 1280, height: 720 },
            winit::dpi::PhysicalSize::new(1920, 2160),
        );
        // 16:9 请求适配 8:9 显示器 → 高度压到 720
        assert_eq!(fit.height, 720);
        assert!(fit.width < 1280);
    }
}
