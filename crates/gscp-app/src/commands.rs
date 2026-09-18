//! Tauri commands：GlassConnect 全部功能移植 + 效果参数与工具链管理。
//!
//! 日志统一走 gscp-core 的 logbus（setup 时订阅转发到 WebView）。

use gscp_core::logbus;
use gscp_core::settings::{self, EffectParams, PlayerOptions};
use gscp_core::userconfig::UserConfig;
use serde_json::json;
use std::sync::Mutex;

pub struct AppState {
    pub config: Mutex<UserConfig>,
}

const CONNECTION_MODE_WIFI: &str = "wifi";
const CONNECTION_MODE_USB: &str = "usb";
const USB_FORWARD_PORT: u16 = 27183;
const DEVICE_MODEL_PREFIX: &str = "RG-";

fn normalize_connection_mode(mode: &str) -> &'static str {
    if mode.eq_ignore_ascii_case(CONNECTION_MODE_USB) {
        CONNECTION_MODE_USB
    } else {
        CONNECTION_MODE_WIFI
    }
}

fn is_valid_ip(ip: &str) -> bool {
    gscp_core::wifi::is_valid_ipv4(ip)
}

struct StreamConfig {
    overlay: bool,
    camera: bool,
    audio: bool,
}

/// 音频 codec 协商：unix 用 opus（系统 libopus），Windows 回退 raw。
/// settings.yaml 显式配置优先。
fn negotiated_audio_codec(audio_enabled: bool) -> String {
    if !audio_enabled {
        return "raw".to_string();
    }
    let shared = settings::load_shared();
    match shared.server_audio_codec.clone() {
        Some(codec) if !codec.is_empty() => codec,
        _ => {
            if cfg!(unix) { "opus".to_string() } else { "raw".to_string() }
        }
    }
}

fn load_shared_or_default() -> PlayerOptions {
    settings::load_shared()
}

/// spawn 自身 `--player` 模式（单二进制双模式）。
fn spawn_player(args: &[String]) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let mut command = std::process::Command::new(&exe);
    command.arg("--player").args(args);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW：播放器是 GUI 窗口，不需要控制台
        command.creation_flags(0x0800_0000);
    }

    logbus::emit(&format!("[preview] 启动播放器: {} --player {}", exe.display(), args.join(" ")));
    let mut child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    if let Some(stdout) = child.stdout.take() {
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                logbus::emit(&line);
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
                logbus::emit(&line);
            }
        });
    }
    std::thread::spawn(move || match child.wait() {
        Ok(status) => {
            if let Some(code) = status.code() {
                logbus::emit(&format!("预览播放器已退出，exit code={code}"));
            } else {
                logbus::emit("预览播放器已退出");
            }
        }
        Err(e) => logbus::emit(&format!("等待预览播放器退出失败: {e}")),
    });
    Ok(())
}

fn build_player_common_args(stream: &StreamConfig) -> Vec<String> {
    let settings = load_shared_or_default();
    vec![
        "--bootstrap-remote".to_string(),
        "true".to_string(),
        "--fullscreen".to_string(),
        "true".to_string(),
        "--enable-overlay".to_string(),
        stream.overlay.to_string(),
        "--enable-camera".to_string(),
        stream.camera.to_string(),
        "--enable-audio".to_string(),
        stream.audio.to_string(),
        "--server-audio-codec".to_string(),
        negotiated_audio_codec(stream.audio),
        "--scrcpy-server-version".to_string(),
        settings
            .scrcpy_server_version
            .clone()
            .unwrap_or_else(|| "3.3.1".to_string()),
    ]
}

// ── 连接管理（移植 GlassConnect）──

#[tauri::command]
pub fn start_wifi(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    ssid: String,
    password: String,
    keepalive: bool,
) -> Result<(), String> {
    if ssid.is_empty() {
        logbus::emit("请输入有效SSID");
        return Err("SSID 不能为空".into());
    }

    {
        let mut config = state.config.lock().unwrap();
        config.add_wifi_network(&ssid, &password, "");
    }

    logbus::emit("连接到Glasses...");
    let handle = app_handle.clone();
    std::thread::spawn(move || {
        match run_start_wifi_sync(&ssid, &password, keepalive) {
            Ok(ip) => {
                use tauri::Emitter;
                let _ = handle.emit("wifi-connected", json!({ "ip": ip }));
            }
            Err(e) => logbus::emit(&format!("错误: {e:#}")),
        }
    });
    Ok(())
}

fn run_start_wifi_sync(ssid: &str, password: &str, keepalive: bool) -> anyhow::Result<String> {
    logbus::emit("清理本地 ADB 连接...");
    match gscp_core::adb::cleanup_local_connections() {
        Ok(actions) => {
            for action in actions {
                logbus::emit(&format!("[adb-cleanup] {action}"));
            }
        }
        Err(e) => logbus::emit(&format!("[adb-cleanup] 忽略清理失败: {e:#}")),
    }
    std::thread::sleep(std::time::Duration::from_millis(300));

    logbus::emit("查找设备...");
    let mut wrapper = gscp_core::adb::AdbWrapper::new()?;
    let mut device = wrapper.connect_device(DEVICE_MODEL_PREFIX)?;

    logbus::emit("Enable Adb...");
    gscp_core::wifi::enable_persistent_adb(&mut device)?;

    logbus::emit("Start Wifi...");
    gscp_core::wifi::enable_wifi(&mut device)?;
    gscp_core::wifi::start_scan(&mut device)?;

    logbus::emit("清除已有网络...");
    gscp_core::wifi::remove_all_networks(&mut device)?;

    logbus::emit("Add Network...");
    gscp_core::wifi::add_network(&mut device, ssid, password)?;

    if !gscp_core::wifi::has_network(&mut device, ssid) {
        anyhow::bail!("无法添加网络");
    }

    logbus::emit("等待获取IP地址...");
    let ip = gscp_core::wifi::get_ip_address(&mut device, 30)?;
    logbus::emit(&format!("连接成功:{ip}"));

    if keepalive {
        logbus::emit("启动wifi保活...");
        match wrapper.connect_device(DEVICE_MODEL_PREFIX) {
            Ok(mut d) => {
                gscp_core::daemon::deploy_wifi_daemon(&mut d)?;
                logbus::emit("守护进程已部署");
            }
            Err(e) => logbus::emit(&format!("守护进程部署失败: {e:#}")),
        }
    }

    logbus::emit("启用 ADB TCP/IP...");
    gscp_core::wifi::enable_adb_tcpip(5555)?;

    logbus::emit("等待 ADB TCP 就绪...");
    let mut connected = false;
    for i in 0..10 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        if let Ok(mut w) = gscp_core::adb::AdbWrapper::new() {
            match w.connect_tcp(&ip, 5555) {
                Ok(()) => {
                    connected = true;
                    logbus::emit(&format!("TCP 连接验证成功 {ip}:5555"));
                    break;
                }
                Err(_) => {
                    if i == 0 || i == 4 || i == 8 {
                        logbus::emit(&format!("等待设备 TCP 就绪 ({}s)...", i + 1));
                    }
                }
            }
        }
    }
    if !connected {
        logbus::emit(&format!("警告: 无法通过 TCP 连接 {ip}:5555, 但 WiFi 已配置"));
    }

    Ok(ip)
}

#[tauri::command]
pub fn stop_wifi(app_handle: tauri::AppHandle) -> Result<(), String> {
    logbus::emit("连接到Glasses...");
    let handle = app_handle.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_stop_wifi_sync() {
            logbus::emit(&format!("错误: {e:#}"));
        }
        let _ = handle;
    });
    Ok(())
}

fn run_stop_wifi_sync() -> anyhow::Result<()> {
    logbus::emit("查找设备...");
    let mut wrapper = gscp_core::adb::AdbWrapper::new()?;
    let mut device = wrapper.connect_device(DEVICE_MODEL_PREFIX)?;

    logbus::emit("Stop Wifi...");
    gscp_core::wifi::enable_persistent_adb(&mut device)?;
    gscp_core::daemon::stop_wifi_daemon(&mut device)?;
    gscp_core::wifi::disable_wifi(&mut device)?;

    logbus::emit("Disable Adb...");
    gscp_core::wifi::disable_persistent_adb(&mut device)?;
    logbus::emit("设置成功");
    Ok(())
}

#[tauri::command]
pub fn preview(
    app_handle: tauri::AppHandle,
    connection_mode: String,
    ip: String,
    overlay: bool,
    camera: bool,
    audio: bool,
) -> Result<(), String> {
    let connection_mode = normalize_connection_mode(&connection_mode);
    let _ = &app_handle;
    let stream_config = StreamConfig { overlay, camera, audio };

    let result = match connection_mode {
        CONNECTION_MODE_USB => {
            logbus::emit("启动有线预览...");
            std::thread::spawn(move || run_usb_preview_sync(&stream_config))
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("预览线程异常退出")))
        }
        _ => {
            if !is_valid_ip(&ip) {
                logbus::emit("无效的IP地址，请先启动WIFI");
                return Err("无效的IP地址".into());
            }
            logbus::emit(&format!("启动预览播放器：{ip}"));
            let ip = ip.clone();
            std::thread::spawn(move || run_wifi_preview_sync(&ip, &stream_config))
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("预览线程异常退出")))
        }
    };
    result.map_err(|e| format!("{e:#}"))
}

fn run_wifi_preview_sync(ip: &str, stream: &StreamConfig) -> anyhow::Result<()> {
    let adb_target = format!("{ip}:5555");
    let mut args = build_player_common_args(stream);
    args.extend([
        "--adb-connect".to_string(),
        adb_target.clone(),
        "--device-serial".to_string(),
        adb_target,
    ]);
    spawn_player(&args)
}

fn run_usb_preview_sync(stream: &StreamConfig) -> anyhow::Result<()> {
    logbus::emit("清理本地 ADB 连接...");
    match gscp_core::adb::cleanup_local_connections() {
        Ok(actions) => {
            for action in actions {
                logbus::emit(&format!("[adb-cleanup] {action}"));
            }
        }
        Err(e) => logbus::emit(&format!("[adb-cleanup] 忽略清理失败: {e:#}")),
    }
    std::thread::sleep(std::time::Duration::from_millis(300));

    logbus::emit("查找 USB 设备...");
    let mut wrapper = gscp_core::adb::AdbWrapper::new()?;
    let serial = wrapper.find_device_identifier(DEVICE_MODEL_PREFIX, true)?;
    logbus::emit(&format!("USB 设备: {serial}"));
    logbus::emit("通过 ADB forward 建立本地 socket...");

    let mut args = build_player_common_args(stream);
    args.extend([
        "--device-serial".to_string(),
        serial,
        "--forward-port".to_string(),
        USB_FORWARD_PORT.to_string(),
        "--remote-host".to_string(),
        "127.0.0.1".to_string(),
    ]);
    spawn_player(&args)
}

// ── 配置（移植 GlassConnect）──

#[tauri::command]
pub fn get_config(state: tauri::State<'_, AppState>) -> serde_json::Value {
    let config = state.config.lock().unwrap();
    let last_ssid = config.get_last_ssid();
    let connection_mode = normalize_connection_mode(&config.get("connection_mode"));
    let overlay = config.get_bool("stream_overlay", true);
    let camera = config.get_bool("stream_camera", true);
    let audio = config.get_bool("stream_audio", true);
    if last_ssid.is_empty() {
        return json!({
            "connectionMode": connection_mode, "ssid": "", "psk": "", "ip": "",
            "overlay": overlay, "camera": camera, "audio": audio,
        });
    }
    let (psk, ip) = config.get_network(&last_ssid).unwrap_or_default();
    json!({
        "connectionMode": connection_mode, "ssid": last_ssid, "psk": psk, "ip": ip,
        "overlay": overlay, "camera": camera, "audio": audio,
    })
}

#[tauri::command]
pub fn save_config(
    state: tauri::State<'_, AppState>,
    connection_mode: String,
    ssid: String,
    psk: String,
    ip: String,
    overlay: bool,
    camera: bool,
    audio: bool,
) {
    let mut config = state.config.lock().unwrap();
    let connection_mode = normalize_connection_mode(&connection_mode);
    if connection_mode == CONNECTION_MODE_WIFI && !ssid.is_empty() {
        config.add_wifi_network(&ssid, &psk, &ip);
    }
    config.set("connection_mode", connection_mode);
    config.set_bool("stream_overlay", overlay);
    config.set_bool("stream_camera", camera);
    config.set_bool("stream_audio", audio);
}

#[tauri::command]
pub fn get_wifi_list(state: tauri::State<'_, AppState>) -> Vec<serde_json::Value> {
    let config = state.config.lock().unwrap();
    config
        .get_wifi_list()
        .into_iter()
        .map(|(ssid, psk, ip)| json!({ "ssid": ssid, "psk": psk, "ip": ip }))
        .collect()
}

#[tauri::command]
pub fn save_wifi_network(
    state: tauri::State<'_, AppState>,
    ssid: String,
    psk: String,
    ip: String,
) {
    let mut config = state.config.lock().unwrap();
    config.add_wifi_network(&ssid, &psk, &ip);
}

// ── 效果参数（播放器热更新）──

#[tauri::command]
pub fn get_effect() -> serde_json::Value {
    let effects = load_shared_or_default().effects();
    effect_to_json(&effects)
}

fn effect_to_json(e: &EffectParams) -> serde_json::Value {
    json!({
        "baseRotateDeg": e.base_rotate_deg,
        "baseBrightness": e.base_brightness,
        "overlayScale": e.overlay_scale,
        "overlayRotateDeg": e.overlay_rotate_deg,
        "overlayAlpha": e.overlay_alpha,
        "overlayBrightness": e.overlay_brightness,
        "dimStrength": e.dim_strength,
        "saturationBoost": e.saturation_boost,
        "overlayBlackKeyLow": e.overlay_black_key_low,
        "overlayBlackKeyHigh": e.overlay_black_key_high,
        "overlayFeatherPower": e.overlay_feather_power,
        "overlayFeatherRadius": e.overlay_feather_radius,
    })
}

#[tauri::command]
#[allow(non_snake_case)]
pub fn set_effect(
    baseRotateDeg: Option<i32>,
    baseBrightness: Option<f32>,
    overlayScale: Option<f32>,
    overlayRotateDeg: Option<i32>,
    overlayAlpha: Option<f32>,
    overlayBrightness: Option<f32>,
    dimStrength: Option<f32>,
    saturationBoost: Option<f32>,
    overlayBlackKeyLow: Option<f32>,
    overlayBlackKeyHigh: Option<f32>,
    overlayFeatherPower: Option<f32>,
    overlayFeatherRadius: Option<u32>,
) -> Result<(), String> {
    let mut opts = load_shared_or_default();
    if let Some(v) = baseRotateDeg { opts.base_rotate_deg = Some(v); }
    if let Some(v) = baseBrightness { opts.base_brightness = Some(v); }
    if let Some(v) = overlayScale { opts.overlay_scale = Some(v); }
    if let Some(v) = overlayRotateDeg { opts.overlay_rotate_deg = Some(v); }
    if let Some(v) = overlayAlpha { opts.overlay_alpha = Some(v); }
    if let Some(v) = overlayBrightness { opts.overlay_brightness = Some(v); }
    if let Some(v) = dimStrength { opts.dim_strength = Some(v); }
    if let Some(v) = saturationBoost { opts.saturation_boost = Some(v); }
    if let Some(v) = overlayBlackKeyLow { opts.overlay_black_key_low = Some(v); }
    if let Some(v) = overlayBlackKeyHigh { opts.overlay_black_key_high = Some(v); }
    if let Some(v) = overlayFeatherPower { opts.overlay_feather_power = Some(v); }
    if let Some(v) = overlayFeatherRadius { opts.overlay_feather_radius = Some(v); }

    let path = gscp_core::paths::settings_path();
    settings::save(&opts, &path).map_err(|e| format!("{e:#}"))
}

// ── 工具链管理（需求 1：专用能力按需下载）──

#[tauri::command]
pub fn tool_status() -> serde_json::Value {
    let adb = gscp_core::adb::find_adb();
    json!({
        "adbPath": adb,
        "adbAvailable": adb.is_some(),
        "embeddedScrcpyServerKb": gscp_core::SCRCPY_SERVER_BYTES.len() as f64 / 1024.0,
    })
}

#[tauri::command]
pub fn install_platform_tools() -> Result<(), String> {
    std::thread::spawn(|| {
        if let Err(err) = gscp_core::downloader::install_platform_tools() {
            logbus::emit(&format!("[downloader] platform-tools 安装失败: {err:#}"));
        }
    });
    Ok(())
}

#[tauri::command]
pub fn get_app_version() -> String {
    gscp_core::VERSION.to_string()
}

// ── 窗口外框随内容自适应 ──

/// 按前端测量的内容高度调整窗口高度（逻辑像素）。
///
/// 外框与内容区的差值（标题栏/边框）按当前缩放系数补偿；
/// 宽度保持不变（控件均为弹性宽度）。移动端窗口由系统管理，直接忽略。
#[tauri::command]
pub fn resize_height_to(
    window: tauri::WebviewWindow,
    content_height: f64,
    inner_height: Option<f64>,
) -> Result<(), String> {
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let _ = (window, content_height, inner_height);
        return Ok(());
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        let scale = window.scale_factor().map_err(|e| e.to_string())?;
        let outer = window.outer_size().map_err(|e| e.to_string())?;
        let inner = window.inner_size().map_err(|e| e.to_string())?;
        let outer_logical = outer.height as f64 / scale;

        // 标题栏高度 = 窗口外框 − 真实客户区。
        // macOS 上 inner_size() 与 outer_size() 同为外框值，因此优先用
        // 前端上报的 innerHeight（真实 WebView 客户区）求差；无上报时
        // 回退到 inner_size()（Windows/Linux 上为客户区）。
        let frame_h = match inner_height {
            Some(h) if h > 0.0 => (outer_logical - h).max(0.0),
            _ => ((outer.height as f64 - inner.height as f64) / scale).max(0.0),
        };

        let target = (content_height + frame_h).round().clamp(260.0, 2200.0);
        if (target - outer_logical).abs() < 1.0 {
            return Ok(());
        }
        gscp_core::logbus::emit(&format!(
            "[window] 高度自适应: 内容 {content_height}px + 标题栏 {frame_h}px → 外框 {target}px"
        ));

        window
            .set_size(tauri::LogicalSize::new(
                outer.width as f64 / scale,
                target,
            ))
            .map_err(|e| e.to_string())
    }
}
