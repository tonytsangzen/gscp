//! gscp-core: 平台无关核心库。
//!
//! 包含 ADB 访问、scrcpy 协议与会话拉起、设备 WiFi 配网、配置持久化、
//! 工具链下载与日志总线。桌面端（gscp-app/gscp-player）与 Android 端
//! （JNI）共同复用本 crate。

pub mod adb;
pub mod bootstrap;
pub mod codec;
pub mod daemon;
pub mod downloader;
pub mod logbus;
pub mod paths;
pub mod scrcpy;
pub mod settings;
pub mod userconfig;
pub mod wifi;

/// 内嵌的 scrcpy-server jar（默认版本，离线可用）。
pub const SCRCPY_SERVER_BYTES: &[u8] = include_bytes!("../../../assets/scrcpy-server");

/// 内嵌的 WiFi 保活 daemon（Android arm64 ELF）。
pub const WIFI_DAEMON_BYTES: &[u8] = include_bytes!("../../../assets/wifi_daemon");

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
