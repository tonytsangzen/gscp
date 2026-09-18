//! 播放器/会话配置。
//!
//! [`PlayerOptions`] 同时用于 settings.yaml（serde）与命令行（clap）：
//! 全字段 `Option`，合并顺序 = 命令行显式提供 > 配置文件 > 内置默认。
//! 效果参数子集 [`EffectParams`] 支持运行中热更新（播放器监视配置文件）。

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_FORWARD_PORT: u16 = 27183;
const DEFAULT_SCRCPY_SERVER_VERSION: &str = "3.3.1";
const DEFAULT_SERVER_REMOTE_PATH: &str = "/data/local/tmp/scrcpy-server.jar";

#[derive(Debug, Clone, Default, Parser, Serialize, Deserialize)]
#[command(name = "gscp-player", about = "gscp 双路视频合成播放器")]
#[serde(default)]
pub struct PlayerOptions {
    // ── 会话 ──
    #[arg(long, action = clap::ArgAction::Set, help = "集成启动远端 scrcpy server/adb forward 流程")]
    pub bootstrap_remote: Option<bool>,
    #[arg(long, help = "ADB daemon 地址，默认读取 ANDROID_ADB_SERVER_ADDRESS 或 127.0.0.1:5037")]
    pub adb_server_address: Option<String>,
    #[arg(long, help = "adb connect 目标地址，例如 192.168.1.10:5555")]
    pub adb_connect: Option<String>,
    #[arg(long, help = "adb 设备序列号")]
    pub device_serial: Option<String>,
    #[arg(long, help = "远端 server jar 推送路径")]
    pub server_remote_path: Option<String>,
    #[arg(long, help = "adb forward 本地端口")]
    pub forward_port: Option<u16>,
    #[arg(long, help = "scrcpy server 版本号")]
    pub scrcpy_server_version: Option<String>,

    // ── 三路流 ──
    #[arg(long, action = clap::ArgAction::Set, help = "是否启用 camera 主视频")]
    pub enable_camera: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "是否启用音频")]
    pub enable_audio: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "是否启用远端 overlay")]
    pub enable_overlay: Option<bool>,
    #[arg(long, help = "远端音频 codec：raw|opus")]
    pub server_audio_codec: Option<String>,
    #[arg(long, help = "远端 camera 目标分辨率，例如 1920x1080")]
    pub server_camera_size: Option<String>,
    #[arg(long, help = "远端 camera 最大边长限制")]
    pub server_max_size: Option<u32>,
    #[arg(long, help = "远端 camera 编码器参数")]
    pub server_video_codec_options: Option<String>,
    #[arg(long, help = "远端 overlay 编码器参数")]
    pub server_overlay_codec_options: Option<String>,

    // ── 远端 socket ──
    #[arg(long, help = "远端 scrcpy 会话主机")]
    pub remote_host: Option<String>,
    #[arg(long, help = "远端 camera 视频端口")]
    pub camera_port: Option<u16>,
    #[arg(long, help = "远端 overlay 视频端口")]
    pub remote_overlay_port: Option<u16>,
    #[arg(long, help = "远端 audio 端口")]
    pub audio_port: Option<u16>,
    #[arg(long, action = clap::ArgAction::Set, help = "是否读取 codec meta（默认 true）")]
    pub send_codec_meta: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "是否读取 frame meta（默认 true）")]
    pub send_frame_meta: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "是否按 raw_stream 处理")]
    pub remote_raw_stream: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "camera 端口是否读取 dummy byte（默认 true）")]
    pub camera_send_dummy_byte: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "camera 端口是否读取 device meta（默认 true）")]
    pub camera_send_device_meta: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "overlay 端口是否读取 dummy byte")]
    pub overlay_send_dummy_byte: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "overlay 端口是否读取 device meta")]
    pub overlay_send_device_meta: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "audio 端口是否读取 dummy byte")]
    pub audio_send_dummy_byte: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "audio 端口是否读取 device meta")]
    pub audio_send_device_meta: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set, help = "输出 scrcpy 协议调试日志")]
    pub remote_debug: Option<bool>,

    // ── 窗口 ──
    #[arg(long, help = "初始画布宽")]
    pub canvas_width: Option<u32>,
    #[arg(long, help = "初始画布高")]
    pub canvas_height: Option<u32>,
    #[arg(long, help = "无实测帧率时的回退节拍")]
    pub fps: Option<f64>,
    #[arg(long, action = clap::ArgAction::Set, help = "启动时自动进入全屏")]
    pub fullscreen: Option<bool>,
    #[arg(long, help = "窗口标题前缀")]
    pub title: Option<String>,

    // ── 渲染效果（支持热更新）──
    #[arg(long, help = "底图旋转角度（90 的倍数）")]
    pub base_rotate_deg: Option<i32>,
    #[arg(long, help = "底图亮度（<1 压暗）")]
    pub base_brightness: Option<f32>,
    #[arg(long, help = "overlay 相对画布高的缩放倍率")]
    pub overlay_scale: Option<f32>,
    #[arg(long, help = "overlay 旋转角度（90 的倍数）")]
    pub overlay_rotate_deg: Option<i32>,
    #[arg(long, help = "overlay 不透明度 0..1")]
    pub overlay_alpha: Option<f32>,
    #[arg(long, help = "overlay 亮度增益")]
    pub overlay_brightness: Option<f32>,
    #[arg(long, help = "overlay 区域底图压暗强度 0..1")]
    pub dim_strength: Option<f32>,
    #[arg(long, help = "overlay 饱和度增益（1 = 不变）")]
    pub saturation_boost: Option<f32>,
    #[arg(long, help = "黑边抠像 luma 下阈值 0..1（0=关闭，需 > key_low 才启用）")]
    pub overlay_black_key_low: Option<f32>,
    #[arg(long, help = "黑边抠像 luma 上阈值 0..1")]
    pub overlay_black_key_high: Option<f32>,
    #[arg(long, help = "抠像羽化曲线幂（>=0.01）")]
    pub overlay_feather_power: Option<f32>,
    #[arg(long, help = "抠像阈值外扩半径（0..=255）")]
    pub overlay_feather_radius: Option<u32>,

    // ── 解码 ──
    #[arg(long, help = "解码后端：auto|videotoolbox|software")]
    pub decoder_backend: Option<String>,

    // ── 文件 ──
    /// 配置文件路径（仅命令行使用；不参与 serde 合并）。
    #[arg(long, help = "配置文件路径（YAML）")]
    #[serde(skip)]
    pub config: Option<PathBuf>,
}

/// 全部解析后的具体配置。
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedConfig {
    pub bootstrap_remote: bool,
    pub adb_server_address: String,
    pub adb_connect: Option<String>,
    pub device_serial: Option<String>,
    pub server_remote_path: String,
    pub forward_port: u16,
    pub scrcpy_server_version: String,
    pub enable_camera: bool,
    pub enable_audio: bool,
    pub enable_overlay: bool,
    pub server_audio_codec: String,
    pub server_camera_size: String,
    pub server_max_size: u32,
    pub server_video_codec_options: String,
    pub server_overlay_codec_options: String,
    pub remote_host: String,
    pub camera_port: Option<u16>,
    pub remote_overlay_port: Option<u16>,
    pub audio_port: Option<u16>,
    pub send_codec_meta: bool,
    pub send_frame_meta: bool,
    pub remote_raw_stream: bool,
    pub camera_send_dummy_byte: bool,
    pub camera_send_device_meta: bool,
    pub overlay_send_dummy_byte: bool,
    pub overlay_send_device_meta: bool,
    pub audio_send_dummy_byte: bool,
    pub audio_send_device_meta: bool,
    pub remote_debug: bool,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub fps: f64,
    pub fullscreen: bool,
    pub title: String,
    pub effects: EffectParams,
    pub decoder_backend: DecoderBackend,
}

impl ResolvedConfig {
    /// bootstrap 模式下按 forward 端口补齐三路 socket 端口（移植 mixplayer 语义）。
    pub fn apply_bootstrap_runtime(&mut self) {
        if self.remote_host.is_empty() {
            self.remote_host = "127.0.0.1".to_string();
        }
        self.camera_port = if self.enable_camera {
            Some(self.camera_port.unwrap_or(self.forward_port))
        } else {
            None
        };
        self.remote_overlay_port = if self.enable_overlay {
            Some(self.remote_overlay_port.unwrap_or(self.forward_port))
        } else {
            None
        };
        self.audio_port = if self.enable_audio {
            Some(self.audio_port.unwrap_or(self.forward_port))
        } else {
            None
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecoderBackend {
    Auto,
    VideoToolbox,
    MediaFoundation,
    Software,
}

impl DecoderBackend {
    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "videotoolbox" => Self::VideoToolbox,
            "mediafoundation" => Self::MediaFoundation,
            "software" => Self::Software,
            _ => Self::Auto,
        }
    }
}

/// 渲染效果参数（可热更新子集）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectParams {
    pub base_rotate_deg: i32,
    pub base_brightness: f32,
    pub overlay_scale: f32,
    pub overlay_rotate_deg: i32,
    pub overlay_alpha: f32,
    pub overlay_brightness: f32,
    pub dim_strength: f32,
    pub saturation_boost: f32,
    pub overlay_black_key_low: f32,
    pub overlay_black_key_high: f32,
    pub overlay_feather_power: f32,
    pub overlay_feather_radius: u32,
}

impl Default for EffectParams {
    /// 中性默认：不改变画面（旧默认沿用 GlassConnect 调优值——overlay 65% 透明
    /// 蒙版 + 压暗底图，新设备上表现为"发灰 + 半透明蒙版"，需要用户重调）。
    fn default() -> Self {
        Self {
            base_rotate_deg: 90,
            base_brightness: 1.0,
            overlay_scale: 1.0,
            overlay_rotate_deg: 0,
            overlay_alpha: 1.0,
            overlay_brightness: 1.0,
            dim_strength: 0.0,
            saturation_boost: 1.0,
            overlay_black_key_low: 0.0,
            overlay_black_key_high: 0.0,
            overlay_feather_power: 1.35,
            overlay_feather_radius: 0,
        }
    }
}

impl EffectParams {
    fn sanitize(mut self) -> Self {
        self.base_brightness = self.base_brightness.clamp(0.0, 4.0);
        self.overlay_scale = self.overlay_scale.clamp(0.1, 8.0);
        self.overlay_alpha = self.overlay_alpha.clamp(0.0, 1.0);
        self.overlay_brightness = self.overlay_brightness.max(0.0);
        self.dim_strength = self.dim_strength.clamp(0.0, 1.0);
        self.saturation_boost = self.saturation_boost.clamp(0.0, 4.0);
        self.overlay_black_key_low = self.overlay_black_key_low.clamp(0.0, 1.0);
        self.overlay_black_key_high = self.overlay_black_key_high.clamp(0.0, 1.0);
        self.overlay_feather_power = self.overlay_feather_power.max(0.01);
        self.overlay_feather_radius = self.overlay_feather_radius.min(255);
        self
    }
}

/// 加载配置文件（可选），合并命令行，解析出最终配置。
pub fn resolve(options: &PlayerOptions, config_path: Option<&std::path::Path>) -> anyhow::Result<ResolvedConfig> {
    let path = match config_path {
        Some(p) => Some(p.to_path_buf()),
        None => options.config.clone().or_else(|| Some(crate::paths::settings_path())),
    };
    let mut file_options = load_file(path.as_deref())?;
    // 命令行 > 文件
    merge(&mut file_options, options);
    file_options.resolve()
}

pub fn load_file(path: Option<&std::path::Path>) -> anyhow::Result<PlayerOptions> {
    let Some(path) = path else {
        return Ok(PlayerOptions::default());
    };
    if !path.exists() {
        return Ok(PlayerOptions::default());
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("读取配置文件失败: {} ({e})", path.display()))?;
    serde_yaml::from_str(&content)
        .map_err(|e| anyhow::anyhow!("解析配置文件失败: {} ({e})", path.display()))
}

/// `target` 缺省字段用 `override` 的 Some 值补齐。
pub fn merge(target: &mut PlayerOptions, over: &PlayerOptions) {
    macro_rules! take {
        ($field:ident) => {
            if over.$field.is_some() {
                target.$field = over.$field.clone();
            }
        };
    }
    take!(bootstrap_remote);
    take!(adb_server_address);
    take!(adb_connect);
    take!(device_serial);
    take!(server_remote_path);
    take!(forward_port);
    take!(scrcpy_server_version);
    take!(enable_camera);
    take!(enable_audio);
    take!(enable_overlay);
    take!(server_audio_codec);
    take!(server_camera_size);
    take!(server_max_size);
    take!(server_video_codec_options);
    take!(server_overlay_codec_options);
    take!(remote_host);
    take!(camera_port);
    take!(remote_overlay_port);
    take!(audio_port);
    take!(send_codec_meta);
    take!(send_frame_meta);
    take!(remote_raw_stream);
    take!(camera_send_dummy_byte);
    take!(camera_send_device_meta);
    take!(overlay_send_dummy_byte);
    take!(overlay_send_device_meta);
    take!(audio_send_dummy_byte);
    take!(audio_send_device_meta);
    take!(remote_debug);
    take!(canvas_width);
    take!(canvas_height);
    take!(fps);
    take!(fullscreen);
    take!(title);
    take!(base_rotate_deg);
    take!(base_brightness);
    take!(overlay_scale);
    take!(overlay_rotate_deg);
    take!(overlay_alpha);
    take!(overlay_brightness);
    take!(dim_strength);
    take!(saturation_boost);
    take!(overlay_black_key_low);
    take!(overlay_black_key_high);
    take!(overlay_feather_power);
    take!(overlay_feather_radius);
    take!(decoder_backend);
    if over.config.is_some() {
        target.config = over.config.clone();
    }
}

impl PlayerOptions {
    pub fn resolve(&self) -> anyhow::Result<ResolvedConfig> {
        let defaults = EffectParams::default().sanitize();
        let effects = EffectParams {
            base_rotate_deg: self.base_rotate_deg.unwrap_or(defaults.base_rotate_deg),
            base_brightness: self.base_brightness.unwrap_or(defaults.base_brightness),
            overlay_scale: self.overlay_scale.unwrap_or(defaults.overlay_scale),
            overlay_rotate_deg: self.overlay_rotate_deg.unwrap_or(defaults.overlay_rotate_deg),
            overlay_alpha: self.overlay_alpha.unwrap_or(defaults.overlay_alpha),
            overlay_brightness: self.overlay_brightness.unwrap_or(defaults.overlay_brightness),
            dim_strength: self.dim_strength.unwrap_or(defaults.dim_strength),
            saturation_boost: self.saturation_boost.unwrap_or(defaults.saturation_boost),
            overlay_black_key_low: self.overlay_black_key_low.unwrap_or(defaults.overlay_black_key_low),
            overlay_black_key_high: self.overlay_black_key_high.unwrap_or(defaults.overlay_black_key_high),
            overlay_feather_power: self.overlay_feather_power.unwrap_or(defaults.overlay_feather_power),
            overlay_feather_radius: self.overlay_feather_radius.unwrap_or(defaults.overlay_feather_radius),
        }
        .sanitize();

        let mut adb_server_address = self
            .adb_server_address
            .clone()
            .unwrap_or_else(crate::adb::adb_server_address);
        if adb_server_address.is_empty() {
            adb_server_address = "127.0.0.1:5037".to_string();
        }

        let bootstrap_remote = self.bootstrap_remote.unwrap_or(false);
        let mut resolved = ResolvedConfig {
            bootstrap_remote,
            adb_server_address,
            adb_connect: self.adb_connect.clone().filter(|s| !s.is_empty()),
            device_serial: self.device_serial.clone().filter(|s| !s.is_empty()),
            server_remote_path: self
                .server_remote_path
                .clone()
                .unwrap_or_else(|| DEFAULT_SERVER_REMOTE_PATH.to_string()),
            forward_port: self.forward_port.unwrap_or(DEFAULT_FORWARD_PORT),
            scrcpy_server_version: self
                .scrcpy_server_version
                .clone()
                .unwrap_or_else(|| DEFAULT_SCRCPY_SERVER_VERSION.to_string()),
            enable_camera: self.enable_camera.unwrap_or(true),
            enable_audio: self.enable_audio.unwrap_or(true),
            enable_overlay: self.enable_overlay.unwrap_or(true),
            server_audio_codec: self.server_audio_codec.clone().unwrap_or_else(|| "raw".to_string()),
            server_camera_size: self
                .server_camera_size
                .clone()
                .unwrap_or_else(|| "1024x768".to_string()),
            server_max_size: self.server_max_size.unwrap_or(1920),
            server_video_codec_options: self
                .server_video_codec_options
                .clone()
                .unwrap_or_else(|| "i-frame-interval=1".to_string()),
            server_overlay_codec_options: self
                .server_overlay_codec_options
                .clone()
                .unwrap_or_else(|| "i-frame-interval=1".to_string()),
            remote_host: self.remote_host.clone().unwrap_or_default(),
            camera_port: self.camera_port,
            remote_overlay_port: self.remote_overlay_port,
            audio_port: self.audio_port,
            send_codec_meta: self.send_codec_meta.unwrap_or(true),
            send_frame_meta: self.send_frame_meta.unwrap_or(true),
            remote_raw_stream: self.remote_raw_stream.unwrap_or(false),
            camera_send_dummy_byte: self.camera_send_dummy_byte.unwrap_or(true),
            camera_send_device_meta: self.camera_send_device_meta.unwrap_or(true),
            overlay_send_dummy_byte: self.overlay_send_dummy_byte.unwrap_or(false),
            overlay_send_device_meta: self.overlay_send_device_meta.unwrap_or(false),
            audio_send_dummy_byte: self.audio_send_dummy_byte.unwrap_or(false),
            audio_send_device_meta: self.audio_send_device_meta.unwrap_or(false),
            remote_debug: self.remote_debug.unwrap_or(false),
            canvas_width: self.canvas_width.unwrap_or(1280),
            canvas_height: self.canvas_height.unwrap_or(720),
            fps: self.fps.unwrap_or(60.0),
            fullscreen: self.fullscreen.unwrap_or(false),
            title: {
                let t = self.title.clone().unwrap_or_else(|| "gscp".to_string());
                if t.is_empty() { "gscp".to_string() } else { t }
            },
            effects,
            decoder_backend: self
                .decoder_backend
                .as_deref()
                .map(DecoderBackend::parse)
                .unwrap_or(DecoderBackend::Auto),
        };
        if resolved.bootstrap_remote {
            if let Some(target) = resolved.adb_connect.clone() {
                if resolved.device_serial.is_none() {
                    resolved.device_serial = Some(target);
                }
            }
            resolved.apply_bootstrap_runtime();
        }
        if !resolved.enable_camera && !resolved.enable_audio && !resolved.enable_overlay {
            anyhow::bail!("至少需要启用 camera、overlay、audio 中的一路流");
        }
        Ok(resolved)
    }

    /// 只更新效果参数字段（管理端 set_effect 使用）。
    pub fn apply_effects(&mut self, effects: &EffectParams) {
        self.base_rotate_deg = Some(effects.base_rotate_deg);
        self.base_brightness = Some(effects.base_brightness);
        self.overlay_scale = Some(effects.overlay_scale);
        self.overlay_rotate_deg = Some(effects.overlay_rotate_deg);
        self.overlay_alpha = Some(effects.overlay_alpha);
        self.overlay_brightness = Some(effects.overlay_brightness);
        self.dim_strength = Some(effects.dim_strength);
        self.saturation_boost = Some(effects.saturation_boost);
        self.overlay_black_key_low = Some(effects.overlay_black_key_low);
        self.overlay_black_key_high = Some(effects.overlay_black_key_high);
        self.overlay_feather_power = Some(effects.overlay_feather_power);
        self.overlay_feather_radius = Some(effects.overlay_feather_radius);
    }

    pub fn effects(&self) -> EffectParams {
        self.resolve().map(|c| c.effects).unwrap_or_default()
    }
}

/// 保存配置到 yaml（管理端使用）。
pub fn save(options: &PlayerOptions, path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_yaml::to_string(options)?;
    std::fs::write(path, content)?;
    Ok(())
}

/// 加载当前共享设置（不存在则返回默认）。
pub fn load_shared() -> PlayerOptions {
    load_file(Some(&crate::paths::settings_path())).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_resolve() {
        let resolved = PlayerOptions::default().resolve().unwrap();
        assert!(resolved.enable_camera);
        assert_eq!(resolved.forward_port, 27183);
        assert_eq!(resolved.send_codec_meta, true);
        assert_eq!(resolved.effects, EffectParams::default());
        assert_eq!(resolved.remote_host, "");
    }

    #[test]
    fn merge_cli_overrides_file() {
        let mut file = PlayerOptions::default();
        file.overlay_alpha = Some(0.3);
        file.title = Some("file-title".into());
        let mut cli = PlayerOptions::default();
        cli.overlay_alpha = Some(0.9);
        merge(&mut file, &cli);
        let resolved = file.resolve().unwrap();
        assert_eq!(resolved.effects.overlay_alpha, 0.9);
        assert_eq!(resolved.title, "file-title");
    }

    #[test]
    fn bootstrap_runtime_fills_ports() {
        let mut opts = PlayerOptions::default();
        opts.bootstrap_remote = Some(true);
        opts.enable_audio = Some(false);
        let resolved = opts.resolve().unwrap();
        assert_eq!(resolved.remote_host, "127.0.0.1");
        assert_eq!(resolved.camera_port, Some(27183));
        assert_eq!(resolved.remote_overlay_port, Some(27183));
        assert_eq!(resolved.audio_port, None);
    }

    #[test]
    fn all_disabled_rejected() {
        let mut opts = PlayerOptions::default();
        opts.enable_camera = Some(false);
        opts.enable_audio = Some(false);
        opts.enable_overlay = Some(false);
        assert!(opts.resolve().is_err());
    }

    #[test]
    fn yaml_roundtrip() {
        let mut opts = PlayerOptions::default();
        opts.overlay_alpha = Some(0.42);
        opts.overlay_black_key_low = Some(0.1);
        let text = serde_yaml::to_string(&opts).unwrap();
        let parsed: PlayerOptions = serde_yaml::from_str(&text).unwrap();
        assert_eq!(parsed.overlay_alpha, Some(0.42));
        let resolved = parsed.resolve().unwrap();
        assert_eq!(resolved.effects.overlay_alpha, 0.42);
    }

    #[test]
    fn effects_sanitize_clamps() {
        let mut opts = PlayerOptions::default();
        opts.overlay_alpha = Some(7.0);
        opts.overlay_feather_power = Some(0.0);
        opts.overlay_feather_radius = Some(999);
        let resolved = opts.resolve().unwrap();
        assert_eq!(resolved.effects.overlay_alpha, 1.0);
        assert_eq!(resolved.effects.overlay_feather_power, 0.01);
        assert_eq!(resolved.effects.overlay_feather_radius, 255);
    }
}
