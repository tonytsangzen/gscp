// release 为 GUI 子系统：不弹 cmd 控制台窗口；debug 保留控制台便于开发调试
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! gscp 桌面端入口。
//!
//! 单二进制双模式：
//! - `gscp`            → 管理界面（Tauri / 系统 WebView）
//! - `gscp --player …` → 独立播放窗口（winit + wgpu，管理端 spawn 自身）

use gscp_app_lib::commands;

use anyhow::Context;
#[cfg(not(target_os = "android"))]
use gscp_core::settings::PlayerOptions;
use gscp_core::logbus;

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    #[cfg(not(target_os = "android"))]
    if args.len() > 1 && args[1] == "--player" {
        // 播放器模式：不初始化 Tauri
        // parse_from 的首元素是程序名，真实参数从 index 2 开始
        env_logger::init();
        let argv = std::iter::once("gscp-player".to_string())
            .chain(args.iter().skip(2).cloned())
            .collect::<Vec<_>>();
        let opts = <PlayerOptions as clap::Parser>::parse_from(argv);
        if let Err(err) = gscp_player::player::run_player(opts) {
            logbus::emit(&format!("[player] 退出: {err:#}"));
            std::process::exit(1);
        }
        return;
    }

    #[cfg(target_os = "android")]
    let _ = &args;

    if args.len() > 1 && (args[1] == "-h" || args[1] == "--help" || args[1] == "help") {
        println!(
            "用法:\n  gscp                 打开管理界面\n  gscp --player [选项]  独立播放器模式\n\n说明:\n  连接 RG- 系列眼镜：WiFi 配网/保活 + USB/WiFi 双路预览。"
        );
        return;
    }

    if let Err(err) = run_gui() {
        let _ = writeln!(std::io::stderr(), "{err:#}");
        std::process::exit(1);
    }
}

fn run_gui() -> anyhow::Result<()> {
    env_logger::init();
    let user_config = gscp_core::userconfig::UserConfig::load();

    use tauri::Manager;
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            let title = format!("GSCP v{}", gscp_core::VERSION);
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&title);
            }
            // 日志总线 → WebView 日志区
            logbus::subscribe(Box::new(move |line| {
                use tauri::Emitter;
                let _ = handle.emit("log-message", serde_json::json!({ "message": line }));
            }));
            logbus::emit(&format!("当前版本: {}", gscp_core::VERSION));
            Ok(())
        })
        .manage(commands::AppState {
            config: std::sync::Mutex::new(user_config),
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_wifi,
            commands::stop_wifi,
            commands::preview,
            commands::get_config,
            commands::save_config,
            commands::get_wifi_list,
            commands::save_wifi_network,
            commands::get_effect,
            commands::set_effect,
            commands::tool_status,
            commands::install_platform_tools,
            commands::get_app_version,
            commands::resize_height_to,
            commands::reset_effect,
        ])
        .run(tauri::generate_context!())
        .context("Tauri 应用运行失败")
}
