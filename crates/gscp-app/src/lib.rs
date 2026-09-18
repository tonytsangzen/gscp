//! gscp-app 库入口：命令与 Android（Tauri mobile）入口。
//!
//! 桌面端入口见 `main.rs`；Android 端由生成的 Kotlin 壳调用
//! [`mobile_entry`]，WebView 内运行与桌面一致的 ui/ 前端（bridge.js
//! 已预留 Android JSI 通道）。视频预览的原生 SurfaceView 渲染
//! （MediaCodec + GLES）按 PLAN.md M5 在 Kotlin 侧实现。

pub mod commands;

#[cfg(target_os = "android")]
#[tauri::mobile_entry_point]
pub fn mobile_entry() {
    env_logger::init();
    let user_config = gscp_core::userconfig::UserConfig::load();

    // Android：GSCP_DATA_DIR 由 Kotlin 侧设置为 app filesDir
    tauri::Builder::default()
        .setup(|app| {
            use tauri::Emitter;
            let handle = app.handle().clone();
            gscp_core::logbus::subscribe(Box::new(move |line| {
                let _ = handle.emit("log-message", serde_json::json!({ "message": line }));
            }));
            gscp_core::logbus::emit(&format!("当前版本: {}", gscp_core::VERSION));
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
        .expect("Tauri 应用运行失败");
}
