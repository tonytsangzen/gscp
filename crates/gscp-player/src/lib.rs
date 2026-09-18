//! gscp-player：桌面端播放器（渲染/解码/音频/三路会话）。
//!
//! 由 gscp-app 以 `gscp --player ...` 子进程方式拉起；也可独立运行。

pub mod audio;
pub mod decode;
pub mod player;
pub mod render;
pub mod session;
pub mod slot;

/// 独立/子进程入口：解析命令行 + 配置文件并运行播放器。
pub fn run_player_main() -> anyhow::Result<()> {
    let opts = <gscp_core::settings::PlayerOptions as clap::Parser>::parse();
    player::run_player(opts)
}
