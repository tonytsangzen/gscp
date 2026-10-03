//! 设备侧 WiFi 保活 daemon 部署（内嵌 ARM64 ELF）。
//! 移植自 GlassConnect daemon.rs；scrcpy-server 改由 crate::SCRCPY_SERVER_BYTES 提供。

use crate::adb::{self, ADBServerDevice};

const DAEMON_REMOTE_PATH: &str = "/data/local/tmp/wifi_daemon";

/// 部署并启动 WiFi 保活 daemon：杀旧进程 → push → chmod → 运行。
///
/// 关键约束：经 adb shell 拉起的进程位于 adbd 的 cgroup 内，`adb tcpip` 切换
/// 会重启 adbd 并连带杀掉它们（setsid 也无法幸免）。因此必须在 adbd 重启
/// （TCP 模式切换）完成之后调用，否则 daemon 启动后几秒内即被杀。
/// 见 `gscp-app` 的 `run_start_wifi_sync` 与 `tests/daemon_deploy_e2e.rs`。
pub fn deploy_wifi_daemon(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    let _ = adb::shell_command(device, "killall wifi_daemon");
    adb::upload_file(device, crate::WIFI_DAEMON_BYTES, DAEMON_REMOTE_PATH)?;
    adb::shell_command(device, &format!("chmod 755 {DAEMON_REMOTE_PATH}"))?;
    adb::shell_command(device, DAEMON_REMOTE_PATH)?;
    Ok(())
}

/// 停止并删除设备上的保活 daemon。
pub fn stop_wifi_daemon(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    let _ = adb::shell_command(device, "killall wifi_daemon");
    let _ = adb::shell_command(device, &format!("rm -rf {DAEMON_REMOTE_PATH}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_assets_are_valid_images() {
        // scrcpy-server jar 是 zip 容器（PK 头）
        assert_eq!(&crate::SCRCPY_SERVER_BYTES[0..2], b"PK");
        assert!(crate::SCRCPY_SERVER_BYTES.len() > 80_000);
        // wifi_daemon 是 ARM64 ELF
        assert_eq!(&crate::WIFI_DAEMON_BYTES[0..4], b"\x7fELF");
        assert!(crate::WIFI_DAEMON_BYTES.len() > 5_000);
    }
}
