//! 设备侧 WiFi 保活 daemon 部署（内嵌 ARM64 ELF）。
//! 移植自 GlassConnect daemon.rs；scrcpy-server 改由 crate::SCRCPY_SERVER_BYTES 提供。

use crate::adb::{self, ADBServerDevice};

const DAEMON_REMOTE_PATH: &str = "/data/local/tmp/wifi_daemon";

/// 部署并启动 WiFi 保活 daemon：杀旧进程 → push → chmod → 运行。
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
