//! 设备侧 WiFi 配网（`cmd wifi` 命令族）。
//! 移植自 GlassConnect wifi.rs。

use crate::adb::{self, AdbDevice, ADBServerDevice};

/// IPv4 格式校验（移植 GlassConnect 的正则）。
pub fn is_valid_ipv4(ip: &str) -> bool {
    let re = regex::Regex::new(
        r"^((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)$",
    )
    .expect("ipv4 regex");
    re.is_match(ip)
}

pub fn enable_wifi(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    adb::shell_command(device, "cmd wifi set-wifi-enabled enabled")?;
    Ok(())
}

pub fn disable_wifi(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    adb::shell_command(device, "cmd wifi set-wifi-enabled disabled")?;
    Ok(())
}

pub fn start_scan(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    adb::shell_command(device, "cmd wifi start-scan")?;
    Ok(())
}

/// 添加 WiFi 网络。密码为空视为开放网络。
pub fn add_network(
    device: &mut ADBServerDevice,
    ssid: &str,
    password: &str,
) -> anyhow::Result<()> {
    let cmd = if password.is_empty() {
        format!("cmd wifi add-network \"{ssid}\" open")
    } else {
        format!("cmd wifi add-network \"{ssid}\" wpa2 \"{password}\"")
    };
    adb::shell_command(device, &cmd)?;
    Ok(())
}

pub fn has_network(device: &mut ADBServerDevice, ssid: &str) -> bool {
    match adb::shell_command(device, "cmd wifi list-networks") {
        Ok(output) => output.contains(ssid),
        Err(_) => false,
    }
}

pub fn remove_all_networks(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    let list = adb::shell_command(device, "cmd wifi list-networks")?;
    for line in list.lines().skip(1) {
        let id = line.split_whitespace().next().unwrap_or("");
        if !id.is_empty() {
            let _ = adb::shell_command(device, &format!("cmd wifi forget-network {id}"));
        }
    }
    Ok(())
}

/// 轮询 wlan0 直到拿到 IPv4 地址（`inet addr:` / `inet x.x.x.x` 两种格式）。
pub fn get_ip_address(device: &mut ADBServerDevice, timeout_secs: u32) -> anyhow::Result<String> {
    for _ in 0..timeout_secs {
        let output = adb::shell_command(device, "ifconfig wlan0")?;
        for line in output.lines() {
            let line = line.trim();
            if line.contains("inet addr:") {
                if let Some(ip) = line
                    .split("inet addr:")
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                {
                    return Ok(ip.to_string());
                }
            } else if line.starts_with("inet ") && !line.contains("inet6") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "inet" {
                    return Ok(parts[1].to_string());
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    anyhow::bail!("获取 IP 地址超时")
}

pub fn enable_adb_tcpip(port: u16) -> anyhow::Result<()> {
    adb::enable_tcpip(port)?;
    Ok(())
}

/// 与 GlassConnect 一致的 `persist.vendor.adb` 持久化开关。
pub fn enable_persistent_adb(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    adb::shell_command(device, "setprop persist.vendor.adb 1")?;
    Ok(())
}

pub fn disable_persistent_adb(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    adb::shell_command(device, "setprop persist.vendor.adb 0")?;
    Ok(())
}

/// 组合：启用持久 ADB（stop 流程也要求先做，保留原顺序）。
pub fn prepare_for_stop(device: &mut ADBServerDevice) -> anyhow::Result<()> {
    enable_persistent_adb(device)
}

/// 便捷方法：直接通过序列号获取 shell 输出（供诊断使用）。
pub fn device_shell(serial: &str, server_addr: &str, cmd: &str) -> anyhow::Result<String> {
    let mut device = AdbDevice::new(server_addr, serial);
    device.shell_exec(cmd)
}
