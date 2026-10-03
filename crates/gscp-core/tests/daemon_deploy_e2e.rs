//! wifi_daemon 部署时序端到端测试（需要真实 RG 眼镜，默认忽略）。
//!
//! 背景：经 `adb shell` 拉起的进程位于 adbd 的 cgroup 内，`adb tcpip` 切换
//! 会重启 adbd 并连带杀掉它们，因此保活 daemon 必须在 TCP 模式切换完成后再
//! 部署（与 `run_start_wifi_sync` 的时序保持一致）。
//!
//! 运行：`cargo test -p gscp-core --test daemon_deploy_e2e -- --ignored`
//! 前置：眼镜已通过 WiFi 连接（本测试不改动网络配置）；结束后 daemon 保持
//! 运行，与真实配网完成后的期望状态一致。

use gscp_core::adb::{self, AdbWrapper};
use gscp_core::daemon::deploy_wifi_daemon;
use gscp_core::wifi;
use std::time::Duration;

const DEVICE_MODEL_PREFIX: &str = "RG-";

/// 查询 wifi_daemon 是否存活；设备短暂不可达时重试（adbd 重启窗口）。
fn daemon_running_with_retry(wrapper: &mut AdbWrapper, tries: u32) -> bool {
    for _ in 0..tries {
        if let Ok(mut device) = wrapper.connect_device(DEVICE_MODEL_PREFIX) {
            if let Ok(out) = adb::shell_command(&mut device, "pgrep wifi_daemon") {
                return !out.trim().is_empty();
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    panic!("设备不可达，无法确认 wifi_daemon 状态");
}

#[test]
#[ignore = "需要真实 RG 眼镜设备"]
fn daemon_survives_when_deployed_after_tcpip_switch() {
    let mut wrapper = AdbWrapper::new().expect("adb server 不可用");
    let mut device = wrapper
        .connect_device(DEVICE_MODEL_PREFIX)
        .expect("未找到 RG 眼镜");

    let ip = wifi::get_ip_address(&mut device, 15).expect("眼镜未连接 WiFi");
    eprintln!("眼镜 IP: {ip}");

    // ── 回归对照：旧时序（tcpip 切换前部署）→ daemon 被 adbd 重启杀掉 ──
    deploy_wifi_daemon(&mut device).expect("部署 daemon 失败");
    assert!(
        daemon_running_with_retry(&mut wrapper, 5),
        "daemon 应已启动"
    );

    wifi::enable_adb_tcpip(5555).expect("adb tcpip 失败");
    assert!(
        !daemon_running_with_retry(&mut wrapper, 10),
        "旧时序下 daemon 应被 adbd 重启杀掉（这正是已修复的 bug）"
    );
    eprintln!("回归对照通过：tcpip 切换杀掉了先部署的 daemon");

    // ── 新时序：等待 TCP 就绪，经重启后的 adbd 部署 daemon ──
    let mut connected = false;
    for _ in 0..10 {
        std::thread::sleep(Duration::from_secs(1));
        if wrapper.connect_tcp(&ip, 5555).is_ok() {
            connected = true;
            break;
        }
    }
    assert!(connected, "TCP 连接 {ip}:5555 未就绪");

    let mut daemon_device = wrapper
        .get_device_by_name(&format!("{ip}:5555"))
        .or_else(|_| wrapper.connect_device(DEVICE_MODEL_PREFIX))
        .expect("无法获得重启后的设备句柄");
    deploy_wifi_daemon(&mut daemon_device).expect("新时序部署 daemon 失败");

    assert!(
        daemon_running_with_retry(&mut wrapper, 5),
        "daemon 应已启动"
    );
    std::thread::sleep(Duration::from_secs(5));
    assert!(
        daemon_running_with_retry(&mut wrapper, 5),
        "daemon 应在部署完成 5s 后仍存活"
    );
    eprintln!("新时序通过：daemon 在 adbd 重启后部署并稳定存活（保留运行）");
}
