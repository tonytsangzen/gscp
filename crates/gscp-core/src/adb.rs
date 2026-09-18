//! ADB 访问层。
//!
//! - [`adb_server_address`] / [`find_adb`]：server 地址与二进制定位（数据目录
//!   下载版优先，其次 PATH 与平台常见路径）。
//! - [`AdbDevice`]：面向 scrcpy 会话的设备操作（connect/push/forward/shell）。
//!   `shell_spawn` 通过 `os_pipe` 保持远端 shell 存活，跨平台。
//! - [`AdbWrapper`]：面向管理端的设备发现（按 `ro.product.model` 前缀）与
//!   本地连接清理。
//! - [`ensure_adb_available`]：定位不到 adb 时触发按需下载。

use adb_client::server::ADBServer;
pub use adb_client::server_device::ADBServerDevice;
use adb_client::ADBDeviceExt;
use anyhow::{anyhow, bail, Context, Result};
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::time::Duration;

const DAEMON_ADDR: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 5037);
const DEVICE_RETRY_COUNT: u32 = 5;
const DEVICE_RETRY_DELAY: Duration = Duration::from_millis(500);

pub const DEFAULT_DAEMON_ADDR: SocketAddrV4 = DAEMON_ADDR;

/// ADB server 地址：环境变量 > 默认 127.0.0.1:5037。
pub fn adb_server_address() -> String {
    if let Ok(addr) = std::env::var("ANDROID_ADB_SERVER_ADDRESS") {
        if !addr.is_empty() {
            return addr;
        }
    }
    if let Ok(port) = std::env::var("ANDROID_ADB_SERVER_PORT") {
        if !port.is_empty() {
            return format!("127.0.0.1:{port}");
        }
    }
    DAEMON_ADDR.to_string()
}

pub(crate) fn parse_daemon_addr(addr: &str) -> Result<SocketAddrV4> {
    let addr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("解析 ADB daemon 地址失败: {addr}"))?;
    match addr {
        std::net::SocketAddr::V4(v4) => Ok(v4),
        std::net::SocketAddr::V6(_) => bail!("不支持 IPv6 ADB daemon 地址"),
    }
}

/// 定位 adb 可执行文件：数据目录 bin/ → PATH → 平台常见路径 → Android SDK。
pub fn find_adb() -> Option<String> {
    let bin_dir = crate::paths::bin_dir();
    #[cfg(target_os = "windows")]
    let bundled = bin_dir.join("adb.exe");
    #[cfg(not(target_os = "windows"))]
    let bundled = bin_dir.join("adb");
    if bundled.exists() {
        return Some(bundled.to_string_lossy().to_string());
    }

    for candidate in platform_adb_candidates() {
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }

    if let Ok(path) = spawn_quiet(if cfg!(windows) { "where" } else { "which" })
        .arg("adb")
        .output()
    {
        let path = String::from_utf8_lossy(&path.stdout).trim().to_string();
        if !path.is_empty() && std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

// Windows 下 spawn 子进程一律加 CREATE_NO_WINDOW，避免 GUI 应用闪出控制台窗口
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

fn spawn_quiet(program: &str) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

fn platform_adb_candidates() -> Vec<String> {    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = std::path::PathBuf::from(home);
        // Android SDK 标准位置
        candidates.push(
            home.join("Library/Android/sdk/platform-tools/adb")
                .to_string_lossy()
                .to_string(),
        );
        candidates.push(
            home.join("Android/Sdk/platform-tools/adb")
                .to_string_lossy()
                .to_string(),
        );
        candidates.push(
            home.join("AppData/Local/Android/Sdk/platform-tools/adb.exe")
                .to_string_lossy()
                .to_string(),
        );
    }
    #[cfg(target_os = "macos")]
    {
        candidates.push("/opt/homebrew/bin/adb".to_string());
        candidates.push("/usr/local/bin/adb".to_string());
    }
    candidates
}

/// 确保 adb 可用：定位不到时下载 platform-tools 并返回其路径。
pub fn ensure_adb_available() -> Result<String> {
    if let Some(path) = find_adb() {
        return Ok(path);
    }
    crate::logbus::emit("未找到 adb，开始下载 platform-tools...");
    crate::downloader::install_platform_tools()?;
    find_adb().ok_or_else(|| anyhow!("platform-tools 安装后仍未找到 adb"))
}

/// 构造带 adb 二进制路径的 server 客户端：
/// adb server 未运行时，adb_client 会用该二进制自动拉起服务。
/// （ADBServer::new 不带路径，server 不在时只能从 PATH 找 adb 而失败。）
pub(crate) fn new_server(addr: &str) -> Result<ADBServer> {
    let daemon_addr = parse_daemon_addr(addr)?;
    Ok(ADBServer::new_from_path(daemon_addr, find_adb()))
}

/// 运行 adb 子进程命令（用于 disconnect/forward 清理等批量操作）。
pub fn run_adb_command(args: &[&str]) -> Result<std::process::Output> {
    let adb_path = ensure_adb_available()?;
    spawn_quiet(&adb_path)
        .args(args)
        .output()
        .map_err(Into::into)
}

/// scrcpy 会话用的设备句柄。
pub struct AdbDevice {
    daemon_addr: SocketAddrV4,
    daemon_addr_str: String,
    device_serial: String,
}

impl AdbDevice {
    pub fn new(server_addr: &str, device_serial: &str) -> Self {
        Self {
            daemon_addr: parse_daemon_addr(server_addr).unwrap_or(DAEMON_ADDR),
            daemon_addr_str: server_addr.to_string(),
            device_serial: device_serial.to_string(),
        }
    }

    pub fn connect(server_addr: &str, host: &str, port: u16) -> Result<()> {
        let adb = find_adb();
        let mut server = new_server(server_addr)?;
        let device_addr = SocketAddrV4::new(host.parse().context("无效的 IP 地址")?, port);

        // 设备不可达（IP 已变 / TCP 模式未开 / 眼镜休眠）时，
        // adb server 的 connect 请求可能长时间无响应，卡死调用线程。
        // 放到独立线程并加 10s 超时，失败信息直接给出用户可执行的处理。
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(server.connect_device(device_addr));
        });
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(result) => result.with_context(|| {
                format!(
                    "adb connect 失败: {host}:{port}（adb 二进制: {}）",
                    adb.as_deref().unwrap_or("未找到")
                )
            }),
            Err(_) => Err(anyhow::anyhow!(
                "adb connect 超时（10s）：眼镜可能不在线（IP 已变或 TCP 模式未开启），请插线重新配网"
            )),
        }
    }

    fn device(&mut self) -> Result<ADBServerDevice> {
        let mut last_err = None;
        for _ in 0..DEVICE_RETRY_COUNT {
            let mut server = new_server(&self.daemon_addr_str)?;
            match server.get_device_by_name(&self.device_serial) {
                Ok(device) => return Ok(device),
                Err(e) => {
                    last_err = Some(e);
                    std::thread::sleep(DEVICE_RETRY_DELAY);
                }
            }
        }
        Err(last_err.unwrap()).with_context(|| format!("未找到设备: {}", self.device_serial))
    }

    pub fn push_bytes(&mut self, data: &[u8], remote_path: &str) -> Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let mut device = self.device()?;
        device
            .push(&mut cursor, remote_path)
            .context("推送文件到设备失败")
    }

    pub fn forward(&mut self, local_port: u16, socket_name: &str) -> Result<()> {
        let mut device = self.device()?;
        device
            .forward(
                format!("localabstract:{socket_name}"),
                format!("tcp:{local_port}"),
            )
            .with_context(|| format!("adb forward tcp:{local_port} -> {socket_name} 失败"))
    }

    /// 同步执行 shell 命令并返回 stdout。
    pub fn shell_exec(&mut self, command: &str) -> Result<String> {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut device = self.device()?;
        device
            .shell_command(&command, Some(&mut stdout), Some(&mut stderr))
            .context("执行设备 shell 命令失败")?;
        if !stderr.is_empty() {
            let stderr_str = String::from_utf8_lossy(&stderr);
            if !stderr_str.trim().is_empty() {
                crate::logbus::emit(&format!("[adb:shell] stderr: {}", stderr_str.trim()));
            }
        }
        Ok(String::from_utf8_lossy(&stdout).trim().to_string())
    }

    pub fn kill_forward(&mut self, local_port: u16) {
        if let Ok(mut device) = self.device() {
            let _ = device.forward_remove(format!("tcp:{local_port}"));
        }
    }

    /// 启动常驻 shell（用于持有远端 scrcpy server 进程）。
    ///
    /// 写端保活期间 shell 持续存在；`AdbShellHandle::kill` drop 写端产生 EOF，
    /// 远端 shell 随之退出。跨平台实现（os_pipe）。
    pub fn shell_spawn(&mut self, command: &str) -> Result<AdbShellHandle> {
        let (mut read_end, write_end) = os_pipe::pipe().context("创建 shell pipe 失败")?;

        let cmd_with_nl = format!("{command}\n");
        write_end
            .try_clone()
            .context("克隆 shell pipe 写端失败")?
            .write_all(cmd_with_nl.as_bytes())
            .context("写入 shell 命令失败")?;
        // 注意：不 flush 丢失问题 —— write_all 到 pipe 直接进入内核缓冲。

        let serial = self.device_serial.clone();
        let daemon_addr = self.daemon_addr;

        let handle = std::thread::Builder::new()
            .name("adb-shell".to_string())
            .spawn(move || {
                let mut device = ADBServerDevice::new(serial, Some(daemon_addr));
                let stderr: Box<dyn Write + Send> = Box::new(std::io::stderr());
                let _ = device.shell(&mut read_end, stderr);
            })
            .context("创建 shell 线程失败")?;

        Ok(AdbShellHandle {
            handle: Some(handle),
            _keepalive: Some(write_end),
        })
    }
}

pub struct AdbShellHandle {
    handle: Option<std::thread::JoinHandle<()>>,
    _keepalive: Option<os_pipe::PipeWriter>,
}

impl AdbShellHandle {
    pub fn try_wait(&mut self) -> Result<bool> {
        if let Some(handle) = &self.handle {
            if handle.is_finished() {
                self.handle.take();
                return Ok(true);
            }
            return Ok(false);
        }
        Ok(true)
    }

    pub fn kill(&mut self) {
        self._keepalive.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ── 管理端封装（移植 GlassConnect adb_wrapper）──

pub struct AdbWrapper {
    server: ADBServer,
}

impl AdbWrapper {
    pub fn new() -> Result<Self> {
        let server = new_server(&adb_server_address())?;
        Ok(AdbWrapper { server })
    }

    /// 连接第一台 `ro.product.model` 以给定前缀开头的设备。
    pub fn connect_device(&mut self, model_prefix: &str) -> Result<ADBServerDevice> {
        let devices = self
            .server
            .devices()
            .map_err(|e| anyhow!("无法列出 ADB 设备: {e}"))?;
        if devices.is_empty() {
            bail!("没有找到已连接的设备");
        }
        for dev in &devices {
            let mut device = match self.server.get_device_by_name(&dev.identifier) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let model = match shell_command(&mut device, "getprop ro.product.model") {
                Ok(output) => output.trim().to_string(),
                Err(_) => continue,
            };
            if model.starts_with(model_prefix) {
                return Ok(device);
            }
        }
        bail!(
            "没有找到 model 以 '{}' 开头的设备。已连接设备: {:?}",
            model_prefix,
            devices.iter().map(|d| &d.identifier).collect::<Vec<_>>()
        )
    }

    /// 查找匹配设备序列号；`usb_only` 时拒绝 TCP 设备。
    pub fn find_device_identifier(
        &mut self,
        model_prefix: &str,
        usb_only: bool,
    ) -> Result<String> {
        let devices = self
            .server
            .devices()
            .map_err(|e| anyhow!("无法列出 ADB 设备: {e}"))?;
        if devices.is_empty() {
            bail!("没有找到已连接的设备");
        }
        let mut tcp_matches = Vec::new();
        for dev in &devices {
            let mut device = match self.server.get_device_by_name(&dev.identifier) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let model = match shell_command(&mut device, "getprop ro.product.model") {
                Ok(output) => output.trim().to_string(),
                Err(_) => continue,
            };
            if !model.starts_with(model_prefix) {
                continue;
            }
            if usb_only && is_tcp_device_identifier(&dev.identifier) {
                tcp_matches.push(dev.identifier.clone());
                continue;
            }
            return Ok(dev.identifier.clone());
        }
        if usb_only && !tcp_matches.is_empty() {
            bail!(
                "找到了目标设备，但当前是 TCP 连接而非 USB 连接: {:?}",
                tcp_matches
            );
        }
        bail!(
            "没有找到 model 以 '{}' 开头的设备。已连接设备: {:?}",
            model_prefix,
            devices.iter().map(|d| &d.identifier).collect::<Vec<_>>()
        )
    }

    pub fn get_device_by_name(&mut self, name: &str) -> Result<ADBServerDevice> {
        Ok(self.server.get_device_by_name(name)?)
    }

    pub fn list_devices(&mut self) -> Result<Vec<String>> {
        let devices = self.server.devices()?;
        Ok(devices.into_iter().map(|d| d.identifier).collect())
    }

    pub fn connect_tcp(&mut self, ip: &str, port: u16) -> Result<()> {
        let addr = SocketAddrV4::new(
            ip.parse::<Ipv4Addr>()
                .map_err(|e| anyhow!("无效的 IP 地址 {ip}: {e}"))?,
            port,
        );
        self.server.connect_device(addr)?;
        Ok(())
    }
}

fn is_tcp_device_identifier(identifier: &str) -> bool {
    if let Some((host, port)) = identifier.rsplit_once(':') {
        return !host.is_empty() && port.parse::<u16>().is_ok();
    }
    false
}

/// 清理本地 ADB 连接（disconnect / forward --remove-all / reverse --remove-all）。
pub fn cleanup_local_connections() -> Result<Vec<String>> {
    let mut actions = Vec::new();

    if let Ok(disconnect) = run_adb_command(&["disconnect"]) {
        if disconnect.status.success() {
            let stdout = String::from_utf8_lossy(&disconnect.stdout).trim().to_string();
            if stdout.is_empty() {
                actions.push("disconnect: ok".to_string());
            } else {
                actions.push(format!("disconnect: {stdout}"));
            }
        }
    }

    if run_adb_command(&["forward", "--remove-all"]).is_ok() {
        actions.push("forward --remove-all: ok".to_string());
    }
    if run_adb_command(&["reverse", "--remove-all"]).is_ok() {
        actions.push("reverse --remove-all: ok".to_string());
    }
    Ok(actions)
}

/// 在设备上执行 shell 命令并返回输出字符串。
pub fn shell_command(device: &mut ADBServerDevice, cmd: &str) -> Result<String> {
    let mut stdout = Vec::new();
    device.shell_command(&cmd, Some(&mut stdout), None)?;
    Ok(String::from_utf8_lossy(&stdout).to_string())
}

/// 推送内存数据到设备。
pub fn upload_file(
    device: &mut ADBServerDevice,
    local_data: &[u8],
    remote_path: &str,
) -> Result<()> {
    let mut cursor = std::io::Cursor::new(local_data);
    device.push(&mut cursor, remote_path)?;
    Ok(())
}

/// 开启设备 adb tcpip 模式（adbd 重启会打断传输，属预期，带超时容忍）。
pub fn enable_tcpip(port: u16) -> Result<()> {
    use std::sync::mpsc;

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = (|| -> Result<()> {
            let mut server = new_server(&adb_server_address())?;
            let devices = server.devices()?;
            if devices.is_empty() {
                bail!("没有找到已连接的设备");
            }
            let mut device = server.get_device_by_name(&devices[0].identifier)?;
            device.tcpip(port)?;
            Ok(())
        })();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(8)) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            let msg = e.to_string();
            if msg.contains("EOF") || msg.contains("broken pipe") || msg.contains("I/O error") {
                Ok(())
            } else {
                Err(e)
            }
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(()),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcp_identifier_detection() {
        assert!(is_tcp_device_identifier("192.168.1.10:5555"));
        assert!(!is_tcp_device_identifier("RG-XXXX-device"));
        assert!(!is_tcp_device_identifier(":5555"));
    }

    #[test]
    fn server_address_env_override() {
        // 不修改进程环境（并行测试不安全），仅验证默认值格式
        let addr = adb_server_address();
        assert!(addr.contains(':'));
        assert!(addr.parse::<std::net::SocketAddr>().is_ok() || addr == "127.0.0.1:5037");
    }

    #[test]
    fn parse_daemon_addr_rejects_garbage() {
        assert!(parse_daemon_addr("not-an-address").is_err());
        assert!(parse_daemon_addr("127.0.0.1:5037").is_ok());
    }
}
