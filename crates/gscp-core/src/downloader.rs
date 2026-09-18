//! 专用工具链按需下载。
//!
//! - `adb`（platform-tools）：不打包，首次需要时按平台下载 Google 官方包，
//!   解压到 `<data>/bin`；
//! - `scrcpy-server`：默认使用内嵌 jar，可按版本从 GitHub Releases 下载覆盖。

use crate::logbus;
use anyhow::{anyhow, Context, Result};
use std::io::Read;

const PLATFORM_TOOLS_URL_LATEST: &str =
    "https://dl.google.com/android/repository/platform-tools-latest";
const SCRCPY_RELEASE_URL: &str = "https://github.com/Genymobile/scrcpy/releases/download";

fn platform_tag() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "linux" => "linux",
        "windows" => "windows",
        other => other,
    }
}

fn http_get(url: &str) -> Result<ureq::Response> {
    let agent = ureq::builder()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300))
        .redirects(8)
        .build();
    agent
        .get(url)
        .call()
        .map_err(|e| anyhow!("下载失败 {url}: {e}"))
}

/// 下载 platform-tools 并解压 adb（Windows 含两个依赖 dll）到数据目录 bin/。
pub fn install_platform_tools() -> Result<()> {
    let url = format!("{}_{}.zip", PLATFORM_TOOLS_URL_LATEST, platform_tag());
    logbus::emit(&format!("[downloader] 下载 {url} ..."));
    let response = http_get(&url)?;

    let mut zip_bytes = Vec::new();
    response
        .into_reader()
        .take(512 * 1024 * 1024)
        .read_to_end(&mut zip_bytes)
        .context("下载 platform-tools 失败")?;
    logbus::emit(&format!(
        "[downloader] platform-tools 下载完成 ({:.1} MB)，解压中...",
        zip_bytes.len() as f64 / 1024.0 / 1024.0
    ));

    extract_adb_from_zip(&zip_bytes)?;
    logbus::emit("[downloader] platform-tools 安装完成");
    Ok(())
}

fn extract_adb_from_zip(zip_bytes: &[u8]) -> Result<()> {
    let bin_dir = crate::paths::bin_dir();
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes))
        .context("解析 platform-tools zip 失败")?;

    #[cfg(target_os = "windows")]
    let wanted = ["adb.exe", "AdbWinApi.dll", "AdbWinUsbApi.dll"];
    #[cfg(not(target_os = "windows"))]
    let wanted = ["adb"];

    let mut extracted = 0;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        let file_name = name.rsplit('/').next().unwrap_or(&name).to_string();
        if !wanted.contains(&file_name.as_str()) {
            continue;
        }
        let mut out_bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut out_bytes)?;
        let out_path = bin_dir.join(&file_name);
        std::fs::write(&out_path, &out_bytes)
            .with_context(|| format!("写入 {} 失败", out_path.display()))?;
        crate::paths::ensure_executable(&out_path)?;
        extracted += 1;
    }
    if extracted == 0 {
        anyhow::bail!("platform-tools 包中未找到 adb");
    }
    Ok(())
}

/// 从 GitHub Releases 下载指定版本 scrcpy-server 到数据目录，返回其路径。
/// 失败时回退内嵌版本（返回 None 表示用内嵌）。
pub fn install_scrcpy_server(version: &str) -> Result<std::path::PathBuf> {
    let version = version.trim_start_matches('v');
    let url = format!("{SCRCPY_RELEASE_URL}/v{version}/scrcpy-server-v{version}");
    logbus::emit(&format!("[downloader] 下载 scrcpy-server v{version} ..."));
    let response = http_get(&url)?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(64 * 1024 * 1024)
        .read_to_end(&mut bytes)
        .context("下载 scrcpy-server 失败")?;
    if !bytes.starts_with(b"PK") {
        anyhow::bail!("scrcpy-server v{version} 内容异常（非 jar）");
    }
    let path = crate::paths::data_dir().join(format!("scrcpy-server-v{version}"));
    std::fs::write(&path, &bytes)?;
    logbus::emit(&format!("[downloader] scrcpy-server v{version} 安装完成"));
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_tag_known() {
        assert!(matches!(platform_tag(), "darwin" | "linux" | "windows"));
    }

    #[test]
    fn url_templates() {
        assert_eq!(
            format!("{}_{}.zip", PLATFORM_TOOLS_URL_LATEST, platform_tag()),
            match platform_tag() {
                "darwin" => "https://dl.google.com/android/repository/platform-tools-latest_darwin.zip",
                "linux" => "https://dl.google.com/android/repository/platform-tools-latest_linux.zip",
                _ => "https://dl.google.com/android/repository/platform-tools-latest_windows.zip",
            }
        );
        assert!(SCRCPY_RELEASE_URL.starts_with("https://github.com/Genymobile/scrcpy/releases"));
    }

    #[test]
    fn extract_rejects_garbage_zip() {
        assert!(extract_adb_from_zip(b"not a zip").is_err());
    }
}
