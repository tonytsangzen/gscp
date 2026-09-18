//! 平台数据目录约定。
//!
//! 所有运行时产物（下载的 adb、settings.yaml、user.ini、下载的
//! scrcpy-server）都放在应用数据目录下，安装包本身保持精简。

use std::path::PathBuf;

/// 应用数据目录（不存在则创建）。
pub fn data_dir() -> std::path::PathBuf {
    let dir = match std::env::var("GSCP_DATA_DIR") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => default_data_dir(),
    };
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn default_data_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = home_dir() {
            return home.join("Library/Application Support/gscp");
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("gscp");
            }
        }
        // Android 上通常没有 HOME：由 JNI 侧通过 GSCP_DATA_DIR 指定 app 目录
        if let Some(home) = home_dir() {
            return home.join(".local/share/gscp");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            if !appdata.is_empty() {
                return PathBuf::from(appdata).join("gscp");
            }
        }
        if let Some(home) = home_dir() {
            return home.join(".gscp");
        }
    }
    std::env::temp_dir().join("gscp")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// 下载/工具二进制目录：`<data>/bin`。
pub fn bin_dir() -> PathBuf {
    let dir = data_dir().join("bin");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 共享设置文件：`<data>/settings.yaml`。
pub fn settings_path() -> PathBuf {
    data_dir().join("settings.yaml")
}

/// 用户配置（INI）：`<data>/user.ini`。
pub fn user_config_path() -> PathBuf {
    data_dir().join("user.ini")
}

#[cfg(unix)]
pub(crate) fn ensure_executable(path: &std::path::Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path)?.permissions();
    let mode = permissions.mode();
    if mode & 0o111 == 0 {
        permissions.set_mode(mode | 0o755);
        std::fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn ensure_executable(_path: &std::path::Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_dir_is_writable() {
        let dir = data_dir();
        assert!(dir.is_dir());
        let probe = dir.join(".probe");
        std::fs::write(&probe, b"ok").unwrap();
        assert_eq!(std::fs::read(&probe).unwrap(), b"ok");
        let _ = std::fs::remove_file(&probe);
    }
}
