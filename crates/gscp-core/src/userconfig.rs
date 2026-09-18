//! 用户配置（INI 格式）。
//!
//! 保存 WiFi 网络列表（SSID/PSK/IP，最多 20 条，去重按 SSID）、
//! 连接模式与三路流开关。移植自 GlassConnect `config.rs` 的 UserConfig，
//! 存储位置改为应用数据目录（兼容读取旧的临时目录 `.config` 文件）。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

pub struct UserConfig {
    file_path: PathBuf,
    entries: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WifiEntry {
    ssid: String,
    psk: String,
    #[serde(default)]
    ip: String,
}

impl UserConfig {
    pub fn load() -> Self {
        let file_path = crate::paths::user_config_path();
        let legacy = std::env::temp_dir().join(".config");
        let entries = if file_path.exists() {
            Self::read_ini(&file_path).unwrap_or_default()
        } else if legacy.exists() {
            Self::read_ini(&legacy).unwrap_or_default()
        } else {
            HashMap::new()
        };
        UserConfig { file_path, entries }
    }

    fn read_ini(path: &PathBuf) -> anyhow::Result<HashMap<String, String>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut map = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            if let Some((key, value)) = line.split_once('=') {
                map.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        Ok(map)
    }

    pub fn get(&self, key: &str) -> String {
        self.entries.get(key).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.entries.insert(key.to_string(), value.to_string());
        self.flush();
    }

    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.entries
            .get(key)
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(default)
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.entries.insert(key.to_string(), value.to_string());
        self.flush();
    }

    /// WiFi 列表，(ssid, psk, ip)。
    pub fn get_wifi_list(&self) -> Vec<(String, String, String)> {
        let raw = self.entries.get("networks").cloned().unwrap_or_default();
        if raw.is_empty() {
            return Vec::new();
        }
        serde_json::from_str::<Vec<WifiEntry>>(&raw)
            .unwrap_or_default()
            .into_iter()
            .map(|e| (e.ssid, e.psk, e.ip))
            .collect()
    }

    pub fn get_network(&self, ssid: &str) -> Option<(String, String)> {
        self.get_wifi_list()
            .into_iter()
            .find(|(s, _, _)| s == ssid)
            .map(|(_, psk, ip)| (psk, ip))
    }

    pub fn get_last_ssid(&self) -> String {
        self.entries.get("last_ssid").cloned().unwrap_or_default()
    }

    fn set_last_ssid(&mut self, ssid: &str) {
        self.entries.insert("last_ssid".to_string(), ssid.to_string());
    }

    /// 添加/更新 WiFi 记录（同 SSID 去重，最新在前，最多 20 条）。
    pub fn add_wifi_network(&mut self, ssid: &str, psk: &str, ip: &str) {
        if ssid.is_empty() {
            return;
        }
        let mut list = self.get_wifi_list();
        list.retain(|(s, _, _)| s != ssid);
        list.insert(0, (ssid.to_string(), psk.to_string(), ip.to_string()));
        list.truncate(20);
        let json = serde_json::to_string(&list).unwrap_or_default();
        self.entries.insert("networks".to_string(), json);
        self.set_last_ssid(ssid);
        self.flush();
    }

    fn flush(&self) {
        let content: String = self
            .entries
            .iter()
            .map(|(k, v)| format!("{}={}\n", k, v))
            .collect();
        let _ = (|| -> anyhow::Result<()> {
            if let Some(parent) = self.file_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::File::create(&self.file_path)?;
            file.write_all(content.as_bytes())?;
            Ok(())
        })();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wifi_list_roundtrip_and_dedup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("user.ini");
        let mut cfg = UserConfig {
            file_path: path.clone(),
            entries: HashMap::new(),
        };
        cfg.add_wifi_network("Glass-5G", "psk1", "192.168.1.2");
        cfg.add_wifi_network("Home", "psk2", "");
        cfg.add_wifi_network("Glass-5G", "psk1-new", "192.168.1.3");

        let list = cfg.get_wifi_list();
        assert_eq!(list.len(), 2);
        // 最新在前
        assert_eq!(list[0].0, "Glass-5G");
        assert_eq!(list[0].1, "psk1-new");
        assert_eq!(list[0].2, "192.168.1.3");

        // 重新加载保持持久化
        let reloaded = UserConfig {
            file_path: path,
            entries: read_ini_legacy_for_test(dir.path().join("user.ini")).unwrap(),
        };
        assert_eq!(reloaded.get_last_ssid(), "Glass-5G");
        assert_eq!(reloaded.get_network("Home").unwrap().0, "psk2");
    }

    fn read_ini_legacy_for_test(path: PathBuf) -> anyhow::Result<HashMap<String, String>> {
        UserConfig::read_ini(&path)
    }

    #[test]
    fn empty_ssid_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = UserConfig {
            file_path: dir.path().join("user.ini"),
            entries: HashMap::new(),
        };
        cfg.add_wifi_network("", "psk", "");
        assert!(cfg.get_wifi_list().is_empty());
    }
}
