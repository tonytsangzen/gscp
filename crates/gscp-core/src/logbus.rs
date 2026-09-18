//! 进程内日志总线。
//!
//! 所有模块通过 [`emit`] 输出运行日志：同时写 stderr（player 子进程由
//! 管理端转发）并广播给已注册的订阅者（管理端转发到 WebView 日志区）。
//! 对控制台进度类输出做 ANSI/控制符清洗，移植自 GlassConnect。

use std::io::Write;
use std::sync::Mutex;

type Subscriber = Box<dyn Fn(&str) + Send + Sync>;

static SUBSCRIBERS: Mutex<Vec<Subscriber>> = Mutex::new(Vec::new());

/// 注册日志订阅者。返回后立即开始接收后续日志（不回放历史）。
pub fn subscribe(sub: Subscriber) {
    SUBSCRIBERS.lock().unwrap().push(sub);
}

/// 输出一条日志（自动清洗、加时间戳前缀）。
pub fn emit(msg: &str) {
    let sanitized = sanitize(msg);
    if sanitized.is_empty() {
        return;
    }
    let line = format!("[{}]:{}", timestamp(), sanitized);
    // stderr 可能已关闭（管理端退出后 player 子进程的管道断裂）：
    // 写失败必须静默忽略 —— eprintln! 在 EPIPE 时会 panic，
    // 配合 release 的 panic=abort 会直接崩溃整个进程。
    let _ = writeln!(std::io::stderr(), "{}", line);
    for sub in SUBSCRIBERS.lock().unwrap().iter() {
        sub(&line);
    }
}

fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        (secs % 86400) / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// 清洗 carriage-return 重写、退格、ANSI 转义与控制字符。
pub fn sanitize(msg: &str) -> String {
    let mut output = String::new();
    let mut chars = msg.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\r' => output.clear(),
            '\u{0008}' => {
                output.pop();
            }
            '\u{001b}' => {
                if matches!(chars.peek(), Some('[')) {
                    chars.next();
                    while let Some(next) = chars.next() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
            }
            c if c.is_control() && c != '\t' => {}
            c => output.push(c),
        }
    }

    output.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_ansi_and_progress_lines() {
        assert_eq!(sanitize("\u{1b}[31m红色\u{1b}[0m"), "红色");
        assert_eq!(sanitize("50%\r100%"), "100%");
        assert_eq!(sanitize("ab\u{0008}c"), "ac");
        assert_eq!(sanitize("\u{0007}"), "");
    }

    #[test]
    fn subscribers_receive_lines() {
        let got = std::sync::Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = got.clone();
        subscribe(Box::new(move |line| sink.lock().unwrap().push(line.to_string())));
        emit("总线测试消息");
        let messages = got.lock().unwrap();
        assert!(
            messages.iter().any(|m| m.contains("总线测试消息")),
            "订阅者未收到日志: {messages:?}"
        );
    }
}
