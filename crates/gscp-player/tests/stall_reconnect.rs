//! 静默断链自动重连集成测试。
//!
//! 模拟「长时间播放后连接静默中断」的场景：mock 服务器发完头部、config 包
//! 和一帧后停止发送（不关闭连接）。修复前：客户端读线程永久阻塞，画面卡死；
//! 修复后：静默超过 stall 阈值判定连接失效，自动重连（服务器会收到第二次连接）。

use gscp_core::scrcpy::VideoSource;
use gscp_core::settings::DecoderBackend;
use gscp_player::session;
use gscp_player::slot::SharedSlot;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

fn frame_header(pts_and_flags: u64, size: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(12);
    header.extend_from_slice(&(pts_and_flags | (1_u64 << 63)).to_be_bytes());
    header.extend_from_slice(&(size as u32).to_be_bytes());
    header
}

#[test]
fn stall_connection_triggers_reconnect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let accept_count = Arc::new(AtomicUsize::new(0));
    let counter = accept_count.clone();

    let server = std::thread::spawn(move || {
        for conn in listener.incoming() {
            counter.fetch_add(1, Ordering::SeqCst);
            let count = counter.load(Ordering::SeqCst);
            if count >= 3 {
                // 已经验证到多次重连，停止接受
                break;
            }
            let mut sock = conn.unwrap();
            // 每个连接单独持有：发送数据后保持打开但不发送（模拟静默断链），
            // accept 循环继续，等待客户端的自动重连
            std::thread::spawn(move || {
                // scrcpy 头：1 dummy + 64B 设备名 + codec(h264) + 宽高
                let mut header = vec![0_u8];
                let mut name = b"RG-stall".to_vec();
                name.resize(64, 0);
                header.extend_from_slice(&name);
                header.extend_from_slice(&0x68323634_u32.to_be_bytes());
                header.extend_from_slice(&WIDTH.to_be_bytes());
                header.extend_from_slice(&HEIGHT.to_be_bytes());
                sock.write_all(&header).unwrap();

                // config 包（假 SPS/PPS，仅结构合法）
                let config = [0_u8, 0, 0, 1, 0x67, 1, 2, 3, 0, 0, 1, 0x68, 4, 5];
                sock.write_all(&frame_header(u64::MAX - 1, config.len())).unwrap();
                sock.write_all(&config).unwrap();
                // 一个普通数据包
                let data = [0_u8, 0, 0, 1, 0x41, 9, 9, 9];
                sock.write_all(&frame_header(1, data.len())).unwrap();
                sock.write_all(&data).unwrap();
                // 静默：不关连接、不再发数据 —— 修复前客户端会永远阻塞在这里
                std::thread::sleep(Duration::from_secs(30));
            });
        }
    });

    let slot = SharedSlot::default();
    let mut source = VideoSource::new(
        addr.ip().to_string(),
        addr.port(),
        true,  // send_device_meta
        true,  // send_frame_meta
        true,  // send_dummy_byte
        true,  // send_codec_meta
        None,
        None,
        false,
        false,
    )
    .unwrap();
    // 测试用短静默阈值（默认 15s 太久）
    source.stall_timeout = Some(Duration::from_secs(1));

    session::spawn_video_source(
        "camera",
        source,
        slot.clone(),
        DecoderBackend::Software,
        None,
    )
    .unwrap();

    // 等待至少 2 次连接（首次 + 静默超时后重连）
    let deadline = Instant::now() + Duration::from_secs(15);
    while accept_count.load(Ordering::SeqCst) < 2 {
        if Instant::now() > deadline {
            panic!(
                "静默 {}s 后未触发重连（accept 次数 = {}）——卡死保护未生效",
                1,
                accept_count.load(Ordering::SeqCst)
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // 会话不应报告致命错误（静默断链属于预期恢复路径）
    server.join().unwrap();
}
