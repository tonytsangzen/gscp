//! mock scrcpy server 集成测试：验证直连模式全链路
//! （socket 握手 → 协议解析 → config 包 → 解码 → SharedSlot 出帧）。
//!
//! 测试内起一个 TCP server 模拟 scrcpy server 的单路 camera socket
//! （dummy byte + 64B 设备名 + codec meta + frame meta/config 包），
//! 码流来自 ffmpeg 生成（不可用时跳过）。

use gscp_core::scrcpy::VideoSource;
use gscp_core::settings::DecoderBackend;
use gscp_player::session;
use gscp_player::slot::{SharedSlot, VIDEO_BUFFER_LOW_WATER};
use std::io::Write;
use std::net::TcpListener;
use std::time::{Duration, Instant};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAME_COUNT: usize = 12;

fn h264_stream() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("gscp-mock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("mock.h264");
    if !out.exists() {
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=12"),
                "-frames:v",
                &FRAME_COUNT.to_string(),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-x264-params",
                "keyint=6:min-keyint=6",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "h264",
            ])
            .arg(&out)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
    }
    // 重新生成以保证长度一致
    let _ = std::fs::remove_file(&out);
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=12"),
            "-frames:v",
            &FRAME_COUNT.to_string(),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-x264-params",
            "keyint=6:min-keyint=6",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
        ])
        .arg(&out)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    std::fs::read(&out).ok()
}

fn frame_header(pts_and_flags: u64, size: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(12);
    header.extend_from_slice(&pts_and_flags.to_be_bytes());
    header.extend_from_slice(&(size as u32).to_be_bytes());
    header
}

const SCRCPY_FLAG_CONFIG: u64 = 1 << 63;

#[test]
fn mock_camera_stream_flows_to_slot() {
    let Some(stream_bytes) = h264_stream() else {
        eprintln!("ffmpeg 不可用，跳过");
        return;
    };
    // 复用 decoder_e2e 的切包方式：把 SPS/PPS 作为一个 config 包，
    // 其余 slice 作为数据包。这里直接按 NAL 切。
    let packets = split_packets(&stream_bytes);
    assert!(packets.len() >= 3);

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let server = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().expect("accept");
        // scrcpy 头：1 dummy + 64 设备名 + 4 codec + 4 w + 4 h
        let mut header = vec![0_u8];
        let mut name = b"RG-mock".to_vec();
        name.resize(64, 0);
        header.extend_from_slice(&name);
        header.extend_from_slice(&0x68323634_u32.to_be_bytes()); // h264
        header.extend_from_slice(&WIDTH.to_be_bytes());
        header.extend_from_slice(&HEIGHT.to_be_bytes());
        sock.write_all(&header).expect("write header");

        for (index, packet) in packets.iter().enumerate() {
            let (flags, data) = if packet.is_config {
                (SCRCPY_FLAG_CONFIG, &packet.data)
            } else {
                ((index as u64) & !(1 << 63), &packet.data)
            };
            sock.write_all(&frame_header(flags, data.len()))
                .expect("write frame meta");
            sock.write_all(data).expect("write body");
        }
        // 保持连接一段时间，等待消费端读完
        std::thread::sleep(Duration::from_millis(800));
    });

    let slot = SharedSlot::default();
    let source = VideoSource::new(
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
    .expect("source");

    // 直连模式（bootstrap=None），软件解码路径保证在无显示环境可用
    session::spawn_video_source(
        "camera",
        source,
        slot.clone(),
        DecoderBackend::Software,
        None,
    )
    .expect("spawn source");

    // 等待低水位起播
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let snap = slot.snapshot_for_render();
        if snap.frame.is_some() && snap.buffer_depth + 1 >= VIDEO_BUFFER_LOW_WATER - 2 {
            break;
        }
        if snap.error.is_some() {
            panic!("会话报错: {}", snap.error.unwrap());
        }
        if Instant::now() > deadline {
            let snap = slot.snapshot_for_render();
            panic!(
                "等待帧超时: info={:?} err={:?} depth={} ended={}",
                snap.info, snap.error, snap.buffer_depth, snap.ended
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let snap = slot.snapshot_for_render();
    let frame = snap.frame.expect("应有帧");
    assert_eq!(frame.width, WIDTH);
    assert_eq!(frame.height, HEIGHT);
    assert_eq!(frame.pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    assert!(snap.bytes_received > 0);

    server.join().expect("server thread");
}

struct SimplePacket {
    is_config: bool,
    data: Vec<u8>,
}

fn split_packets(data: &[u8]) -> Vec<SimplePacket> {
    use gscp_core::codec::{nal_units_annexb, strip_start_code};
    let nals: Vec<Vec<u8>> = nal_units_annexb(data)
        .into_iter()
        .map(|(off, len)| strip_start_code(&data[off..off + len]).to_vec())
        .filter(|n| !n.is_empty())
        .collect();
    let nal_type = |n: &[u8]| n[0] & 0x1F;
    let config: Vec<Vec<u8>> = nals
        .iter()
        .filter(|n| matches!(nal_type(n), 7 | 8))
        .cloned()
        .collect();
    let mut packets = vec![SimplePacket {
        is_config: true,
        data: join_nals(&config),
    }];
    for nal in &nals {
        if matches!(nal_type(nal), 7 | 8) {
            continue;
        }
        packets.push(SimplePacket {
            is_config: false,
            data: join_nals(&[nal.clone()]),
        });
    }
    packets
}

fn join_nals(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}
