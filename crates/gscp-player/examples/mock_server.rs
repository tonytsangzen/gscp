//! mock scrcpy server（三路，常驻）：给 GUI 冒烟测试喂真实码流。
//!
//! 用法：`cargo run -p gscp-player --example mock_server`
//! 输出三个端口：camera / overlay / audio，配合
//! `gscp --player --remote-host 127.0.0.1 --camera-port C --remote-overlay-port O --audio-port A`

use gscp_core::codec::{nal_units_annexb, strip_start_code};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::Duration;

const SCRCPY_FLAG_CONFIG: u64 = 1 << 63;

fn gen_stream(size: &str, frames: usize, path: &std::path::Path) -> Option<Vec<u8>> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={size}:rate=12"),
            "-frames:v",
            &frames.to_string(),
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
        .arg(path)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    std::fs::read(path).ok()
}

fn split_packets(data: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let nals: Vec<Vec<u8>> = nal_units_annexb(data)
        .into_iter()
        .map(|(off, len)| strip_start_code(&data[off..off + len]).to_vec())
        .filter(|n| !n.is_empty())
        .collect();
    let nal_type = |n: &[u8]| n[0] & 0x1F;
    let mut config = Vec::new();
    let mut frames = Vec::new();
    let mut pending = Vec::new();
    for nal in &nals {
        match nal_type(nal) {
            7 | 8 => {
                config.extend_from_slice(&[0, 0, 0, 1]);
                config.extend_from_slice(nal);
            }
            5 => {
                let mut frame = config.clone();
                frame.extend_from_slice(&[0, 0, 0, 1]);
                frame.extend_from_slice(nal);
                frames.push(frame);
            }
            1 => {
                pending.extend_from_slice(&[0, 0, 0, 1]);
                pending.extend_from_slice(nal);
                frames.push(std::mem::take(&mut pending));
            }
            _ => {
                pending.extend_from_slice(&[0, 0, 0, 1]);
                pending.extend_from_slice(nal);
            }
        }
    }
    (config, frames)
}

fn write_video_header(
    sock: &mut TcpStream,
    send_dummy: bool,
    send_device_meta: bool,
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    if send_dummy {
        sock.write_all(&[0])?;
    }
    if send_device_meta {
        let mut name = b"RG-mock".to_vec();
        name.resize(64, 0);
        sock.write_all(&name)?;
    }
    sock.write_all(&0x68323634_u32.to_be_bytes())?; // h264
    sock.write_all(&width.to_be_bytes())?;
    sock.write_all(&height.to_be_bytes())?;
    Ok(())
}

fn send_packets(sock: &mut TcpStream, config: &[u8], frames: &[Vec<u8>]) -> std::io::Result<()> {
    sock.write_all(&frame_header(SCRCPY_FLAG_CONFIG, config.len()))?;
    sock.write_all(config)?;
    let mut pts = 0_u64;
    loop {
        for frame in frames {
            sock.write_all(&frame_header(pts & !SCRCPY_FLAG_CONFIG, frame.len()))?;
            sock.write_all(frame)?;
            pts = pts.wrapping_add(1);
        }
        std::thread::sleep(Duration::from_millis(1000 / 12));
    }
}

fn frame_header(pts_and_flags: u64, size: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(12);
    header.extend_from_slice(&pts_and_flags.to_be_bytes());
    header.extend_from_slice(&(size as u32).to_be_bytes());
    header
}

/// 生成 440Hz 正弦 s16le 立体声 PCM 包。
fn sine_pcm(sample_index: &mut usize, samples: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples * 4);
    for _ in 0..samples {
        let value =
            (2.0 * std::f64::consts::PI * 440.0 * (*sample_index as f64) / 48000.0).sin();
        let s = (value * 0.3 * 32767.0) as i16;
        data.extend_from_slice(&s.to_le_bytes());
        data.extend_from_slice(&s.to_le_bytes());
        *sample_index += 1;
    }
    data
}

fn main() {
    let tmp = std::env::temp_dir().join("gscp-mock-server");
    std::fs::create_dir_all(&tmp).expect("tmp dir");

    let camera_stream = gen_stream("640x480", 12, &tmp.join("camera.h264"))
        .expect("ffmpeg 生成 camera 码流失败");
    let overlay_stream = gen_stream("320x240", 12, &tmp.join("overlay.h264"))
        .expect("ffmpeg 生成 overlay 码流失败");
    let (camera_config, camera_frames) = split_packets(&camera_stream);
    let (overlay_config, overlay_frames) = split_packets(&overlay_stream);

    let camera_listener = TcpListener::bind("127.0.0.1:0").expect("bind camera");
    let overlay_listener = TcpListener::bind("127.0.0.1:0").expect("bind overlay");
    let audio_listener = TcpListener::bind("127.0.0.1:0").expect("bind audio");

    println!(
        "MOCK_READY camera={} overlay={} audio={}",
        camera_listener.local_addr().unwrap().port(),
        overlay_listener.local_addr().unwrap().port(),
        audio_listener.local_addr().unwrap().port(),
    );

    // camera：全 meta
    let handle_cam = std::thread::spawn(move || {
        for conn in camera_listener.incoming() {
            let mut sock = conn.expect("accept");
            write_video_header(&mut sock, true, true, 640, 480).ok();
            if let Err(err) = send_packets(&mut sock, &camera_config, &camera_frames) {
                eprintln!("[mock] camera: {err}");
            }
        }
    });

    // overlay：无 dummy/device meta（与播放器 overlay 默认一致）
    let handle_ov = std::thread::spawn(move || {
        for conn in overlay_listener.incoming() {
            let mut sock = conn.expect("accept");
            write_video_header(&mut sock, false, false, 320, 240).ok();
            if let Err(err) = send_packets(&mut sock, &overlay_config, &overlay_frames) {
                eprintln!("[mock] overlay: {err}");
            }
        }
    });

    // audio：raw PCM s16le 48k stereo
    let handle_au = std::thread::spawn(move || {
        for conn in audio_listener.incoming() {
            let mut sock = conn.expect("accept");
            sock.write_all(&0x00726177_u32.to_be_bytes()).ok(); // raw
            let mut sample_index = 0_usize;
            loop {
                let pcm = sine_pcm(&mut sample_index, 1024);
                if sock
                    .write_all(&frame_header(0, pcm.len()))
                    .and_then(|_| sock.write_all(&pcm))
                    .is_err()
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1024 * 1000 / 48000 / 2));
            }
        }
    });

    let _ = (handle_cam, handle_ov, handle_au);
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
