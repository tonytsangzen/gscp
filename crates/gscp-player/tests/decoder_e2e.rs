//! 端到端解码器集成测试。
//!
//! 用 ffmpeg 生成真实 H264 码流（testsrc2 彩条运动画面），按 access unit
//! 切包后分别喂给软件解码器（OpenH264）与 VideoToolbox（macOS），
//! 验证：config 包/extradata 构造正确、解出帧数与尺寸、像素非平凡。
//!
//! 环境无 ffmpeg 时测试自动跳过。

use gscp_core::codec::{nal_units_annexb, strip_start_code, VideoDecoder};
use gscp_core::scrcpy::VideoCodec;
use gscp_player::decode::software::SoftwareDecoder;
use std::path::PathBuf;
use std::process::Command;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAME_COUNT: usize = 12;

struct Packet {
    is_config: bool,
    data: Vec<u8>,
}

/// 把连续 annex-B 码流切成 scrcpy 风格的包：
/// config = SPS/PPS；每帧 = 一个（或多个）slice NAL，keyframe 附带 SPS/PPS。
fn split_annexb_into_packets(data: &[u8]) -> Vec<Packet> {
    let nals: Vec<Vec<u8>> = nal_units_annexb(data)
        .into_iter()
        .map(|(off, len)| strip_start_code(&data[off..off + len]).to_vec())
        .filter(|n| !n.is_empty())
        .collect();

    let nal_type_h264 = |n: &[u8]| n[0] & 0x1F;
    let sps_pps: Vec<Vec<u8>> = nals
        .iter()
        .filter(|n| matches!(nal_type_h264(n), 7 | 8))
        .cloned()
        .collect();

    let mut packets = Vec::new();
    packets.push(Packet {
        is_config: true,
        data: nals_to_annexb(&sps_pps),
    });

    let mut current: Vec<Vec<u8>> = Vec::new();
    for nal in &nals {
        match nal_type_h264(nal) {
            7 | 8 => {}
            5 => {
                // 关键帧：SPS/PPS + slice
                let mut frame = sps_pps.clone();
                frame.push(nal.clone());
                packets.push(Packet {
                    is_config: false,
                    data: nals_to_annexb(&frame),
                });
                current.clear();
            }
            1 => {
                current.push(nal.clone());
                packets.push(Packet {
                    is_config: false,
                    data: nals_to_annexb(&current),
                });
                current.clear();
            }
            _ => {
                // AUD/SEI 等附加到下一包
                current.push(nal.clone());
            }
        }
    }
    packets
}

fn nals_to_annexb(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

fn generate_test_stream() -> Option<PathBuf> {
    let ffmpeg = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let _ = ffmpeg;

    let dir = std::env::temp_dir().join(format!("gscp-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("testsrc.h264");
    let status = Command::new("ffmpeg")
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
    Some(out)
}

fn pixel_variance(rgba: &[u8]) -> f64 {
    let luma: Vec<f64> = rgba
        .chunks_exact(4)
        .map(|px| 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64)
        .collect();
    let mean = luma.iter().sum::<f64>() / luma.len() as f64;
    luma.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / luma.len() as f64
}

fn decode_all(decoder: &mut dyn VideoDecoder, packets: &[Packet]) -> (usize, Vec<u8>) {
    let mut count = 0;
    let mut last_frame = Vec::new();
    for packet in packets {
        if packet.is_config {
            decoder
                .configure(VideoCodec::H264, &packet.data, WIDTH, HEIGHT)
                .expect("configure 失败");
            continue;
        }
        if let Some(frame) = decoder.decode(&packet.data).expect("decode 失败") {
            assert_eq!(frame.width, WIDTH);
            assert_eq!(frame.height, HEIGHT);
            assert_eq!(frame.pixels.len(), (WIDTH * HEIGHT * 4) as usize);
            count += 1;
            last_frame = frame.pixels.to_vec();
        }
    }
    for frame in decoder.flush().expect("flush 失败") {
        count += 1;
        last_frame = frame.pixels.to_vec();
    }
    (count, last_frame)
}

fn available() -> Option<PathBuf> {
    generate_test_stream()
}

#[test]
fn software_decoder_decodes_ffmpeg_stream() {
    let Some(stream) = available() else {
        eprintln!("ffmpeg 不可用，跳过");
        return;
    };
    let data = std::fs::read(&stream).unwrap();
    assert!(!data.is_empty());
    let packets = split_annexb_into_packets(&data);
    assert!(packets.len() >= FRAME_COUNT + 1);

    let mut decoder = SoftwareDecoder::new();
    assert_eq!(decoder.name(), "openh264");
    let (count, last_frame) = decode_all(&mut decoder, &packets);
    eprintln!("software decoded {count} frames");
    assert!(count >= FRAME_COUNT - 2, "软件解码帧数不足: {count}");
    // testsrc2 是运动彩条，像素应当非平凡
    assert!(pixel_variance(&last_frame) > 100.0, "画面像素过于平坦");
}

#[cfg(target_os = "macos")]
#[test]
fn videotoolbox_decodes_ffmpeg_stream() {
    let Some(stream) = available() else {
        eprintln!("ffmpeg 不可用，跳过");
        return;
    };
    let data = std::fs::read(&stream).unwrap();
    let packets = split_annexb_into_packets(&data);

    let mut decoder =
        match gscp_player::decode::videotoolbox::VideoToolboxDecoder::new() {
            Ok(d) => d,
            Err(err) => panic!("VideoToolbox 初始化失败: {err:#}"),
        };
    let (count, last_frame) = decode_all(&mut decoder, &packets);
    eprintln!("videotoolbox decoded {count} frames");
    assert!(count >= FRAME_COUNT - 2, "VideoToolbox 帧数不足: {count}");
    assert!(pixel_variance(&last_frame) > 100.0, "画面像素过于平坦");
}

#[cfg(target_os = "windows")]
#[test]
fn mediafoundation_decodes_ffmpeg_stream() {
    let Some(stream) = available() else {
        eprintln!("ffmpeg 不可用，跳过");
        return;
    };
    let data = std::fs::read(&stream).unwrap();
    let packets = split_annexb_into_packets(&data);

    let mut decoder = match gscp_player::decode::mediafoundation::MediaFoundationDecoder::new() {
        Ok(d) => d,
        Err(err) => {
            // CI 精简环境可能没有 Media Foundation 平台，优雅跳过
            eprintln!("Media Foundation 不可用，跳过: {err:#}");
            return;
        }
    };
    let (count, last_frame) = decode_all(&mut decoder, &packets);
    eprintln!("mediafoundation({}) decoded {count} frames", decoder.name());
    assert!(count >= FRAME_COUNT - 2, "Media Foundation 帧数不足: {count}");
    assert!(pixel_variance(&last_frame) > 100.0, "画面像素过于平坦");

    // 与软件解码对比平均亮度：两端口径（BT.601 limited range）应一致
    let mut software = SoftwareDecoder::new();
    let (_, sw_last) = decode_all(&mut software, &packets);
    let mean = |px: &[u8]| -> f64 {
        let luma: Vec<f64> = px
            .chunks_exact(4)
            .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
            .collect();
        luma.iter().sum::<f64>() / luma.len() as f64
    };
    let (mf_mean, sw_mean) = (mean(&last_frame), mean(&sw_last));
    eprintln!("mean luma: mf={mf_mean:.1} sw={sw_mean:.1}");
    assert!(
        (mf_mean - sw_mean).abs() < 8.0,
        "MF 与软件解码平均亮度差过大: mf={mf_mean:.1} sw={sw_mean:.1}"
    );
}

#[cfg(test)]
mod units {
    use super::*;

    #[test]
    fn variance_of_flat_image_is_zero() {
        assert!(pixel_variance(&vec![128_u8; 16 * 4]) == 0.0);
    }

    #[test]
    fn split_packets_groups_slices() {
        let sps = [0x67_u8, 1, 2, 3];
        let pps = [0x68_u8, 4, 5];
        let idr = [0x65_u8, 6, 7, 8, 9];
        let slice = [0x41_u8, 10, 11];
        let mut data = Vec::new();
        for nal in [&sps[..], &pps[..], &idr[..], &slice[..]] {
            data.extend_from_slice(&[0, 0, 0, 1]);
            data.extend_from_slice(nal);
        }
        let packets = split_annexb_into_packets(&data);
        // 注意：尾部的 P slice 长度 3 无法与起始码区分出问题 —— 这里只验证分组
        assert!(packets.len() >= 2); // config + IDR（+ P slice）
        assert!(packets[0].is_config);
        assert!(!packets[1].is_config);
        // IDR 包含 SPS/PPS
        assert!(packets[1].data.windows(4).any(|w| w == [0, 0, 0, 1]));
    }
}
