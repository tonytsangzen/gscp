//! 视频编解码辅助：annex-B 解析、AVCC/HVCC extradata 构造、解码器 trait。
//!
//! extradata 构造移植自 mixplayer decoder.rs（系统解码器 VideoToolbox /
//! Media Foundation 与软件解码器同样需要这些配置记录）。

use crate::scrcpy::VideoCodec;
use anyhow::Result;
use std::sync::Arc;

/// 解码输出的 RGBA 帧（零拷贝跨线程）。
#[derive(Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub seq: u64,
    pub pixels: Arc<[u8]>,
}

pub fn rgba_bytes(width: u32, height: u32) -> Result<usize> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| anyhow::anyhow!("视频尺寸过大"))?;
    pixels
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("RGBA 帧尺寸过大"))
}

/// 跨平台视频解码器接口。
///
/// 会话语义（与 mixplayer 一致）：
/// - 收到 config 包（SPS/PPS annex-B）时调用 [`VideoDecoder::configure`]；
/// - 之后的每个 packet 调用 [`VideoDecoder::decode`]；
/// - 会话结束调用 [`VideoDecoder::flush`] 排空残留帧。
pub trait VideoDecoder: Send {
    /// 收到 config 包时调用。`width`/`height` 来自 scrcpy codec meta，
    /// 硬解器（VideoToolbox/MediaFoundation）创建会话时需要。
    fn configure(
        &mut self,
        codec: VideoCodec,
        config_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()>;
    /// 返回 Some 表示解出一帧 RGBA。
    fn decode(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>>;
    /// 排空解码器内部缓冲的帧。
    fn flush(&mut self) -> Result<Vec<DecodedFrame>>;
    fn name(&self) -> &'static str;
}

pub fn rgba_frame(width: u32, height: u32, seq: u64, rgba: Vec<u8>) -> Result<DecodedFrame> {
    let expected = rgba_bytes(width, height)?;
    if rgba.len() != expected {
        anyhow::bail!(
            "RGBA 帧尺寸不匹配: 期望 {expected} 字节，实际 {}",
            rgba.len()
        );
    }
    Ok(DecodedFrame {
        width,
        height,
        seq,
        pixels: Arc::<[u8]>::from(rgba),
    })
}

// ── annex-B 解析 ──

/// 找出全部起始码位置（优先识别 4 字节起始码）。
fn find_start_codes(data: &[u8]) -> Vec<(usize, usize)> {
    let len = data.len();
    let mut positions = Vec::new();
    let mut i = 0;
    while i + 3 <= len {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            if i > 0 && data[i - 1] == 0 {
                positions.push((i - 1, 4));
            } else {
                positions.push((i, 3));
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    positions
}

/// 切分 annex-B 数据为 NAL 单元（返回的切片含起始码）。
///
/// 与 mixplayer 原实现语义一致，但修正了紧贴数据末尾的 NAL 被截断的问题
/// （scrcpy config 包的 PPS 恰好总在末尾）。
pub fn nal_units_annexb(data: &[u8]) -> Vec<(usize, usize)> {
    let codes = find_start_codes(data);
    codes
        .iter()
        .enumerate()
        .map(|(idx, &(start, _code_len))| {
            let end = codes.get(idx + 1).map(|&(next, _)| next).unwrap_or(data.len());
            (start, end - start)
        })
        .collect()
}

/// 去掉起始码，返回裸 NAL 切片。
pub fn strip_start_code(nal_data: &[u8]) -> &[u8] {
    if nal_data.len() > 4
        && nal_data[0] == 0
        && nal_data[1] == 0
        && nal_data[2] == 0
        && nal_data[3] == 1
    {
        &nal_data[4..]
    } else if nal_data.len() > 3 && nal_data[0] == 0 && nal_data[1] == 0 && nal_data[2] == 1 {
        &nal_data[3..]
    } else {
        nal_data
    }
}

/// 列出 annex-B 数据中的裸 NAL（无起始码）。
pub fn raw_nals_annexb(data: &[u8]) -> Vec<&[u8]> {
    nal_units_annexb(data)
        .into_iter()
        .map(|(offset, length)| strip_start_code(&data[offset..offset + length]))
        .filter(|nalu| !nalu.is_empty())
        .collect()
}

/// 把 annex-B 转成 4 字节长度前缀（AVCC 封包，供 VideoToolbox/Media Foundation 使用）。
pub fn annexb_to_length_prefixed(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for nalu in raw_nals_annexb(data) {
        out.extend_from_slice(&(nalu.len() as u32).to_be_bytes());
        out.extend_from_slice(nalu);
    }
    out
}

// ── extradata 构造 ──

/// 从 annex-B SPS/PPS 构造 AVCDecoderConfigurationRecord。
/// 已是 AVCC（首字节 0x01）时原样返回。
pub fn build_avcc_extradata(config_data: &[u8]) -> Vec<u8> {
    if config_data.is_empty() {
        return Vec::new();
    }
    if config_data[0] == 0x01 {
        return config_data.to_vec();
    }

    let units = nal_units_annexb(config_data);
    if units.is_empty() {
        return config_data.to_vec();
    }

    let mut sps_list: Vec<&[u8]> = Vec::new();
    let mut pps_list: Vec<&[u8]> = Vec::new();
    for &(offset, length) in &units {
        let nalu = strip_start_code(&config_data[offset..offset + length]);
        if nalu.is_empty() {
            continue;
        }
        match nalu[0] & 0x1F {
            7 => sps_list.push(nalu),
            8 => pps_list.push(nalu),
            _ => {}
        }
    }

    let Some(sps) = sps_list.first() else {
        return config_data.to_vec();
    };

    let mut avcc = Vec::new();
    avcc.push(0x01); // configurationVersion
    avcc.push(sps[1]); // AVCProfileIndication
    avcc.push(sps[2]); // profile_compatibility
    avcc.push(sps[3]); // AVCLevelIndication
    avcc.push(0xFF); // reserved(6) + lengthSizeMinusOne(3)
    avcc.push(0xE0 | (sps_list.len() as u8).min(0x1F)); // reserved(3) + num_sps(5)
    for sps_nal in &sps_list {
        let slen = sps_nal.len();
        avcc.push((slen >> 8) as u8);
        avcc.push(slen as u8);
        avcc.extend_from_slice(sps_nal);
    }
    avcc.push(pps_list.len() as u8);
    for pps_nal in &pps_list {
        let plen = pps_nal.len();
        avcc.push((plen >> 8) as u8);
        avcc.push(plen as u8);
        avcc.extend_from_slice(pps_nal);
    }
    avcc
}

/// 从 annex-B VPS/SPS/PPS 构造 HEVCDecoderConfigurationRecord（H.265）。
pub fn build_hvcc_extradata(config_data: &[u8]) -> Vec<u8> {
    if config_data.is_empty() {
        return Vec::new();
    }
    if config_data[0] == 0x01 && config_data.len() > 22 {
        return config_data.to_vec();
    }

    let units = nal_units_annexb(config_data);
    if units.is_empty() {
        return config_data.to_vec();
    }

    let mut vps_list: Vec<&[u8]> = Vec::new();
    let mut sps_list: Vec<&[u8]> = Vec::new();
    let mut pps_list: Vec<&[u8]> = Vec::new();
    for &(offset, length) in &units {
        let nalu = strip_start_code(&config_data[offset..offset + length]);
        if nalu.is_empty() {
            continue;
        }
        match (nalu[0] >> 1) & 0x3F {
            32 => vps_list.push(nalu),
            33 => sps_list.push(nalu),
            34 => pps_list.push(nalu),
            _ => {}
        }
    }

    let Some(sps) = sps_list.first() else {
        return config_data.to_vec();
    };

    let mut hvcc = Vec::new();
    hvcc.push(0x01); // configurationVersion
    hvcc.push(sps[0]); // general_profile_space + tier_flag + profile_idc
    hvcc.extend_from_slice(&sps[1..5]); // general_profile_compatibility_flags
    hvcc.extend_from_slice(&sps[5..11]); // general_constraint_indicator_flags + level
    hvcc.push(0xF0); // min_spatial_segmentation_idc
    hvcc.push(0x00);
    hvcc.push(0xFC); // parallelismType
    hvcc.push(0xFC | 0x03); // chroma_format_idc = 1 (4:2:0)
    hvcc.push(0xF8); // bit_depth_luma_minus8 = 0
    hvcc.push(0xF8); // bit_depth_chroma_minus8 = 0
    hvcc.push(0x00); // avg_frame_rate
    hvcc.push(0x00);
    hvcc.push(0x00); // constantFrameRate + numTemporalLayers + temporalIdNested
    hvcc.push(0x00);
    hvcc.push(0x00); // lengthSizeMinusOne + numOfArrays（回填）
    let len_size_plus_arrays_pos = hvcc.len() - 1;

    let mut arrays: Vec<(u8, &Vec<&[u8]>)> = Vec::new();
    if !vps_list.is_empty() {
        arrays.push((32, &vps_list));
    }
    arrays.push((33, &sps_list));
    arrays.push((34, &pps_list));

    let array_count = arrays.len() as u8;
    for (nal_type, nals) in &arrays {
        hvcc.push(((*nal_type & 0x3F) << 1) | 0x01);
        hvcc.push((nals.len() >> 8) as u8);
        hvcc.push(nals.len() as u8);
        for nal in *nals {
            hvcc.push((nal.len() >> 8) as u8);
            hvcc.push(nal.len() as u8);
            hvcc.extend_from_slice(nal);
        }
    }

    hvcc[len_size_plus_arrays_pos] = (3 << 1) | (array_count & 0x07);
    hvcc
}

/// 按 codec 构造对应 extradata。
pub fn build_extradata(codec: VideoCodec, config_data: &[u8]) -> Vec<u8> {
    match codec {
        VideoCodec::H264 => build_avcc_extradata(config_data),
        VideoCodec::H265 => build_hvcc_extradata(config_data),
        VideoCodec::Av1 => config_data.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 结构性假 NAL：nal_type 正确即可，不要求合法码流
    fn sps() -> Vec<u8> {
        vec![0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40, 0x50]
    }
    fn pps() -> Vec<u8> {
        vec![0x68, 0xeb, 0xec, 0xb2, 0x2c]
    }
    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn nal_splitting_3_and_4_byte_start_codes() {
        let mut data = vec![0, 0, 0, 1];
        data.extend_from_slice(&sps());
        data.extend_from_slice(&[0, 0, 1]);
        data.extend_from_slice(&pps());
        let units = nal_units_annexb(&data);
        assert_eq!(units.len(), 2);
        let (off, len) = units[0];
        assert_eq!(strip_start_code(&data[off..off + len]), sps().as_slice());
        let (off, len) = units[1];
        assert_eq!(strip_start_code(&data[off..off + len]), pps().as_slice());
    }

    #[test]
    fn avcc_record_structure() {
        let config = annexb(&[&sps(), &pps()]);
        let avcc = build_avcc_extradata(&config);
        assert_eq!(avcc[0], 0x01);
        assert_eq!(avcc[1], 0x64); // profile from SPS[1]
        assert_eq!(avcc[2], 0x00);
        assert_eq!(avcc[3], 0x1f); // level from SPS[3]
        assert_eq!(avcc[4], 0xFF); // lengthSizeMinusOne=3
        assert_eq!(avcc[5], 0xE1); // 1 个 SPS
        assert_eq!(&avcc[6..8], &8_u16.to_be_bytes());
        assert_eq!(&avcc[8..16], sps().as_slice());
        assert_eq!(avcc[16], 1); // 1 个 PPS
        assert_eq!(&avcc[17..19], &5_u16.to_be_bytes());
        assert_eq!(&avcc[19..24], pps().as_slice());
        assert_eq!(avcc.len(), 24);
    }

    #[test]
    fn avcc_passthrough_when_already_avcc() {
        let already = vec![0x01, 0x64, 0x00, 0x1f, 0xff, 0xe1, 0x00, 0x08];
        assert_eq!(build_avcc_extradata(&already), already);
    }

    #[test]
    fn hvcc_record_structure() {
        // H.265: VPS nal_type=32 → 首字节 (32<<1)=64, SPS=66, PPS=68
        let vps = vec![64_u8, 0x01, 0x02, 0x03];
        let hevc_sps = vec![66_u8, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xbc, 0xd0];
        let hevc_pps = vec![68_u8, 0x01, 0xc0];
        let config = annexb(&[&vps, &hevc_sps, &hevc_pps]);
        let hvcc = build_hvcc_extradata(&config);
        assert_eq!(hvcc[0], 0x01);
        // 定长头 23 字节，index 22 = lengthSizeMinusOne(=3)<<1 | numOfArrays(=3)
        assert_eq!(hvcc[22], (3 << 1) | 3);
        // 数组类型标记: (32<<1)|1=65, (33<<1)|1=67, (34<<1)|1=69
        assert!(hvcc.windows(1).any(|w| w[0] == 65));
        assert!(hvcc.windows(1).any(|w| w[0] == 67));
        assert!(hvcc.windows(1).any(|w| w[0] == 69));
    }

    #[test]
    fn length_prefix_conversion() {
        let config = annexb(&[&sps(), &pps()]);
        let converted = annexb_to_length_prefixed(&config);
        let expect_len = (sps().len() as u32).to_be_bytes();
        assert_eq!(&converted[0..4], &expect_len);
        assert_eq!(&converted[4..12], sps().as_slice());
        assert_eq!(&converted[12..16], &(pps().len() as u32).to_be_bytes());
        assert_eq!(&converted[16..], pps().as_slice());
    }

    #[test]
    fn rgba_frame_size_check() {
        assert!(rgba_frame(2, 2, 1, vec![0; 16]).is_ok());
        assert!(rgba_frame(2, 2, 1, vec![0; 15]).is_err());
        assert!(rgba_frame(0xffff_ffff, 0xffff_ffff, 1, vec![]).is_err());
    }
}
