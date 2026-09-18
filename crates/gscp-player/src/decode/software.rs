//! OpenH264 软件解码（openh264 crate，默认 feature 从源码编译，无系统依赖）。
//!
//! OpenH264 不需要 extradata：config 包（SPS/PPS annex-B）作为普通 packet
//! 喂入即可完成初始化。

use anyhow::{Context, Result};
use gscp_core::codec::{DecodedFrame, VideoDecoder};
use gscp_core::logbus;
use gscp_core::scrcpy::VideoCodec;
use openh264::decoder::Decoder as H264Decoder;
use openh264::formats::YUVSource;

pub struct SoftwareDecoder {
    inner: Option<H264Decoder>,
    seq: u64,
}

impl SoftwareDecoder {
    pub fn new() -> Self {
        Self {
            inner: None,
            seq: 0,
        }
    }

    fn ensure_decoder(&mut self) -> Result<&mut H264Decoder> {
        if self.inner.is_none() {
            let decoder = H264Decoder::new().context("初始化 OpenH264 解码器失败")?;
            self.inner = Some(decoder);
        }
        Ok(self.inner.as_mut().expect("decoder just set"))
    }

    fn frame_from_yuv(seq: &mut u64, yuv: &openh264::decoder::DecodedYUV<'_>) -> Option<DecodedFrame> {
        let (w, h) = yuv.dimensions();
        if w == 0 || h == 0 {
            return None;
        }
        let rgba = yuv420_to_rgba(yuv.y(), yuv.u(), yuv.v(), yuv.strides(), w, h);
        *seq += 1;
        Some(DecodedFrame {
            width: w as u32,
            height: h as u32,
            seq: *seq,
            pixels: std::sync::Arc::<[u8]>::from(rgba),
        })
    }
}

/// YUV420（I420，独立 stride）→ RGBA（BT.601 全量程，色度最近邻上采样）。
pub fn yuv420_to_rgba(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    strides: (usize, usize, usize),
    width: usize,
    height: usize,
) -> Vec<u8> {
    let (y_stride, u_stride, v_stride) = strides;
    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..height {
        let y_row = &y[row * y_stride..];
        let chroma_row = row / 2;
        let u_row = &u[chroma_row * u_stride..];
        let v_row = &v[chroma_row * v_stride..];
        let dst = &mut rgba[row * width * 4..(row + 1) * width * 4];
        for col in 0..width {
            let yv = y_row[col] as f32;
            let cb = u_row[col / 2] as f32 - 128.0;
            let cr = v_row[col / 2] as f32 - 128.0;
            let r = yv + 1.402 * cr;
            let g = yv - 0.344_136 * cb - 0.714_136 * cr;
            let b = yv + 1.772 * cb;
            dst[col * 4] = clamp_u8(r);
            dst[col * 4 + 1] = clamp_u8(g);
            dst[col * 4 + 2] = clamp_u8(b);
            dst[col * 4 + 3] = 255;
        }
    }
    rgba
}

fn clamp_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

impl VideoDecoder for SoftwareDecoder {
    fn configure(
        &mut self,
        codec: VideoCodec,
        config_data: &[u8],
        _width: u32,
        _height: u32,
    ) -> Result<()> {
        if codec != VideoCodec::H264 {
            anyhow::bail!("软件解码当前仅支持 H264，远端下发的是 {}", codec.as_str());
        }
        self.ensure_decoder()?;
        // SPS/PPS 作为首个输入喂入，完成码流参数初始化
        let _ = self.decode(config_data)?;
        Ok(())
    }

    fn decode(&mut self, packet: &[u8]) -> Result<Option<DecodedFrame>> {
        let decoder = self.ensure_decoder()?;
        let yuv = match decoder.decode(packet) {
            Ok(yuv) => yuv,
            Err(err) => {
                logbus::emit(&format!("[decoder] OpenH264 解码错误（跳过包）: {err}"));
                return Ok(None);
            }
        };
        let Some(yuv) = yuv else {
            return Ok(None);
        };
        // DecodedYUV 借用解码器内部缓冲：拷出平面后立即解除借用
        let dims = yuv.dimensions();
        let strides = yuv.strides();
        if dims.0 == 0 || dims.1 == 0 {
            return Ok(None);
        }
        let planes = (yuv.y().to_vec(), yuv.u().to_vec(), yuv.v().to_vec());
        drop(yuv);
        self.seq += 1;
        let rgba = yuv420_to_rgba(&planes.0, &planes.1, &planes.2, strides, dims.0, dims.1);
        Ok(Some(DecodedFrame {
            width: dims.0 as u32,
            height: dims.1 as u32,
            seq: self.seq,
            pixels: std::sync::Arc::<[u8]>::from(rgba),
        }))
    }

    fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        let Some(decoder) = self.inner.as_mut() else {
            return Ok(vec![]);
        };
        let frames = decoder.flush_remaining().unwrap_or_default();
        let mut out = Vec::new();
        for yuv in &frames {
            if let Some(frame) = Self::frame_from_yuv(&mut self.seq, yuv) {
                out.push(frame);
            }
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "openh264"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv_conversion_solid_gray() {
        // 全 128 的 Y、全 128 的 UV → 中性灰（约 128）
        let w = 4;
        let h = 4;
        let y = vec![128_u8; w * h];
        let u = vec![128_u8; w * h / 4];
        let v = vec![128_u8; w * h / 4];
        let rgba = yuv420_to_rgba(&y, &u, &v, (w, w / 2, w / 2), w, h);
        for px in rgba.chunks_exact(4) {
            assert!((px[0] as i32 - 128).abs() <= 2);
            assert!((px[1] as i32 - 128).abs() <= 2);
            assert!((px[2] as i32 - 128).abs() <= 2);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn yuv_conversion_respects_stride() {
        // stride 大于宽度时仍能正确取行
        let w = 2;
        let h = 2;
        let stride = 8;
        let y = vec![255_u8; h * stride]; // Y=255 → 白
        let u = vec![128_u8; h * stride];
        let v = vec![128_u8; h * stride];
        let rgba = yuv420_to_rgba(&y, &u, &v, (stride, stride, stride), w, h);
        for px in rgba.chunks_exact(4) {
            assert!(px[0] >= 250 && px[1] >= 250 && px[2] >= 250);
        }
    }
}
