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

/// 色度平面布局：I420 为独立 U/V 平面，NV12 为交错 UV 平面。
pub enum ChromaPlanes<'a> {
    Planar {
        u: &'a [u8],
        v: &'a [u8],
        u_stride: usize,
        v_stride: usize,
    },
    Interleaved {
        uv: &'a [u8],
        uv_stride: usize,
    },
}

/// YUV420 → RGBA，色度布局由 `ChromaPlanes` 决定。
///
/// 口径与 macOS VideoToolbox 硬件输出对齐：BT.601 limited range（视频范围
/// 16-235 扩展为全量程）+ 色度双线性上采样。Android 编码器输出的 H.264 是
/// limited range，按全量程直解会导致 Windows/Linux 黑位抬升、对比度/饱和度
/// 偏低，渲染效果与 macOS 不一致。
fn yuv_to_rgba_impl(
    y: &[u8],
    y_stride: usize,
    chroma: ChromaPlanes<'_>,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let (u_stride, v_stride) = match &chroma {
        ChromaPlanes::Planar { u_stride, v_stride, .. } => (*u_stride, *v_stride),
        ChromaPlanes::Interleaved { uv_stride, .. } => (*uv_stride, *uv_stride),
    };
    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..height {
        let y_row = &y[row * y_stride..];
        let dst = &mut rgba[row * width * 4..(row + 1) * width * 4];
        // 像素中心在色度平面上的连续坐标，clamp-to-edge 双线性
        let fy = ((row as f32 + 0.5) * 0.5 - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(ch - 1);
        let y1 = (y0 + 1).min(ch - 1);
        let ty = fy - y0 as f32;
        let u_row0 = match &chroma {
            ChromaPlanes::Planar { u, .. } => &u[y0 * u_stride..],
            ChromaPlanes::Interleaved { uv, .. } => &uv[y0 * u_stride..],
        };
        let u_row1 = match &chroma {
            ChromaPlanes::Planar { u, .. } => &u[y1 * u_stride..],
            ChromaPlanes::Interleaved { uv, .. } => &uv[y1 * u_stride..],
        };
        let v_row0 = match &chroma {
            ChromaPlanes::Planar { v, .. } => &v[y0 * v_stride..],
            ChromaPlanes::Interleaved { .. } => u_row0, // NV12 中与 U 同行，偏移由列索引处理
        };
        let v_row1 = match &chroma {
            ChromaPlanes::Planar { v, .. } => &v[y1 * v_stride..],
            ChromaPlanes::Interleaved { .. } => u_row1,
        };
        for col in 0..width {
            let yv = (y_row[col] as f32 - 16.0) * (255.0 / 219.0);
            let fx = ((col as f32 + 0.5) * 0.5 - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(cw - 1);
            let x1 = (x0 + 1).min(cw - 1);
            let tx = fx - x0 as f32;
            // NV12：UV 交错，U 在偶数列、V 在奇数列（列索引 ×2）
            let (u_off, v_off) = match &chroma {
                ChromaPlanes::Planar { .. } => (x0, x0),
                ChromaPlanes::Interleaved { .. } => (x0 * 2, x0 * 2 + 1),
            };
            let (u_off1, v_off1) = match &chroma {
                ChromaPlanes::Planar { .. } => (x1, x1),
                ChromaPlanes::Interleaved { .. } => (x1 * 2, x1 * 2 + 1),
            };
            let cb = lerp(
                lerp(u_row0[u_off] as f32, u_row0[u_off1] as f32, tx),
                lerp(u_row1[u_off] as f32, u_row1[u_off1] as f32, tx),
                ty,
            ) - 128.0;
            let cr = lerp(
                lerp(v_row0[v_off] as f32, v_row0[v_off1] as f32, tx),
                lerp(v_row1[v_off] as f32, v_row1[v_off1] as f32, tx),
                ty,
            ) - 128.0;
            let r = yv + 1.596 * cr;
            let g = yv - 0.392 * cb - 0.813 * cr;
            let b = yv + 2.017 * cb;
            dst[col * 4] = clamp_u8(r);
            dst[col * 4 + 1] = clamp_u8(g);
            dst[col * 4 + 2] = clamp_u8(b);
            dst[col * 4 + 3] = 255;
        }
    }
    rgba
}

/// YUV420（I420，独立 stride）→ RGBA。见 [`yuv_to_rgba_impl`] 的色彩口径说明。
pub fn yuv420_to_rgba(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    strides: (usize, usize, usize),
    width: usize,
    height: usize,
) -> Vec<u8> {
    let (y_stride, u_stride, v_stride) = strides;
    yuv_to_rgba_impl(
        y,
        y_stride,
        ChromaPlanes::Planar { u, v, u_stride, v_stride },
        width,
        height,
    )
}

/// NV12（Y 平面 + 交错 UV 平面，各自 stride）→ RGBA。
/// Media Foundation MFT 输出格式，色彩口径与 [`yuv420_to_rgba`] 一致。
pub fn nv12_to_rgba(
    y: &[u8],
    uv: &[u8],
    strides: (usize, usize),
    width: usize,
    height: usize,
) -> Vec<u8> {
    let (y_stride, uv_stride) = strides;
    yuv_to_rgba_impl(
        y,
        y_stride,
        ChromaPlanes::Interleaved { uv, uv_stride },
        width,
        height,
    )
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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
        // 全 128 的 Y、全 128 的 UV → 中性灰。
        // limited range 下 Y=128 扩展为 1.164×(128−16) ≈ 130。
        let w = 4;
        let h = 4;
        let y = vec![128_u8; w * h];
        let u = vec![128_u8; w * h / 4];
        let v = vec![128_u8; w * h / 4];
        let rgba = yuv420_to_rgba(&y, &u, &v, (w, w / 2, w / 2), w, h);
        for px in rgba.chunks_exact(4) {
            assert!((px[0] as i32 - 130).abs() <= 1);
            assert!((px[1] as i32 - 130).abs() <= 1);
            assert!((px[2] as i32 - 130).abs() <= 1);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn yuv_limited_range_expands_blacks_and_whites() {
        // Y=16 → 纯黑，Y=235 → 纯白（视频范围边界正确扩展）
        let w = 2;
        let h = 2;
        for (yv, lo, hi) in [(16_u8, 0_u8, 2_u8), (235_u8, 253_u8, 255_u8)] {
            let y = vec![yv; w * h];
            let u = vec![128_u8; w * h / 4];
            let v = vec![128_u8; w * h / 4];
            let rgba = yuv420_to_rgba(&y, &u, &v, (w, w / 2, w / 2), w, h);
            for px in rgba.chunks_exact(4) {
                assert!(px[0] >= lo && px[0] <= hi);
                assert!(px[1] >= lo && px[1] <= hi);
                assert!(px[2] >= lo && px[2] <= hi);
            }
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
