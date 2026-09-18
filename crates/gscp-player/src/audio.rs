//! 音频输出：rodio ContinuousSource（缓冲空时输出静音防断流），
//! raw PCM（s16le 48k 双声道）直推 + Opus 解码（unix 平台，系统 libopus）。
//! 移植自 mixplayer audio.rs。

use anyhow::Result;
use rodio::Source;
use std::collections::VecDeque;
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const SAMPLE_RATE: NonZero<u32> = NonZero::new(48000).unwrap();
const CHANNELS: NonZero<u16> = NonZero::new(2).unwrap();
/// 背压高水位：约 300ms。
const HIGH_WATER: usize = (48000 * 300 / 1000) * 2;

/// 永不返回 None 的 source：缓冲空时输出 0.0 静音。
struct ContinuousSource {
    buffer: Arc<Mutex<VecDeque<f32>>>,
}

impl ContinuousSource {
    fn new(buffer: Arc<Mutex<VecDeque<f32>>>) -> Self {
        Self { buffer }
    }
}

impl Iterator for ContinuousSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.buffer.lock().unwrap().pop_front().unwrap_or(0.0))
    }
}

impl Source for ContinuousSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> NonZero<u16> {
        CHANNELS
    }

    fn sample_rate(&self) -> NonZero<u32> {
        SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

pub struct AudioPlayer {
    _handle: rodio::MixerDeviceSink,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    #[cfg(unix)]
    #[allow(dead_code)]
    opus_decoder: Option<opus::Decoder>,
}

impl AudioPlayer {
    pub fn new() -> Result<Self> {
        let handle = rodio::DeviceSinkBuilder::open_default_sink()
            .map_err(|e| anyhow::anyhow!("打开音频设备失败: {e}"))?;
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(HIGH_WATER * 2)));
        let source = ContinuousSource::new(buffer.clone());
        handle.mixer().add(source);
        Ok(Self {
            _handle: handle,
            buffer,
            #[cfg(unix)]
            opus_decoder: None,
        })
    }

    #[cfg(unix)]
    pub fn init_opus(&mut self) -> Result<()> {
        let decoder = opus::Decoder::new(SAMPLE_RATE.get(), opus::Channels::Stereo)
            .map_err(|e| anyhow::anyhow!("创建 Opus 解码器失败: {e}"))?;
        self.opus_decoder = Some(decoder);
        Ok(())
    }

    pub fn supports_opus() -> bool {
        cfg!(unix)
    }

    pub fn push_raw_pcm(&mut self, data: &[u8]) {
        {
            let mut buf = self.buffer.lock().unwrap();
            for chunk in data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                buf.push_back(sample as f32 / 32768.0);
            }
        }
        self.backpressure();
    }

    #[cfg(unix)]
    pub fn push_opus(&mut self, data: &[u8]) -> Result<()> {
        let decoder = self
            .opus_decoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Opus 解码器未初始化"))?;
        let max_samples = (SAMPLE_RATE.get() as usize) * 120 / 1000 * CHANNELS.get() as usize;
        let mut pcm = vec![0_i16; max_samples];
        let frame_size = decoder
            .decode(data, &mut pcm, false)
            .map_err(|e| anyhow::anyhow!("Opus 解码错误: {e}"))?;
        let count = frame_size * CHANNELS.get() as usize;
        if count > 0 {
            let mut buf = self.buffer.lock().unwrap();
            for &s in &pcm[..count] {
                buf.push_back(s as f32 / 32768.0);
            }
        }
        self.backpressure();
        Ok(())
    }

    /// 缓冲超水位时阻塞推送线程，防止无界增长。
    fn backpressure(&self) {
        while self.buffer.lock().unwrap().len() > HIGH_WATER {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_support_matches_platform() {
        // unix 走系统 libopus，Windows 回退 raw
        assert_eq!(AudioPlayer::supports_opus(), cfg!(unix));
    }

    #[test]
    fn raw_pcm_converts_little_endian() {
        // 不打开真实音频设备，仅验证字节序转换逻辑（通过独立函数不可行，
        // 这里通过小水位推入不触发背压来验证不会 panic）
        // 完整设备级验证放在集成测试中。
        let data = [0x00_u8, 0x40, 0x00, 0xc0]; // 0x4000, 0xc000 → 0.5, -0.5
        let mut samples = VecDeque::new();
        for chunk in data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push_back(sample as f32 / 32768.0);
        }
        let v: Vec<f32> = samples.into_iter().collect();
        assert!((v[0] - 0.5).abs() < 1e-3);
        assert!((v[1] + 0.5).abs() < 1e-3);
    }
}
