//! 双路帧缓冲：16 帧容量槽、低水位起播、EMA 实测帧间隔、
//! 缓冲深度自适应播放速率。移植自 mixplayer。

use gscp_core::codec::DecodedFrame;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const VIDEO_BUFFER_CAPACITY: usize = 16;
pub const VIDEO_BUFFER_LOW_WATER: usize = 4;
pub const VIDEO_PLAYBACK_RATE_MIN: f64 = 0.50;
pub const VIDEO_PLAYBACK_RATE_MAX: f64 = 1.50;
pub const OVERLAY_HIDE_AFTER_IDLE: Duration = Duration::from_secs(5);

pub struct FrameSlot {
    frame: Option<DecodedFrame>,
    pending_frames: VecDeque<DecodedFrame>,
    last_frame_at: Option<Instant>,
    /// EMA 实测帧间隔（秒），0.0 = 未测得。
    avg_frame_interval_secs: f64,
    bytes_received: u64,
    ended: bool,
    error: Option<String>,
    info: Option<String>,
    buffering: bool,
}

impl Default for FrameSlot {
    fn default() -> Self {
        Self {
            frame: None,
            pending_frames: VecDeque::new(),
            last_frame_at: None,
            avg_frame_interval_secs: 0.0,
            bytes_received: 0,
            ended: false,
            error: None,
            info: None,
            buffering: true,
        }
    }
}

#[derive(Clone, Default)]
pub struct SharedSlot {
    inner: Arc<Mutex<FrameSlot>>,
}

#[derive(Clone)]
pub struct SlotSnapshot {
    pub frame: Option<DecodedFrame>,
    pub last_frame_at: Option<Instant>,
    pub ended: bool,
    pub error: Option<String>,
    pub info: Option<String>,
    pub buffer_depth: usize,
    pub avg_frame_interval_secs: f64,
    pub bytes_received: u64,
}

impl SharedSlot {
    pub fn publish(&self, frame: DecodedFrame) {
        let mut guard = self.inner.lock().expect("slot poisoned");
        let now = Instant::now();
        if let Some(last) = guard.last_frame_at {
            let elapsed = now.duration_since(last).as_secs_f64();
            let alpha = 0.03; // EMA 平滑抗抖动
            if guard.avg_frame_interval_secs > 0.0 {
                guard.avg_frame_interval_secs =
                    guard.avg_frame_interval_secs * (1.0 - alpha) + elapsed * alpha;
            } else {
                guard.avg_frame_interval_secs = elapsed;
            }
        }
        if guard.pending_frames.len() >= VIDEO_BUFFER_CAPACITY {
            guard.pending_frames.pop_front();
        }
        guard.pending_frames.push_back(frame);
        guard.last_frame_at = Some(now);
    }

    pub fn set_info(&self, info: String) {
        self.inner.lock().expect("slot poisoned").info = Some(info);
    }

    pub fn set_error(&self, error: String) {
        self.inner.lock().expect("slot poisoned").error = Some(error);
    }

    pub fn finish(&self) {
        self.inner.lock().expect("slot poisoned").ended = true;
    }

    pub fn add_bytes_received(&self, n: u64) {
        self.inner.lock().expect("slot poisoned").bytes_received += n;
    }

    pub fn snapshot_for_render(&self) -> SlotSnapshot {
        let mut guard = self.inner.lock().expect("slot poisoned");
        if guard.buffering {
            if guard.pending_frames.len() >= VIDEO_BUFFER_LOW_WATER
                || (guard.ended && !guard.pending_frames.is_empty())
            {
                guard.buffering = false;
            }
        }
        if !guard.buffering {
            if let Some(frame) = guard.pending_frames.pop_front() {
                guard.frame = Some(frame);
            }
        }
        let buffer_depth = guard.pending_frames.len();
        SlotSnapshot {
            frame: guard.frame.clone(),
            last_frame_at: guard.last_frame_at,
            ended: guard.ended,
            error: guard.error.clone(),
            info: guard.info.clone(),
            buffer_depth,
            avg_frame_interval_secs: guard.avg_frame_interval_secs,
            bytes_received: guard.bytes_received,
        }
    }
}

/// 缓冲深度 → 播放速率（≤低水位放慢防欠载，接近满时加快消化积压）。
pub fn playback_rate_for_buffer_depth(buffer_depth: usize) -> f64 {
    let depth = buffer_depth as f64;
    let low = VIDEO_BUFFER_LOW_WATER as f64;
    let cap = VIDEO_BUFFER_CAPACITY as f64;

    if depth <= low {
        let t = depth / low;
        VIDEO_PLAYBACK_RATE_MIN + (1.0 - VIDEO_PLAYBACK_RATE_MIN) * t
    } else {
        let t = (depth - low) / (cap - low);
        1.0 + (VIDEO_PLAYBACK_RATE_MAX - 1.0) * t
    }
}

pub fn playback_interval_for_buffer_depth(
    frame_interval: Duration,
    buffer_depth: usize,
) -> Duration {
    frame_interval.div_f64(playback_rate_for_buffer_depth(buffer_depth).max(0.01))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(seq: u64) -> DecodedFrame {
        DecodedFrame {
            width: 2,
            height: 2,
            seq,
            pixels: Arc::<[u8]>::from(vec![0_u8; 16]),
        }
    }

    #[test]
    fn buffering_until_low_water() {
        let slot = SharedSlot::default();
        for seq in 0..(VIDEO_BUFFER_LOW_WATER - 1) as u64 {
            slot.publish(frame(seq));
        }
        let snap = slot.snapshot_for_render();
        // 未达低水位：不起播
        assert!(snap.frame.is_none());
        assert_eq!(snap.buffer_depth, VIDEO_BUFFER_LOW_WATER - 1);

        slot.publish(frame(99));
        let snap = slot.snapshot_for_render();
        assert!(snap.frame.is_some());
        assert_eq!(snap.buffer_depth, VIDEO_BUFFER_LOW_WATER - 1);
    }

    #[test]
    fn capacity_drops_oldest() {
        let slot = SharedSlot::default();
        for seq in 0..(VIDEO_BUFFER_CAPACITY * 2) as u64 {
            slot.publish(frame(seq));
        }
        let snap = slot.snapshot_for_render();
        assert!(snap.buffer_depth <= VIDEO_BUFFER_CAPACITY);
    }

    #[test]
    fn ema_interval_measured() {
        let slot = SharedSlot::default();
        slot.publish(frame(0));
        std::thread::sleep(Duration::from_millis(30));
        slot.publish(frame(1));
        std::thread::sleep(Duration::from_millis(30));
        slot.publish(frame(2));
        let snap = slot.snapshot_for_render();
        assert!(
            snap.avg_frame_interval_secs > 0.0 && snap.avg_frame_interval_secs < 0.5,
            "EMA 未测得合理帧间隔: {}",
            snap.avg_frame_interval_secs
        );
    }

    #[test]
    fn rate_curve_bounds() {
        let r0 = playback_rate_for_buffer_depth(0);
        assert!((r0 - VIDEO_PLAYBACK_RATE_MIN).abs() < 1e-9);
        let rlow = playback_rate_for_buffer_depth(VIDEO_BUFFER_LOW_WATER);
        assert!((rlow - 1.0).abs() < 1e-9);
        let rcap = playback_rate_for_buffer_depth(VIDEO_BUFFER_CAPACITY);
        assert!((rcap - VIDEO_PLAYBACK_RATE_MAX).abs() < 1e-9);
    }

    #[test]
    fn ended_flushes_pending() {
        let slot = SharedSlot::default();
        slot.publish(frame(0));
        slot.finish();
        let snap = slot.snapshot_for_render();
        assert!(snap.frame.is_some());
        assert!(snap.ended);
    }
}
