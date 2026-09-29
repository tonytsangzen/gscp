//! Android 眼镜音频：opus 软解 JNI 出口。
//!
//! 与 PC 端 gscp-player 使用同一 Rust `opus` 解码器（`Decoder::new(48000, Stereo)` +
//! `decode(data, &mut pcm)`），绕开本机不可用的 MediaCodec `c2.android.opus.decoder`
//! （该组件对有效 opus 帧也会直接 self-release）。解码器全局单实例，由
//! Kotlin `AudioNative` 调用。

#![cfg(target_os = "android")]

use jni::objects::{JByteArray, JObject};
use jni::sys::{jboolean, jbyteArray, jint, JNI_TRUE};
use jni::JNIEnv;
use std::sync::Mutex;

pub const SAMPLE_RATE: u32 = 48000;
pub const CHANNELS: u32 = 2; // 与眼镜流 OpusHead 一致（stereo）

/// 全局 opus 解码器（本进程一个播放通路），启动态由 Kotlin start/stop 控制。
static DECODER: Mutex<Option<opus::Decoder>> = Mutex::new(None);

fn lock_decoder() -> std::sync::MutexGuard<'static, Option<opus::Decoder>> {
    DECODER.lock().unwrap_or_else(|e| e.into_inner())
}

#[no_mangle]
pub extern "system" fn Java_com_gscp_desktop_AudioNative_nativeOpusStart<'local>(
    mut _env: JNIEnv<'local>,
    _this: JObject<'local>,
) -> jboolean {
    let mut g = lock_decoder();
    if g.is_none() {
        *g = match opus::Decoder::new(SAMPLE_RATE, opus::Channels::Stereo) {
            Ok(d) => Some(d),
            Err(e) => {
                log::debug!("opus decoder create failed: {e:?}");
                None
            }
        };
    }
    if g.is_some() { JNI_TRUE } else { 0 }
}

#[no_mangle]
pub extern "system" fn Java_com_gscp_desktop_AudioNative_nativeOpusDecode<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
    data: jbyteArray,
    pcm: jbyteArray,
) -> jint {
    let input = match env.convert_byte_array(unsafe { JByteArray::from_raw(data) }) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    if input.is_empty() {
        return 0;
    }
    // 单帧最多：60ms 立体声 → 48000*60/1000*2 i16 = 5760 i16
    let mut out = vec![0i16; (SAMPLE_RATE as usize) * 120 / 1000 * CHANNELS as usize];
    let frame_size = {
        let mut g = lock_decoder();
        match g.as_mut() {
            Some(d) => match d.decode(&input, &mut out, false) {
                Ok(n) => n,
                Err(_) => 0,
            },
            None => 0,
        }
    };
    if frame_size <= 0 {
        return 0;
    }
    let n = frame_size * CHANNELS as usize; // 总采样数（立体声交织）
    let bytes = n * 2; // i16 → 2 字节
    if bytes > out.len() * 2 {
        return 0;
    }
    // i16 LE → i8
    let mut i8buf = Vec::with_capacity(bytes);
    for &s in &out[..n] {
        i8buf.push((s & 0xff) as i8);
        i8buf.push(((s >> 8) & 0xff) as i8);
    }
    let _ = env.set_byte_array_region(unsafe { JByteArray::from_raw(pcm) }, 0, &i8buf);
    bytes as jint
}

#[no_mangle]
pub extern "system" fn Java_com_gscp_desktop_AudioNative_nativeOpusStop<'local>(
    mut _env: JNIEnv<'local>,
    _this: JObject<'local>,
) {
    let mut g = lock_decoder();
    *g = None;
}