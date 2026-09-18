//! 三路流会话：camera/overlay 视频源线程 + audio 音频源线程。
//! 连接、协议解析（frame meta / config 包）与重连循环移植自 mixplayer。

use crate::audio::AudioPlayer;
use crate::decode;
use crate::slot::SharedSlot;
use gscp_core::bootstrap::{self, PreparedAudioStream, PreparedVideoStream, RemoteBootstrapManager};
use gscp_core::codec::{DecodedFrame, rgba_bytes};
use gscp_core::logbus;
use gscp_core::scrcpy::{self, AudioCodec, AudioSource, BufferedStream, VideoSource};
use gscp_core::settings::DecoderBackend;
use anyhow::Context;
use std::io::ErrorKind;
use std::sync::Arc;
use std::time::{Duration, Instant};

const RECONNECT_DELAY: Duration = bootstrap::RECONNECT_DELAY;
const RESTART_RETRY_DELAY: Duration = bootstrap::RESTART_RETRY_DELAY;
const VIDEO_DIAG_HEADER_BLOCK_LOG_THRESHOLD: Duration = Duration::from_millis(50);
const VIDEO_DIAG_BODY_BLOCK_LOG_THRESHOLD: Duration = Duration::from_millis(50);
const VIDEO_DIAG_PACKET_GAP_LOG_THRESHOLD: Duration = Duration::from_millis(80);
/// 解码器无帧输出判定：码流在到达但持续解不出帧，视为解码器失效。
const FRAME_STALL_SECS: u64 = 30;

fn is_bootstrap_generation_changed_error(err: &anyhow::Error) -> bool {
    bootstrap::is_bootstrap_generation_changed_error(err)
}

// ── 视频源 ──

pub fn spawn_video_source(
    role: &'static str,
    source: VideoSource,
    slot: SharedSlot,
    backend: DecoderBackend,
    bootstrap: Option<Arc<RemoteBootstrapManager>>,
) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name(format!("{role}-source"))
        .spawn(move || {
            let result =
                run_video_source_loop(role, &source, &slot, backend, bootstrap.as_deref());
            if let Err(err) = result {
                logbus::emit(&format!("[{role}] {err:#}"));
                slot.set_error(format!("{err:#}"));
            }
            slot.finish();
        })
        .map(|_| ())
        .context("创建视频源线程失败")
}

fn run_video_source_loop(
    role: &str,
    source: &VideoSource,
    slot: &SharedSlot,
    backend: DecoderBackend,
    bootstrap_mgr: Option<&RemoteBootstrapManager>,
) -> anyhow::Result<()> {
    let mut bootstrap_generation = bootstrap_mgr.map(|m| m.wait_until_ready());
    loop {
        let result = run_video_source_session(
            role,
            source,
            slot,
            backend,
            bootstrap_generation.unwrap_or(0),
            bootstrap_mgr,
        );
        match result {
            Ok(()) => {
                logbus::emit(&format!(
                    "[{role}] socket 已断开，{} 秒后尝试重连",
                    RECONNECT_DELAY.as_secs()
                ));
            }
            Err(err) => {
                if let Some(manager) = bootstrap_mgr {
                    if is_bootstrap_generation_changed_error(&err) {
                        bootstrap_generation = Some(manager.wait_until_ready());
                        continue;
                    }
                }
                logbus::emit(&format!(
                    "[{role}] socket 异常，{} 秒后尝试重连: {err:#}",
                    RECONNECT_DELAY.as_secs()
                ));
            }
        }
        if let Some(manager) = bootstrap_mgr {
            match manager.execute_restart_plan(manager.schedule_restart(
                bootstrap_generation.unwrap_or(0),
            )) {
                Ok(next_generation) => bootstrap_generation = Some(next_generation),
                Err(restart_err) => {
                    logbus::emit(&format!(
                        "[{role}] 远端 bootstrap 重建失败，{} 秒后重试: {restart_err:#}",
                        RESTART_RETRY_DELAY.as_secs()
                    ));
                    std::thread::sleep(RESTART_RETRY_DELAY);
                }
            }
        } else {
            std::thread::sleep(RECONNECT_DELAY);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_video_source_session(
    role: &str,
    source: &VideoSource,
    slot: &SharedSlot,
    backend: DecoderBackend,
    bootstrap_generation: u64,
    bootstrap_mgr: Option<&RemoteBootstrapManager>,
) -> anyhow::Result<()> {
    if source.debug {
        logbus::emit(&format!(
            "[{role}] connect={} raw_stream={} dummy={} device_meta={} codec_meta={} frame_meta={}",
            source.address(),
            source.raw_stream,
            source.send_dummy_byte,
            source.send_device_meta,
            source.send_codec_meta,
            source.send_frame_meta
        ));
    }
    let address = source.address();
    let prepared = if let Some(manager) = bootstrap_mgr {
        manager.acquire_video_stream(role, &address, bootstrap_generation, source.debug)?
    } else {
        bootstrap::prepare_direct_video_stream(source)?
    };
    let PreparedVideoStream {
        mut stream,
        device_name,
        codec,
        size,
    } = prepared;

    let info = format!(
        "{} {}x{}{}",
        codec.as_str(),
        size.width,
        size.height,
        device_name
            .as_ref()
            .map(|name| format!(" {name}"))
            .unwrap_or_default()
    );
    slot.set_info(info);

    let mut decoder = decode::create_decoder(backend)?;
    let mut decoder_open = false;
    let mut seq = 0_u64;
    let mut packet_index = 0_u64;
    let mut last_packet_received_at: Option<Instant> = None;
    let expected_rgba = rgba_bytes(size.width, size.height)?;

    // 卡死保护：静默超过阈值即判定连接失效并自动重连
    let stall = source.stall_timeout.unwrap_or_else(scrcpy::stream_stall_timeout);
    scrcpy::enable_stream_mode(stream.stream_ref(), stall)?;
    logbus::emit(&format!(
        "[{role}] 流模式已就绪（静默 {}s 未收到数据将自动重连）",
        stall.as_secs()
    ));
    let frame_stall = std::time::Duration::from_secs(FRAME_STALL_SECS);
    let mut last_frame_at = Instant::now();

    if !source.send_frame_meta {
        anyhow::bail!("raw_stream 模式暂不支持，请使用默认的 send_frame_meta=true 模式");
    }

    loop {
        let mut packet_header = [0_u8; 12];
        if last_frame_at.elapsed() >= frame_stall {
            anyhow::bail!(
                "已 {}s 未解出新帧（码流仍在到达），解码器可能已失效，触发重连",
                frame_stall.as_secs()
            );
        }
        let header_wait = match read_exact_timed(&mut stream, &mut packet_header) {
            Ok(wait) => wait,
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) if scrcpy::is_stall_error(&err) => {
                logbus::emit(&format!(
                    "[{role}] 数据静默 {}s，连接可能已中断，准备重连",
                    stall.as_secs()
                ));
                break;
            }
            Err(err) => return Err(err).context("读取 scrcpy frame meta 失败"),
        };

        let (pts_and_flags, packet_size) = scrcpy::parse_frame_header(&packet_header);
        let mut packet = vec![0_u8; packet_size];
        let body_wait = match read_exact_timed(&mut stream, &mut packet) {
            Ok(wait) => wait,
            Err(err) if scrcpy::is_stall_error(&err) => {
                logbus::emit(&format!(
                    "[{role}] 包体读取静默 {}s，连接可能已中断，准备重连",
                    stall.as_secs()
                ));
                break;
            }
            Err(err) => return Err(err).context("读取 scrcpy packet body 失败"),
        };
        let packet_received_at = Instant::now();
        let packet_gap =
            last_packet_received_at.map(|at| packet_received_at.duration_since(at));
        last_packet_received_at = Some(packet_received_at);

        slot.add_bytes_received(12 + packet_size as u64);

        let is_config = scrcpy::is_config_packet(pts_and_flags);
        packet_index += 1;
        if source.debug
            && (header_wait >= VIDEO_DIAG_HEADER_BLOCK_LOG_THRESHOLD
                || body_wait >= VIDEO_DIAG_BODY_BLOCK_LOG_THRESHOLD
                || packet_gap.is_some_and(|gap| gap >= VIDEO_DIAG_PACKET_GAP_LOG_THRESHOLD))
        {
            logbus::emit(&format!(
                "[{role}] video packet #{packet_index} wait: header={}ms body={}ms gap={} size={} type={}",
                header_wait.as_millis(),
                body_wait.as_millis(),
                packet_gap
                    .map(|gap| gap.as_millis().to_string())
                    .unwrap_or_else(|| "-".to_string()),
                packet_size,
                if is_config { "config" } else { "data" }
            ));
        }

        if is_config {
            if !decoder_open {
                decoder
                    .configure(codec, &packet, size.width, size.height)
                    .context("初始化视频解码器失败")?;
                decoder_open = true;
                logbus::emit(&format!(
                    "[{role}] 解码器 {} 已就绪 (config {} bytes)",
                    decoder.name(),
                    packet.len()
                ));
            }
            // config 包继续喂给解码器，支持中途重配置
            let _ = decoder.decode(&packet);
            continue;
        }

        if !decoder_open {
            // 尚未收到 config，丢弃普通包
            continue;
        }

        if let Some(frame) = decoder.decode(&packet)? {
            // 尺寸不匹配（如设备旋转）的帧丢弃
            if frame.width == size.width
                && frame.height == size.height
                && frame.pixels.len() == expected_rgba
            {
                seq += 1;
                last_frame_at = Instant::now();
                slot.publish(DecodedFrame {
                    width: size.width,
                    height: size.height,
                    seq,
                    pixels: frame.pixels,
                });
            }
        }
    }

    // 会话结束，排空解码器残留帧
    if decoder_open {
        for frame in decoder.flush()? {
            if frame.width == size.width && frame.height == size.height {
                seq += 1;
                slot.publish(DecodedFrame {
                    width: size.width,
                    height: size.height,
                    seq,
                    pixels: frame.pixels,
                });
            }
        }
    }
    Ok(())
}

fn read_exact_timed(
    stream: &mut BufferedStream,
    buf: &mut [u8],
) -> std::io::Result<Duration> {
    let started_at = Instant::now();
    stream.read_exact_buffered(buf)?;
    Ok(started_at.elapsed())
}

// ── 音频源 ──

pub fn spawn_audio_source(
    role: &'static str,
    source: AudioSource,
    bootstrap: Option<Arc<RemoteBootstrapManager>>,
) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name(format!("{role}-source"))
        .spawn(move || {
            if let Err(err) =
                run_audio_source_loop(role, &source, bootstrap.as_deref())
            {
                logbus::emit(&format!("[{role}] {err:#}"));
            }
        })
        .map(|_| ())
        .context("创建音频源线程失败")
}

fn run_audio_source_loop(
    role: &str,
    source: &AudioSource,
    bootstrap_mgr: Option<&RemoteBootstrapManager>,
) -> anyhow::Result<()> {
    let mut bootstrap_generation = bootstrap_mgr.map(|m| m.wait_until_ready());
    loop {
        let result = run_audio_source_session(
            role,
            source,
            bootstrap_generation.unwrap_or(0),
            bootstrap_mgr,
        );
        match result {
            Ok(()) => {
                logbus::emit(&format!(
                    "[{role}] audio socket 已断开，{} 秒后尝试重连",
                    RECONNECT_DELAY.as_secs()
                ));
            }
            Err(err) => {
                if let Some(manager) = bootstrap_mgr {
                    if is_bootstrap_generation_changed_error(&err) {
                        bootstrap_generation = Some(manager.wait_until_ready());
                        continue;
                    }
                }
                logbus::emit(&format!(
                    "[{role}] audio socket 异常，{} 秒后尝试重连: {err:#}",
                    RECONNECT_DELAY.as_secs()
                ));
            }
        }
        if let Some(manager) = bootstrap_mgr {
            match manager.execute_restart_plan(manager.schedule_restart(
                bootstrap_generation.unwrap_or(0),
            )) {
                Ok(next_generation) => bootstrap_generation = Some(next_generation),
                Err(restart_err) => {
                    logbus::emit(&format!(
                        "[{role}] 远端 bootstrap 重建失败，{} 秒后重试: {restart_err:#}",
                        RESTART_RETRY_DELAY.as_secs()
                    ));
                    std::thread::sleep(RESTART_RETRY_DELAY);
                }
            }
        } else {
            std::thread::sleep(RECONNECT_DELAY);
        }
    }
}

fn run_audio_source_session(
    role: &str,
    source: &AudioSource,
    bootstrap_generation: u64,
    bootstrap_mgr: Option<&RemoteBootstrapManager>,
) -> anyhow::Result<()> {
    let address = source.address();
    let prepared = if let Some(manager) = bootstrap_mgr {
        manager.acquire_audio_stream(&address, bootstrap_generation, source.debug)?
    } else {
        bootstrap::prepare_direct_audio_stream(source)?
    };
    let PreparedAudioStream {
        mut stream,
        codec,
        device_name,
    } = prepared;
    if source.debug {
        logbus::emit(&format!(
            "[{role}] codec={} device_meta={}",
            codec.as_str(),
            device_name.as_deref().unwrap_or("<empty>")
        ));
    }

    let mut player = AudioPlayer::new().context("创建音频播放器失败")?;
    match codec {
        AudioCodec::Opus => {
            if !AudioPlayer::supports_opus() {
                anyhow::bail!(
                    "当前平台不支持 Opus 音频，请使用 audio_codec=raw（管理端会自动回退）"
                );
            }
            player.init_opus()?;
        }
        AudioCodec::Raw => {}
        AudioCodec::Aac | AudioCodec::Flac => {
            anyhow::bail!("音频 codec {} 暂不支持，当前仅支持 raw/opus", codec.as_str());
        }
    }

    let stall = source.stall_timeout.unwrap_or_else(scrcpy::stream_stall_timeout);
    scrcpy::enable_stream_mode(stream.stream_ref(), stall)?;

    if source.send_frame_meta {
        loop {
            let mut packet_header = [0_u8; 12];
            match stream.read_exact_buffered(&mut packet_header) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) if scrcpy::is_stall_error(&err) => {
                    logbus::emit(&format!(
                        "[{role}] 音频数据静默 {}s，连接可能已中断，准备重连",
                        stall.as_secs()
                    ));
                    break;
                }
                Err(err) => return Err(err).context("读取 audio frame meta 失败"),
            }

            let (_pts_and_flags, packet_size) = scrcpy::parse_frame_header(&packet_header);
            let mut packet = vec![0_u8; packet_size];
            match stream.read_exact_buffered(&mut packet) {
                Ok(()) => {}
                Err(err) if scrcpy::is_stall_error(&err) => {
                    logbus::emit(&format!(
                        "[{role}] 音频包体静默 {}s，连接可能已中断，准备重连",
                        stall.as_secs()
                    ));
                    break;
                }
                Err(err) => return Err(err).context("读取 audio packet body 失败"),
            }

            match codec {
                AudioCodec::Raw => player.push_raw_pcm(&packet),
                AudioCodec::Opus => {
                    if let Err(e) = player.push_opus(&packet) {
                        if source.debug {
                            logbus::emit(&format!("[{role}] Opus 解码失败: {e}"));
                        }
                    }
                }
                AudioCodec::Aac | AudioCodec::Flac => unreachable!(),
            }
        }
    } else {
        // raw stream 直通
        let mut buf = vec![0_u8; 65536];
        loop {
            let n = match stream.read_buffered(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) if scrcpy::is_stall_error(&err) => {
                    logbus::emit(&format!(
                        "[{role}] 音频数据静默 {}s，连接可能已中断，准备重连",
                        stall.as_secs()
                    ));
                    break;
                }
                Err(err) => return Err(err).context("读取 audio socket 数据失败"),
            };
            player.push_raw_pcm(&buf[..n]);
        }
    }
    Ok(())
}
