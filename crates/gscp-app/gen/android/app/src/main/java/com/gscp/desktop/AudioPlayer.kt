package com.gscp.desktop

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * 眼镜端音频播放：opus 帧 —— Rust(`gscp-app::opus_jni`)软解 —— PCM —— AudioTrack。
 * （本机 MediaCodec `c2.android.opus.decoder` 组件不可用，故与 PC gscp-player 同源走软件解码。）
 */
class AudioPlayer {

    companion object {
        private const val TAG = "ar-audio"
        private const val SAMPLE_RATE = 48000
        // 单帧最多 60ms → 48000*0.06*2ch*2B = 23040 字节
        private const val PCM_BUF = 23040
    }

    private var audioTrack: AudioTrack? = null
    private var started = false

    /** 用眼镜流首帧 OpusHead(csd-0) 配置（无需内容，Rust 解码器按 48k 立体声建）。 */
    @Synchronized
    fun start(csd0: ByteArray) {
        if (started) return
        try {
            val channelCount = if (csd0.size > 9) csd0[9].toInt() and 0xFF else 2
            var sampleRate = 48000
            if (csd0.size >= 16) {
                sampleRate = (csd0[12].toInt() and 0xFF) or
                    ((csd0[13].toInt() and 0xFF) shl 8) or
                    ((csd0[14].toInt() and 0xFF) shl 16) or
                    ((csd0[15].toInt() and 0xFF) shl 24)
            }
            val ok = AudioNative.nativeOpusStart()
            Log.i(TAG, "opus native start=$ok ch=$channelCount sr=$sampleRate")
            val minBuf = AudioTrack.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT)
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build())
                .setBufferSizeInBytes(minOf(maxOf(minBuf, PCM_BUF), PCM_BUF * 2))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
            audioTrack = track
            track.play()
            started = true
            Log.i(TAG, "audio player started ch=$channelCount sr=$sampleRate")
        } catch (e: Exception) {
            started = false
            Log.w(TAG, "audio start fail", e)
        }
    }

    @Synchronized
    fun play(buffer: ByteArray, offset: Int, length: Int) {
        if (!started) { Log.w(TAG, "audio play before config, drop ${length}b"); return }
        try {
            val pcmBuf = ByteArray(PCM_BUF)
            val frame = if (offset == 0 && length == buffer.size) buffer
                else buffer.copyOfRange(offset, offset + length)
            val n = AudioNative.nativeOpusDecode(frame, pcmBuf)
            if (n > 0) {
                val track = audioTrack
                if (track != null) track.write(pcmBuf, 0, n)
            }
        } catch (e: Exception) {
            Log.w(TAG, "audio play error: ${e.message}")
        }
    }

    @Synchronized
    fun stop() {
        started = false
        try { AudioNative.nativeOpusStop() } catch (_: Exception) {}
        try { audioTrack?.release() } catch (_: Exception) {}
        audioTrack = null
    }
}