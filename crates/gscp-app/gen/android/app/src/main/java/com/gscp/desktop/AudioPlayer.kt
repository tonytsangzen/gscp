package com.gscp.desktop

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaFormat
import java.nio.ByteBuffer

/** scrcpy 音频（Opus）解码 + AudioTrack 播放。 */
class AudioPlayer(val rate: Int, val channel: Int, val format: Int) {
    private val audioTrack: AudioTrack
    private val minBufferSize: Int = AudioTrack.getMinBufferSize(rate, channel, format)
    private val mediaCodec: MediaCodec

    init {
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(format)
                    .setSampleRate(rate)
                    .setChannelMask(channel)
                    .build()
            )
            .setBufferSizeInBytes(minBufferSize * 2) // 双倍缓冲
            .build()
        mediaCodec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_AUDIO_OPUS)
    }

    @Synchronized
    fun start() {
        val mediaFormat = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_OPUS, rate, channel)
        mediaFormat.setInteger(MediaFormat.KEY_BIT_RATE, 102000)
        // Opus Identification Header：单声道降混由 Mapping Family 0 处理
        val csd0bytes = byteArrayOf(
            0x4f, 0x70, 0x75, 0x73,  // "OpusHead"
            0x48, 0x65, 0x61, 0x64,
            0x01,  // Version
            0x02,  // Channel Count
            0x38, 0x01,  // Pre-skip
            0x80.toByte(), 0xbb.toByte(), 0x00, 0x00,  // Input Sample Rate 48000
            0x00, 0x00,  // Output Gain (Q7.8)
            0x00,  // Mapping Family
            0x00
        )
        val csd1bytes = ByteArray(8)
        val csd2bytes = ByteArray(8)
        mediaFormat.setByteBuffer("csd-0", ByteBuffer.wrap(csd0bytes))
        mediaFormat.setByteBuffer("csd-1", ByteBuffer.wrap(csd1bytes))
        mediaFormat.setByteBuffer("csd-2", ByteBuffer.wrap(csd2bytes))

        mediaCodec.reset()
        mediaCodec.configure(mediaFormat, null, null, 0)
        mediaCodec.start()
        audioTrack.play()
    }

    @Synchronized
    fun stop() {
        try {
            audioTrack.stop()
            mediaCodec.stop()
        } catch (_: Exception) {
        }
    }

    @Synchronized
    fun play(buffer: ByteArray, offset: Int, length: Int) {
        val decodeBufferInfo = MediaCodec.BufferInfo()
        try {
            val inputBufferId = mediaCodec.dequeueInputBuffer(-1)
            if (inputBufferId >= 0) {
                val inputBuffer = mediaCodec.getInputBuffer(inputBufferId)
                inputBuffer!!.clear()
                inputBuffer.put(buffer, offset, length)
                mediaCodec.queueInputBuffer(inputBufferId, 0, length, 0, 0)
            }
            while (true) {
                val outputBufferId = mediaCodec.dequeueOutputBuffer(decodeBufferInfo, 0)
                if (outputBufferId < 0) break
                val outputBuffer = mediaCodec.getOutputBuffer(outputBufferId)
                val pcm = ByteArray(decodeBufferInfo.size)
                outputBuffer!!.get(pcm)
                outputBuffer.clear()
                audioTrack.write(pcm, 0, pcm.size)
                mediaCodec.releaseOutputBuffer(outputBufferId, false)
            }
        } catch (_: Exception) {
        }
    }
}
