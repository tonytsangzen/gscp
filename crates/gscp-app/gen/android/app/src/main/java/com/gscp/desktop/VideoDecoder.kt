package com.gscp.desktop

import android.media.MediaCodec
import android.media.MediaFormat
import android.view.Surface

/** H.264 硬解（MediaCodec）直渲染到合成器 Surface。 */
class VideoDecoder {
    private val mediaCodec: MediaCodec = try {
        MediaCodec.createDecoderByType("video/avc")
    } catch (e: Exception) {
        MediaCodec.createByCodecName("OMX.google.h264.decoder")
    }

    @Synchronized
    fun start(width: Int, height: Int, surface: Surface) {
        val format = MediaFormat.createVideoFormat("video/avc", width, height)
        format.setInteger(MediaFormat.KEY_PRIORITY, 0)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 15)
        format.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT601_NTSC)
        format.setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
        mediaCodec.configure(format, surface, null, 0)
        mediaCodec.start()
    }

    @Synchronized
    fun stop() {
        try {
            val bufferInfo = MediaCodec.BufferInfo()
            mediaCodec.flush()
            val idx = mediaCodec.dequeueInputBuffer(1000)
            if (idx >= 0) {
                val inputBuffer = mediaCodec.getInputBuffer(idx)
                inputBuffer!!.clear()
                mediaCodec.queueInputBuffer(
                    idx, 0, 0, 0,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                )
            }
            while (true) {
                val outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 100000)
                if (outputBufferIndex < 0) break
                mediaCodec.releaseOutputBuffer(outputBufferIndex, true)
            }
            mediaCodec.stop()
            mediaCodec.reset()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    @Synchronized
    fun decode(buffer: ByteArray, offset: Int, length: Int) {
        val bufferInfo = MediaCodec.BufferInfo()
        try {
            val idx = mediaCodec.dequeueInputBuffer(100000)
            if (idx >= 0) {
                val inputBuffer = mediaCodec.getInputBuffer(idx)
                inputBuffer!!.clear()
                inputBuffer.put(buffer, offset, length)
                mediaCodec.queueInputBuffer(idx, offset, length, System.currentTimeMillis(), 0)
            }
            while (true) {
                val outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 0)
                if (outputBufferIndex < 0) break
                mediaCodec.releaseOutputBuffer(outputBufferIndex, true) // 渲染到 Surface
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
