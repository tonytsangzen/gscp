package com.gscp.desktop

import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.view.Surface

/**
 * H.264 硬解（MediaCodec）。
 * 两种输出模式：
 *  - surface != null：直渲染到合成器 Surface（旧路径）；
 *  - surface == null：ByteBuffer 输出 + getOutputImage()，每帧回调 CPU 可读的
 *    I420 Image（供 Canvas 架构取帧；规避厂商 ImageReader 平铺/压缩格式 CPU 不可读的问题）。
 */
class VideoDecoder {
    private val mediaCodec: MediaCodec = try {
        MediaCodec.createDecoderByType("video/avc")
    } catch (e: Exception) {
        MediaCodec.createByCodecName("OMX.google.h264.decoder")
    }
    private var toSurface = true

    /** 无 Surface 解码时每输出一帧回调一次（Image 在回调内消费，返回后即失效）。 */
    var frameCallback: ((Image) -> Unit)? = null

    @Synchronized
    fun start(width: Int, height: Int, surface: Surface?) {
        val format = MediaFormat.createVideoFormat("video/avc", width, height)
        format.setInteger(MediaFormat.KEY_PRIORITY, 0)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 15)
        format.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT601_NTSC)
        format.setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
        toSurface = surface != null
        if (surface != null) {
            mediaCodec.configure(format, surface, null, 0)
        } else {
            format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )
            mediaCodec.configure(format, null, null, 0)
        }
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
                mediaCodec.releaseOutputBuffer(outputBufferIndex, toSurface)
            }
            mediaCodec.stop()
            mediaCodec.reset()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    var diagOut = 0L   // 诊断：已渲染输出帧计数
    var diagTag = "vdec"

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
                if (!toSurface) {
                    // 无 Surface：getOutputImage 给出线性 I420 平面（CPU 可读）
                    val img = mediaCodec.getOutputImage(outputBufferIndex)
                    if (img != null) frameCallback?.invoke(img)
                }
                mediaCodec.releaseOutputBuffer(outputBufferIndex, toSurface)
                if (toSurface && ++diagOut % 150L == 1L)
                    android.util.Log.i("ar-ui", "$diagTag out #$diagOut")
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
