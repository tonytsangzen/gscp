package com.gscp.desktop

/**
 * 眼镜音频 opus 软解 JNI 接口（Rust 端 gscp-app::opus_jni，与 PC gscp-player 同源 opus 解码器）。
 * 绕开本机不可用的 MediaCodec `c2.android.opus.decoder`。
 */
object AudioNative {
    init {
        System.loadLibrary("gscp_app_lib")
    }

    /** 创建全局 opus 解码器（48k 立体声），返回是否就绪。 */
    @JvmStatic external fun nativeOpusStart(): Boolean

    /** 解码一帧 opus -> PCM(i16 LE 交织)。pcm 缓冲需 ≥ 23040 字节；返回写入字节数，出错返回 0。 */
    @JvmStatic external fun nativeOpusDecode(data: ByteArray, pcm: ByteArray): Int

    /** 释放解码器。 */
    @JvmStatic external fun nativeOpusStop()
}