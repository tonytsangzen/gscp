package com.gscp.desktop

/**
 * AR native 管线 JNI 桥（移植自 ncnn-benchmark NativeBridge，lib 为 ncnnshim）。
 * 完整管线见 tools/ncnn/ar_face_live.cpp：det_10g(448) + FaceMesh(468) 9 点 Kabsch
 * + hopenet 融合 + One-Euro + 几何锚定 overlay。
 */
object ArNative {
    init {
        System.loadLibrary("ncnnshim")
    }

    /**
     * 加载检测网（backend: 0=CPU FP32, 1=CPU FP16, 2/3=Vulkan 自动调优），
     * 返回 JSON {"ok","backend","threads","input","variant","gpuMs",...}。
     */
    external fun nativeFaceOpen(assets: android.content.res.AssetManager, filesDir: String,
                                backend: Int, input: Int): String

    /**
     * 一帧处理：YUV 三平面（保留 rowStride 布局）转 BGR 并顺时针旋转 rot 度，直立图
     * RGBA 写入 rgbaOut（direct buffer，大小 = 直立图宽×高×4）；返回 JSON：
     * {"ok","convMs","ms","w","h","rot","maxScore","faces":[
     *   {"box":[4],"score","kps":[10],"pose":[3],"normal":[2][2],
     *    "overlay":[4][2],"ovC":[2],"ovD":cm}]}
     */
    external fun nativeFaceDetect(
        y: ByteArray, u: ByteArray, v: ByteArray,
        w: Int, h: Int,
        yStride: Int, uStride: Int, vStride: Int,
        uPix: Int, vPix: Int,
        rot: Int, detect: Boolean,
        rgbaOut: java.nio.ByteBuffer,
    ): String

    /** 释放检测网（Vulkan 全局设备保留）。 */
    external fun nativeFaceClose()

    /** 姿态算法：1=hopenet（对照） 3=融合（默认主路径）；其余一律回融合。 */
    external fun nativeFaceSetPoseAlgo(algo: Int)

    /** overlay 显示位图尺寸（内容包围盒裁剪后）——quad 纵横比按实际内容锁定。 */
    external fun nativeSetOverlaySize(w: Int, h: Int)
}
