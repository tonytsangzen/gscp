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

    // ── overlay 超分（ncnn-benchmark sr_benchmark ESPCN x2 灰度）──────────

    /**
     * 加载超分网：assets/espcn_x2.ncnn.param/bin（480² 亮度 → 960²）。
     * vulkan=true 优先 Vulkan FP16（需检测网先建好 GPU 设备则自动创建），
     * 失败回退 safe-CPU（fp32/普通卷积，pnnx 模型真机稳健配置）。
     * 返回 JSON {"ok","backend":"vulkan"|"cpu","scale":2}。
     */
    external fun nativeSrOpen(assets: android.content.res.AssetManager, threads: Int, vulkan: Boolean): String

    /**
     * 亮度超分：lumaIn = w*h 字节（direct），lumaOut = 4*w*h 字节（direct，2x2 放大）。
     * 返回 JSON {"ok","ms","w","h","ow","oh"}。调用线程需串行。
     */
    external fun nativeSrProcess(lumaIn: java.nio.ByteBuffer, w: Int, h: Int, lumaOut: java.nio.ByteBuffer): String

    /** 释放超分网（Vulkan 全局设备保留）。 */
    external fun nativeSrClose()
}
