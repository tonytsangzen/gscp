package com.gscp.desktop

import android.util.Log

/**
 * NCNN(-vulkan) inference for SCRFD det_10g (GPU fp16) + 1k3d68 (fp32)，mirroring the ORT CPU
 * path's output contract (scores/boxes/kps per stride for det; out0 flattened for landmarks).
 */
object NcnnEngine {
    private const val TAG = "gscp-ncnn"
    // det outputs (name order matches InsightPose constants)
    internal val DET_NAMES = arrayOf("out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8")

    private var detNet: Long = 0
    private var lmNet: Long = 0
    private var detDump = 0

    external fun nativeInit(param: String, bin: String, useGpu: Boolean, allowFp16: Boolean): Long
    external fun nativeRun(handle: Long, inputName: String, data: FloatArray,
                           names: Array<String>, dims: IntArray): FloatArray?
    external fun nativeRelease(handle: Long)
    /** 清空跨帧锁定（离线逐帧测试用；设备端不需要）。 */
    external fun nativeResetTrack()
    /** nativePose 内部 6 阶段平均耗时（µs）+ 成功帧数：det前处理/det前向/解码+NMS/lm裁剪/lm前向/解算。 */
    external fun nativeProf(): DoubleArray
    external fun nativeProfReset()
    // 整条 det→landmark→Procrustes→pose 链（C++ 确定性移植）。frame 为 BGR(w×h×3)；
    // out 需 ≥26：out[0..8]=R 的行(r1,r2,r3；landmark 系 L=X右/Y下/Z朝相机，人脸轴取 R 的列),
    // [9..11]=pos3(cm,x右/y上/z朝前), [12..25]=bbox14(x1,y1,x2,y2,5×2 kps, 全帧 px)。返回 1=成功。
    external fun nativePose(det: Long, lm: Long, frame: ByteArray, w: Int, h: Int, out: FloatArray): Int

    /** 一次调用拿整条 head-pose：输入「正立 BGR」帧，返回 [26]（见 nativePose）或 null。 */
    private var frameBuf: ByteArray? = null
    private val outBuf = FloatArray(26)

    fun poseOf(bgr: org.opencv.core.Mat): FloatArray? {
        if (detNet == 0L || lmNet == 0L) return null
        val w = bgr.cols().toInt(); val h = bgr.rows().toInt()
        if (w <= 0 || h <= 0) return null
        val n = w * h * 3
        // 像素拷贝缓冲按尺寸复用（每帧 ~1MB，30fps 下新建会让 GC 成为抖动源）；
        // 结果数组同样复用——调用方在同一个 analyzer 回调内消费完毕。
        val bytes = frameBuf?.takeIf { it.size == n } ?: ByteArray(n).also { frameBuf = it }
        bgr.get(0, 0, bytes)
        val hit = try {
            nativePose(detNet, lmNet, bytes, w, h, outBuf)
        } catch (t: Throwable) { Log.w("gscp-ncnn", "nativePose fail", t); 0 }
        return if (hit == 1) outBuf else null
    }

    fun initLoad() {
        try {
            // order matters: libncnn must be loadable before the shim resolves its symbols
            System.loadLibrary("c++_shared")
            System.loadLibrary("ncnn")
            System.loadLibrary("ncnnshim")
        } catch (t: Throwable) { Log.w(TAG, "native load fail", t) }
    }

    fun initDet(param: String, bin: String, useGpu: Boolean): Boolean {
        if (detNet != 0L) { nativeRelease(detNet); detNet = 0 }
        // Mali(实测 vivo/mt6993) 上 fp16 激活存储会让 SCRFD 分数/框逐帧闪跳（框宽波动
        // p50 ~40%），fp32 存储后稳定；Apple GPU 无此现象。故 det 也走 fp32。
        detNet = nativeInit(param, bin, useGpu, false)
        return detNet != 0L
    }
    fun initLm(param: String, bin: String, useGpu: Boolean): Boolean {
        if (lmNet != 0L) { nativeRelease(lmNet); lmNet = 0 }
        // 1k3d68 的 landmark 输出对 fp16 极其敏感（权重+算术均须 fp32），见 InsightPose
        lmNet = nativeInit(param, bin, useGpu, false)
        return lmNet != 0L
    }
    fun isDetReady(): Boolean = detNet != 0L
    fun isLmReady(): Boolean = lmNet != 0L

    /** Run det on a [1,3,480,480] CHW blob. Returns name→flat tensor (score/box/kps ×3). */
    fun runDet(data: FloatArray): Map<String, FloatArray>? {
        if (detNet == 0L) return null
        val dims = IntArray(DET_NAMES.size)
        val cat = nativeRun(detNet, "in0", data, DET_NAMES, dims) ?: return null
        if (detDump++ % 15 == 0) Log.i(TAG, "det dims=" + java.util.Arrays.toString(dims))
        val out = HashMap<String, FloatArray>(DET_NAMES.size)
        var off = 0
        for (i in DET_NAMES.indices) {
            val n = dims[i]
            out[DET_NAMES[i]] = if (n > 0) cat.copyOfRange(off, off + n) else FloatArray(0)
            off += n
        }
        return out
    }

    /** Run 1k3d68 → out0 flat (3309). */
    fun runLm(data: FloatArray): FloatArray? {
        if (lmNet == 0L) return null
        val dims = IntArray(1)
        return nativeRun(lmNet, "in0", data, arrayOf("out0"), dims)
    }
}