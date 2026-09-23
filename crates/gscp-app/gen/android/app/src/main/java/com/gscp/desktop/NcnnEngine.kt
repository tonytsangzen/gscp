package com.gscp.desktop

import android.util.Log

/**
 * NCNN(-vulkan, fp16) inference for SCRFD det_10g + 1k3d68, mirroring the ORT CPU path's
 * output contract (scores/boxes/kps per stride for det; fc1 flattened for landmarks).
 */
object NcnnEngine {
    private const val TAG = "gscp-ncnn"
    // det outputs (name order matches InsightPose constants)
    internal val DET_NAMES = arrayOf("out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8")

    private var detNet: Long = 0
    private var lmNet: Long = 0
    private var detDump = 0

    external fun nativeInit(param: String, bin: String, useGpu: Boolean): Long
    external fun nativeRun(handle: Long, inputName: String, data: FloatArray,
                           names: Array<String>, dims: IntArray): FloatArray?
    external fun nativeRelease(handle: Long)
    // 整条 det→landmark→Procrustes→pose 链（C++ 确定性移植）。frame 为 BGR(w×h×3)；
    // out 需 ≥26：out[0..8]=basis(right,up,normal), [9..11]=pos3(cm,z 朝前),
    // [12..25]=bbox14(x1,y1,x2,y2,5×2 kps, 全帧 px)。返回 1=成功。
    external fun nativePose(det: Long, lm: Long, frame: ByteArray, w: Int, h: Int, out: FloatArray): Int

    /** 一次调用拿整条 head-pose：输入「正立 BGR」帧，返回 [26]（见 nativePose）或 null。 */
    fun poseOf(bgr: org.opencv.core.Mat): FloatArray? {
        if (detNet == 0L || lmNet == 0L) return null
        val w = bgr.cols().toInt(); val h = bgr.rows().toInt()
        if (w <= 0 || h <= 0) return null
        val bytes = ByteArray(w * h * 3)
        bgr.get(0, 0, bytes)
        val out = FloatArray(26)
        val hit = try {
            nativePose(detNet, lmNet, bytes, w, h, out)
        } catch (t: Throwable) { Log.w("gscp-ncnn", "nativePose fail", t); 0 }
        return if (hit == 1) out else null
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
        val h = nativeInit(param, bin, useGpu)
        detNet = h
        return h != 0L
    }
    fun initLm(param: String, bin: String, useGpu: Boolean): Boolean {
        if (lmNet != 0L) { nativeRelease(lmNet); lmNet = 0 }
        val h = nativeInit(param, bin, useGpu)
        lmNet = h
        return h != 0L
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

    /** Run 1k3d68 → fc1 flat (3309). */
    fun runLm(data: FloatArray): FloatArray? {
        if (lmNet == 0L) return null
        val dims = IntArray(1)
        return nativeRun(lmNet, "data", data, arrayOf("fc1"), dims)
    }
}