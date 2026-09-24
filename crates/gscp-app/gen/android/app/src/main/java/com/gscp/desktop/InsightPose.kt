package com.gscp.desktop

import android.content.res.AssetManager
import android.opengl.GLES11Ext
import android.util.Log
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream

/**
 * face→landmark→Procrustes→head-pose 的**确定性 C++ 移植**（nativePose）薄壳。
 *
 * 整条链（SCRFD det_10g 640 letterbox → 解码+NMS → 单人脸 → 仿射裁剪 192 →
 * 1k3d68 → canonical 模板 Procrustes → 旋转 R + 世界位 cm）
 * 全部在 ncnn_engine.cpp 的 Java_com_gscp_desktop_NcnnEngine_nativePose 里一次完成，
 * 与 ht 项目 insight_pipe 的固化配置一致（lightmode=false；两网 GPU 上均 fp32 激活存储——
 * fp16 权重使 landmark 漂移 ~14px+（ht §5.13b），fp16 激活存储使 SCRFD 框逐帧闪跳（Mali 真机实测））。
 * 本类只负责：asset→filesDir 拷贝模型 + 一个入参(正立 BGR 帧)调用。
 *
 * 返回 [26]：out[0..8]=R 的行 (r1,r2,r3)，R 把 canonical 帧旋到 landmark 系 L(X右/Y下/Z朝相机)，
 * 人脸三轴取 R 的列（横轴=(o0,o3,o6)、下轴=(o1,o4,o7)、朝外法线=(o2,o5,o8)）；
 * 换算到渲染器系的工作在 ArActivity 完成。姿态角用 rot_to_ypr(行式)：
 * yaw=-atan2(-o6,√(o0²+o3²))、pitch=atan2(o7,o8)、roll=atan2(o3,o0)（yaw 为报告约定）。
 * out[9..11]=pos3(cm, x右/y上/z朝前)，out[12..25]=bbox14(x1,y1,x2,y2, 5×2 kps, 全帧 px)。
 */
class InsightPose(private val assets: AssetManager, private val filesDir: File) {
    companion object {
        private const val TAG = "gscp-ar"
    }

    @Volatile
    private var ready = false

    fun isReady(): Boolean = ready

    fun ensureLoaded() {
        if (ready) return
        synchronized(this) {
            if (ready) return
            NcnnEngine.initLoad()
            ensureModel("det_ht.param"); ensureModel("det_ht.bin")
            ensureModel("1k3d68_192_fp32.param"); ensureModel("1k3d68_192_fp32.bin")
            // fp16 权重的 1k3d68 导出会让 landmark 漂移 ~14px+（ht §5.13b 实测），换 fp32 后清掉旧缓存
            File(filesDir, "1k3d68_192_f16.param").delete()
            File(filesDir, "1k3d68_192_f16.bin").delete()
            try {
                val detOk = NcnnEngine.initDet(
                    File(filesDir, "det_ht.param").absolutePath,
                    File(filesDir, "det_ht.bin").absolutePath, true)
                val lmOk = NcnnEngine.initLm(
                    File(filesDir, "1k3d68_192_fp32.param").absolutePath,
                    File(filesDir, "1k3d68_192_fp32.bin").absolutePath, true)
                if (!detOk || !lmOk) {
                    Log.w(TAG, "ncnn init fail detOk=" + detOk + " lmOk=" + lmOk)
                    return
                }
            } catch (t: Throwable) {
                Log.w(TAG, "ncnn init fail", t)
                return
            }
            ready = true
        }
    }

    private fun ensureModel(name: String) {
        val f = File(filesDir, name)
        if (f.exists() && f.length() > 1000) return
        assets.open(name).use { input ->
            FileOutputStream(f).use { out ->
                val buf = ByteArray(1 shl 16)
                while (true) { val n = input.read(buf); if (n < 0) break; out.write(buf, 0, n) }
            }
        }
    }

    /** 整条 pose 链（nativePose）。输入「正立 BGR」帧，返回 [26]；无人脸或失败 null。 */
    fun pose(bgr: Mat): FloatArray? = NcnnEngine.poseOf(bgr)
}