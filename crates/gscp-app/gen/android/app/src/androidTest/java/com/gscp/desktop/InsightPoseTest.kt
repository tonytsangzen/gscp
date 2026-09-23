package com.gscp.desktop

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlin.math.abs
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * 用 ht 测试集图片在模拟器上离线校验 insight_procrustes_coreml 移植（SCRFD+1k3d68+Procrustes）
 * 与 Python 参考一致。gold 值来自离线校验脚本（误差=0）。
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class InsightPoseTest {
    companion object {
        private const val TAG = "insighttest"
        private const val TOL_BBOX = 4f
        private const val TOL_LM = 4f
        private const val TOL_Z = 5f
        private const val TOL_N = 0.03f

        @JvmStatic
        @BeforeClass
        fun init() {
            OpenCVLoader.initLocal()
        }
    }

    /** 从 app assets 加载 PNG → BGR Mat（不旋转）。 */
    private fun loadBgr(ctx: Context, name: String): Mat {
        val bmp = ctx.assets.open(name).use { BitmapFactory.decodeStream(it) }
        val rgba = Mat(); Utils.bitmapToMat(bmp, rgba)
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release(); bmp.recycle()
        return bgr
    }

    private fun poseOn(name: String, expectBbox: FloatArray,
                      expectLm0: FloatArray, expectLm30: FloatArray, expectN: FloatArray) {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val ip = InsightPose(ctx.assets, ctx.filesDir)
        ip.ensureLoaded()
        assertTrue("model load", ip.isReady())
        val bgr = loadBgr(ctx, name)
        val sum = Core.sumElems(bgr)
        val np = (bgr.rows() * bgr.cols()).toFloat()
        Log.i(TAG, "$name img=${bgr.cols()}x${bgr.rows()} meanc=" +
            String.format("%.1f,%.1f,%.1f", (sum.`val`[0] / np), (sum.`val`[1] / np), (sum.`val`[2] / np)))
        val det = ip.detect(bgr)
        assertTrue("detect hit", det != null)
        assertTrue("bbox", abs(det!![0] - expectBbox[0]) < TOL_BBOX &&
            abs(det[1] - expectBbox[1]) < TOL_BBOX &&
            abs(det[2] - expectBbox[2]) < TOL_BBOX &&
            abs(det[3] - expectBbox[3]) < TOL_BBOX)
        val bbox = floatArrayOf(det[0], det[1], det[2], det[3])
        val lmk = ip.landmarks(bgr, bbox)!!
        Log.i(TAG, "$name bbox=${bbox.contentToString()}")
        Log.i(TAG, "$name lm0=(${lmk[0]},${lmk[1]},${lmk[2]}) lm30=(${lmk[90]},${lmk[91]},${lmk[92]})")
        assertTrue("lm0", abs(lmk[0] - expectLm0[0]) < TOL_LM && abs(lmk[1] - expectLm0[1]) < TOL_LM &&
            abs(lmk[2] - expectLm0[2]) < TOL_Z)
        assertTrue("lm30", abs(lmk[90] - expectLm30[0]) < TOL_LM && abs(lmk[91] - expectLm30[1]) < TOL_LM &&
            abs(lmk[92] - expectLm30[2]) < 0.5f)
        var n = 0f; for (i in 0 until 68) n += lmk[i * 3 + 2]
        Log.i(TAG, "$name meanz=${(n / 68f)}")
        val pose = ip.pose(bgr, bbox)
        assertTrue("pose null", pose != null)
        val p = pose!!
        Log.i(TAG, "$name basis/pos = " + java.util.Arrays.toString(p))
        Log.i(TAG, "$name normal=(${p[6]},${p[7]},${p[8]})")
        assertTrue("normal", abs(p[6] - expectN[0]) < TOL_N && abs(p[7] - expectN[1]) < TOL_N &&
            abs(p[8] - expectN[2]) < TOL_N)
        Log.i(TAG, "$name PASS normal=(${p[6]},${p[7]},${p[8]}) pos=(${p[9]},${p[10]},${p[11]})")
    }

    @Test
    fun a0_test() {
        poseOn("t_a0.png",
            floatArrayOf(1.63f, -6.08f, 90.09f, 103.06f),
            floatArrayOf(6.91f, 62.91f, 62.83f),
            floatArrayOf(52.81f, 52.60f, 0f),
            floatArrayOf(-0.083f, -0.011f, 0.994f))
    }

    @Test
    fun landmark_check() {
        poseOn("t_lm.png",
            floatArrayOf(0.33f, -10.14f, 107.63f, 121.06f),
            floatArrayOf(7.15f, 73.94f, 75.23f),
            floatArrayOf(62.31f, 63.83f, 0f),
            floatArrayOf(-0.068f, -0.016f, 0.996f))
    }
}