package com.gscp.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.*

/**
 * 移植自 ht 项目 `insight_procrustes_coreml`，替代现有 YuNet+EPNP+启发式：
 * SCRFD det_10g → 1k3d68（192 对齐裁剪）→ canonical 模板 Procrustes。
 * 用 ONNX Runtime（enforcing? 参考，与 insightface/ht 一致）+ OpenCV 预处理/解码。
 * 离线校验与 insightface 参考完全一致（det/kps、68 地标 px/mm、R、normal、depth 误差 0）。
 *
 * 约定：输入「正立人像 BGR」帧（宽=W 高=H，全帧坐标）；1k3d68 归一化 = 均0/标1（0..255 RGB）。
 */
class InsightPose(private val assets: AssetManager, private val filesDir: File) {

    companion object {
        private const val IPD_CM = 6.2f
        private const val TAG = "gscp-ar"
        // 真机实测：NNAPI 被 OS(SELinux)封锁、GPU 峰值 ~31 GFLOPS 不足以加速 640；
        // CPU 下 640→480 约两倍提速（~274ms→~140ms），故 live 用 480。
        private const val DET = 480
        private const val LM = 192
        private const val THR = 0.3f
        private const val NMS = 0.4f
        private val STRIDES = intArrayOf(8, 16, 32)
        // det_10g 输出（batched=False，2-D）：scores / boxes / kps，按 stride 8,16,32 分组命名
        private val SCORES_OUT = arrayOf("448", "471", "494")
        private val BOXES_OUT = arrayOf("451", "474", "497")
        private val KPS_OUT = arrayOf("454", "477", "500")
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var detSess: OrtSession? = null
    private var lmSess: OrtSession? = null
    private var lmWarned = false
    // 观测：det / lm 每 30 次打印一次平均耗时（ms）
    private var detCnt = 0; private var detAcc = 0L
    private var lmCnt = 0; private var lmAcc = 0L
    @Volatile
    private var ready = false

    fun isReady(): Boolean = ready

    private val canonical = floatArrayOf(
        -0.626695f,-0.2927f,-0.314002f,-0.599665f,-0.122503f,-0.292441f,-0.571026f,0.051187f,-0.254947f,-0.533857f,0.212848f,-0.186633f,-0.479733f,0.350646f,-0.0477f,
        -0.395756f,0.446512f,0.073287f,-0.298808f,0.510661f,0.179778f,-0.188387f,0.55444f,0.31666f,0.001471f,0.584439f,0.388416f,0.190991f,0.551707f,0.314331f,
        0.326929f,0.489576f,0.1684f,0.440026f,0.402358f,0.035962f,0.506879f,0.311625f,-0.094761f,0.540895f,0.204526f,-0.202671f,0.574118f,0.045703f,-0.284176f,
        0.599142f,-0.145859f,-0.296496f,0.627544f,-0.307748f,-0.301995f,-0.474668f,-0.437605f,0.236487f,-0.41666f,-0.471756f,0.315993f,-0.347539f,-0.484078f,0.366113f,
        -0.26064f,-0.476391f,0.399232f,-0.167123f,-0.457775f,0.416609f,0.123174f,-0.45874f,0.425062f,0.206362f,-0.480419f,0.415784f,0.286673f,-0.490149f,0.391954f,
        0.362397f,-0.47686f,0.352786f,0.425568f,-0.450058f,0.295319f,-0.007628f,-0.323089f,0.461944f,-0.007877f,-0.25574f,0.510469f,-0.007689f,-0.19918f,0.552546f,
        -0.007345f,-0.142617f,0.598667f,-0.144961f,0.033127f,0.419691f,-0.084343f,0.031274f,0.473174f,-0.0055f,0.039751f,0.51467f,0.063463f,0.046135f,0.479224f,
        0.133984f,0.02204f,0.419079f,-0.386751f,-0.313398f,0.259632f,-0.316661f,-0.350072f,0.328527f,-0.234138f,-0.354914f,0.333493f,-0.155162f,-0.315249f,0.314328f,
        -0.230922f,-0.28427f,0.325583f,-0.317509f,-0.285165f,0.309882f,0.138957f,-0.309824f,0.318284f,0.219456f,-0.353192f,0.338028f,0.301747f,-0.349665f,0.333102f,
        0.376653f,-0.313518f,0.263229f,0.296695f,-0.287144f,0.322013f,0.214624f,-0.290528f,0.331242f,-0.201438f,0.237361f,0.379537f,-0.137321f,0.185786f,0.465253f,
        -0.076486f,0.151193f,0.503572f,-0.002536f,0.168727f,0.516438f,0.064421f,0.15088f,0.504524f,0.126465f,0.179476f,0.468592f,0.218248f,0.238992f,0.375674f,
        0.132883f,0.283928f,0.440051f,0.068022f,0.297354f,0.477404f,-0.000469f,0.300041f,0.487109f,-0.069343f,0.296969f,0.48054f,-0.14252f,0.274303f,0.43808f,
        -0.178135f,0.230591f,0.396359f,-0.074031f,0.214719f,0.465326f,-0.002636f,0.214142f,0.48323f,0.059816f,0.210764f,0.472244f,0.1669f,0.231271f,0.396789f,
        0.059809f,0.223764f,0.466413f,-0.001434f,0.225759f,0.475244f,-0.075221f,0.230656f,0.4671475f,
    )

    fun ensureLoaded() {
        if (ready) return
        synchronized(this) {
            if (ready) return
            ensureModel("det_10g.onnx"); ensureModel("1k3d68.onnx")
            try {
                val dOpts = OrtSession.SessionOptions()
                val lOpts = OrtSession.SessionOptions()
                // CPU EP：全核利用 + 图形级优化，最大化 CPU 吞吐（NNAPI/GPU 在本机均无效）。
                val cores = Runtime.getRuntime().availableProcessors()
                dOpts.setIntraOpNumThreads(cores); dOpts.setInterOpNumThreads(1)
                dOpts.setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT)
                lOpts.setIntraOpNumThreads(cores); lOpts.setInterOpNumThreads(1)
                lOpts.setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // ONNX Runtime：与 ht 参考一致（CPU EP，稳定可靠）
                detSess = env.createSession(File(filesDir, "det_10g.onnx").absolutePath, dOpts)
                lmSess = env.createSession(File(filesDir, "1k3d68.onnx").absolutePath, lOpts)
            } catch (t: Throwable) {
                Log.w(TAG, "insight ort session fail", t)
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

    // ---------------- SCRFD 检测 ----------------
    /** 检测最大/最明显人脸。返回全帧坐标 {x1,y1,x2,y2, 5×关键点(右眼,左眼,鼻,右嘴,左嘴 ×(x,y))}（14 浮点），失败 null。 */
    fun detect(bgr: Mat): FloatArray? {
        val sess = detSess ?: return null
        val h = bgr.rows(); val w = bgr.cols()
        val r = h.toFloat() / w
        val nh: Int; val nw: Int
        if (r > 1f) { nh = DET; nw = (nh / r).toInt() } else { nw = DET; nh = (nw * r).toInt() }
        val scale = nh.toFloat() / h
        val res = Mat()
        Imgproc.resize(bgr, res, Size(nw.toDouble(), nh.toDouble()))
        val detImg = Mat.zeros(DET, DET, CvType.CV_8UC3)
        res.copyTo(detImg.submat(0, nh, 0, nw))
        val blob = Dnn.blobFromImage(detImg, 1.0 / 128.0, Size(DET.toDouble(), DET.toDouble()),
            Scalar(127.5, 127.5, 127.5), true, false)
        val data = blobF(blob)
        res.release(); detImg.release()
        val bx = ArrayList<FloatArray>(); val sc = ArrayList<Float>(); val kp = ArrayList<FloatArray>()
        var allScoreMax = 0f
        val t0 = System.nanoTime()
        val result = runOrt(sess, data, longArrayOf(1, 3, DET.toLong(), DET.toLong())) ?: run {
            detAcc += (System.nanoTime() - t0) / 1_000_000; if (++detCnt % 30 == 0) Log.i(TAG, "perf det=" + (detAcc / detCnt) + "ms")
            return null
        }
        detAcc += (System.nanoTime() - t0) / 1_000_000; if (++detCnt % 30 == 0) Log.i(TAG, "perf det=" + (detAcc / detCnt) + "ms")
        try {
            val map = java.util.HashMap<String, ai.onnxruntime.OnnxTensor>()
            for (e in result) if (e.value is ai.onnxruntime.OnnxTensor) map[e.key] = e.value as ai.onnxruntime.OnnxTensor
            for (k in 0..2) {
                val sv = flattenF(map[SCORES_OUT[k]]?.getValue() ?: continue)
                val bv = flattenF(map[BOXES_OUT[k]]?.getValue() ?: continue)
                val kv = flattenF(map[KPS_OUT[k]]?.getValue() ?: continue)
                if (sv != null) for (v in sv) if (v > allScoreMax) allScoreMax = v
                if (sv == null || bv == null || kv == null) continue
                decodeStride(sv, bv, kv, STRIDES[k], bx, sc, kp)
            }
        } finally { result.close() }
        if (sc.isEmpty()) return null
        val n = sc.size
        // NMS（按 score 降序）
        val keep = ArrayList<Int>()
        val undone = (0 until n).sortedByDescending { sc[it] }.toMutableList()
        while (undone.isNotEmpty()) {
            val i = undone.removeAt(0); keep.add(i)
            val aI = (bx[i][2] - bx[i][0]) * (bx[i][3] - bx[i][1]) + 1f
            val it = undone.iterator()
            while (it.hasNext()) {
                val j = it.next()
                val xx1 = max(bx[i][0], bx[j][0]); val yy1 = max(bx[i][1], bx[j][1])
                val xx2 = min(bx[i][2], bx[j][2]); val yy2 = min(bx[i][3], bx[j][3])
                val ww = max(0f, xx2 - xx1 + 1f); val hh = max(0f, yy2 - yy1 + 1f)
                if (ww * hh / (aI + 1e-6f) > NMS) it.remove()
            }
        }
        var bestScore = -1f
        for (k in keep) bestScore = max(bestScore, sc[k])
        val thr = max(THR, 0.9f * bestScore)
        var pick = keep[0]; var pickArea = -1f
        for (k in keep) if (sc[k] >= thr) {
            val a = (bx[k][2] - bx[k][0]) * (bx[k][3] - bx[k][1])
            if (a > pickArea) { pickArea = a; pick = k }
        }
        val b = bx[pick]; val pk = kp[pick]
        val bw = b[2] - b[0]; val bh = b[3] - b[1]
        if (!b[0].isFinite() || bw <= 0f || bh <= 0f) { Log.d(TAG, "guard bbox b=" + java.util.Arrays.toString(b)); return null }
        if (bw * bh < 8f) return null   // 拒绝 1~2px 幻影框（缩放帧坐标）
        // 拒绝「5 关键点塌缩成一点」的退化人脸（真脸在缩放帧内至少几十像素跨度）
        var kx0 = pk[0]; var kx1 = pk[0]; var ky0 = pk[1]; var ky1 = pk[1]
        for (i in 0 until 5) {
            val px = pk[i * 2]; val py = pk[i * 2 + 1]
            if (px < kx0) kx0 = px; if (px > kx1) kx1 = px
            if (py < ky0) ky0 = py; if (py > ky1) ky1 = py
        }
        if (kx1 - kx0 < 12f || ky1 - ky0 < 12f) return null
        val out = FloatArray(14)
        out[0] = b[0] / scale; out[1] = b[1] / scale; out[2] = b[2] / scale; out[3] = b[3] / scale
        for (i in 0 until 10) out[4 + i] = pk[i] / scale
        return out
    }

    /** 反序列化 onnxruntime 输出（batched=False 各为 2-D [N,·]）。anchor 交错：cell = p/2，col=cell%g,row=cell/g。 */
    private fun decodeStride(sv: FloatArray, bv: FloatArray, kv: FloatArray, s: Int,
                             bx: MutableList<FloatArray>, sc: MutableList<Float>, kp: MutableList<FloatArray>) {
        val g = DET / s
        val per = g * g
        val total = per * 2
        val strideF = s.toFloat()
        for (p in 0 until total) {
            val v = sv[p]; if (v < THR) continue
            val cell = p / 2      // per-anchor 单元
            val col = cell % g; val row = cell / g
            val ax = col * strideF; val ay = row * strideF
            val b4 = p * 4
            if (b4 + 3 >= bv.size) continue
            bx.add(floatArrayOf(ax - bv[b4] * strideF, ay - bv[b4 + 1] * strideF,
                ax + bv[b4 + 2] * strideF, ay + bv[b4 + 3] * strideF))
            val k10 = FloatArray(10)
            val k4 = p * 10
            if (k4 + 9 < kv.size) for (i in 0 until 5) {
                k10[i * 2] = ax + kv[k4 + i * 2] * strideF
                k10[i * 2 + 1] = ay + kv[k4 + i * 2 + 1] * strideF
            }
            kp.add(k10)
            sc.add(v)
        }
    }

    private fun runOrt(sess: OrtSession, data: FloatArray, shape: LongArray): OrtSession.Result? {
        return try {
            val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
            val names = java.util.HashMap<String, OnnxTensor>(); names.put(sess.inputNames.first(), input)
            val out = sess.run(names)
            input.close()
            out
        } catch (t: Throwable) {
            Log.w(TAG, "ort run fail", t)
            null
        }
    }

    private fun flatFloat(getValue: Any): FloatArray? = flattenF(getValue)

    /** 递归展平 N-D float 数组（适配 onnxruntime 各种 shape 返回的 Object）。 */
    private fun flattenF(o: Any): FloatArray? {
        val out = ArrayList<Float>()
        fun f(x: Any) {
            val comp = x.javaClass.componentType
            if (comp == java.lang.Float.TYPE) { for (v in x as FloatArray) out.add(v) }
            else { val len = java.lang.reflect.Array.getLength(x); for (i in 0 until len) f(java.lang.reflect.Array.get(x, i)!!) }
        }
        f(o)
        return if (out.isEmpty()) null else out.toFloatArray()
    }

    // ---------------- 1k3d68 + Procrustes + 渲染器姿态 ----------------
    /** 68 地标（x,y 全帧 px；z 相对头深 mm）。 */
    fun landmarks(bgr: Mat, bbox: FloatArray): FloatArray? {
        val sess = lmSess ?: return null
        val w0 = bbox[2] - bbox[0]; val h0 = bbox[3] - bbox[1]
        val cx = (bbox[2] + bbox[0]) / 2f; val cy = (bbox[3] + bbox[1]) / 2f
        val scaleS = LM / (max(w0, h0) * 1.5f)
        val M = floatArrayOf(
            scaleS, 0f, LM / 2f - cx * scaleS,
            0f, scaleS, LM / 2f - cy * scaleS,
        )
        val mMat = Mat(2, 3, CvType.CV_32FC1); mMat.put(0, 0, M)
        val aimg = Mat()
        Imgproc.warpAffine(bgr, aimg, mMat, Size(LM.toDouble(), LM.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0))
        val blob = Dnn.blobFromImage(aimg, 1.0, Size(LM.toDouble(), LM.toDouble()), Scalar(0.0, 0.0, 0.0), true, false)
        val data = blobF(blob)
        mMat.release(); aimg.release()
        val t0 = System.nanoTime()
        val result = runOrt(sess, data, longArrayOf(1, 3, LM.toLong(), LM.toLong())) ?: return null
        val pred: FloatArray
        try {
            // fc1 输出 [1,3309]
            var fc: ai.onnxruntime.OnnxTensor? = null
            outer@ for (e in result) if (e.value is ai.onnxruntime.OnnxTensor) { fc = e.value as ai.onnxruntime.OnnxTensor; break@outer }
            pred = flattenF(fc?.getValue() ?: return null) ?: return null
        } finally { result.close() }
        lmAcc += (System.nanoTime() - t0) / 1_000_000; if (++lmCnt % 30 == 0) Log.i(TAG, "perf lm=" + (lmAcc / lmCnt) + "ms")
        if (pred.size < 3309) { if (!lmWarned) { lmWarned = true; Log.w(TAG, "1k3d68 bad size=" + pred.size) }; return null }
        val base = pred.size / 3 - 68
        if (base < 0) return null
        val half = LM / 2f
        // 反变换（trans_points3d）：z 按 IM 缩放。M 为 2x3 行主序 [a00,a01,a02, a10,a11,a12]。
        val det = M[0] * M[4] - M[1] * M[3]
        val i00 = M[4] / det; val i01 = -M[1] / det
        val i02 = (M[1] * M[5] - M[2] * M[4]) / det
        val i10 = -M[3] / det; val i11 = M[0] / det
        val i12 = (M[2] * M[3] - M[0] * M[5]) / det
        val zS = sqrt(i00 * i00 + i01 * i01)
        val out = FloatArray(68 * 3)
        for (i in 0 until 68) {
            val j = base + i
            val x = (pred[j * 3] + 1f) * half
            val y = (pred[j * 3 + 1] + 1f) * half
            val z = pred[j * 3 + 2] * half
            out[i * 3] = i00 * x + i01 * y + i02
            out[i * 3 + 1] = i10 * x + i11 * y + i12
            out[i * 3 + 2] = z * zS
        }
        return out
    }

    /** 渲染器姿态：返回 basis9（right/up/normal，单位，法线朝相机）+ pos3（cm）。失败 null。 */
    fun pose(bgr: Mat, bbox: FloatArray): FloatArray? {
        val w = bgr.cols().toFloat(); val h = bgr.rows().toFloat()
        val lmk = landmarks(bgr, bbox) ?: return null
        for (i in 0 until 68) if (!lmk[i * 3].isFinite() || !lmk[i * 3 + 1].isFinite() || !lmk[i * 3 + 2].isFinite()) {
            if (!lmWarned) { lmWarned = true
                Log.w(TAG, "pose nonfinite lmk i=" + i + " v=" + lmk[i * 3] + "," + lmk[i * 3 + 1] + "," + lmk[i * 3 + 2]) }
            return null
        }
        // Procrustes（法方程）→ P(3×4)：r1,r2,r3 = P 行归一
        val aM = Array(4) { DoubleArray(4) }
        val rhs = Array(4) { DoubleArray(3) }
        for (i in 0 until 68) {
            val c = canonical
            val s0 = c[i * 3].toDouble(); val s1 = c[i * 3 + 1].toDouble(); val s2 = c[i * 3 + 2].toDouble()
            val src = doubleArrayOf(s0, s1, s2, 1.0)
            val dst = doubleArrayOf(lmk[i * 3].toDouble(), lmk[i * 3 + 1].toDouble(), lmk[i * 3 + 2].toDouble())
            for (r in 0 until 4) for (t in 0 until 4) aM[r][t] += src[r] * src[t]
            for (r in 0 until 4) for (t in 0 until 3) rhs[r][t] += src[r] * dst[t]
        }
        val B = solve4(aM, rhs)   // 4×3
        if (B == null) return null
        val r1 = doubleArrayOf(B[0][0], B[1][0], B[2][0])
        val r2 = doubleArrayOf(B[0][1], B[1][1], B[2][1])
        val n1 = sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2])
        val n2 = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2])
        if (n1 < 1e-9f || n2 < 1e-9f) return null
        val r1n = doubleArrayOf(r1[0] / n1, r1[1] / n1, r1[2] / n1)
        val r2n = doubleArrayOf(r2[0] / n2, r2[1] / n2, r2[2] / n2)
        val r3n = cross3(r1n, r2n)
        var nx = r3n[0].toFloat(); var ny = r3n[1].toFloat(); var nz = r3n[2].toFloat()
        val nl = sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-6f); nx /= nl; ny /= nl; nz /= nl
        if (nz < 0f) { nx = -nx; ny = -ny; nz = -nz }
        val rx = r1n[0].toFloat(); val ry = r1n[1].toFloat(); val rz = r1n[2].toFloat()
        val ux = r2n[0].toFloat(); val uy = r2n[1].toFloat(); val uz = r2n[2].toFloat()
        // 深度：眼距（右/左眼中心）
        val rightEye = eyeCent(lmk, 36, 41); val leftEye = eyeCent(lmk, 42, 47)
        val ipdPx = hypot(rightEye[0] - leftEye[0], rightEye[1] - leftEye[1]).coerceAtLeast(1f)
        val focal = (h / 2f) / tan(Math.toRadians(25.0)).toFloat()
        val depth = IPD_CM * focal / ipdPx
        val bx = (bbox[0] + bbox[2]) / 2f; val by = (bbox[1] + bbox[3]) / 2f
        val wx = (bx - w / 2f) / focal * depth
        val wy = (h / 2f - by) / focal * depth
        val wz = depth
        val d3 = 3.0f * IPD_CM
        val px3 = wx + nx * d3; val py3 = wy + ny * d3; val pz3 = wz + nz * d3
        val out = floatArrayOf(rx, ry, rz, ux, uy, uz, nx, ny, nz, px3, py3, pz3)
        for (v in out) if (!v.isFinite()) return null
        return out
    }

    private fun eyeCent(l: FloatArray, i0: Int, i1: Int): FloatArray {
        var sx = 0f; var sy = 0f
        for (i in i0..i1) { sx += l[i * 3]; sy += l[i * 3 + 1] }
        val n = (i1 - i0 + 1)
        return floatArrayOf(sx / n, sy / n)
    }

    private fun cross3(a: DoubleArray, b: DoubleArray): DoubleArray =
        doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    /** 高斯消元解 4×3（pivot）。返回 null 奇异。 */
    private fun solve4(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray>? {
        val n = 4
        val m = Array(n) { a[it].copyOf() }
        val x = Array(n) { java.util.Arrays.copyOf(b[it], 3) }
        for (col in 0 until n) {
            var piv = col; var best = abs(m[col][col])
            for (r in col + 1 until n) { val v = abs(m[r][col]); if (v > best) { best = v; piv = r } }
            if (best < 1e-12) return null
            if (piv != col) { val t = m[col]; m[col] = m[piv]; m[piv] = t; val tb = x[col]; x[col] = x[piv]; x[piv] = tb }
            val pivVal = m[col][col]
            for (c in 0 until n) m[col][c] /= pivVal
            for (c in 0 until 3) x[col][c] /= pivVal
            for (r in 0 until n) if (r != col) {
                val f = m[r][col]
                if (abs(f) < 1e-15) continue
                for (c in 0 until n) m[r][c] -= f * m[col][c]
                for (c in 0 until 3) x[r][c] -= f * x[col][c]
            }
        }
        return x
    }

    private fun matFloat(m: Mat): FloatArray {
        val a = FloatArray((m.total() * m.channels()).toInt().coerceAtMost(64_000_000))
        m.get(0, 0, a)
        return a
    }

    /** 4D blob → 1 维 Float32（reshape 后读取，避免 4D Mat.get 丢数据）。 */
    private fun blobF(m: Mat): FloatArray {
        val one = m.reshape(1, 1)
        val a = FloatArray((one.total() * one.channels()).toInt().coerceAtMost(64_000_000))
        one.get(0, 0, a)
        return a
    }
}