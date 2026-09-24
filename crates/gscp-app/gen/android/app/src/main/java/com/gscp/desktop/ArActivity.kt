package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.RectF
import android.os.Bundle
import android.util.Patterns
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import android.graphics.Bitmap
import android.graphics.Matrix
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.Size
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.tan

/**
 * AR 试验模式：仅拉取眼镜 overlay 流，手机摄像头拍摄现实画面，
 * YuNet（OpenCV FaceDetectorYN）检测人脸 + 5 点几何解算头部位姿，
 * 把 overlay 作为虚拟平面绘制在第一张稳定追踪人脸的正前方
 * （平面法线与人脸法线重合，距离/大小可调）。
 *
 * 检测与渲染严格对齐：分析帧顺时针旋转 90° 到正立显示方向后喂给 YuNet；
 * 背景 Preview 流经同一旋转（逆变换 + SurfaceTexture 变换矩阵）渲染。
 */
class ArActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    private lateinit var settingsPanel: View
    private lateinit var arPanel: View
    private lateinit var glSurface: android.opengl.GLSurfaceView
    private lateinit var statusText: TextView
    private lateinit var poseHud: TextView
    private lateinit var progressView: View
    private lateinit var ipEdit: EditText
    private lateinit var renderer: ArOverlayRenderer
    private var cameraSurface: Surface? = null

    private var connection: ScrcpyConnection? = null
    private var overlayDecoder: VideoDecoder? = null
    private var detector: FaceDetectorYN? = null
    /** insight_procrustes_coreml 管线（SCRFD det_10g → 1k3d68 → Procrustes） */
    private var insight: InsightPose? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var playing = false
    private var lastFaceAtMillis = 0L
    private var firstFaceAtMillis = 0L
    private var faceLocked = false
    private var lastMatrixLogAt = 0L
    private var hudLogAt = 0L
    // 速率统计（FPS + 原生引擎单帧耗时）
    private var fpsFrames = 0
    private var fpsStartUp = 0L
    @Volatile private var fpsNow = 0f
    private var natMsSum = 0L
    private var natMsCnt = 0
    @Volatile private var natMsAvg = 0f
    private var stabLogAt = 0L
    private var detectCount = 0
    private val lock = Any()

    // 可调参数已全部移除（大小改为按眼距自动推算、深度固定）。
    /** 检测工作尺寸：分析帧缩放到此长边后喂给 YuNet（面积≈该值²，大幅提速）。 */
    private val detectMaxDim = 640

    /** overlay 宽度 = 实测眼距 × 此倍数（约 2.2~2.4 时覆盖整个脸部宽度）。 */
    private val OVERLAY_EYE_MULT = 2.2f

    // 人脸防抖：一阶低通（EMA）。alpha 越小越稳（但滞后越大）、越大越跟手。
    // 首帧锚定或丢脸后重置，避免从头带旧值造成明显滞后。
    private val SMOOTH_ALPHA = 0.3f

    /** EPNP 姿态 EMA 系数（更小 = 更平滑；0.06 强防抖）。 */
    private val POSE_SMOOTH_ALPHA = 0.06f

    /** 瞳距（cm），金字塔/3 瞳距球半径的统一基准。 */
    private val IPD_CM = 6.2
    private var filterInit = false
    private var sCx = 0.5
    private var sCy = 0.5
    private var sDist = 100.0
    private var sRoll = 0.0
    private var sSize = 14.0

    /** EPNP 解算输出的 EMA 平滑状态（位置 + 9 维姿态基）。 */
    private var poseInit = false
    private var sPx = 0f
    private var sPy = 0f
    private var sPz = 0f
    private val sB = FloatArray(9)
    private var lastPoseMode = "heur"
    private var epnpFail = 0

    // 朝向符号校正（固定默认：图像 y 向下而 GL 渲染 y 向上，roll 需翻号）。
    private var invertYaw = false
    private var invertPitch = false
    private var invertRoll = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_ar)
        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        // OpenCV 本地库（AAR 自带）
        OpenCVLoader.initLocal()

        // insight_procrustes_coreml 模型（assets → filesDir，异步加载不阻塞相机分析）
        insight = InsightPose(assets, filesDir)
        insight?.let { ip ->
            Thread {
                try {
                    ip.ensureLoaded()
                    // 最小测试：模拟器（无相机）跑 det/lm 合成输入验证输出张量形状；真机走正常相机链路
                    if (android.os.Build.HARDWARE == "ranchu" && ip.isReady()) runSyntheticTest(ip)
                } catch (t: Throwable) {
                    android.util.Log.w("gscp-ar", "insight model init fail", t)
                }
            }.start()
        }

        settingsPanel = findViewById(R.id.settings_panel)
        arPanel = findViewById(R.id.ar_panel)
        glSurface = findViewById(R.id.ar_gl_surface)
        statusText = findViewById(R.id.ar_status)
        poseHud = findViewById(R.id.pose_hud)
        progressView = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        val connectButton = findViewById<Button>(R.id.button_connect)

        renderer = ArOverlayRenderer()
        renderer.onCameraSurfaceReady = { surface ->
            runOnUiThread {
                cameraSurface = surface
                if (checkSelfPermission(android.Manifest.permission.CAMERA) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
                ) {
                    startCamera(surface)
                }
            }
        }
        glSurface.setEGLContextClientVersion(2)
        glSurface.setRenderer(renderer)
        glSurface.renderMode = android.opengl.GLSurfaceView.RENDERMODE_CONTINUOUSLY

        ipEdit.setText(prefs.getString("ip", ""))
        connectButton.setOnClickListener { startAr() }
        findViewById<Button>(R.id.button_exit).setOnClickListener { exitAr() }

        requestCameraPermission()
    }

    // ── 相机权限与预览 ────────────────────────────────────────

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            statusText.text = if (granted) "相机就绪" else "未授予相机权限，无法追踪人脸"
            val surface = cameraSurface
            if (granted && surface != null) startCamera(surface)
        }

    private fun requestCameraPermission() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            cameraSurface?.let { startCamera(it) }
        } else {
            cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startCamera(surface: Surface) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build()
                preview.setSurfaceProvider { request ->
                    renderer.setCameraResolution(
                        request.resolution.width,
                        request.resolution.height,
                    )
                    request.provideSurface(
                        surface,
                        ContextCompat.getMainExecutor(this),
                    ) { }
                }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                androidx.camera.core.resolutionselector.ResolutionStrategy(
                                    android.util.Size(2160, 1620),
                                    androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                                )
                            )
                            .build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }
                provider.unbindAll()
                // 模拟器只有后置相机(virtualscene 有人脸场景)；真机用前置
                val isEmulator = android.os.Build.HARDWARE == "ranchu"
                val camSel = if (isEmulator)
                    CameraSelector.DEFAULT_BACK_CAMERA
                else
                    CameraSelector.DEFAULT_FRONT_CAMERA
                provider.bindToLifecycle(
                    this,
                    camSel,
                    preview,
                    analysis,
                )
                statusText.text = "等待人脸…"
            } catch (e: Exception) {
                android.util.Log.e("gscp-ar", "camera start fail", e)
                statusText.text = "相机启动失败: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── YuNet 人脸检测 + 位姿（正立显示空间）──────────────────

    private fun ensureModelFile(): String {
        val f = File(filesDir, "face_detection_yunet_2023mar.onnx")
        if (!f.exists()) {
            assets.open("face_detection_yunet_2023mar.onnx").use { input ->
                java.io.FileOutputStream(f).use { output ->
                    val buf = ByteArray(8192)
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        output.write(buf, 0, n)
                    }
                }
            }
        }
        return f.absolutePath
    }

    private fun ensureDetector(w: Int, h: Int): FaceDetectorYN {
        val existing = detector
        if (existing != null) {
            existing.setInputSize(Size(w.toDouble(), h.toDouble()))
            return existing
        }
        val d = FaceDetectorYN.create(
            ensureModelFile(), "", Size(w.toDouble(), h.toDouble()), 0.6f, 0.3f, 3,
        )
        detector = d
        return d
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val full = image.toBitmap()
            // 一次矩阵变换同时完成「旋转到正立 + 缩放到检测工作尺寸」：
            // 避免先整帧旋转拷贝、再整帧 bitmapToMat/cvtColor，检测输入面积降至
            // ≈ detectMaxDim²，YuNet 与像素拷贝耗时随之大幅下降 → 跟踪更跟手。
            // 坐标仍在同一帧空间内归一，焦点/距离均用同尺寸 H，缩放不改变位姿数值。
            val sx = full.width
            val sy = full.height
            val scale = (detectMaxDim / maxOf(sx, sy).toFloat()).coerceIn(0.0f, 1f)
            val m = Matrix()
            val deg = image.imageInfo.rotationDegrees.toFloat()
            if (deg != 0f) m.postRotate(deg)
            m.postScale(scale, scale)
            val work = Bitmap.createBitmap(full, 0, 0, sx, sy, m, true)
            if (work !== full) full.recycle()
            val W = work.width
            val H = work.height

            val detector = ensureDetector(W, H)
            val bgr = Mat()
            Utils.bitmapToMat(work, bgr)
            work.recycle()
            // FaceDetectorYN 要求 BGR 3 通道；bitmapToMat 产出 RGBA 4 通道
            org.opencv.imgproc.Imgproc.cvtColor(
                bgr, bgr, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR,
            )
            // —— 整条 pose 链在 native C++ 完成（det→SCRFD→landmark→Procrustes）——
            val ip = insight
            // 原生引擎单帧耗时 + FPS 统计
            val natT0 = android.os.SystemClock.uptimeMillis()
            val ins = if (ip?.isReady() == true) ip.pose(bgr) else null
            val natMs = (android.os.SystemClock.uptimeMillis() - natT0).toFloat()
            natMsSum += natMs.toLong(); natMsCnt++
            natMsAvg = natMsSum.toFloat() / maxOf(natMsCnt.toLong(), 1)
            fpsFrames++
            if (fpsStartUp == 0L) fpsStartUp = android.os.SystemClock.uptimeMillis()
            else if (android.os.SystemClock.uptimeMillis() - fpsStartUp >= 1000) {
                fpsNow = fpsFrames * 1000.0f / (android.os.SystemClock.uptimeMillis() - fpsStartUp)
                fpsFrames = 0; fpsStartUp = android.os.SystemClock.uptimeMillis()
            }
            detectCount++
            if (detectCount % 60 == 1) {
                android.util.Log.i(
                    "gscp-ar",
                    String.format(java.util.Locale.US,
                        "detect#%d %dx%d insight=%s fps=%.1f nat=%.0fms",
                        detectCount, W, H, if (ins != null) "hit" else "miss", fpsNow, natMsAvg),
                )
            }
            // nativePose: [0..8]=basis, [9..11]=pos3(cm), [12..25]=bbox14(全帧 px)
            val hasFace = ins != null
            synchronized(lock) {
                val now = java.lang.System.currentTimeMillis()
                if (hasFace) {
                    val p = ins!!
                    var bad = false
                    for (i in 0..11) if (p[i] != p[i]) { bad = true; break }
                    if (bad) { renderer.faceMatrix = null; return@synchronized }
                    // 稳定性诊断：每 200ms 打当前 pick 的人脸框中心，观察是否在多处跳
                    if (now - stabLogAt > 200) {
                        stabLogAt = now
                        val bcx = (p[12] + p[14]) / 2f
                        val bcy = (p[13] + p[15]) / 2f
                        android.util.Log.i(
                            "gscp-trk",
                            String.format(java.util.Locale.US,
                                "pick c=(%5.1f,%5.1f) s=(%4.0fx%4.0f) pos=(%6.1f,%6.1f,%6.1f) fps=%.1f",
                                bcx, bcy, p[14] - p[12], p[15] - p[13],
                                p[9], p[10], -p[11], fpsNow),
                        )
                    }
                    if (firstFaceAtMillis == 0L) firstFaceAtMillis = now
                    lastFaceAtMillis = now
                    if (!faceLocked && now - firstFaceAtMillis >= 500) {
                        faceLocked = true
                        poseInit = false   // 锚定时重置 EMA，避免旧值滞后
                        runOnUiThread { statusText.text = "已锁定人脸" }
                    }
                    if (faceLocked) {
                        // nativePose 输出：[0..8]=basis, [9..11]=pos3(cm, z 朝前)
                        val candBasis = FloatArray(9)
                        for (i in 0..8) candBasis[i] = p[i]
                        val cpx = p[9]; val cpy = p[10]; val cpz = -p[11]  // 渲染器负朝前系
                        // 唯一低通出口：EMA 平滑 native 残余抖动
                        if (!poseInit) {
                            sPx = cpx; sPy = cpy; sPz = cpz
                            for (i in 0..8) sB[i] = candBasis[i]
                            poseInit = true
                        } else {
                            val a = POSE_SMOOTH_ALPHA
                            sPx += (cpx - sPx) * a; sPy += (cpy - sPy) * a; sPz += (cpz - sPz) * a
                            for (i in 0..8) sB[i] += (candBasis[i] - sB[i]) * a
                        }
                        // 归一化 basis，保证法线朝相机
                        val rl = kotlin.math.sqrt(sB[0]*sB[0]+sB[1]*sB[1]+sB[2]*sB[2]).toFloat().coerceAtLeast(1e-6f)
                        sB[0]/=rl; sB[1]/=rl; sB[2]/=rl
                        val ul = kotlin.math.sqrt(sB[3]*sB[3]+sB[4]*sB[4]+sB[5]*sB[5]).toFloat().coerceAtLeast(1e-6f)
                        sB[3]/=ul; sB[4]/=ul; sB[5]/=ul
                        val nl = kotlin.math.sqrt(sB[6]*sB[6]+sB[7]*sB[7]+sB[8]*sB[8]).toFloat().coerceAtLeast(1e-6f)
                        sB[6]/=nl; sB[7]/=nl; sB[8]/=nl
                        if (sB[8] < 0f) { sB[6] = -sB[6]; sB[7] = -sB[7]; sB[8] = -sB[8] }
                        // 位置钳位在画面内
                        val halfW = (sPz * tan(Math.toRadians(25.0)) * 0.9f).toFloat().coerceAtLeast(6f)
                        sPx = sPx.coerceIn(-halfW, halfW)
                        sPy = sPy.coerceIn(-halfW * (H / W.toFloat()), halfW * (H / W.toFloat()))
                        sPz = sPz.coerceIn(-150f, -10f)
                        renderer.faceBasis = sB.copyOf()
                        val tm = FloatArray(16)
                        tm[12] = sPx; tm[13] = sPy; tm[14] = sPz; tm[15] = 1f
                        renderer.faceMatrix = tm
                        renderer.faceRoll = 0f
                        renderer.overlayWidthCm = 14f

                        // HUD：绘「原始 ncnn 引擎」输出的 pos + 姿态 + 速率统计
                        if (now - hudLogAt > 200) {
                            hudLogAt = now
                            val deg = 180.0 / kotlin.math.PI
                            val nYaw = kotlin.math.asin(p[6].toDouble().coerceIn(-1.0, 1.0)) * deg
                            val nPit = kotlin.math.asin(p[7].toDouble().coerceIn(-1.0, 1.0)) * deg
                            val nRol = kotlin.math.atan2(p[1].toDouble(), p[0].toDouble()) * deg
                            val txt = String.format(
                                java.util.Locale.US,
                                "ncnn  pos (%6.1f,%6.1f,%6.1f)cm\n" +
                                    "      yaw %+5.0f°  pitch %+5.0f°  roll %+5.0f°\n" +
                                    "rate  %4.1f fps · avg %4.1f ms/帧",
                                p[9], p[10], -p[11],
                                nYaw, nPit, nRol,
                                fpsNow, natMsAvg,
                            )
                            runOnUiThread { poseHud.text = txt }
                        }
                        if (now - lastMatrixLogAt > 500) {
                            lastMatrixLogAt = now
                            val msg = String.format(
                                java.util.Locale.US,
                                "face native d=%.0fcm t=(%.1f,%.1f,%.1f) n=(%.2f,%.2f,%.2f)",
                                -sPz, sPx, sPy, sPz, sB[6], sB[7], sB[8],
                            )
                            android.util.Log.i("gscp-ar", msg)
                        }
                    }
                } else {
                    firstFaceAtMillis = 0L
                    // 短暂(single-frame)miss 不清 overlay，避免闪烁；连续丢脸 >1.2s 才复位
                    if (faceLocked && now - lastFaceAtMillis > 1200) {
                        faceLocked = false
                        poseInit = false
                        renderer.faceMatrix = null
                        runOnUiThread { statusText.text = "等待人脸…" }
                        runOnUiThread { poseHud.text = "head: --" }
                    } else if (!faceLocked) {
                        renderer.faceMatrix = null
                    }
                }
            }
            bgr.release()
        } catch (e: Exception) {
            android.util.Log.w("gscp-ar", "analyze failed", e)
        } finally {
            image.close()
        }
    }

    // ── scrcpy overlay-only 连接 / 测试模式 ───────────────────

    private fun startAr() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this, audioEnabled = false, overlayOnly = true)
        connection!!.connectAsync(ip, 5555, callback)
        enterArScreen("连接眼镜中…")
    }

    /* 无眼镜测试模式已移除（含参考半透明平面）。 */

    private fun enterArScreen(status: String) {
        settingsPanel.visibility = View.GONE
        arPanel.visibility = View.VISIBLE
        progressView.visibility = View.VISIBLE
        statusText.text = status
        playing = true
    }

    private fun exitAr() {
        playing = false
        connection?.disconnect()
        runOnUiThread {
            progressView.visibility = View.GONE
            overlayDecoder?.stop()
            overlayDecoder = null
            faceLocked = false
            renderer.faceMatrix = null
            arPanel.visibility = View.GONE
            settingsPanel.visibility = View.VISIBLE
        }
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            runOnUiThread {
                progressView.visibility = View.GONE
                statusText.text = "已连接，等待 overlay…"
            }
        }

        override fun onVideoPrepare(codec: String, width: Int, height: Int) {}

        override fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int) {}

        override fun onAudioPrepare(codec: String, frameRate: Int, channel: Int) {}

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {}

        override fun onOverlayPrepare(codec: String, width: Int, height: Int) {
            // 叠加区域固定为正方形 480×480：宽高比用 1.0，避免发生拉伸。
            renderer.overlayAspect = 1f
            runOnUiThread {
                statusText.text = "overlay ${width}×${height}，请正对手机摄像头"
                try {
                    val surf = renderer.getOverlaySurfaceOrNull()
                    if (surf != null) {
                        overlayDecoder?.start(width, height, surf)
                    } else {
                        android.util.Log.w("gscp-ar", "overlay GL 未就绪，跳过解码（避免崩溃）")
                    }
                } catch (e: Exception) {
                    android.util.Log.w("gscp-ar", "overlay prepare 失败", e)
                }
            }
        }

        override fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int) {
            try {
                overlayDecoder?.decode(buffer, offset, length)
            } catch (e: Exception) {
                android.util.Log.w("gscp-ar", "overlay 解码中断", e)
            }
        }

        override fun onDisconnect() {
            if (playing) {
                playing = false
                runOnUiThread {
                    overlayDecoder?.stop()
                    overlayDecoder = null
                    faceLocked = false
                    renderer.faceMatrix = null
                    arPanel.visibility = View.GONE
                    settingsPanel.visibility = View.VISIBLE
                    statusText.text = "连接已断开"
                }
            }
        }

        override fun onError() {
            playing = false
            runOnUiThread {
                progressView.visibility = View.GONE
                statusText.text = "连接失败，请检查眼镜 IP 与网络"
                Toast.makeText(this@ArActivity, "连接失败", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ── 通用 ─────────────────────────────────────────────────

    /**
     * 用 5 点重建金字塔底面平面姿态（投影法）：
     * - 周围 4 点（眼/嘴）作为底面投影、中间点（鼻尖）作为顶点投影，顶点高度 = 0.5 瞳距；
     * - solvePnP 解出底面平面在相机系的姿态；
     * - 返回 [右,上,法线]（渲染器世界系、单位向量，法线朝向相机，9 个 float）
     *   与 overlay 平面中心（cm）——取「底面沿法线前移 3 个瞳距」；
     * - 失败返回 null（调用方回退到平行画面 roll 贴纸）。
     */
    private fun solveOverlayPose(
        rex: Double, rey: Double, lex: Double, ley: Double,
        nx: Double, ny: Double, rmx: Double, rmy: Double,
        lmx: Double, lmy: Double, focalPx: Double, w: Int, h: Int,
    ): Pair<FloatArray, FloatArray>? {
        val ipd = IPD_CM         // 瞳距（cm）
        val half = ipd / 2.0
        val noseH = ipd / 2.0    // 顶点（鼻尖）高度 = 0.5 瞳距（用户指定固定值）
        val noseY = ipd * 0.5    // 鼻尖在眼线下 0.5 瞳距（真实地标比例标定）
        val mouthHalf = ipd * 0.44
        val mouthY = -ipd * 0.95 // 嘴角到眼线垂直 = 0.95 瞳距（真实地标比例标定）
        val obj = MatOfPoint3f(
            Point3(half, 0.0, 0.0),             // 右眼
            Point3(-half, 0.0, 0.0),            // 左眼
            Point3(0.0, -noseY, noseH),          // 鼻尖 = 顶点（眼线下、朝相机 +z）
            Point3(mouthHalf, mouthY, 0.0),      // 右嘴角
            Point3(-mouthHalf, mouthY, 0.0),     // 左嘴角
        )
        val img = MatOfPoint2f(
            Point(rex, rey), Point(lex, ley), Point(nx, ny),
            Point(rmx, rmy), Point(lmx, lmy),
        )
        val cam = Mat(3, 3, CvType.CV_64FC1)
        cam.put(0, 0, focalPx, 0.0, w / 2.0)
        cam.put(1, 0, 0.0, focalPx, h / 2.0)
        cam.put(2, 0, 0.0, 0.0, 1.0)
        val dist = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)
        val rvec = Mat()
        val tvec = Mat()
        val ok = try {
            // EPNP：支持 4+ 点、无外部位姿初值（ITERATIVE 无初值时内部 DLT 需 ≥6 点，5 点会抛异常）
            Calib3d.solvePnP(obj, img, cam, dist, rvec, tvec, false, Calib3d.SOLVEPNP_EPNP)
        } catch (e: Throwable) {
            android.util.Log.w("gscp-ar", "solvePnP 异常", e)
            false
        }
        if (ok) {
            try {
                val rmat = Mat(3, 3, CvType.CV_64FC1)
                Calib3d.Rodrigues(rvec, rmat)
                // 抗「侧脸瞳距缩小被误读为距离变大」：
                // 用竖向眼-嘴跨度定深（yaw 不改变竖向投影长度），
                // depth = f·|mouthY·R[1][1]| / projV_px，再把横向分量按该深度等比修正。
                val projV =
                    kotlin.math.abs((rmy + lmy) / 2.0 - (rey + ley) / 2.0).coerceAtLeast(1.0)
                val rvY = kotlin.math.abs(rmat.get(1, 1)[0]).coerceAtLeast(0.05)
                val depth = focalPx * (ipd * 0.95) * rvY / projV
                val tz = tvec.get(2, 0)[0]
                val scale = if (tz > 1.0) depth / tz else 1.0
                // renderer 世界系：x 同向、y 向上(z=-sp.z)、z 朝相机。先把 tvec 折好转到渲染器
                val bx = tvec.get(0, 0)[0] * scale
                val by = -tvec.get(1, 0)[0] * scale
                val bz = -depth
                // 旋转矩阵各轴（模型: x右,y上,z朝相机；统一按渲染器 T=(x,-y,-z) 变换）
                var rx = rmat.get(0, 0)[0]; var ry = -rmat.get(1, 0)[0]; var rz = -rmat.get(2, 0)[0]
                var ux = rmat.get(0, 1)[0]; var uy = -rmat.get(1, 1)[0]; var uz = -rmat.get(2, 1)[0]
                var qx = rmat.get(0, 2)[0]; var qy = -rmat.get(1, 2)[0]; var qz = -rmat.get(2, 2)[0]
                // 法线朝相机（+渲染器 z）
                val ql = kotlin.math.sqrt(qx * qx + qy * qy + qz * qz).coerceAtLeast(1e-6)
                qx /= ql; qy /= ql; qz /= ql
                if (qz < 0) { qx = -qx; qy = -qy; qz = -qz }
                // 归一化右/上
                var rl = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz).coerceAtLeast(1e-6)
                rx /= rl; ry /= rl; rz /= rl
                var ul = kotlin.math.sqrt(ux * ux + uy * uy + uz * uz).coerceAtLeast(1e-6)
                ux /= ul; uy /= ul; uz /= ul
                // overlay 中心 = 底面中心 + 3 瞳距 × 法线（朝相机）
                val ox = bx + 3 * ipd * qx
                val oy = by + 3 * ipd * qy
                val oz = bz + 3 * ipd * qz
                rmat.release()
                rvec.release(); tvec.release(); cam.release(); dist.release()
                obj.release(); img.release()
                return Pair(
                    floatArrayOf(rx.toFloat(), ry.toFloat(), rz.toFloat(),
                                 ux.toFloat(), uy.toFloat(), uz.toFloat(),
                                 qx.toFloat(), qy.toFloat(), qz.toFloat()),
                    floatArrayOf(ox.toFloat(), oy.toFloat(), oz.toFloat()),
                )
            } catch (e: Throwable) {
                android.util.Log.w("gscp-ar", "pose 构造异常", e)
            }
        }
        android.util.Log.w("gscp-ar", "solvePnP 返回 false，回退平面")
        rvec.release(); tvec.release(); cam.release(); dist.release()
        obj.release(); img.release()
        return null
    }

    /** 列主序 4×4 矩阵乘法 out = a·b（与渲染器 multiply 同一约定）。 */
    private fun multiplyCm(a: FloatArray, b: FloatArray, out: FloatArray) {
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                out[col * 4 + row] = sum
            }
        }
    }

    @Deprecated("Deprecated in Java")
    @SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        if (playing) {
            exitAr()
        } else {
            finish()
        }
    }

    override fun onDestroy() {
        playing = false
        connection?.disconnect()
        analysisExecutor.shutdown()
        detector = null
        super.onDestroy()
    }

    /** 最小测试（无相机依赖）：合成输入直接跑 det/lm，验证各输出张量形状。 */
    private fun runSyntheticTest(ip: InsightPose) {
        try {
            val detN = 640 * 640 * 3
            val detData = FloatArray(detN)
            for (i in 0 until detN) detData[i] = ((i * 31) % 255).toFloat() - 127.5f
            // 中央放一个“脸状”亮块，便于检测分支有真实响应
            val half = 70
            for (y in 320 - half until 320 + half) for (x in 320 - half until 320 + half) {
                val b = ((y * 640 + x) * 3)
                detData[b] = 100f; detData[b + 1] = 80f; detData[b + 2] = 90f
            }
            val t0 = System.nanoTime()
            val tensors = NcnnEngine.runDet(detData)
            val dt = (System.nanoTime() - t0) / 1_000_000
            if (tensors == null) { android.util.Log.i("gscp-ar", "syntest det=null"); return }
            val sb = StringBuilder()
            for (k in NcnnEngine.DET_NAMES) {
                val v = tensors[k]
                sb.append(k).append(":").append(if (v == null) -1 else v.size).append(" ")
            }
            android.util.Log.i("gscp-ar", "syntest det ms=" + dt + " dims=" + sb.toString())

            val lmN = 192 * 192 * 3
            val lmData = FloatArray(lmN)
            for (i in 0 until lmN) lmData[i] = ((i * 17) % 255).toFloat() - 128f
            for (y in 96 - 40 until 96 + 40) for (x in 96 - 40 until 96 + 40) {
                val b = ((y * 192 + x) * 3)
                lmData[b] = 90f; lmData[b + 1] = 90f; lmData[b + 2] = 100f
            }
            val t1 = System.nanoTime()
            val pred = NcnnEngine.runLm(lmData)
            val lt = (System.nanoTime() - t1) / 1_000_000
            android.util.Log.i("gscp-ar", "syntest lm ms=" + lt + " size=" + (pred?.size ?: -1))
        } catch (t: Throwable) {
            android.util.Log.e("gscp-ar", "syntest fail", t)
        }
    }
}
