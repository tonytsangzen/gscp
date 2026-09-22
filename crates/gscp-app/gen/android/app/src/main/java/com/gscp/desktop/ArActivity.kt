package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.Matrix
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
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.objdetect.FaceDetectorYN
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.tan

/**
 * AR 试验模式：仅拉取眼镜 overlay 流，手机摄像头拍摄现实画面，
 * YuNet（OpenCV FaceDetectorYN）检测人脸 + 5 点 DLT-PnP 解算头部位姿，
 * 把 overlay 作为虚拟平面绘制在第一张稳定追踪人脸的正前方
 * （平面法线与人脸法线重合，距离/大小可调）。
 *
 * 小脸优化：分析流 1280×720 + 动态 ROI 裁剪（锁定后分析区域放大到人脸附近，
 * 位姿在正立显示坐标系下解算，直接可用于渲染）。
 */
class ArActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    private lateinit var settingsPanel: View
    private lateinit var arPanel: View
    private lateinit var glSurface: android.opengl.GLSurfaceView
    private lateinit var statusText: TextView
    private lateinit var progressView: View
    private lateinit var ipEdit: EditText
    private lateinit var renderer: ArOverlayRenderer
    private var cameraSurface: Surface? = null

    private var connection: ScrcpyConnection? = null
    private var overlayDecoder: VideoDecoder? = null
    private var detector: FaceDetectorYN? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var playing = false
    private var lastFaceAtMillis = 0L
    private var firstFaceAtMillis = 0L
    private var faceLocked = false
    private var lastMatrixLogAt = 0L
    private var detectCount = 0
    private val lock = Any()

    // 动态 ROI（正立显示空间归一化 [0,1]）：小脸时裁剪放大分析区域
    private val roiRect = RectF(0f, 0f, 1f, 1f)
    private var roiActive = false
    private var roiLostFrames = 0

    // 可调参数
    private var distanceCm = 40      // 平面目标深度（厘米，20..120）
    private var sizePercent = 100    // 平面大小系数（50..200，基准宽 14cm）

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_ar)
        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        // OpenCV 本地库（AAR 自带）
        OpenCVLoader.initLocal()

        settingsPanel = findViewById(R.id.settings_panel)
        arPanel = findViewById(R.id.ar_panel)
        glSurface = findViewById(R.id.ar_gl_surface)
        statusText = findViewById(R.id.ar_status)
        progressView = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        val connectButton = findViewById<Button>(R.id.button_connect)

        distanceCm = prefs.getInt("arDistanceCm", 40)
        sizePercent = prefs.getInt("arSizePercent", 100)

        renderer = ArOverlayRenderer().apply {
            planeDistance = distanceCm.toFloat()
            planeScale = sizePercent / 100f
        }
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
        findViewById<Button>(R.id.button_test).setOnClickListener { startTestTracking() }
        findViewById<Button>(R.id.button_exit).setOnClickListener { exitAr() }
        bindSeekBar(R.id.ar_distance, distanceCm, 20, 120) { v ->
            distanceCm = v
            renderer.planeDistance = v.toFloat()
            prefs.edit().putInt("arDistanceCm", v).apply()
        }
        bindSeekBar(R.id.ar_size, sizePercent, 50, 200) { v ->
            sizePercent = v
            renderer.planeScale = v / 100f
            prefs.edit().putInt("arSizePercent", v).apply()
        }

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
                                    android.util.Size(1280, 720),
                                    androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis,
                )
                statusText.text = "等待人脸…"
            } catch (e: Exception) {
                statusText.text = "相机启动失败: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── YuNet 人脸检测 + DLT-PnP 位姿（动态 ROI）─────────────

    private fun ensureModelFile(): String {
        val f = File(filesDir, "face_detection_yunet_2023mar.onnx")
        if (!f.exists()) {
            assets.open("face_detection_yunet_2023mar.onnx").use { input ->
                java.io.FileOutputStream(f).use { output -> input.copyTo(output) }
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
            // 正立显示空间：传感器横向帧顺时针旋转 90°
            val m = Matrix().apply { postRotate(90f) }
            val upright = Bitmap.createBitmap(full, 0, 0, full.width, full.height, m, true)
            if (upright !== full) full.recycle()
            val W = upright.width
            val H = upright.height

            // 动态 ROI 裁剪（正立空间）
            val bmp: Bitmap
            var cropX = 0
            var cropY = 0
            if (roiActive) {
                cropX = (roiRect.left * W).toInt().coerceIn(0, W - 2)
                cropY = (roiRect.top * H).toInt().coerceIn(0, H - 2)
                val cw = (roiRect.width() * W).toInt().coerceIn(16, W - cropX)
                val ch = (roiRect.height() * H).toInt().coerceIn(16, H - cropY)
                bmp = Bitmap.createBitmap(upright, cropX, cropY, cw, ch)
                if (bmp !== upright) upright.recycle()
            } else {
                bmp = upright
            }

            val detector = ensureDetector(bmp.width, bmp.height)
            val bgr = Mat()
            Utils.bitmapToMat(bmp, bgr)
            // FaceDetectorYN 要求 BGR 3 通道；bitmapToMat 产出 RGBA 4 通道
            org.opencv.imgproc.Imgproc.cvtColor(
                bgr, bgr, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR,
            )
            val faces = Mat()
            detector.detect(bgr, faces)
            detectCount++
            if (detectCount % 30 == 1) {
                android.util.Log.i(
                    "gscp-ar",
                    "detect#" + detectCount + " " + bmp.width + "x" + bmp.height +
                        " faces=" + faces.rows() + " roi=" + roiActive,
                )
            }
            if (bmp !== upright) bmp.recycle()
            upright.recycle()

            // 取置信度最高的人脸（行 = [x,y,w,h, 右眼,左眼,鼻尖,右嘴,左嘴, score]）
            var best = -1
            var bestScore = 0.0
            for (r in 0 until faces.rows()) {
                val score = faces.get(r, 14)[0]
                if (score > bestScore) { bestScore = score; best = r }
            }
            val hasFace = best >= 0
            synchronized(lock) {
                if (hasFace) {
                    roiLostFrames = 0
                    val g = { c: Int -> faces.get(best, c)[0] }
                    // 裁剪坐标 → 正立全帧坐标
                    val bx = g(0) + cropX; val by = g(1) + cropY
                    val bw = g(2); val bh = g(3)
                    val ptsPx = listOf(
                        Pair(g(4) + cropX, g(5) + cropY),
                        Pair(g(6) + cropX, g(7) + cropY),
                        Pair(g(8) + cropX, g(9) + cropY),
                        Pair(g(10) + cropX, g(11) + cropY),
                        Pair(g(12) + cropX, g(13) + cropY),
                    )

                    val now = System.currentTimeMillis()
                    if (firstFaceAtMillis == 0L) firstFaceAtMillis = now
                    lastFaceAtMillis = now
                    if (!faceLocked && now - firstFaceAtMillis >= 500) {
                        faceLocked = true
                        runOnUiThread { statusText.text = "已锁定人脸" }
                    }

                    if (faceLocked) {
                        // 针孔模型（正立空间）：垂直 FOV 50°
                        val focal = (H / 2.0) / tan(Math.toRadians(25.0))
                        val distance = (focal * 15.0 / bw).coerceIn(10.0, 300.0)

                        // 位置：脸中心（bbox 中心）在深度 distance 的视线上
                        val cxN = (bx + bw / 2.0) / W
                        val cyN = (by + bh / 2.0) / H
                        val halfHcm = distance * tan(Math.toRadians(25.0))
                        val halfWcm = halfHcm * W / H
                        val tx = (cxN - 0.5) * 2 * halfWcm
                        val ty = (0.5 - cyN) * 2 * halfHcm
                        val tz = -distance

                        // 朝向（5 点几何启发式）：
                        // roll = 眼线角度；yaw = 鼻尖水平偏移；pitch = 鼻尖垂直偏移
                        val (rex, rey) = ptsPx[0]
                        val (lex, ley) = ptsPx[1]
                        val (nx, ny) = ptsPx[2]
                        val eyeMidX = (rex + lex) / 2.0
                        val eyeMidY = (rey + ley) / 2.0
                        val facePxW = maxOf(abs(lex - rex), 1.0)
                        val roll = atan2(ley - rey, lex - rex)
                        val yaw = kotlin.math.asin(
                            ((nx - eyeMidX) / facePxW * 1.8).coerceIn(-1.0, 1.0),
                        )
                        val pitch = kotlin.math.asin(
                            (((ny - eyeMidY) / facePxW) * 1.5).coerceIn(-1.0, 1.0),
                        )

                        // GL 旋转矩阵 R = Ry(yaw)·Rx(pitch)·Rz(roll)（列主序）
                        val cy2 = kotlin.math.cos(yaw); val sy2 = kotlin.math.sin(yaw)
                        val cp = kotlin.math.cos(pitch); val sp = kotlin.math.sin(pitch)
                        val cr = kotlin.math.cos(roll); val sr = kotlin.math.sin(roll)
                        val ry = floatArrayOf(
                            cy2.toFloat(), 0f, -sy2.toFloat(), 0f,
                            0f, 1f, 0f, 0f,
                            sy2.toFloat(), 0f, cy2.toFloat(), 0f,
                            0f, 0f, 0f, 1f,
                        )
                        val rx = floatArrayOf(
                            1f, 0f, 0f, 0f,
                            0f, cp.toFloat(), sp.toFloat(), 0f,
                            0f, -sp.toFloat(), cp.toFloat(), 0f,
                            0f, 0f, 0f, 1f,
                        )
                        val rz = floatArrayOf(
                            cr.toFloat(), sr.toFloat(), 0f, 0f,
                            -sr.toFloat(), cr.toFloat(), 0f, 0f,
                            0f, 0f, 1f, 0f,
                            0f, 0f, 0f, 1f,
                        )
                        fun mul3(a: FloatArray, b: FloatArray): FloatArray {
                            val o = FloatArray(16)
                            for (col in 0 until 4) for (row in 0 until 4) {
                                var sum = 0f
                                for (k in 0 until 4) sum += a[k * 4 + row] * b[col * 4 + k]
                                o[col * 4 + row] = sum
                            }
                            return o
                        }
                        val rot = mul3(ry, mul3(rx, rz))
                        val matrix = FloatArray(16)
                        for (i in 0 until 16) matrix[i] = rot[i]
                        matrix[12] = tx.toFloat()
                        matrix[13] = ty.toFloat()
                        matrix[14] = tz.toFloat()
                        renderer.faceMatrix = matrix

                        if (now - lastMatrixLogAt > 500) {
                            lastMatrixLogAt = now
                            android.util.Log.i(
                                "gscp-ar",
                                "face d=%.0fcm t=(%.1f,%.1f,%.1f) ypr=(%.2f,%.2f,%.2f)".format(
                                    distance, tx, ty, tz, yaw, pitch, roll,
                                ),
                            )
                        }
                    }

                    // 更新 ROI（正立空间归一化，方形 3 倍人脸宽度）
                    val faceWN = (bw / W).coerceAtLeast(0.02)
                    val side = (faceWN * 3.0).coerceIn(0.15, 1.0)
                    val cx = bx / W
                    val cy = by / H
                    val rx0 = (cx - side / 2).coerceIn(0.0, 1.0 - side)
                    val ry0 = (cy - side / 2).coerceIn(0.0, 1.0 - side)
                    roiRect.set(rx0.toFloat(), ry0.toFloat(), (rx0 + side).toFloat(), (ry0 + side).toFloat())
                    roiActive = true
                } else {
                    firstFaceAtMillis = 0L
                    if (roiActive) {
                        roiLostFrames++
                        if (roiLostFrames > 12) {
                            roiRect.set(0f, 0f, 1f, 1f)
                            roiActive = false
                            roiLostFrames = 0
                        }
                    }
                    if (faceLocked && now2() - lastFaceAtMillis > 1000) {
                        faceLocked = false
                        renderer.faceMatrix = null
                        runOnUiThread { statusText.text = "等待人脸…" }
                    }
                }
            }
            faces.release()
            bgr.release()
        } catch (e: Exception) {
            android.util.Log.w("gscp-ar", "analyze failed", e)
        } finally {
            image.close()
        }
    }

    private fun now2(): Long = System.currentTimeMillis()

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

    /** 无眼镜：跳过连接，用半透明测试图层验证人脸追踪与锚定。 */
    private fun startTestTracking() {
        renderer.useTestPattern = true
        renderer.overlayAspect = 4f / 3f
        enterArScreen("测试模式：请正对手机摄像头")
    }

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
            renderer.overlayAspect = width.toFloat() / height
            runOnUiThread {
                statusText.text = "overlay ${width}×${height}，请正对手机摄像头"
                overlayDecoder?.start(width, height, renderer.getOverlaySurface())
            }
        }

        override fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int) {
            overlayDecoder?.decode(buffer, offset, length)
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

    private fun bindSeekBar(id: Int, initial: Int, min: Int, max: Int, onChange: (Int) -> Unit) {
        val bar = findViewById<SeekBar>(id)
        bar.max = max
        bar.progress = initial.coerceIn(min, max)
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(b: SeekBar, value: Int, fromUser: Boolean) {
                onChange(value.coerceIn(min, max))
            }

            override fun onStartTrackingTouch(b: SeekBar) {}
            override fun onStopTrackingTouch(b: SeekBar) {}
        })
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
}
