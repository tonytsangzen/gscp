package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Patterns
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
 * 模型输入 = 屏幕显示：同一张分析 Bitmap（旋转到正立显示方向）既渲染背景
 * 又喂给 YuNet，锚定与画面像素级一致。
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
            if (granted) startCamera()
        }

    private fun requestCameraPermission() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                // 仅 ImageAnalysis：分析帧同时用于 YuNet 与背景渲染（数据一致）
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                androidx.camera.core.resolutionselector.ResolutionStrategy(
                                    android.util.Size(720, 1280),
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
                    analysis,
                )
                statusText.text = "等待人脸…"
            } catch (e: Exception) {
                statusText.text = "相机启动失败: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── YuNet 人脸检测 + DLT-PnP 位姿 ─────────────────────────

    private fun ensureModelFile(): String {
        val f = File(filesDir, "face_detection_yunet_2023mar.onnx")
        if (!f.exists()) {
            assets.open("face_detection_yunet_2023mar.onnx").use { input ->
                val out = java.io.FileOutputStream(f)
                input.copyTo(out)
                out.close()
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
            // 正立显示空间：传感器横向帧顺时针旋转 90°（与屏幕显示方向一致）
            val m = Matrix().apply { postRotate(90f) }
            val upright = Bitmap.createBitmap(full, 0, 0, full.width, full.height, m, true)
            if (upright !== full) full.recycle()
            val W = upright.width
            val H = upright.height

            // 同一 Bitmap：先渲染背景（GL 线程上传），再喂 YuNet（数据一致）
            renderer.latestFrame = upright

            val detector = ensureDetector(upright.width, upright.height)
            val bgr = Mat()
            Utils.bitmapToMat(upright, bgr)
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
                    "detect#" + detectCount + " " + upright.width + "x" + upright.height +
                        " faces=" + faces.rows(),
                )
            }

            // 取置信度最高的人脸（行 = [x,y,w,h, 右眼,左眼,鼻尖,右嘴,左嘴, score]）
            var best = -1
            var bestScore = 0.0
            for (r in 0 until faces.rows()) {
                val score = faces.get(r, 14)[0]
                if (score > bestScore) { bestScore = score; best = r }
            }
            val hasFace = best >= 0
            synchronized(lock) {
                val now = System.currentTimeMillis()
                if (hasFace) {
                    val g = { c: Int -> faces.get(best, c)[0] }
                    val bx = g(0); val by = g(1)
                    val bw = g(2); val bh = g(3)
                    val ptsPx = listOf(
                        Pair(g(4), g(5)),
                        Pair(g(6), g(7)),
                        Pair(g(8), g(9)),
                        Pair(g(10), g(11)),
                        Pair(g(12), g(13)),
                    )

                    if (firstFaceAtMillis == 0L) firstFaceAtMillis = now
                    lastFaceAtMillis = now
                    // 稳定判定：连续追踪 ~0.5s 后锚定第一张人脸
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
                        // 朝向（5 点几何启发式）：
                        // roll = 眼线角度；yaw = 鼻尖水平偏移；pitch = 鼻尖垂直偏移
                        val (rex, rey) = ptsPx[0]
                        val (lex, ley) = ptsPx[1]
                        val (nx, ny) = ptsPx[2]
                        val eyeMidX = (rex + lex) / 2.0
                        val eyeMidY = (rey + ley) / 2.0
                        val facePxW = maxOf(abs(lex - rex), 1.0)
                        val roll = kotlin.math.atan2(ley - rey, lex - rex)
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
                        val rym = floatArrayOf(
                            cy2.toFloat(), 0f, -sy2.toFloat(), 0f,
                            0f, 1f, 0f, 0f,
                            sy2.toFloat(), 0f, cy2.toFloat(), 0f,
                            0f, 0f, 0f, 1f,
                        )
                        val rxm = floatArrayOf(
                            1f, 0f, 0f, 0f,
                            0f, cp.toFloat(), sp.toFloat(), 0f,
                            0f, -sp.toFloat(), cp.toFloat(), 0f,
                            0f, 0f, 0f, 1f,
                        )
                        val rzm = floatArrayOf(
                            cr.toFloat(), sr.toFloat(), 0f, 0f,
                            -sr.toFloat(), cr.toFloat(), 0f, 0f,
                            0f, 0f, 1f, 0f,
                            0f, 0f, 0f, 1f,
                        )
                        val matrix = FloatArray(16)
                        // rot = Ry·Rx·Rz
                        for (col in 0 until 4) {
                            for (row in 0 until 4) {
                                var sum = 0f
                                for (k in 0 until 4) {
                                    var sum2 = 0f
                                    for (kk in 0 until 4) {
                                        sum2 += rym[kk * 4 + row] * rxm[col * 4 + kk]
                                    }
                                    sum += sum2 * rzm[col * 4 + k]
                                }
                                matrix[col * 4 + row] = sum
                            }
                        }
                        // 位置：脸中心（厘米，深度 distance 的视线上）
                        val halfHcm2 = distance * tan(Math.toRadians(25.0)).toFloat()
                        val halfWcm2 = halfHcm2 * W / H
                        matrix[12] = ((cxN - 0.5) * 2 * halfWcm2).toFloat()
                        matrix[13] = ((0.5 - cyN) * 2 * halfHcm2).toFloat()
                        matrix[14] = (-distance).toFloat()
                        renderer.faceMatrix = matrix

                        if (now - lastMatrixLogAt > 500) {
                            lastMatrixLogAt = now
                            android.util.Log.i(
                                "gscp-ar",
                                "face d=%.0fcm t=(%.1f,%.1f,%.1f)".format(
                                    distance, matrix[12], matrix[13], matrix[14],
                                ),
                            )
                        }
                    }
                } else {
                    firstFaceAtMillis = 0L
                    if (faceLocked && now - lastFaceAtMillis > 1000) {
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
