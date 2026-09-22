package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
import android.graphics.Bitmap
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
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.util.concurrent.Executors

/**
 * AR 试验模式：仅拉取眼镜 overlay 流，手机摄像头拍摄现实画面，
 * MediaPipe Face Landmarker 追踪人脸，把 overlay 作为虚拟平面绘制在
 * 第一张稳定追踪人脸的正前方（平面法线与人脸法线重合，距离/大小可调）。
 *
 * 小脸优化：追踪锁定后分析区域自动裁剪放大到人脸附近（动态 ROI），
 * 位姿矩阵经仿射逆变换映射回全帧坐标；丢失后回到全帧重新捕获。
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
    private var landmarker: FaceLandmarker? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var playing = false
    private var lastFaceAtMillis = 0L
    private var firstFaceAtMillis = 0L
    private var faceLocked = false
    private var lastMatrixLogAt = 0L
    private val lock = Any()

    // 动态 ROI（归一化 [0,1]）：小脸时裁剪放大分析区域
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

    // ── MediaPipe 人脸追踪（动态 ROI）─────────────────────────

    private fun ensureLandmarker(): FaceLandmarker {
        landmarker?.let { return it }
        val base = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(3)
            .setMinFaceDetectionConfidence(0.4f)
            .setMinFacePresenceConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            .setOutputFacialTransformationMatrixes(true)
            .setResultListener { result, _ ->
                val now = System.currentTimeMillis()
                synchronized(lock) {
                    val hasFace = result.faceLandmarks().isNotEmpty()
                    if (hasFace) {
                        roiLostFrames = 0
                        if (firstFaceAtMillis == 0L) firstFaceAtMillis = now
                        lastFaceAtMillis = now
                        // 稳定判定：连续追踪 ~0.5s 后锚定第一张人脸
                        if (!faceLocked && now - firstFaceAtMillis >= 500) {
                            faceLocked = true
                            runOnUiThread { statusText.text = "已锁定人脸" }
                        }
                        if (faceLocked) {
                            val matrix = result.facialTransformationMatrixes()
                            if (matrix.isPresent) {
                                val matrices = matrix.get()
                                if (matrices.isNotEmpty()) {
                                    renderer.faceMatrix = toFullScreenMatrix(matrices[0])
                                }
                            }
                        }
                        // 从首张人脸 landmark 更新 ROI（方形，含 3 倍人脸宽度余量）
                        updateRoiFromFace(result.faceLandmarks()[0])
                    } else {
                        firstFaceAtMillis = 0L
                        if (roiActive) {
                            roiLostFrames++
                            // ROI 内连续丢失：回到全帧重新捕获
                            if (roiLostFrames > 12) {
                                roiRect.set(0f, 0f, 1f, 1f)
                                roiActive = false
                                roiLostFrames = 0
                            }
                        }
                        if (faceLocked && now - lastFaceAtMillis > 1000) {
                            faceLocked = false
                            renderer.faceMatrix = null
                            runOnUiThread { statusText.text = "等待人脸…" }
                        }
                    }
                }
            }
            .setErrorListener { err ->
                runOnUiThread { statusText.text = "追踪错误: ${err.message}" }
            }
            .build()
        return FaceLandmarker.createFromOptions(this, options).also { landmarker = it }
    }

    private fun analyzeFrame(image: ImageProxy) {
        val lm = try {
            ensureLandmarker()
        } catch (e: Exception) {
            runOnUiThread { statusText.text = "初始化追踪失败: ${e.message}" }
            image.close()
            return
        }
        val timestamp = image.imageInfo.timestamp / 1_000_000
        try {
            val full = image.toBitmap()
            val bmp: Bitmap = if (roiActive) {
                val W = full.width.toFloat()
                val H = full.height.toFloat()
                val x = (roiRect.left * W).toInt().coerceIn(0, full.width - 2)
                val y = (roiRect.top * H).toInt().coerceIn(0, full.height - 2)
                val w = (roiRect.width() * W).toInt().coerceIn(16, full.width - x)
                val h = (roiRect.height() * H).toInt().coerceIn(16, full.height - y)
                Bitmap.createBitmap(full, x, y, w, h)
            } else {
                full
            }
            lm.detectAsync(BitmapImageBuilder(bmp).build(), timestamp)
            if (bmp !== full) bmp.recycle()
            full.recycle()
        } catch (_: Exception) {
        } finally {
            image.close()
        }
    }

    /** 从首张人脸 landmark（当前输入空间）更新 ROI：方形、3 倍人脸宽度余量。 */
    private fun updateRoiFromFace(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>) {
        if (landmarks.isEmpty()) return
        var minX = 1f; var maxX = 0f
        var minY = 1f; var maxY = 0f
        for (p in landmarks) {
            if (p.x() < minX) minX = p.x()
            if (p.x() > maxX) maxX = p.x()
            if (p.y() < minY) minY = p.y()
            if (p.y() > maxY) maxY = p.y()
        }
        // 裁剪空间 → 全帧空间
        val fx0 = roiRect.left + minX * roiRect.width()
        val fx1 = roiRect.left + maxX * roiRect.width()
        val fy0 = roiRect.top + minY * roiRect.height()
        val fy1 = roiRect.top + maxY * roiRect.height()
        val cx = (fx0 + fx1) / 2f
        val cy = (fy0 + fy1) / 2f
        val faceW = (fx1 - fx0).coerceAtLeast(0.02f)
        val side = (faceW * 3f).coerceIn(0.15f, 1f)
        val rx0 = (cx - side / 2).coerceIn(0f, 1f - side)
        val ry0 = (cy - side / 2).coerceIn(0f, 1f - side)
        roiRect.set(rx0, ry0, rx0 + side, ry0 + side)
        roiActive = true
    }

    /** 裁剪空间的位姿矩阵 → 全帧空间（ROI 平移/缩放的仿射逆）。 */
    private fun toFullScreenMatrix(mCrop: FloatArray): FloatArray {
        val iw = 1f / roiRect.width()
        val ih = 1f / roiRect.height()
        // C⁻¹ = [rw,0,0,rx; 0,rh,0,ry; 0,0,1,0; 0,0,0,1]（列主序）
        val cinv = floatArrayOf(
            iw, 0f, 0f, 0f,
            0f, ih, 0f, 0f,
            0f, 0f, 1f, 0f,
            roiRect.left, roiRect.top, 0f, 1f,
        )
        val out = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += cinv[k * 4 + row] * mCrop[col * 4 + k]
                }
                out[col * 4 + row] = sum
            }
        }
        return out
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
                statusText.text = "overlay ${'$'}{width}×${'$'}{height}，请正对手机摄像头"
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
        landmarker?.close()
        landmarker = null
        super.onDestroy()
    }
}
