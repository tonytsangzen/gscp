package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
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
 * 相机画面与 overlay 平面在同一 GLSurfaceView 内渲染（单 Surface，无分层）。
 * 稳定判定：人脸连续追踪约 0.5s 才锚定；丢失约 1s 后隐藏平面。
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
    private var rotationQuadrant = 1
    private var mirror = true

    // 可调参数
    private var distanceCm = 40      // 平面目标深度（厘米，20..120）
    private var sizePercent = 100    // 平面大小系数（50..200，基准宽 14cm）
    private var audioEnabled = true

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
        rotationQuadrant = prefs.getInt("arRotationQuadrant", 1)
        mirror = prefs.getBoolean("arMirror", true)

        renderer = ArOverlayRenderer().apply {
            planeDistance = distanceCm.toFloat()
            planeScale = sizePercent / 100f
            outputRotationQuadrant = rotationQuadrant
            outputMirror = mirror
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
        findViewById<Button>(R.id.button_rotate).setOnClickListener {
            rotationQuadrant = (rotationQuadrant + 1) % 4
            renderer.outputRotationQuadrant = rotationQuadrant
            prefs.edit().putInt("arRotationQuadrant", rotationQuadrant).apply()
            statusText.text = "旋转=${rotationQuadrant * 90}° 镜像=${if (mirror) "开" else "关"}"
        }
        findViewById<Button>(R.id.button_mirror).setOnClickListener {
            mirror = !mirror
            renderer.outputMirror = mirror
            prefs.edit().putBoolean("arMirror", mirror).apply()
            statusText.text = "旋转=${rotationQuadrant * 90}° 镜像=${if (mirror) "开" else "关"}"
        }
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
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }
                provider.unbindAll()
                val camera = provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis,
                )
                // 相机传感器与竖屏显示的旋转差：背景与 overlay 投影同象限旋转，
                // 保证画面方向正确且锚定不错位。
                val delta = camera.cameraInfo.getSensorRotationDegrees(0)
                renderer.outputRotationQuadrant = (delta / 90) % 4
                renderer.outputMirror = true
                statusText.text = "等待人脸…"
            } catch (e: Exception) {
                statusText.text = "相机启动失败: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── MediaPipe 人脸追踪 ────────────────────────────────────

    private fun ensureLandmarker(): FaceLandmarker {
        landmarker?.let { return it }
        val base = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(3)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setOutputFacialTransformationMatrixes(true)
            .setResultListener { result, _ ->
                val now = System.currentTimeMillis()
                synchronized(lock) {
                    if (result.faceLandmarks().isNotEmpty()) {
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
                                    renderer.faceMatrix = matrices[0]
                                    val m = matrices[0]
                                    if (now - lastMatrixLogAt > 500) {
                                        lastMatrixLogAt = now
                                        android.util.Log.i(
                                            "gscp-ar",
                                            "face n=(%.2f,%.2f,%.2f) t=(%.2f,%.2f,%.2f)"
                                                .format(m[8], m[9], m[10], m[12], m[13], m[14]),
                                        )
                                    }
                                }
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
            val bitmap = image.toBitmap()
            lm.detectAsync(BitmapImageBuilder(bitmap).build(), timestamp)
        } catch (_: Exception) {
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
        connection = ScrcpyConnection(this, audioEnabled = audioEnabled, overlayOnly = true)
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
        landmarker?.close()
        landmarker = null
        super.onDestroy()
    }
}
