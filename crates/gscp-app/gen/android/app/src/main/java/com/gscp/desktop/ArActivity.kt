package com.gscp.desktop

import android.annotation.SuppressLint
import android.content.SharedPreferences
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
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.util.concurrent.Executors

/**
 * AR 试验模式：仅拉取眼镜 overlay 流，用手机前摄拍摄现实画面，
 * MediaPipe Face Landmarker 追踪人脸，把 overlay 作为虚拟平面绘制在
 * 第一张稳定追踪人脸的正前方（平面法线与人脸法线重合，距离/大小可调）。
 *
 * 稳定判定：人脸需被连续追踪约 0.5 秒才锚定；丢失超过约 1 秒后隐藏平面。
 */
class ArActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences
    private lateinit var connectPanel: View
    private lateinit var arPanel: View
    private lateinit var glSurface: android.opengl.GLSurfaceView
    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var progressView: View
    private lateinit var ipEdit: EditText
    private lateinit var renderer: ArOverlayRenderer

    private var connection: ScrcpyConnection? = null
    private var overlayDecoder: VideoDecoder? = null
    private var landmarker: FaceLandmarker? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var playing = false
    private var lastFaceAtMillis = 0L
    private var firstFaceAtMillis = 0L
    private var faceLocked = false

    // 可调参数
    private var distancePercent = 40    // 20..120 → 归一化距离（×0.01）
    private var sizePercent = 100       // 50..200 → 平面大小系数
    private var flipNormal = false
    private var frontCamera = true

    private val lock = Any()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_ar)
        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        connectPanel = findViewById(R.id.settings_panel)
        arPanel = findViewById(R.id.ar_panel)
        glSurface = findViewById(R.id.ar_gl_surface)
        previewView = findViewById(R.id.ar_preview)
        statusText = findViewById(R.id.ar_status)
        progressView = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        val connectButton = findViewById<Button>(R.id.button_connect)

        distancePercent = prefs.getInt("arDistance", 40)
        sizePercent = prefs.getInt("arSize", 100)
        flipNormal = prefs.getBoolean("arFlipNormal", false)
        frontCamera = prefs.getBoolean("arFrontCamera", true)

        renderer = ArOverlayRenderer().apply {
            planeDistance = distancePercent / 100f
            planeScale = sizePercent / 100f
            this.flipNormal = this@ArActivity.flipNormal
        }
        glSurface.setEGLContextClientVersion(2)
        glSurface.setEGLConfigChooser(8, 8, 8, 8, 0, 0) // 透明背景
        glSurface.setRenderer(renderer)
        glSurface.renderMode = android.opengl.GLSurfaceView.RENDERMODE_CONTINUOUSLY

        ipEdit.setText(prefs.getString("ip", ""))
        connectButton.setOnClickListener { startAr() }
        findViewById<Button>(R.id.button_exit).setOnClickListener { exitAr() }
        findViewById<Button>(R.id.button_flip).setOnClickListener {
            flipNormal = !flipNormal
            renderer.flipNormal = flipNormal
            prefs.edit().putBoolean("arFlipNormal", flipNormal).apply()
        }
        bindSeekBar(R.id.ar_distance, distancePercent, 20, 120) { v ->
            distancePercent = v
            renderer.planeDistance = v / 100f
            prefs.edit().putInt("arDistance", v).apply()
        }
        bindSeekBar(R.id.ar_size, sizePercent, 50, 200) { v ->
            sizePercent = v
            renderer.planeScale = v / 100f
            prefs.edit().putInt("arSize", v).apply()
        }

        requestCameraPermission()
    }

    // ── 相机权限与预览 ────────────────────────────────────────

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            statusText.text = if (granted) "相机就绪" else "未授予相机权限，无法追踪人脸"
        }

    @SuppressLint("MissingPermission")
    private fun requestCameraPermission() {
        val granted = checkSelfPermission(android.Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        if (granted) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }
                provider.unbindAll()
                try {
                    bindCamera(provider, preview, analysis, frontCamera)
                } catch (e: Exception) {
                    // 部分机型前摄枚举异常：回退另一颗相机
                    statusText.text = "切换相机重试…"
                    bindCamera(provider, preview, analysis, !frontCamera)
                }
            } catch (e: Exception) {
                statusText.text = "相机启动失败: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("MissingPermission")
    private fun bindCamera(
        provider: ProcessCameraProvider,
        preview: Preview,
        analysis: ImageAnalysis,
        useFront: Boolean,
    ) {
        provider.bindToLifecycle(
            this,
            if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA
            else CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis,
        )
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
                    val hasFace = result.faceLandmarks().isNotEmpty()
                    if (hasFace) {
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

    // ── scrcpy overlay-only 连接 ──────────────────────────────

    private fun startAr() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        connectPanel.visibility = View.GONE
        arPanel.visibility = View.VISIBLE
        progressView.visibility = View.VISIBLE
        statusText.text = "连接眼镜中…"
        playing = true

        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this, audioEnabled = false, overlayOnly = true)
        connection!!.connectAsync(ip, 5555, callback)
    }

    private fun exitAr() {
        playing = false
        connection?.disconnect()
        handleStopped()
        connectPanel.visibility = View.VISIBLE
        arPanel.visibility = View.GONE
    }

    private fun handleStopped() {
        runOnUiThread {
            progressView.visibility = View.GONE
            overlayDecoder?.stop()
            overlayDecoder = null
            faceLocked = false
            renderer.faceMatrix = null
        }
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            runOnUiThread { statusText.text = "已连接，等待 overlay…"; progressView.visibility = View.GONE }
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
            if (playing) handleStopped()
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
