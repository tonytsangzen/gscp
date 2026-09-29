package com.gscp.desktop

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.RectF
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Patterns
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * AR 模式（ncnn-benchmark CameraActivity 移植）：
 * Camera2 (YUV_420_888) → ArNative.nativeFaceDetect（det_10g + FaceMesh 融合姿态，
 * 全部在 native）→ SurfaceView Canvas 绘制：相机帧（前置镜像）+ 检测框 + 5 关键点 +
 * 法线 + 眼镜 overlay 屏（scrcpy 帧按 overlay 四角 drawBitmapMesh）+ 调试面板。
 * 显示与检测用同一份像素，叠加层与检测输入严格对应（所见即所测）。
 */
class ArActivity : Activity() {

    private lateinit var prefs: android.content.SharedPreferences
    private lateinit var settingsPanel: android.view.View
    private lateinit var arPanel: android.view.View
    private lateinit var cameraView: SurfaceView
    private val MESH_N = 4                 // 内容透视网格密度
    private lateinit var statusText: TextView
    private lateinit var progressView: android.view.View
    private lateinit var ipEdit: EditText

    private var connection: ScrcpyConnection? = null
    // 眼镜端音频：Opus 解码 → AudioTrack 播放；解码出的 PCM 经 onPcm 转发给录像混音
    private var glassesAudio: AudioPlayer? = null
    private var overlayDecoder: VideoDecoder? = null
    private val overlayLock = Any()
    private var overlayBmp: Bitmap? = null
    private var overlayPix: IntArray? = null
    private var yScr: ByteArray? = null
    private var uScr: ByteArray? = null
    private var vScr: ByteArray? = null
    private var overlayNew = false

    // ── 后摄子系统：完全复用普通模式（MainActivity）管线，独立于前摄 Canvas 管线 ──
    // GlassesPlayer 自持连接/双硬解/音频；rearMixer 输出到 ar_gl_surface 显示，
    // 录像时编码器面直接挂 mixer 输出。与前摄不共享任何连接/解码状态。
    private var rearPlayer: GlassesPlayer? = null
    private var rearMixer: SurfaceMixer? = null
    private lateinit var glSurface: SurfaceView
    private var glAttached: Surface? = null
    // 子系统活动守卫：仅当对应子系统应处于活动态时，其断开事件才会触发退出预览
    @Volatile private var frontActive = false
    @Volatile private var rearActive = false

    @Volatile private var recorder: ArRecorder? = null
    @Volatile private var recStarting = false
    @Volatile private var recStopping = false
    // 预热的 AVC 编码器 + 输入 Surface（复用，压低首次录像启动的偶发 1~5s）
    private val warmLock = Any()
    @Volatile private var warmCodec: MediaCodec? = null
    @Volatile private var warmSurf: android.view.Surface? = null
    @Volatile private var warmW = 0
    @Volatile private var warmH = 0

    private val ui = Handler(Looper.getMainLooper())
    private var camThread: HandlerThread? = null
    private var camHandler: Handler? = null
    private val det: ExecutorService = Executors.newSingleThreadExecutor()
    private val openExec: ExecutorService = Executors.newSingleThreadExecutor()
    // 自定义合成绘制（相机+overlay 单应投影）专用渲染线程：从主线程彻底挪走，
    // 避免每帧的绘制阻塞按钮/动画/触摸等 UI 响应；它与人脸检测(det)并行。
    private val renderExec: ExecutorService = Executors.newSingleThreadExecutor()
    // 顶部录制指示：录象中→闪烁红点+时长(常驻)；非录像→信息 5s 后隐藏
    private lateinit var recOverlay: android.widget.TextView
    private val recTimerHandler = Handler(Looper.getMainLooper())
    private var recTimerRunnable: Runnable? = null
    private var recDotOn = false
    private var recDotOnLogs = 0
    @Volatile private var recStartEt = 0L

    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var surfaceReady = false
    private val busy = AtomicBoolean(false)

    // 帧数据（生产者：det 线程；消费者：UI 线程绘制；bmp 并发访问用 frameLock 保护）
    private val frameLock = Any()
    private var bmp: Bitmap? = null
    private var rgbaBuf: ByteBuffer? = null

    @Volatile private var debugLines = arrayOf<String>()
    @Volatile private var faceKps0: FloatArray? = null   // face0 五点（面板显示）
    @Volatile private var faceYpr0: FloatArray? = null   // face0 姿态角（面板显示）
    @Volatile private var faceOv = arrayOf<FloatArray>()       // 每脸 17 值：四角 8 + 背板 8 + 距离 cm

    // ── 录像按钮出现条件：三层画面（相机 / 眼镜overlay首帧 / 人脸锚点）都就绪才显示 ──
    @Volatile private var camLayerReady = false
    @Volatile private var overLayerReady = false
    @Volatile private var anchorLayerReady = false
    private var lastRecBtnShow = false   // bottom_controls 上次可见态（布局初始 hidden），避免每帧重复更新 View

    // 眼镜连接状态：0=未连接 1=连接中 2=已连接
    @Volatile private var glassesState = 0

    // 相机参数
    private var cameraId: String? = null
    private var sensorOrientation = 90
    @Volatile private var frontCamera = true   // true=手机前摄 AR(现用场景)；false=眼镜画面做基底(后摄模式)
    private var lastVideoUri: android.net.Uri? = null   // 最近录像的 MediaStore 内容 URI（左侧缩略图点开播放）
    private var frameW = 0
    private var frameH = 0
    private var yStride = 0
    private var uStride = 0
    private var vStride = 0
    private var uPix = 0
    private var vPix = 0
    private var yBuf: ByteArray? = null
    private var uBuf: ByteArray? = null
    private var vBuf: ByteArray? = null

    // ── 合成参数（移植第一页设置，与 MainActivity 同名同 prefs key）──
    private var overlayScalePct = 100          // overlay_scale ×100（100 = contain 铺满）
    private var overlayAlphaPct = 100          // overlay_alpha ×100
    private var overlayBrightnessPct = 100     // overlay_brightness ×100
    private var overlaySaturationPct = 100     // saturation_boost ×100
    private var dimStrengthPct = 0             // dim_strength ×100
    private var keyLowPct = 0                  // overlay_black_key_low ×100
    private var keyHighPct = 0                 // overlay_black_key_high ×100
    private var featherPowerPct = 135          // overlay_feather_power ×100
    private var featherRadiusPx = 0            // overlay_feather_radius（px/255）
    private var baseBrightnessPct = 100        // base_brightness ×100
    private var bottomRotationDeg = 90
    private var bottomMirror = false
    private var topRotationDeg = 0
    private var topMirror = false
    private var audioEnabled = true

    private val detectOn = true
    private val backendIdx = 2      // Vulkan FP16（自动调优，失败回退 CPU FP32）
    private var playing = false
    private var openSent = false
    private var openErr: String? = null
    private var backendName = "…"
    private var cornerDbg = 0
    private var threadInfo = ""
    private var dbgFrames = 0
    private val fpsEma = doubleArrayOf(0.0)
    private val lastFrameT = doubleArrayOf(0.0)
    @Volatile private var lastInferMs = 0f
    @Volatile private var lastConvMs = 0f
    @Volatile private var lastCopyMs = 0f
    @Volatile private var lastProcMs = 0f
    @Volatile private var lastDrawMs = 0f

    private lateinit var textPaint: Paint
    private lateinit var panelPaint: Paint
    private lateinit var connPaint: Paint
    private lateinit var meshPaint: Paint
    private val camDimPaint = Paint().apply {
        // 相机压暗 70%（ColorMatrix 缩放 RGB）：overlay 成为画面最亮处 → “比 camera 高很多”
        colorFilter = android.graphics.ColorMatrixColorFilter(
            android.graphics.ColorMatrix(floatArrayOf(
                0.70f, 0f, 0f, 0f, 0f,
                0f, 0.70f, 0f, 0f, 0f,
                0f, 0f, 0.70f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f,
            )))
    }
    // lazy：构造期 resources 未挂载，不能在字段初始化器里取密度
    private val glowPaint: Paint by lazy {
        // 背板边缘柔光（亮屏边框受光感）：纯绿
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 6 * dp0()
            color = 0x0A00FF00.toInt()
            maskFilter = android.graphics.BlurMaskFilter(
                5 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
        }
    }
    private val vertsBuf = FloatArray(8)   // 保留：背板/诊断备用
    private lateinit var dimPaint: Paint

    // 黑键 LUT（= PC 合成器 blackKeyAlpha）：luma < keyLow-feather → 全透明，
    // > keyHigh+feather → 不透明，中间按 pow(t,1.2) 羽化。褐色/暗背景即被抠透。
    private val keyLut = IntArray(256).also { lut ->
        val lo = maxOf(0.08f - 0.06f, 0f) * 255f
        val hi = minOf(0.28f + 0.06f, 1f) * 255f
        for (i in 0..255) {
            val t = ((i - lo) / (hi - lo)).coerceIn(0f, 1f)
            lut[i] = (Math.pow(t.toDouble(), 1.2).toFloat() * 255f).toInt()
        }
    }

    // overlay 发光（真实发光感，金字塔 bloom）：
    //  远距光晕 ob 1/16、中距光晕 ob 1/4，都为模糊副本 + ADDITIVE 加光（只加光不叠字形）；
    //  内容本身保持普通 SRC_OVER 只画一次（文字不再叠亮重影）。
    private fun Paint.addBlend() {
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            blendMode = android.graphics.BlendMode.PLUS
        } else {
            @Suppress("DEPRECATION")
            xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.ADD)
        }
    }
    private val bloomMidPaint: Paint by lazy {
        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            maskFilter = android.graphics.BlurMaskFilter(
                2 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
            addBlend()
        }
    }
    private val bloomFarPaint: Paint by lazy {
        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            maskFilter = android.graphics.BlurMaskFilter(
                5 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
            addBlend()
        }
    }
    private val glowAlphaPaint = Paint().apply { alpha = 200 }        // 中距光晕强度
    private val bloomFarAlphaPaint = Paint().apply { alpha = 150 }    // 远距光晕强度
    private var glowBmp: Bitmap? = null       // ob 1/4 降采样（中距光晕）
    private var glowCanvas: Canvas? = null
    private var bloomFarBmp: Bitmap? = null   // ob 1/16 降采样（远距光晕）
    private var bloomFarCanvas: Canvas? = null
    private var glowSrc = Rect(0, 0, 0, 0)
    private var glowDst = Rect(0, 0, 0, 0)
    private var farDst = Rect(0, 0, 0, 0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_ar)
        prefs = getSharedPreferences("gscp", MODE_PRIVATE)

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        settingsPanel = findViewById(R.id.settings_panel)
        arPanel = findViewById(R.id.ar_panel)
        cameraView = findViewById(R.id.ar_surface)
        statusText = findViewById(R.id.ar_status)
        progressView = findViewById(R.id.progress_bar)
        ipEdit = findViewById(R.id.ip_address)
        glSurface = findViewById(R.id.ar_gl_surface)
        // 与普通模式完全相同的 3:4 显示平面（MainActivity 同款计算）：SurfaceMixer 的
        // overlay contain+cropMargin 公式仅在 3:4 画布下无失真，全屏画布会把画面拉长
        val sz = android.graphics.Point()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealSize(sz)
        val (gw, gh) = if (sz.x <= sz.y) sz.x to sz.x * 4 / 3
                       else sz.y to sz.y * 3 / 4
        glSurface.layoutParams = android.widget.FrameLayout.LayoutParams(gw, gh).apply {
            gravity = android.view.Gravity.CENTER
        }

        // 后摄 GL 合成输出面：随可见性创建/销毁 surface，挂到/摘出后摄 SurfaceMixer
        glSurface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(h: SurfaceHolder) {}
            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {
                val mx = ensureRearMixer(w, ht)
                val s = h.surface
                if (glAttached !== s) {
                    glAttached?.let { mx.detachOutputSurface(it) }
                    glAttached = s
                    mx.attachOutputSurface(s)
                }
                startRearPlayerIfReady()
            }

            override fun surfaceDestroyed(h: SurfaceHolder) {
                glAttached?.let { rearMixer?.detachOutputSurface(it) }
                glAttached = null
            }
        })

        loadSettings()   // 与第一页同一组合成参数（同名 prefs key）

        cameraView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(h: SurfaceHolder) { surfaceReady = true; tryStart() }
            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}
            override fun surfaceDestroyed(h: SurfaceHolder) { surfaceReady = false; stopCamera() }
        })

        findViewById<Button>(R.id.button_connect).setOnClickListener { startAr() }
        recOverlay = findViewById(R.id.rec_overlay)
        val lastThumb = findViewById<ImageView>(R.id.last_thumb)
        lastThumb.setOnClickListener {
            val u = lastVideoUri ?: return@setOnClickListener
            try {
                val i = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                    setDataAndType(u, "video/mp4")
                    addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(i)
            } catch (e: Exception) { android.util.Log.w(TAG, "open last video fail", e) }
        }
        findViewById<ImageButton>(R.id.btn_switch_cam).setOnClickListener {
            if (recorder?.recording == true) { showStatusText("录像中请勿切换摄像头"); return@setOnClickListener }
            frontCamera = !frontCamera
            updateCamSwitchUi()
            if (frontCamera) enterFrontMode() else enterRearMode()
        }
        findViewById<ImageButton>(R.id.button_record).setOnClickListener {
            if (recStarting || recStopping) return@setOnClickListener   // 启动/停止中忽略点击
            if (recorder?.recording == true) stopRec() else startRec()
        }
        ipEdit.setText(prefs.getString("ip", ""))
        loadRecentLastVideo()            // 左侧显示上次录像缩略图（跨会话）

        initPaints()

        camThread = HandlerThread("cam-bg").also { it.start() }
        camHandler = Handler(camThread!!.looper)
        ArNative.nativeFaceSetPoseAlgo(3)   // 融合（mesh Kabsch + hopenet 大角度）

        if (checkSelfPermission(android.Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            tryStart()
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), REQ_CAM)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAM &&
            grantResults.isNotEmpty() && grantResults[0] ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            tryStart()
        } else if (requestCode == REQ_CAM) {
            showStatusText("未授予相机权限，无法追踪人脸")
        } else if (requestCode == REQ_MIC) {
            if (grantResults.isNotEmpty() && grantResults[0] ==
                android.content.pm.PackageManager.PERMISSION_GRANTED) {
                startRec()
            } else {
                showStatusText("未授予麦克风权限，录像将不含手机录音")
            }
        }
    }

    private fun micGranted(): Boolean =
        checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun initPaints() {
        val dp = resources.displayMetrics.density
        textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE; typeface = Typeface.MONOSPACE; textSize = 12 * dp
        }
        // 背景板上的"连接中..."：绿色，板较大用 16dp 保证可见
        connPaint = Paint(textPaint).apply {
            color = 0xFF90EE90.toInt()
            textSize = 16 * dp
        }
        meshPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
            // overlay 内容提亮（发光体感）：RGB ×1.4 + 12，透明区/alpha 不受影响
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix(floatArrayOf(
                    1.40f, 0f, 0f, 0f, 12f,
                    0f, 1.40f, 0f, 0f, 12f,
                    0f, 0f, 1.40f, 0f, 12f,
                    0f, 0f, 0f, 1f, 0f,
                )))
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                blendMode = android.graphics.BlendMode.SRC_OVER
            } else {
                @Suppress("DEPRECATION")
                xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.SRC_OVER)
            }
        }
        // 背板（亮屏面板）投影区衬底：微绿低透明压暗
        dimPaint = Paint().apply { color = 0x0800FF00.toInt() }
        panelPaint = Paint().apply { color = 0xB30d1117.toInt() }
    }

    private fun tryStart() {
        if (surfaceReady &&
            checkSelfPermission(android.Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED &&
            cameraDevice == null
        ) {
            openBackend()
            camHandler?.post { startCamera() }
        }
    }

    // ── 合成参数 + GL 合成器（移植第一页连接逻辑）────────────────

    private fun loadSettings() {
        overlayScalePct = prefs.getInt("overlayScalePct", 100)
        overlayAlphaPct = prefs.getInt("overlayAlphaPct", 100)
        overlayBrightnessPct = prefs.getInt("overlayBrightnessPct", 100)
        overlaySaturationPct = prefs.getInt("overlaySaturationPct", 100)
        dimStrengthPct = prefs.getInt("dimStrengthPct", 0)
        keyLowPct = prefs.getInt("keyLowPct", 0)
        keyHighPct = prefs.getInt("keyHighPct", 0)
        featherPowerPct = prefs.getInt("featherPowerPct", 135)
        featherRadiusPx = prefs.getInt("featherRadiusPx", 0)
        baseBrightnessPct = prefs.getInt("baseBrightnessPct", 100)
        bottomRotationDeg = prefs.getInt("bottomRotationDeg", 90)
        bottomMirror = prefs.getBoolean("bottomMirror", false)
        topRotationDeg = prefs.getInt("topRotationDeg", 0)
        topMirror = prefs.getBoolean("topMirror", false)
        audioEnabled = prefs.getBoolean("audioEnabled", true)
    }

    /** 与 MainActivity.applySettingsToMixer 同一套参数下发；仅黑边抠像特殊：
     *  页一未启用抠像（keyHigh ≤ keyLow）时回退到 AR 原校准值——旧 CPU 路径 keyLut
     *  恒为 lo=0.08-0.06、hi=0.28+0.06、pow1.2（暗色底即抠透）。否则 shader 直接关闭
     *  抠像，overlay 的褐色底会不透明盖在基底上（表现为颜色不对）。 */
    private fun applySettingsToMixer(mx: SurfaceMixer) {
        mx.setOverlayScale(overlayScalePct / 100f)
        var keyLow = keyLowPct / 100f
        var keyHigh = keyHighPct / 100f
        var featherPower = featherPowerPct / 100f
        var featherRadius = featherRadiusPx
        if (keyHigh <= keyLow) {
            keyLow = 0.08f; keyHigh = 0.28f; featherPower = 1.2f; featherRadius = 15
        }
        mx.setOverlayParams(
            overlayAlphaPct / 100f,
            overlayBrightnessPct / 100f,
            overlaySaturationPct / 100f,
            dimStrengthPct / 100f,
            keyLow,
            keyHigh,
            featherPower,
            featherRadius,
        )
        mx.setBaseBrightness(baseBrightnessPct / 100f)
        mx.setBottomRotation(bottomRotationDeg.toFloat(), bottomMirror)
        mx.setTopRotation(topRotationDeg, topMirror)
    }

    /** 后摄首次切入时建 GL 合成器（尺寸取 GL 面实际大小，保证 1:1 输出）。 */
    private fun ensureRearMixer(w: Int, h: Int): SurfaceMixer {
        var mx = rearMixer
        if (mx == null) {
            mx = SurfaceMixer(this, w, h)
            rearMixer = mx
            applySettingsToMixer(mx)
        }
        return mx
    }

    /** 首次加载实时检测网：默认 Vulkan FP16（自动调优），失败自动回退 CPU FP32。 */
    private fun openBackend() {
        if (openSent) return
        openSent = true
        val am = assets
        val dir = filesDir.absolutePath
        openExec.execute {
            val j = ArNative.nativeFaceOpen(am, dir, backendIdx, LIVE_INPUT)
            if (!parseOpen(j)) {
                ArNative.nativeFaceOpen(am, dir, 0, LIVE_INPUT)
            }
        }
    }

    private fun parseOpen(j: String): Boolean {
        return try {
            val o = JSONObject(j)
            val ok = o.optBoolean("ok")
            if (ok) {
                backendName = o.optString("backend", "?")
                val t = o.optInt("threads", 0)
                threadInfo = if (t > 0) " · 线程 $t" else ""
            } else {
                openErr = o.optString("err", "open_failed")
                android.util.Log.w(TAG, "det net open failed: $j")
            }
            ok
        } catch (e: Exception) {
            false
        }
    }

    // ── Camera2 ───────────────────────────────────────────────

    @SuppressLint("MissingPermission")
    private fun startCamera() {
        try {
            if (cameraDevice != null) return
            cameraId = pickCamera()
            val cc = cameraManager!!.getCameraCharacteristics(cameraId!!)
            sensorOrientation = cc.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            val map = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                as StreamConfigurationMap
            val sz = pickSize(map.getOutputSizes(ImageFormat.YUV_420_888))
            frameW = sz.width
            frameH = sz.height

            // 同步关闭旧 reader（不能用 stopCamera：其 closeStream 是 post 到队尾，
            // 会把本函数刚创建的新 reader 置空，帧投递被掐断——蓝本此处即 closeStream）
            closeStream()
            reader = ImageReader.newInstance(frameW, frameH, ImageFormat.YUV_420_888, 3)
            reader!!.setOnImageAvailableListener({ r -> onImage(r) }, camHandler)
            cameraManager!!.openCamera(cameraId!!, object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) {
                    cameraDevice = d
                    try {
                        val outs = listOf(reader!!.surface)
                        d.createCaptureSession(outs, object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(s: CameraCaptureSession) {
                                session = s
                                try {
                                    val b = d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                                    b.addTarget(reader!!.surface)
                                    s.setRepeatingRequest(b.build(), null, camHandler)
                                    showStatus(statusTextFor())
                                } catch (e: Exception) {
                                    showStatus("预览启动失败: $e")
                                }
                            }

                            override fun onConfigureFailed(s: CameraCaptureSession) {
                                showStatus("capture session 配置失败")
                            }
                        }, camHandler)
                    } catch (e: Exception) {
                        showStatus("创建会话失败: $e")
                    }
                }

                override fun onDisconnected(d: CameraDevice) {
                    d.close(); if (cameraDevice == d) cameraDevice = null
                }

                override fun onError(d: CameraDevice, error: Int) {
                    d.close()
                    if (cameraDevice == d) cameraDevice = null
                    showStatus("相机打开失败 error=$error")
                }
            }, camHandler)
        } catch (e: Exception) {
            showStatus("相机启动失败: $e")
        }
    }

    private fun stopCamera() {
        val ch = camHandler ?: return
        ch.post { closeCameraNow() }
    }

    /** 在 camHandler 线程内同步关闭当前相机会话/设备/流。 */
    private fun closeCameraNow() {
        try { session?.close() } catch (_: Exception) {}
        try { cameraDevice?.close() } catch (_: Exception) {}
        session = null
        cameraDevice = null
        closeStream()
    }

    private fun closeStream() {
        try { reader?.close() } catch (_: Exception) {}
        reader = null
        busy.set(false)
    }

    private fun pickCamera(): String {
        var front: String? = null
        var back: String? = null
        var any: String? = null
        try {
            for (id in cameraManager!!.cameraIdList) {
                if (any == null) any = id
                val f = cameraManager!!.getCameraCharacteristics(id).get(
                    CameraCharacteristics.LENS_FACING) ?: continue
                if (f == CameraMetadata.LENS_FACING_FRONT && front == null) front = id
                if (f == CameraMetadata.LENS_FACING_BACK && back == null) back = id
            }
        } catch (_: Exception) {}
        return (if (frontCamera) (front ?: back) else (back ?: front)) ?: any ?: "0"
    }

    /** 选最接近 640x480、比例 4:3 的分析帧尺寸。 */
    private fun pickSize(sizes: Array<android.util.Size>): android.util.Size {
        var best: android.util.Size? = null
        var bestScore = Long.MAX_VALUE
        for (s in sizes) {
            val area = s.width.toLong() * s.height
            val aspect = s.width.toFloat() / s.height
            val score = Math.abs(area - 640L * 480L) +
                (if (Math.abs(aspect - 4f / 3f) > 0.05f) 10_000_000L else 0L)
            if (score < bestScore) { bestScore = score; best = s }
        }
        return best ?: android.util.Size(640, 480)
    }

    // ── 帧处理 ────────────────────────────────────────────────

    private fun onImage(r: ImageReader) {
        var img: Image? = null
        try { img = r.acquireLatestImage() } catch (_: Exception) {}
        if (img == null) return
        if (busy.getAndSet(true)) { img.close(); return }
        val t0 = System.nanoTime()
        try {
            val okCopy = copyPlanes(img)
            lastCopyMs = ((System.nanoTime() - t0) / 1e6).toFloat()
            if (!okCopy) {
                android.util.Log.e(TAG, "copyPlanes failed")
                busy.set(false)
                img.close()
                return
            }
        } finally {
            img.close()
        }
        det.execute { processFrame() }
    }

    /** 三个平面按行拷贝（保留原 rowStride 布局），供 native 读取。 */
    private fun copyPlanes(img: Image): Boolean {
        return try {
            val p = img.planes
            val w = img.width
            val h = img.height
            yStride = p[0].rowStride
            uStride = p[1].rowStride
            vStride = p[2].rowStride
            uPix = p[1].pixelStride
            vPix = p[2].pixelStride
            val ch = (h + 1) / 2
            val cwBytes = ((w + 1) / 2 - 1) * uPix + 1
            val vwBytes = ((w + 1) / 2 - 1) * vPix + 1
            val yNeed = (h - 1) * yStride + w
            val uNeed = (ch - 1) * uStride + cwBytes
            val vNeed = (ch - 1) * vStride + vwBytes
            val yCap = p[0].buffer.limit()
            val uCap = p[1].buffer.limit()
            val vCap = p[2].buffer.limit()
            if (yBuf == null || yBuf!!.size != maxOf(yNeed, yCap)) yBuf = ByteArray(maxOf(yNeed, yCap))
            if (uBuf == null || uBuf!!.size != maxOf(uNeed, uCap)) uBuf = ByteArray(maxOf(uNeed, uCap))
            if (vBuf == null || vBuf!!.size != maxOf(vNeed, vCap)) vBuf = ByteArray(maxOf(vNeed, vCap))
            copyPlane(p[0].buffer, yBuf!!, h, w, yStride)
            copyPlane(p[1].buffer, uBuf!!, ch, cwBytes, uStride)
            copyPlane(p[2].buffer, vBuf!!, ch, vwBytes, vStride)
            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "copyPlanes exception", e)
            false
        }
    }

    private fun copyPlane(src: ByteBuffer, dst: ByteArray, rows: Int, rowBytes: Int, rowStride: Int) {
        val cap = src.limit()
        for (r in 0 until rows) {
            val pos = r * rowStride
            if (pos >= cap) break
            val len = minOf(rowBytes, cap - pos)
            src.limit(pos + len).position(pos)
            src.get(dst, pos, len)
        }
        src.clear()
    }

    private fun processFrame() {
        try {
            if (!camLayerReady) { camLayerReady = true; updateRecBtnVisible() }   // 相机帧到达 → 第一层就绪
            val now = System.nanoTime() / 1e6
            if (lastFrameT[0] > 0) {
                val inst = 1000.0 / maxOf(1e-3, now - lastFrameT[0])
                fpsEma[0] = if (fpsEma[0] == 0.0) inst else fpsEma[0] * 0.9 + inst * 0.1
            }
            lastFrameT[0] = now

            val disp = windowManager.defaultDisplay
            val displayRot = (disp?.rotation ?: 0) * 90
            val rot = (sensorOrientation - displayRot + 360) % 360
            val rw = if (rot == 90 || rot == 270) frameH else frameW
            val rh = if (rot == 90 || rot == 270) frameW else frameH

            val rb: ByteBuffer = synchronized(frameLock) {
                val cur = rgbaBuf
                if (cur == null || cur.capacity() != rw * rh * 4) {
                    val nb = ByteBuffer.allocateDirect(rw * rh * 4).order(ByteOrder.nativeOrder())
                    rgbaBuf = nb
                    bmp?.recycle()
                    bmp = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
                    nb
                } else cur
            }

            val tn = System.nanoTime()
            val j = ArNative.nativeFaceDetect(
                yBuf!!, uBuf!!, vBuf!!, frameW, frameH,
                yStride, uStride, vStride, uPix, vPix, rot, detectOn, rb)

            if (++dbgFrames <= 3)
                android.util.Log.i(TAG, "frame#$dbgFrames rot=$rot upright=${rw}x$rh json=" +
                    (if (j.length > 160) j.substring(0, 160) else j))

            parseResult(j, rw, rh)
            lastProcMs = ((System.nanoTime() - tn) / 1e6).toFloat()

            synchronized(frameLock) {
                val b0 = bmp
                if (b0 != null) {
                    rb.rewind()
                    b0.copyPixelsFromBuffer(rb)
                }
            }
            // 后台渲染线程合成绘制（SurfaceView 可从任意线程 lockCanvas），主线程不再被拖慢
            renderExec.execute { render() }
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "帧处理异常", t)
        } finally {
            busy.set(false)
        }
    }

    /** 提取面板数据（face0 关键点/姿态）与 overlay 四角锚点；人脸丢失时 faceOv 为空。 */
    private fun parseResult(j: String, rw: Int, rh: Int) {
        val lines = ArrayList<String>()
        val ovList = ArrayList<FloatArray>()
        var k0: FloatArray? = null
        var p0: FloatArray? = null
        try {
            val o = JSONObject(j)
            lastConvMs = o.optDouble("convMs", 0.0).toFloat()
            lastInferMs = o.optDouble("ms", 0.0).toFloat()
            val total = lastCopyMs + lastProcMs + lastDrawMs
            lines.add(String.format(Locale.ROOT, "FPS %.1f · 帧耗时 %.1fms", fpsEma[0], total))
            lines.add(String.format(Locale.ROOT, "耗时: 拷贝 %.1f · 转换 %.1f · 推理 %.1f ms",
                lastCopyMs, lastConvMs, lastInferMs))
            lines.add(String.format(Locale.ROOT, "绘制 %.1f ms", lastDrawMs))
            lines.add(String.format(Locale.ROOT, "后端 %s%s · 输入 %d×%d",
                o.optString("backend", backendName), threadInfo,
                o.optInt("input", 0), o.optInt("input", 0)))
            lines.add(String.format(Locale.ROOT, "相机 %d×%d · 直立 %d×%d · 旋转 %d°",
                frameW, frameH, o.optInt("w", rw), o.optInt("h", rh), o.optInt("rot", 0)))
            if (!o.optBoolean("ok", true)) {
                lines.add("错误: " + o.optString("err", "?") + (if (detectOn) "" else "（检测关）"))
            } else {
                val fs = o.optJSONArray("faces")
                val n = fs?.length() ?: 0
                val mx = o.optDouble("maxScore", -1.0).toFloat()
                val mxStr = if (mx >= 0) String.format(Locale.ROOT, " · 最高分 %.2f", mx) else ""
                lines.add(String.format(Locale.ROOT, "人脸 %d · 阈值 %.2f%s%s",
                    n, SCORE_THRESH, if (detectOn) "" else " · 检测:关", mxStr))
                if (n > 0) {
                    val f = fs?.optJSONObject(0)
                    if (f != null) {
                        val k = f.optJSONArray("kps")
                        if (k != null && k.length() >= 10)
                            k0 = FloatArray(10) { q -> k.optDouble(q, 0.0).toFloat() }
                        val po = f.optJSONArray("pose")
                        if (po != null && po.length() == 3)
                            p0 = FloatArray(3) { q -> po.optDouble(q, 0.0).toFloat() }
                        val ovArr = f.optJSONArray("overlay")
                        val bkArr = f.optJSONArray("back")
                        if (ovArr != null && ovArr.length() == 4 && bkArr != null && bkArr.length() == 4) {
                            val ov = FloatArray(17)   // 前四角 8 + 背板四角 8 + 距离 cm
                            for (q in 0 until 4) {
                                val pt = ovArr.optJSONArray(q)
                                ov[2 * q] = pt?.optDouble(0, 0.0)?.toFloat() ?: 0f
                                ov[2 * q + 1] = pt?.optDouble(1, 0.0)?.toFloat() ?: 0f
                            }
                            for (q in 0 until 4) {
                                val pt = bkArr.optJSONArray(q)
                                ov[8 + 2 * q] = pt?.optDouble(0, 0.0)?.toFloat() ?: 0f
                                ov[9 + 2 * q] = pt?.optDouble(1, 0.0)?.toFloat() ?: 0f
                            }
                            ov[16] = f.optDouble("ovD", 30.0).toFloat()
                            ovList.add(ov)
                        }
                    }
                    val kf = k0
                    if (kf != null) {
                        val sb = StringBuilder("face0 点: ")
                        val names = arrayOf("眼", "眼", "鼻", "嘴角", "嘴角")
                        for (q in 0 until 5)
                            sb.append(names[q]).append(q).append("(")
                                .append(Math.round(kf[2 * q])).append(",")
                                .append(Math.round(kf[2 * q + 1])).append(") ")
                        lines.add(sb.toString())
                        val pf = p0
                        if (pf != null) {
                            lines.add(String.format(Locale.ROOT,
                                "face0 姿态: yaw %.0f° pitch %.0f° roll %.0f°", pf[1], pf[0], pf[2]))
                        }
                    }
                }
                lines.add("算法:" + o.optString("algo", "fusion"))
            }
        } catch (e: Exception) {
            lines.add("JSON 解析失败: $e")
        }
        if (openErr != null) lines.add("网络加载失败: $openErr")
        debugLines = lines.toTypedArray()
        faceKps0 = k0
        faceYpr0 = p0
        faceOv = ovList.toTypedArray()
        val anchored = ovList.isNotEmpty()          // 第三层：人脸追到 overlay 四角锚点
        if (anchored != anchorLayerReady) { anchorLayerReady = anchored; updateRecBtnVisible() }
    }

    // ── 绘制 ──────────────────────────────────────────────────

    private fun render() {
        if (!surfaceReady) return
        val td = System.nanoTime()
        var sc: Canvas? = null
        try { sc = cameraView.holder.lockCanvas() } catch (_: Exception) {}
        if (sc == null) return
        try {
            // 直接画到 Surface（硬件加速、SurfaceFlinger 帧同步交换 → 不撕裂、不卡顿）。
            paintFrame(sc, sc.width.toFloat(), sc.height.toFloat())
            // 通知编码线程取一帧（其自身按 live 状态渲染，卡顿不会阻塞本线程）
            try { recorder?.tick() } catch (_: Exception) {}
        } finally {
            try { cameraView.holder.unlockCanvasAndPost(sc) } catch (_: Exception) {}
            lastDrawMs = ((System.nanoTime() - td) / 1e6).toFloat()
        }
    }

    /** 画一帧（相机 + 背板 + overlay + 发光）。UI 显示与录像编码线程均调用。
     *  可变 scratch（Matrix/Path/顶点数组）均为局部，避免两线程互踩 */
    private fun paintFrame(c: Canvas, vw: Float, vh: Float) {
        c.drawColor(Color.BLACK)
        // 后摄模式（frontCamera=false）：显示与录像均走 GL 合成（SurfaceMixer 直出，同第一页
        // 连接模式的渲染管线）；此 Canvas 路径仅在 GL 面未就绪时兜底提示连接状态。
        if (!frontCamera) {
            drawConnectionStatus(c, vw / 2f, vh / 2f)
            return
        }
        val b: Bitmap? = synchronized(frameLock) { bmp }
            var sc = 1f; var dx = 0f; var dy = 0f
            if (b != null) {
                // 等比例缩放完整显示相机画面（letterbox 居中，不裁剪）；
                // 标注（框/关键点/法线/overlay）共用同一变换，自动跟随
                sc = minOf(vw / b.width, vh / b.height)
                dx = (vw - b.width * sc) / 2f
                dy = (vh - b.height * sc) / 2f
                val m = Matrix()
                m.reset()
                if (frontCamera) {
                    // 前置预览镜像，坐标随变换同步镜像
                    m.setScale(-1f, 1f)
                    m.postTranslate(b.width.toFloat(), 0f)
                    m.postScale(sc, sc)
                    m.postTranslate(dx, dy)
                } else {
                    m.setScale(sc, sc)
                    m.postTranslate(dx, dy)
                }
                c.drawBitmap(b, m, camDimPaint)
            }
            val fSc = sc; val fDx = dx; val fDy = dy
            val fImgW = (bmp?.width ?: 1).toFloat()
            fun mapX(x: Float) = (if (frontCamera) (fImgW - x) else x) * fSc + fDx
            fun mapY(y: Float) = y * fSc + fDy

            // 眼镜屏：有解码帧 → 真实屏幕 drawBitmapMesh 投影到锚点四角（透视近似）；
            // 没有帧/未连接 → 在锚点处显示连接状态。四角描边 + 距离标签为调试信息。
            val ovs = faceOv
            val ob: Bitmap? = synchronized(overlayLock) { overlayBmp }
            if (ovs.isNotEmpty()) {
                val ov = ovs[0]
                // 背景平面：沿 overlay 中心子四边形描出（内容中心 480×480，上下净空 80px；
                // 法线/旋转与人脸同步），圆角用顶点内插实现（二次贝塞尔连接）。
                val qx = floatArrayOf(mapX(ov[0]), mapX(ov[2]), mapX(ov[4]), mapX(ov[6]))
                val qy = floatArrayOf(mapY(ov[1]), mapY(ov[3]), mapY(ov[5]), mapY(ov[7]))
                var ccx2 = 0f; var ccy2 = 0f
                for (q in 0 until 4) { ccx2 += qx[q] / 4f; ccy2 += qy[q] / 4f }
                // 背景板与 overlay 内容 quad 完全重合：overlay 位图已是裁好的
                // 480×480 正方形（convertOverlayFrame 裁掉了上下 80px 空带），
                // 并铺满整个 quad（bt=0 不再按旧全屏 640 内缩）。正脸时 quad=1:1。
                val bt = 0f
                val bb = 1f - bt
                val bgx = FloatArray(4); val bgy = FloatArray(4)
                bgx[0] = bt * qx[3] + bb * qx[0]; bgy[0] = bt * qy[3] + bb * qy[0]   // 左下'
                bgx[1] = bt * qx[2] + bb * qx[1]; bgy[1] = bt * qy[2] + bb * qy[1]   // 右下'
                bgx[2] = bb * qx[2] + bt * qx[1]; bgy[2] = bb * qy[2] + bt * qy[1]   // 右上'
                bgx[3] = bb * qx[3] + bt * qx[0]; bgy[3] = bb * qy[3] + bt * qy[0]   // 左上'
                // 角点顺序（实测）：c0=左下 c1=右下 c2=右上 c3=左上；
                // 屏幕顺时针 TL→TR→BR→BL = c3→c2→c1→c0
                val ord = intArrayOf(3, 2, 1, 0)
                // 圆角四边形：沿两条邻边各退边长的 18%（同一比例 → 切角均匀），
                // quadTo(控制=角点) 圆滑过渡。背板与边框共用同一条路径，边缘完全重合。
                fun roundedQuad(): Path {
                    val p = Path()
                    val ox = FloatArray(4); val oy = FloatArray(4)   // 各角沿入边方向的退点
                    val ix = FloatArray(4); val iy = FloatArray(4)   // 各角沿出边方向的退点
                    for (q in 0 until 4) {
                        val kp = ord[q]
                        // 入边：kp ← 上一角；出边：kp → 下一角（各退边长 10%，小圆角）
                        val prevIdx = ord[(q + 3) % 4]
                        val nextIdx = ord[(q + 1) % 4]
                        ox[q] = (bgx[kp] + (bgx[prevIdx] - bgx[kp]) * 0.10f)
                        oy[q] = (bgy[kp] + (bgy[prevIdx] - bgy[kp]) * 0.10f)
                        ix[q] = (bgx[kp] + (bgx[nextIdx] - bgx[kp]) * 0.10f)
                        iy[q] = (bgy[kp] + (bgy[nextIdx] - bgy[kp]) * 0.10f)
                    }
                    p.moveTo(ox[0], oy[0])
                    for (q in 0 until 4) {
                        val kp = ord[q]
                        // 圆角 q：入点 A_q --(控制=角点)--> 出点 B_q
                        p.quadTo(bgx[kp], bgy[kp], ix[q], iy[q])
                        // 边：出点 B_q --直线--> 下一角入点 A_{q+1}
                        val n = (q + 1) % 4
                        p.lineTo(ox[n], oy[n])
                    }
                    p.close()
                    return p
                }
                val platePath = roundedQuad()
                // 发光（无边线）：模糊光晕画在背板之下，沿轮廓向外溢出形成柔光
                c.drawPath(platePath, glowPaint)
                // 半透明背板（与 overlay 平面同姿态、同圆角轮廓）：浅绿低透明
                c.drawPath(platePath, dimPaint)
                if (ob != null) {
                    // drawBitmapMesh 顶点序：(0,0),(w,0),(0,h),(w,h) → 四角 TL,TR,BL,BR；
                    // 内容去除全部特效，普通 SRC_OVER 贴到四角（键控透明区透出相机）
                    // 实测角点布局：c0=左下 c1=右下 c2=右上 c3=左上
                    // mesh 槽位 TL,TR,BL,BR ← c3,c2,c0,c1
                    // 透视校正网格：1×1 drawBitmapMesh 是双线性映射，强透视（躺看/低头）
                    // 时四边形内区按双线性变形 → 内容被拉伸。单应映射内容角点到屏幕四角，
                    // 4×4 网格顶点取单应像（Heckbert square→quad），逐格双线性即可精确复现透视。
                    val tlX = mapX(ov[6]); val tlY = mapY(ov[7])   // 位图 TL ← c3
                    val trX = mapX(ov[4]); val trY = mapY(ov[5])   // TR ← c2
                    val brX = mapX(ov[2]); val brY = mapY(ov[3])   // BR ← c1
                    val blX = mapX(ov[0]); val blY = mapY(ov[1])   // BL ← c0
                    val d1x = trX - brX; val d1y = trY - brY
                    val d2x = blX - brX; val d2y = blY - brY
                    val sxq = tlX - trX + brX - blX
                    val syq = tlY - trY + brY - blY
                    val den = d1x * d2y - d1y * d2x
                    val vn = FloatArray((MESH_N + 1) * (MESH_N + 1) * 2)
                    if (kotlin.math.abs(den) > 1e-6f) {
                        val gh = (sxq * d2y - syq * d2x) / den
                        val hv = (d1x * syq - d1y * sxq) / den
                        val av = trX - tlX + gh * trX
                        val bv = blX - tlX + hv * blX
                        val dv = trY - tlY + gh * trY
                        val ev = blY - tlY + hv * blY
                        var k2 = 0
                        for (j in 0..MESH_N) {
                            val vv = j.toFloat() / MESH_N
                            for (i in 0..MESH_N) {
                                val uu = i.toFloat() / MESH_N
                                val wq = gh * uu + hv * vv + 1f
                                vn[k2++] = (av * uu + bv * vv + tlX) / wq
                                vn[k2++] = (dv * uu + ev * vv + tlY) / wq
                            }
                        }
                    } else {
                        // 退化（近似仿射）：回退四角双线性
                        var k2 = 0
                        for (j in 0..MESH_N) {
                            val vv = j.toFloat() / MESH_N
                            val ly0 = blX + (brX - blX) * vv; val ly1 = tlX + (trX - tlX) * vv
                            val lx0 = blY + (brY - blY) * vv; val lx1 = tlY + (trY - tlY) * vv
                            for (i in 0..MESH_N) {
                                val uu = i.toFloat() / MESH_N
                                vn[k2++] = ly0 + (ly1 - ly0) * uu
                                vn[k2++] = lx0 + (lx1 - lx0) * uu
                            }
                        }
                    }
                    // 金字塔 bloom：远距光晕（1/16，ADD）→ 中距光晕（1/4，ADD）→ 内容（普通合成）。
                    // 只加光不叠字形；与内容同一 mesh 投影、完全重合，随姿态同步。
                    val fb = synchronized(overlayLock) { bloomFarBmp }
                    val gb = synchronized(overlayLock) { glowBmp }
                    if (fb != null) {
                        c.drawBitmapMesh(fb, MESH_N, MESH_N, vn, 0, null, 0, bloomFarPaint)
                    }
                    if (gb != null) {
                        c.drawBitmapMesh(gb, MESH_N, MESH_N, vn, 0, null, 0, bloomMidPaint)
                    }
                    // overlay 内容：普通合成贴到四角（键控透明区透出相机；meshPaint 已提亮）
                    c.drawBitmapMesh(ob, MESH_N, MESH_N, vn, 0, null, 0, meshPaint)
                }
                // 无内容/未连接：在 overlay 四角形心处显示绿色"连接中..."
                if (ob == null) {
                    val t = "连接中..."
                    c.drawText(t, ccx2 - connPaint.measureText(t) / 2f,
                        ccy2 + connPaint.textSize / 3f, connPaint)
                }
            } else {
                drawConnectionStatus(c, vw / 2f, vh / 2f)
            }
    }

    private fun dp0(): Float = resources.displayMetrics.density

    private fun startRec() {
        android.util.Log.i(TAG, "rec: startRec called, cameraView=${cameraView.width}x${cameraView.height}")
        // 录像分辨率取显示面：前摄=全屏 Canvas 面；后摄=3:4 GL 面（与 mixer 输出一致）
        val sw: Int; val sh: Int
        if (frontCamera) { sw = cameraView.width; sh = cameraView.height }
        else { sw = glSurface.width; sh = glSurface.height }
        if (sw <= 0 || sh <= 0) { showStatusText("画面未就绪，无法录像"); return }
        // 录像音轨需要麦克风权限（眼镜音频不依赖此权限，但混音需要手机录音）
        if (!micGranted()) {
            requestPermissions(
                arrayOf(android.Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        if (recStarting || recStopping) return
        recStarting = true
        showStatusText("正在启动录像…")
        setRecAnim(true)                 // 旋转动画 = 启动中，且忽略点击
        val btn = findViewById<ImageButton>(R.id.button_record)
        // 取预热好的视频编码器（尺寸匹配才复用），复用则省去偶发 1~5s 的编码器创建
        var preC: MediaCodec? = null; var preS: android.view.Surface? = null
        synchronized (warmLock) {
            if (warmCodec != null && warmSurf != null && warmW == sw && warmH == sh) {
                preC = warmCodec; preS = warmSurf
                warmCodec = null; warmSurf = null
            }
        }
        val r = ArRecorder()
        // 编码器/muxer/麦克风/AAC 初始化重且耗时（这端媒体栈偶发数秒），
        // 放后台线程，按钮即时响应，完成后再回主线程更新状态。
        Thread {
            var ok = false
            try {
                r.start(sw, sh, preC, preS)
                ok = r.recording
                recorder = r
                // 后摄 GL 合成：编码器面直接挂后摄 mixer 输出（合成结果 GPU 直喂编码器，免 Canvas 重画）
                val mx = rearMixer
                val inSurf = r.inputSurface
                if (ok && !frontCamera && mx != null && inSurf != null) {
                    mx.attachOutputSurface(inSurf)
                }
            } catch (e: Exception) {
                android.util.Log.w(TAG, "rec start fail", e)
                try { r.release() } catch (_: Exception) {}
            }
            recStarting = false
            val finalOk = ok
            runOnUiThread {
                setRecAnim(false)
                btn.setBackgroundResource(
                    if (finalOk) R.drawable.ic_record_stop else R.drawable.ic_record_idle)
                statusText.text = if (finalOk) "录像中… (Movies 文件夹)" else "录像启动失败"
                if (finalOk) {
                    recStartEt = r.startedAtEt
                    showRecTimer()                 // 顶部常驻：闪烁红点+时长
                } else {
                    showStatusText("录像启动失败")   // 非录像：overlay 5s 后隐藏
                }
            }
        }.start()
    }

    /** 录像按钮忙碌指示：启动中/停止中 → 在按钮上叠加同圆心同外径的旋转环，并禁用点击；调速 → 隐藏环复原。
     *  按钮图标保留在环下（叠加效果），RingSpinner 可见即自转。 */
    private fun setRecAnim(on: Boolean) {
        val btn = findViewById<ImageButton>(R.id.button_record)
        findViewById<com.gscp.desktop.RingSpinner>(R.id.record_spinner).visibility =
            if (on) android.view.View.VISIBLE else android.view.View.GONE
        btn.isEnabled = !on
        btn.isClickable = !on
    }

    private fun stopRec() {
        if (recStopping || recStarting) return
        val r = recorder ?: return
        recStopping = true
        recorder = null
        showStatusText("正在停止录像…")
        findViewById<ImageButton>(R.id.button_record)
            .setBackgroundResource(R.drawable.ic_record_idle)   // 停止时切回红点再旋转
        setRecAnim(true)                 // 旋转动画 = 停止中，且忽略点击
        val btn = findViewById<ImageButton>(R.id.button_record)
        // 预热尺寸与录像面一致：前摄=全屏 Canvas 面；后摄=3:4 GL 面
        val ws = if (frontCamera) cameraView.width else glSurface.width
        val hs = if (frontCamera) cameraView.height else glSurface.height
        Thread {
            var name: String? = null
            try {
                // 后摄 GL 模式：先把编码器面从后摄 mixer 摘掉（同步、不 release），再停编码器，
                // 避免 GL 线程向已失效的编码器面 swap
                val mx = rearMixer
                val inSurf = r.inputSurface
                if (mx != null && inSurf != null) mx.detachOutputSurface(inSurf, releaseSurface = false)
                r.stop()
                name = r.displayName
            } catch (e: Exception) {
                android.util.Log.w(TAG, "rec stop fail", e)
            }
            recStopping = false
            armWarmVideo(ws, hs)         // 停止后立即后台预热，供下次录像复用
            runOnUiThread {
                setRecAnim(false)
                btn.setBackgroundResource(R.drawable.ic_record_idle)
                showStatusText(if (name != null) "已保存到系统录像: $name" else "录像未保存")
                if (r.lastUri != null) { lastVideoUri = r.lastUri; loadLastThumb() }
            }
        }.start()
    }

    /** 后台预热一个 AVC 编码器+Surface（与录像同参），供下次录像复用，省去首次编码器创建偶发慢。 */
    private fun armWarmVideo(w: Int, h: Int) {
        if (w <= 0 || h <= 0) return
        synchronized(warmLock) { if (warmCodec != null) return }
        Thread({
            var c: MediaCodec? = null
            var s: android.view.Surface? = null
            try {
                c = MediaCodec.createEncoderByType("video/avc")
                val fmt = MediaFormat.createVideoFormat("video/avc", w, h)
                fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                fmt.setInteger(MediaFormat.KEY_BIT_RATE, 8_000_000)
                fmt.setInteger(MediaFormat.KEY_FRAME_RATE, 15)
                fmt.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                c.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                s = c.createInputSurface()
                c.start()
                synchronized (warmLock) { warmCodec = c; warmSurf = s; warmW = w; warmH = h }
                android.util.Log.i(TAG, "rec: warm avc ready ${w}x${h}")
            } catch (e: Exception) {
                android.util.Log.w(TAG, "rec: warm fail", e)
                try { if (s != null && s.isValid) s.release() } catch (_: Exception) {}
                try { if (c != null) c.release() } catch (_: Exception) {}
                synchronized (warmLock) { warmCodec = null; warmSurf = null }
            }
        }, "rec-warm").start()
    }

    /** 统一状态路由（主线程安全）：
     *  原顶部文字(ar_status)停用后不再单独显示，改为——录像中→顶部 overlay 常驻闪烁计时；
     *  非录像→把这段文字放到 overlay，5s 后自动隐藏。可安全地从任意线程调用。 */
    private var hideRunnable: Runnable? = null   // 顶部 overlay 的延迟隐藏（可取消）

    // 取消顶部 overlay 的延迟隐藏（进入录像等场景时调用，避免旧的 5s 隐藏把计时藏掉）
    private fun cancelOverlayHide() {
        hideRunnable?.let { recTimerHandler.removeCallbacks(it) }
        hideRunnable = null
    }

    /** 底部相机控件条可见性：录像中始终显示（可停止）；否则仅当三层画面都就绪才显示。
     *  录像按钮/缩略图/切换钮在同一个 bottom_controls 容器里，一起显示/隐藏。主线程安全。 */
    private fun updateRecBtnVisible() {
        val show = playing   // 进入预览状态显示、退出预览隐藏（与录像按钮一起）
        if (show == lastRecBtnShow) return
        lastRecBtnShow = show
        recTimerHandler.post {
            findViewById<android.view.View>(R.id.bottom_controls).visibility =
                if (show) android.view.View.VISIBLE else android.view.View.GONE
        }
    }

    /** 前后摄切换反馈。 */
    private fun updateCamSwitchUi() {
        showStatusText(if (frontCamera) "前摄：手机 AR 人脸追踪" else "后摄：眼镜画面作基底")
    }

    /** 前台载入最近一条录像（跨会话）：查询 Movies 里最新的视频设为 lastVideoUri 并生成缩略图。 */
    private fun loadRecentLastVideo() {
        Thread {
            try {
                val cols = arrayOf(
                    MediaStore.Video.Media._ID,
                    MediaStore.Video.Media.DISPLAY_NAME,
                    MediaStore.Video.Media.RELATIVE_PATH)
                val c = contentResolver.query(MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                    cols, null, null, MediaStore.Video.Media.DATE_ADDED + " DESC") ?: return@Thread
                if (c.moveToFirst() && isMoviesRow(c)) {
                    lastVideoUri = android.content.ContentUris.withAppendedId(
                        MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                        c.getLong(c.getColumnIndexOrThrow(MediaStore.Video.Media._ID)))
                    loadLastThumb()
                }
                c.close()
            } catch (e: Exception) { android.util.Log.w(TAG, "load recent video fail", e) }
        }.start()
    }

    private fun isMoviesRow(c: android.database.Cursor): Boolean {
        return try {
            val idx = c.getColumnIndex(MediaStore.Video.Media.RELATIVE_PATH)
            if (idx >= 0 && !c.isNull(idx)) c.getString(idx).contains("Movies") else true
        } catch (_: Exception) { true }
    }

    /** 后台给最近录像生成缩略图并显示到左侧。 */
    private fun loadLastThumb() {
        val u = lastVideoUri ?: return
        Thread {
            try {
                val mmr = android.media.MediaMetadataRetriever()
                mmr.setDataSource(this, u)
                val bmp = mmr.getFrameAtTime(0,
                    android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                try { mmr.release() } catch (_: Exception) {}
                if (bmp != null) runOnUiThread {
                    findViewById<ImageView>(R.id.last_thumb).apply {
                        setImageBitmap(bmp)
                        visibility = android.view.View.VISIBLE
                    }
                }
            } catch (e: Exception) { android.util.Log.w(TAG, "load thumb fail", e) }
        }.start()
    }

    private fun showStatusText(msg: String) {
        recTimerHandler.post {
            statusText.text = msg
            if (msg.isBlank()) {                 // 空提示（如已连接的等待画面）→ 直接隐藏，不显示空框
                recTimerRunnable?.let { recTimerHandler.removeCallbacks(it) }
                cancelOverlayHide()
                recOverlay.visibility = android.view.View.GONE
                return@post
            }
            if (recorder?.recording == true) return@post   // 录像中：顶部保持闪烁计时，不刷新隐藏
            // 非录像：显示文字，有更新就顺延 5s，5s 无更新才自动隐藏
            cancelOverlayHide()
            recTimerRunnable?.let { recTimerHandler.removeCallbacks(it) }
            recOverlay.visibility = android.view.View.VISIBLE
            recOverlay.text = msg
            android.util.Log.i(TAG, "recOverlay: $msg")
            val hide = Runnable { recOverlay.visibility = android.view.View.GONE }
            hideRunnable = hide
            recTimerHandler.postDelayed(hide, 5000)
        }
    }

    /** 录像中：顶部显示【闪烁红点 + 录制时长】，常驻不隐藏。 */
    private fun showRecTimer() {
        recTimerHandler.post {
            cancelOverlayHide()                 // 取消先前"正在启动录像…"排的 5s 隐藏
            recTimerRunnable?.let { recTimerHandler.removeCallbacks(it) }
            recOverlay.visibility = android.view.View.VISIBLE
            val runnable = object : Runnable {
                override fun run() {
                    if (recorder?.recording != true) return
                    val ms = SystemClock.elapsedRealtime() - recStartEt
                    val s = ms / 1000
                    val h = s / 3600; val m = (s % 3600) / 60; val sec = s % 60
                    recDotOn = !recDotOn                     // 每秒亮/灭一次（亮 1 秒、灭 1 秒）
                    val dot = if (recDotOn) "●" else "○"          // 实/空心切换=闪烁且宽度不变
                    val t = if (h > 0) String.format("%d:%02d:%02d", h, m, sec)
                             else String.format("%02d:%02d", m, sec)
                    val sp = android.text.SpannableString("$dot $t")
                    sp.setSpan(android.text.style.ForegroundColorSpan(Color.RED), 0, 1,
                        android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    recOverlay.text = sp
                    if (recDotOnLogs < 3) { android.util.Log.i(TAG, "recOverlay: dot=$dot time=$t"); recDotOnLogs++ }
                    recTimerHandler.postDelayed(this, 1000)
                }
            }
            recTimerRunnable = runnable
            recTimerHandler.post(runnable)
        }
    }

    // ── 录像器：H.264 Surface 输入 + MediaMuxer 封装 MP4 ─────────────
    // 直接在编码器输入 Surface 上画合成位图（免手动 YUV），帧级所见即所录。
    // 输出写入系统录象文件夹（MediaStore Movies）。
    private inner class ArRecorder {
        @Volatile var recording = false; private set
        var displayName: String? = null
        var lastUri: android.net.Uri? = null            // 本段录像的 MediaStore 内容 URI（左侧缩略图/播放）
        val startedAtEt: Long get() = startMs           // 录制起始墙钟(elapsedRealtime)，顶部计时用
        /** 编码器输入 Surface（后摄 GL 模式挂到 SurfaceMixer 作输出；生命周期归编码器）。 */
        val inputSurface: android.view.Surface? get() = inSurf
        private var codec: MediaCodec? = null
        private var muxer: MediaMuxer? = null
        private var pfd: ParcelFileDescriptor? = null
        private var inSurf: android.view.Surface? = null
        private var track = -1
        private var muxStarted = false
        private var lastPtsUs = Long.MIN_VALUE
        private var pendingPtsUs = 0L
        private var startMs = 0L
        private var w = 0; private var h = 0
        private var encThread: Thread? = null
        private val tickLock = Object()

        // ---- 混音（眼镜端 PCM + 手机麦克风 → AAC 音轨）----
        // 主视频编码线程在 rec-encoder；音频混音编码在 audio-mix 线程，二者共享 muxer。
        private val muxerLock = Object()   // mediaMuxer 非线程安全，写样本需互斥
        private var wantsAudio = false       // 本路是否拥有音轨（麦克风权限 OK 且 AAC 就绪）
        private var audioTrack = -1           // 音频轨
        private var glassBytes = ByteArray(1 shl 16) // 眼镜 PCM 累积缓冲（交织 i16）
        private var glassLen = 0
        private var mic: AudioRecord? = null
        private var aac: MediaCodec? = null
        private var audioThread: Thread? = null
        private var audioPtsUs = 0L
        // 手机麦克风对数增强查表（样本异 → 增强后样本）：小音量提升、大音量压缩、满幅映射满幅不溢出
        private val micLut = ShortArray(65536)

        init {
            // y = sign*32767 * log1p(C·|x|) / log1p(C·32767)
            //  |x|→32767 时 y→32767（满幅不削顶）；|x| 很小时斜率≈32767·C/log1p(C·32767) 即放大倍数
            val c = 0.0005
            val denom = Math.log1p((c * 32767.0))
            for (idx in 0 until 65536) {
                val v = idx - 32768
                val ax = if (v < 0) -v else v
                val y = if (ax == 0) 0F
                        else (if (v > 0) 1F else -1F) * 32767F *
                            (Math.log1p(c * ax) / denom).toFloat()
                micLut[idx] = if (y > 32767F) 32767.toShort()
                              else if (y < -32768F) (-32768).toShort()
                              else y.toInt().toShort()
            }
        }

        fun start(w: Int, h: Int, preC: MediaCodec? = null, preS: android.view.Surface? = null) {
            val t0 = SystemClock.elapsedRealtime()
            fun pt(tag: String, t: Long) { android.util.Log.i(TAG, "rec: t $tag ${t - t0}ms") }
            val name = "rec_${System.currentTimeMillis()}.mp4"
            // 系统录象文件夹（Movies）建 MediaStore 条目，无需存储权限即可写入
            val cv = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                if (Build.VERSION.SDK_INT >= 29) {
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES)
                } else {
                    @Suppress("DEPRECATION")
                    put(MediaStore.Video.Media.DATA,
                        Environment.getExternalStoragePublicDirectory(
                            Environment.DIRECTORY_MOVIES).absolutePath + "/" + name)
                }
            }
            val uri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, cv)
                ?: throw IllegalStateException("MediaStore insert failed")
            lastUri = uri
            pfd = contentResolver.openFileDescriptor(uri, "rw")
                ?: throw IllegalStateException("openFileDescriptor failed")
            pt("mediaStore", t0)
            val c: MediaCodec
            val s: android.view.Surface
            if (preC != null && preS != null) {
                c = preC; s = preS                       // 直接复用预热好的编码器
            } else {
                c = MediaCodec.createEncoderByType("video/avc")
                val fmt = MediaFormat.createVideoFormat("video/avc", w, h)
                fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                fmt.setInteger(MediaFormat.KEY_BIT_RATE, 8_000_000)
                fmt.setInteger(MediaFormat.KEY_FRAME_RATE, 15)
                fmt.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                c.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                s = c.createInputSurface()
                c.start()
            }
            pt("avc-build", t0)
            codec = c; inSurf = s
            muxer = MediaMuxer(pfd!!.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            pt("muxer", t0)
            displayName = name
            track = -1; muxStarted = false
            this.w = w; this.h = h
            lastPtsUs = Long.MIN_VALUE
            startMs = SystemClock.elapsedRealtime()
            recording = true
            android.util.Log.i(TAG, "rec: start e(${w}x${h}) name=$name")
            // 麦克风权限 OK 则开混音音轨（眼镜 PCM 由 onAudioPackage→feedGlassesPcm 持续喂入）
            wantsAudio = micGranted() && initMicAndAac()
            pt("audio", t0)
            encThread = Thread({ encLoop() }, "rec-encoder").apply { start() }
        }

        /** Apple 帧级喂入眼镜端 PCM（decode 线程直投；仅录像时）。与 [onAudioPackage] 同路线。 */
        @Synchronized
        fun feedGlassesPcm(pcm: ByteArray) {
            if (!recording || pcm.isEmpty()) return
            val cap = glassBytes.size
            val keep = cap - glassLen
            if (keep > 0) {
                val n = minOf(pcm.size, keep)
                System.arraycopy(pcm, 0, glassBytes, glassLen, n)
                glassLen += n
            }
            // keep==0 表示混音线程消费不及，丢弃最旧（实时混音可接受）
        }

        /** 初始化麦克风(48k 单声道)+AAC(48k 立体声)编码器；失败返回 false（退化为无音轨）。 */
        private fun initMicAndAac(): Boolean {
            return try {
                val tA = SystemClock.elapsedRealtime()
                val sr = 48000
                val mr = try {
                    val ar = AudioRecord(
                        MediaRecorder.AudioSource.UNPROCESSED, sr,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                        AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT) * 2)
                    if (ar.state != AudioRecord.STATE_INITIALIZED) {
                        try { ar.release() } catch (_: Exception) {}
                        null
                    } else ar
                } catch (e: Exception) {
                    val ar2 = try {
                        val a = AudioRecord(
                            MediaRecorder.AudioSource.MIC, sr,
                            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                            AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO,
                                AudioFormat.ENCODING_PCM_16BIT) * 2)
                        if (a.state != AudioRecord.STATE_INITIALIZED) { null } else a
                    } catch (_: Exception) { null }
                    ar2
                }
                mic = mr
                android.util.Log.i(TAG, "rec: t mic-create ${SystemClock.elapsedRealtime() - tA}ms")
                val ac = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
                val af = MediaFormat.createAudioFormat(
                    MediaFormat.MIMETYPE_AUDIO_AAC, sr, 2)
                af.setInteger(MediaFormat.KEY_BIT_RATE, 128_000)
                af.setInteger(MediaFormat.KEY_AAC_PROFILE,
                    MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                af.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 8192)
                ac.configure(af, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                android.util.Log.i(TAG, "rec: t aac-build ${SystemClock.elapsedRealtime() - tA}ms")
                ac.start()
                aac = ac
                audioTrack = -1
                audioPtsUs = 0L
                glassLen = 0
                try { mr?.startRecording() } catch (_: Exception) {}
                android.util.Log.i(TAG, "rec: t mic-start ${SystemClock.elapsedRealtime() - tA}ms")
                audioThread = Thread({ mixLoop() }, "audio-mix").apply { start() }
                android.util.Log.i(TAG, "rec: audio mix on (mic=${mr != null})")
                true
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "rec: audio mix disabled", t)
                false
            }
        }

        /** 每帧由 render() 调用：唤醒编码线程采一帧（自身卡顿不会阻塞显示线程） */
        fun tick() {
            if (!recording) return
            synchronized(tickLock) { tickLock.notifyAll() }
        }

        /** 编码线程：>15fps 采样 live 画面 → 编码器输入 Surface（GPU 加速）→ muxer。
         *  后摄 GL 模式下合成结果由 SurfaceMixer 直接推到编码器面，本线程只做取包+PTS。 */
        private fun encLoop() {
            while (recording) {
                val s = inSurf
                if (s == null) { sleepTick(); continue }
                val glFeed = !frontCamera && rearMixer != null
                if (!glFeed) {
                    var cv: Canvas? = null
                    try {
                        cv = if (Build.VERSION.SDK_INT >= 26) s.lockHardwareCanvas()
                            else @Suppress("DEPRECATION") s.lockCanvas(Rect(0, 0, 0, 0))
                        // 直接重画合成画面（局部 scratch，可与显示线程并发）
                        paintFrame(cv, w.toFloat(), h.toFloat())
                    } catch (_: Exception) {
                    } finally {
                        try { s.unlockCanvasAndPost(cv) } catch (_: Exception) {}
                    }
                }
                // 每帧取真实墙钟作为该帧 PTS（单调），时长=真实录制时长
                pendingPtsUs = (SystemClock.elapsedRealtime() - startMs) * 1000L
                try { drain(false) } catch (_: Exception) {}
                synchronized(tickLock) {
                    try { tickLock.wait(62) } catch (_: InterruptedException) {}
                }
            }
        }

        private fun sleepTick() {
            synchronized(tickLock) { try { tickLock.wait(62) } catch (_: InterruptedException) {} }
        }

        private fun drain(eosWait: Boolean) {
            val c = codec ?: return
            val mx = muxer ?: return
            val bi = MediaCodec.BufferInfo()
            var guard = 0
            while (true) {
                val oi = c.dequeueOutputBuffer(bi, if (eosWait) 20000 else 0)
                if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (track < 0) { track = mx.addTrack(c.outputFormat); tryStartMuxer() }
                } else if (oi >= 0) {
                    if (track >= 0 && muxStarted) {
                        // 用本帧墙钟 PTS（pendingPtsUs），强制单调：<= 前一帧则 +1 顶上来
                        var p = pendingPtsUs
                        if (p <= lastPtsUs) { p = lastPtsUs + 1; pendingPtsUs = p }
                        bi.presentationTimeUs = p
                        lastPtsUs = p
                        synchronized(muxerLock) {
                            mx.writeSampleData(track, c.getOutputBuffer(oi)!!, bi)
                        }
                    }
                    val eosFlag = (bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                    c.releaseOutputBuffer(oi, false)
                    if (eosFlag) break
                } else if (oi == MediaCodec.INFO_TRY_AGAIN_LATER) {
                    if (!eosWait) break
                    if (++guard > 150) break   // ~3s 兜底，避免卡死
                } else {
                    break
                }
            }
        }

        /** muxer.start() 必须在所有音视频轨 addTrack 之后；全部就绪才启动全局写样本。 */
        private fun tryStartMuxer() {
            val mx = muxer ?: return
            if (muxStarted) return
            val audioReady = !wantsAudio || audioTrack >= 0
            if (track >= 0 && audioReady) {
                synchronized(muxerLock) {
                    if (!muxStarted) {
                        try { mx.start(); muxStarted = true; android.util.Log.i(TAG, "rec: muxer started (vt=$track at=$audioTrack)") } catch (e: Exception) { android.util.Log.w(TAG, "rec: muxer.start fail", e) }
                    }
                }
            }
        }

        /** 混音线程：麦克风(单声道) + 眼镜PCM(立体声) → 立体声 → AAC → muxer 音轨。 */
        private fun mixLoop() {
            val frameSamples = 960          // 20ms @48k
            val ac = aac ?: return
            val micRec = mic
            val micShort = ShortArray(frameSamples)
            val glass = ByteArray(frameSamples * 4) // 本帧眼镜立体声 i16 交织（L0 R0 L1 R1…）
            val outBi = ByteBuffer.allocateDirect(frameSamples * 4).order(ByteOrder.LITTLE_ENDIAN)
            val bf = MediaCodec.BufferInfo()
            while (recording) {
                val samplesGot = if (micRec != null) {
                    val got = micRec.read(micShort, 0, frameSamples); if (got > 0) got else 0
                } else 0
                // 取一帧眼镜 PCM（交错 i16），不足空隙补零
                val consumed = synchronized(this) {
                    val n = minOf(glassLen, glass.size)
                    if (n > 0) System.arraycopy(glassBytes, 0, glass, 0, n)
                    shiftGlass(n)
                    n
                }
                outBi.clear()
                var k = 0
                for (i in 0 until frameSamples) {
                    val l: Int; val r: Int
                    if (k + 4 <= consumed) {
                        l = (glass[k].toInt() and 0xFF) or (glass[k + 1].toInt() shl 8)
                        r = (glass[k + 2].toInt() and 0xFF) or (glass[k + 3].toInt() shl 8)
                        k += 4
                    } else { l = 0; r = 0 }
                    val m = if (i < samplesGot) micLut[(micShort[i].toInt() and 0xFFFF)].toInt() else 0
                    outBi.putShort(clamp(l + m))
                    outBi.putShort(clamp(r + m))
                }
                outBi.rewind()
                val idx = ac.dequeueInputBuffer(10000)
                if (idx >= 0) {
                    val buf = ac.getInputBuffer(idx) ?: continue
                    buf.clear()
                    buf.put(outBi)
                    ac.queueInputBuffer(idx, 0, frameSamples * 4, audioPtsUs, 0)
                    audioPtsUs += 20000L   // 20ms/帧
                }
                drainAac(ac, bf)
                // 无麦克风时无阻塞读，需自定节奏(20ms/帧)
                if (micRec == null) {
                    synchronized(tickLock) { try { tickLock.wait(20) } catch (_: InterruptedException) {} }
                }
            }
        }

        /** 把已消费的眼镜 PCM 字节从累积缓冲移除（前移剩余）。调用方须持 synchronized(this)。 */
        private fun shiftGlass(consumed: Int) {
            if (consumed <= 0) return
            val remain = glassLen - consumed
            if (remain > 0) System.arraycopy(glassBytes, consumed, glassBytes, 0, remain)
            glassLen = remain
            java.util.Arrays.fill(glassBytes, remain, glassBytes.size, (0).toByte())
        }

        private fun clamp(v: Int): Short =
            if (v > 32767) 32767 else if (v < -32768) -32768 else v.toShort()

        private fun drainAac(ac: MediaCodec, bi: MediaCodec.BufferInfo) {
            val mx = muxer ?: return
            while (true) {
                val oi = ac.dequeueOutputBuffer(bi, 0)
                if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (audioTrack < 0) { audioTrack = mx.addTrack(ac.outputFormat); tryStartMuxer() }
                } else if (oi >= 0) {
                    if (audioTrack >= 0 && muxStarted) {
                        synchronized(muxerLock) {
                            try { mx.writeSampleData(audioTrack, ac.getOutputBuffer(oi)!!, bi) } catch (_: Exception) {}
                        }
                    }
                    ac.releaseOutputBuffer(oi, false)
                } else break
            }
        }

        fun stop() {
            recording = false
            val t = encThread
            if (t != null && t.isAlive) {
                synchronized(tickLock) { tickLock.notifyAll() }
                try { t.join(2000) } catch (_: InterruptedException) {}
            }
            encThread = null
            // 停混音线程（它用 tickLock 做无麦克风时的等待节奏）
            val at = audioThread
            if (at != null && at.isAlive) {
                synchronized(tickLock) { tickLock.notifyAll() }
                try { at.join(1500) } catch (_: InterruptedException) {}
            }
            audioThread = null
            finishAudio()
            val c = codec
            if (c == null) { release(); return }
            try {
                try { c.signalEndOfInputStream() } catch (_: Exception) {}
                drain(true)
            } catch (_: Exception) {
            }
            release()
        }

        /** 收尾音轨：给 AAC 送 EOS 并 drain 残留样本（须已停 audioThread）。 */
        private fun finishAudio() {
            val ac = aac ?: return
            try {
                val bi = MediaCodec.BufferInfo()
                val idx = ac.dequeueInputBuffer(5000)
                if (idx >= 0) {
                    ac.queueInputBuffer(idx, 0, 0, audioPtsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                }
                var guard = 0
                while (true) {
                    val oi = ac.dequeueOutputBuffer(bi, 20000)
                    if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        if (audioTrack < 0) { audioTrack = muxer?.addTrack(ac.outputFormat) ?: -1; tryStartMuxer() }
                    } else if (oi >= 0) {
                        if (audioTrack >= 0 && muxStarted) {
                            synchronized(muxerLock) {
                                try { muxer!!.writeSampleData(audioTrack, ac.getOutputBuffer(oi)!!, bi) } catch (_: Exception) {}
                            }
                        }
                        ac.releaseOutputBuffer(oi, false)
                        if ((bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                    } else if (oi == MediaCodec.INFO_TRY_AGAIN_LATER) {
                        if (++guard > 100) break
                    } else break
                }
            } catch (_: Exception) {
            }
        }

        fun release() {
            // 停视频编码线程
            val t = encThread
            if (t != null && t.isAlive) {
                synchronized(tickLock) { tickLock.notifyAll() }
                try { t.join(500) } catch (_: InterruptedException) {}
            }
            encThread = null
            // 停混音线程 + 关麦克风/AAC
            val at = audioThread
            if (at != null && at.isAlive) {
                synchronized(tickLock) { tickLock.notifyAll() }
                try { at.join(500) } catch (_: InterruptedException) {}
            }
            audioThread = null
            try { mic?.stop() } catch (_: Exception) {}
            try { mic?.release() } catch (_: Exception) {}
            mic = null
            try { aac?.stop() } catch (_: Exception) {}
            try { aac?.release() } catch (_: Exception) {}
            aac = null
            try { codec?.stop() } catch (_: Exception) {}
            try { codec?.release() } catch (_: Exception) {}
            try { muxer?.stop() } catch (e: Exception) { android.util.Log.w(TAG, "rec: muxer.stop fail", e) }
            try { muxer?.release() } catch (_: Exception) {}
            try { pfd?.close() } catch (_: Exception) {}
            codec = null; muxer = null; pfd = null; inSurf = null; track = -1; muxStarted = false
        }
    }

    private fun statusTextFor(): String = when (glassesState) {
        0 -> "未连接眼镜"
        1 -> "连接眼镜中…"
        else -> ""          // 已连接：不再提示"等待画面"，避免遮挡
    }

    /** 在锚点（或屏幕中心）画半透明底的连接状态。 */
    private fun drawConnectionStatus(c: Canvas, cx: Float, cy: Float) {
        val s = statusTextFor()
        val tw = textPaint.measureText(s)
        val th = textPaint.textSize
        val pad = 10 * resources.displayMetrics.density
        c.drawRoundRect(
            RectF(cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad),
            10f, 10f, panelPaint)
        c.drawText(s, cx - tw / 2, cy + th / 3, textPaint)
    }

    /** 解码线程回调：overlay I420 Image → RGBA Bitmap（BT.601 limited，兼容 planar/semiplanar + cropRect）。
     *  行数据先 bulk 拷到 scratch 数组再逐像素索引（直接读 direct ByteBuffer 逐次都是 JNI）。 */
    private fun convertOverlayFrame(img: Image) {
        try {
            // 用 cropRect 的真实可见区：厂商解码可能带 16/32 对齐填充 + 裁剪偏移，
            // img.width/height 含 padding，若直接用它取“中段 strip”会把填充行也算进去，
            // 取到的就是错位的条带 → 字形错位/残影。
            val crop = img.cropRect
            val w = crop.width()
            val hFull = crop.height()
            // 背景层恒为 480×480 内容方区（眼镜流 480×640，上下各 80px 恒空带）；
            // 未连接/无内容时同样绘制背景板并显示绿色"连接中..."
            val topRow = if (hFull > w) (hFull - w) / 2 else 0
            val h = if (hFull > w) w else hFull
            val pl = img.planes
            val yB = pl[0].buffer
            val uB = pl[1].buffer
            val vB = pl[2].buffer
            val yRs = pl[0].rowStride
            val uRs = pl[1].rowStride
            val vRs = pl[2].rowStride
            val uPs = pl[1].pixelStride
            val vPs = pl[2].pixelStride
            val cw = (w + 1) shr 1
            // scratch 复用（解码线程串行调用）
            var ys = yScr
            if (ys == null || ys.size < yRs * (h + 1)) { ys = ByteArray(yRs * (h + 1)); yScr = ys }
            var us = uScr
            if (us == null || us.size < cw * uPs * ((h shr 1) + 2)) {
                us = ByteArray(cw * uPs * ((h shr 1) + 2)); uScr = us
            }
            var vs = vScr
            if (vs == null || vs.size < cw * vPs * ((h shr 1) + 2)) {
                vs = ByteArray(cw * vPs * ((h shr 1) + 2)); vScr = vs
            }
            synchronized(overlayLock) {
                var pix = overlayPix
                if (pix == null || pix.size != w * h) { pix = IntArray(w * h); overlayPix = pix }
                // 防“残影”：pix 跨帧复用，若本帧因 limit 提前 break 未写满，残留上一帧内容会
                // 整块拷入 ob → 字符拖影。填零（全透明）确保未写到的行是空、而非旧帧。
                java.util.Arrays.fill(pix, 0)
                if (overlayBmp == null || overlayBmp!!.width != w || overlayBmp!!.height != h) {
                    overlayBmp?.recycle()
                    overlayBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                }
                if (overlayBmp == null || overlayBmp!!.width != w || overlayBmp!!.height != h) {
                    overlayBmp?.recycle()
                    overlayBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    ArNative.nativeSetOverlaySize(w, h)
                }
                val cropL = crop.left
                val cropT = crop.top
                val uBytes = cw * uPs
                var o = 0
                for (j in 0 until h) {
                    val yRow = (j + topRow + cropT) * yRs + cropL
                    if (yRow + w > yB.limit()) break
                    yB.position(yRow)
                    yB.get(ys, 0, w)                       // Y 行 bulk 拷贝
                    if ((j and 1) == 0) {
                        // 色度行随 crop 同步偏移（4:2:0：cTop = (y+crop.top)/2 + crop.left/2）
                        val cTop = ((j + topRow + cropT) shr 1) + (cropL shr 1)
                        val uRow = cTop * uRs
                        if (uRow + uBytes > uB.limit() || uRow + uBytes > vB.limit()) break
                        uB.position(uRow)
                        uB.get(us, 0, uBytes)              // 按 pixelStride 读满（planar=1 / NV12=2）
                        vB.position(cTop * vRs)
                        vB.get(vs, 0, cw * vPs)
                    }
                    for (i in 0 until w) {
                        // 水平镜像：观察者看到的眼镜屏与内容左右相反（旧 GL 路径 u→1-u 同理）
                        val si = w - 1 - i
                        val c = si shr 1
                        val y = (ys[si].toInt() and 0xFF) - 16
                        // planar：像素间距 1 直读；semiplanar(NV12)：UV 交错，U 在偶位 V 在奇位
                        val u = if (uPs == 1) (us[c].toInt() and 0xFF) - 128
                                else (us[c * uPs].toInt() and 0xFF) - 128
                        val v = if (vPs == 1) (vs[c].toInt() and 0xFF) - 128
                                else (vs[c * vPs + 1].toInt() and 0xFF) - 128
                        var r = (298 * y + 409 * v + 128) shr 8
                        var g = (298 * y - 100 * u - 208 * v + 128) shr 8
                        var b = (298 * y + 516 * u + 128) shr 8
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255
                        // 黑键抠像：luma 查 LUT 得 alpha（暗褐 → 0）
                        val luma = (299 * r + 587 * g + 114 * b) / 1000
                        val a = keyLut[luma.coerceIn(0, 255)]
                        // 饱和度 ×1.12（l = Rec.601 luma；c' = l + (c-l)×28/25）
                        r = (luma + (r - luma) * 28 / 25).coerceIn(0, 255)
                        g = (luma + (g - luma) * 28 / 25).coerceIn(0, 255)
                        b = (luma + (b - luma) * 28 / 25).coerceIn(0, 255)
                        pix[o] = (a shl 24) or (r shl 16) or (g shl 8) or b
                        o++
                    }
                }
                overlayBmp!!.setPixels(pix, 0, w, 0, 0, w, h)
                // 发光底图：ob 降采样（1/4 中距 + 1/16 远距），与内容同源、随姿态同步
                val gw = maxOf(8, w shr 2); val gh = maxOf(8, h shr 2)
                val fw = maxOf(4, w shr 4); val fh = maxOf(4, h shr 4)
                if (glowBmp == null || glowBmp!!.width != gw || glowBmp!!.height != gh) {
                    glowBmp?.recycle()
                    glowBmp = Bitmap.createBitmap(gw, gh, Bitmap.Config.ARGB_8888)
                    glowCanvas = Canvas(glowBmp!!)
                }
                if (bloomFarBmp == null || bloomFarBmp!!.width != fw || bloomFarBmp!!.height != fh) {
                    bloomFarBmp?.recycle()
                    bloomFarBmp = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
                    bloomFarCanvas = Canvas(bloomFarBmp!!)
                }
                glowSrc.set(0, 0, w, h); glowDst.set(0, 0, gw, gh); farDst.set(0, 0, fw, fh)
                glowCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                glowCanvas!!.drawBitmap(overlayBmp!!, glowSrc, glowDst, glowAlphaPaint)
                bloomFarCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                bloomFarCanvas!!.drawBitmap(overlayBmp!!, glowSrc, farDst, bloomFarAlphaPaint)
                overlayNew = true
                if (!overLayerReady) { overLayerReady = true; updateRecBtnVisible() }   // 第二层：overlay 首帧解码
            }
        } catch (t: Throwable) {
            // Throwable：JNI 方法不匹配等 Error 也不得杀死解码线程
            android.util.Log.w(TAG, "overlay frame convert fail", t)
        } finally {
            try { img.close() } catch (_: Exception) {}
        }
    }

    private fun showStatus(s: String?) { showStatusText(s ?: statusTextFor()) }

    // ── scrcpy overlay 连接 ───────────────────────────────────

    private fun startAr() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        settingsPanel.visibility = android.view.View.GONE
        arPanel.visibility = android.view.View.VISIBLE
        progressView.visibility = android.view.View.VISIBLE
        playing = true
        updateRecBtnVisible()   // 进入预览 → 显示底部三键（缩略图-录像-切换）
        startFrontConnection()
        showStatusText(statusTextFor())
        // 预览就绪后后台预热录像编码器，避免首次点录像时偶发 1~5s 初始化
        camHandler?.postDelayed({
            // 预热尺寸与录像面一致：前摄=全屏 Canvas 面；后摄=3:4 GL 面
        val ws = if (frontCamera) cameraView.width else glSurface.width
        val hs = if (frontCamera) cameraView.height else glSurface.height
            if (ws > 0 && hs > 0) armWarmVideo(ws, hs)
        }, 1800)
    }

    // ── 子系统切换：前摄（overlay-only + Canvas AR）与后摄（普通模式管线 + GL）互相独立 ──

    /** 起前摄 overlay-only 连接（CPU overlay 帧供人脸锚点投影）。 */
    private fun startFrontConnection() {
        val ip = prefs.getString("ip", null) ?: ipEdit.text.toString().trim()
        if (ip.isEmpty()) { showStatusText("未配置眼镜 IP"); return }
        frontActive = true
        glassesState = 1
        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this, audioEnabled = audioEnabled, overlayOnly = true)
        connection!!.connectAsync(ip, 5555, callback)
    }

    /** 切前摄：先起新的 overlay-only 连接，再异步停掉后摄管线（killall 误杀窗口与旧实现相同）。 */
    private fun enterFrontMode() {
        frontCamera = true
        frontActive = true
        glassesState = 1
        camHandler?.post { closeCameraNow(); startCamera() }
        glSurface.visibility = android.view.View.GONE
        startFrontConnection()
        showStatusText(statusTextFor())
        val old = rearPlayer
        rearPlayer = null
        rearActive = false
        if (old != null) Thread { try { old.stop() } catch (_: Exception) {} }.start()
    }

    /** 切后摄：停前摄子系统，起完全复用普通模式的 GlassesPlayer（全流硬解 → GL 合成）。 */
    private fun enterRearMode() {
        rearActive = true
        glassesState = 1
        camHandler?.post { closeCameraNow() }   // 后摄基底=眼镜视频，手机相机整体停掉省电
        glSurface.visibility = android.view.View.VISIBLE   // surfaceChanged → ensureRearMixer → startRearPlayerIfReady
        startRearPlayerIfReady()
        showStatusText("后摄：连接眼镜画面…")
        val oldConn = connection
        val oldDec = overlayDecoder
        connection = null
        overlayDecoder = null
        frontActive = false
        if (oldConn != null || oldDec != null) Thread {
            try { oldConn?.disconnect() } catch (_: Exception) {}
            try { oldDec?.stop() } catch (_: Exception) {}
        }.start()
        try { glassesAudio?.stop() } catch (_: Exception) {}
        glassesAudio = null
        resetOverlay()
    }

    /** 后摄管线就绪即连接（等 GL 面尺寸出来建好 mixer）。 */
    private fun startRearPlayerIfReady() {
        if (!rearActive || rearPlayer != null) return
        val mx = rearMixer ?: return
        val ip = prefs.getString("ip", null) ?: ipEdit.text.toString().trim()
        if (ip.isEmpty()) { showStatusText("未配置眼镜 IP"); return }
        mx.reset()   // 重建两路输入面，清掉上一会话残留帧
        val player = GlassesPlayer(
            this, mx, audioEnabled, bottomRotationDeg, bottomMirror,
            applySettings = { applySettingsToMixer(it) },
            events = rearEvents,
        )
        rearPlayer = player
        player.connect(ip)
    }

    /** 停后摄子系统（幂等）：断开管线、藏 GL 面。异步释放，不阻塞调用线程。 */
    private fun exitRearMode() {
        rearActive = false
        val old = rearPlayer
        rearPlayer = null
        if (old != null) Thread { try { old.stop() } catch (_: Exception) {} }.start()
        glSurface.visibility = android.view.View.GONE
    }

    /** 断开/出错后收尾：停录像、复位两个子系统、回到设置面板。可在任意线程调用。 */
    private fun teardownToSettings(msg: String) {
        runOnUiThread {
            if (recorder?.recording == true) stopRec()
            overlayDecoder?.stop()
            overlayDecoder = null
            resetOverlay()
            exitRearMode()
            arPanel.visibility = android.view.View.GONE
            settingsPanel.visibility = android.view.View.VISIBLE
            findViewById<android.view.View>(R.id.bottom_controls).visibility = android.view.View.GONE
            lastRecBtnShow = false
            camLayerReady = false; overLayerReady = false; anchorLayerReady = false
            glassesState = 0
            showStatusText(msg)
            restoreFrontMode()
        }
    }

    private fun resetOverlay() {
        synchronized(overlayLock) {
            overlayBmp?.recycle()
            overlayBmp = null
            overlayNew = false
        }
    }

    private fun exitAr() {
        playing = false
        frontActive = false
        val c = connection
        connection = null
        if (c != null) Thread { try { c.disconnect() } catch (_: Exception) {} }.start()
        try { glassesAudio?.stop() } catch (_: Exception) {}
        glassesAudio = null
        teardownToSettings("")
    }

    /** 恢复前摄 AR 基准态：隐藏 GL 合成面、重开手机前摄。退出预览/断连时调用。 */
    private fun restoreFrontMode() {
        frontCamera = true
        glSurface.visibility = android.view.View.GONE
        camHandler?.post {
            closeCameraNow()
            startCamera()
        }
    }

    // ── 前摄子系统回调：overlay-only 连接（无视频流），overlay CPU 帧供人脸投影 ──
    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            glassesState = 2
            runOnUiThread {
                progressView.visibility = android.view.View.GONE
                showStatusText(statusTextFor())
            }
        }

        override fun onVideoPrepare(codec: String, width: Int, height: Int) {}
        override fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int) {}

        override fun onAudioPrepare(codec: String) {
            // 与 onAudioConfig 同在音频 Handler 线程上顺序执行，须在此同步建好播放器，
            // 不能再走 UI 线程（否则 onAudioConfig 先到会拿不到 glassesAudio）。
            try {
                val prev = glassesAudio
                try { prev?.stop() } catch (_: Exception) {}
                val ap = AudioPlayer()
                // 解码出的眼镜 PCM（统一立体声）同时喂给录像混音
                ap.onPcm = { pcm -> recorder?.feedGlassesPcm(pcm) }
                glassesAudio = ap
            } catch (e: Exception) {
                android.util.Log.w(TAG, "眼镜音频就绪失败", e)
            }
        }

        override fun onAudioConfig(csd0: ByteArray) {
            try {
                val ap = glassesAudio
                if (ap != null) ap.start(csd0) else android.util.Log.w(TAG, "audio config 无播放器，丢弃")
            } catch (e: Exception) {
                android.util.Log.w(TAG, "眼镜音频配置失败", e)
            }
        }

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {
            try {
                glassesAudio?.play(buffer, offset, length)
            } catch (e: Exception) {
                android.util.Log.w(TAG, "眼镜音频播放中断", e)
            }
        }

        override fun onOverlayPrepare(codec: String, width: Int, height: Int) {
            runOnUiThread {
                showStatusText("overlay ${width}×${height}，请正对手机摄像头")
                try {
                    if (overlayDecoder != null) {
                        runCatching { overlayDecoder?.stop() }   // 重连/复用前停掉旧解码器
                        overlayDecoder = null
                    }
                    resetOverlay()
                    val od = VideoDecoder()
                    od.frameCallback = { img -> convertOverlayFrame(img) }
                    // 无 Surface 解码：getOutputImage 给出 CPU 可读 I420（绕开厂商平铺格式）
                    od.start(width, height, null)
                    overlayDecoder = od
                } catch (e: Exception) {
                    android.util.Log.w(TAG, "overlay prepare 失败", e)
                }
            }
        }

        override fun onOverlayPackage(buffer: ByteArray, offset: Int, length: Int) {
            try {
                overlayDecoder?.decode(buffer, offset, length)
            } catch (e: Exception) {
                android.util.Log.w(TAG, "overlay 解码中断", e)
            }
        }

        override fun onDisconnect() {
            if (playing && frontActive) {
                playing = false
                teardownToSettings("连接已断开")
            }
        }

        override fun onError() {
            if (playing && frontActive) {
                playing = false
                runOnUiThread {
                    progressView.visibility = android.view.View.GONE
                    showStatusText("连接失败，请检查眼镜 IP 与网络")
                    Toast.makeText(this@ArActivity, "连接失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ── 后摄子系统事件（GlassesPlayer = 普通模式管线）──
    private val rearEvents = object : GlassesPlayer.Events {
        override fun onConnect() {
            glassesState = 2
            runOnUiThread {
                progressView.visibility = android.view.View.GONE
                showStatusText(statusTextFor())
            }
        }

        override fun onDisconnect() {
            if (playing && rearActive) {
                playing = false
                teardownToSettings("连接已断开")
            }
        }

        override fun onError() {
            if (rearActive) {
                playing = false
                exitRearMode()
                runOnUiThread {
                    progressView.visibility = android.view.View.GONE
                    showStatusText("连接失败，请检查眼镜 IP 与网络")
                    Toast.makeText(this@ArActivity, "连接失败", Toast.LENGTH_SHORT).show()
                    restoreFrontMode()   // 后摄失败时手机相机已关，重开前摄保持可预览
                }
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (playing) {
            exitAr()
        } else {
            finish()
        }
    }

    override fun onDestroy() {
        playing = false
        frontActive = false
        rearActive = false
        connection?.disconnect()
        stopCamera()
        val player = rearPlayer
        rearPlayer = null
        if (player != null) Thread { try { player.stop() } catch (_: Exception) {} }.start()
        det.shutdown()
        openExec.shutdown()
        resetOverlay()
        synchronized (warmLock) {
            try { warmSurf?.let { if (it.isValid) it.release() } } catch (_: Exception) {}
            try { warmCodec?.release() } catch (_: Exception) {}
            warmCodec = null; warmSurf = null
        }
        try { rearMixer?.release() } catch (_: Exception) {}
        rearMixer = null
        renderExec.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "ar-java"
        private const val REQ_CAM = 1
        private const val REQ_MIC = 2
        // det_10g 图内写死了 448 输入的 FPN 上采样尺寸，实时检测同样固定 448
        private const val LIVE_INPUT = 448
        private const val SCORE_THRESH = 0.50f
    }
}
