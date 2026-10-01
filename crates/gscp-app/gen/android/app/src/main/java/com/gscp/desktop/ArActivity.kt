package com.gscp.desktop

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * AR 模式（ncnn-benchmark CameraActivity 移植）：
 * 前摄：Camera2 (YUV_420_888) → ArNative.nativeFaceDetect（det_10g + FaceMesh 融合姿态，
 * 全部在 native）→ 检测直写的直立 RGBA 帧与 overlay 四角锚点交给 ArFrontGl(GLES2) 上屏：
 * 相机纹理 + 背板 + overlay 单应网格 + bloom 全部 GPU 合成（与后摄 SurfaceMixer 同构，
 * 录像编码器面直挂输出，同一份合成结果上屏与录制）。检测与上屏用同一份像素，
 * 叠加层与检测输入严格对应（所见即所测）。
 * 后摄：眼镜画面作基底，走 GlassesPlayer → SurfaceMixer GL 管线（普通模式管线复用）。
 */
class ArActivity : Activity() {

    private lateinit var prefs: android.content.SharedPreferences
    private lateinit var settingsPanel: android.view.View
    private lateinit var arPanel: android.view.View
    private lateinit var cameraView: SurfaceView
    private lateinit var statusText: TextView
    private lateinit var progressView: android.view.View
    private lateinit var ipEdit: EditText

    // 前摄 overlay 会话（后摄同款 GlassesPlayer 管线，overlayOnly 连接）：
    // 连接/解码/音频生命周期全部由 player 自管，ArActivity 只消费事件。
    private var frontPlayer: GlassesPlayer? = null
    private val overlayLock = Any()
    private var overlayBmp: Bitmap? = null
    private var overlayPix: IntArray? = null
    private var yScr: ByteArray? = null
    private var uScr: ByteArray? = null
    private var vScr: ByteArray? = null
    private var overlayNew = false
    // overlay 内容版本号：convertOverlayFrame/resetOverlay 递增，GL 合成器据此刷新纹理
    @Volatile private var overlayVersion = 0
    @Volatile var diagOverlayPkg = 0L   // 诊断：overlay 网络帧计数
    private var dumpCount = 0

    // ── 后摄子系统：完全复用普通模式（MainActivity）管线 ──
    // GlassesPlayer 自持连接/双硬解/音频；rearGl 输出到 ar_gl_surface 显示，
    // 录像时编码器面直接挂 mixer 输出。与前摄不共享任何连接/解码状态。
    private var rearPlayer: GlassesPlayer? = null
    private var rearGl: ArRearGl? = null
    private lateinit var glSurface: SurfaceView
    private var glAttached: Surface? = null

    // ── 前摄子系统 GL 合成器（ArFrontGl）：与后摄 SurfaceMixer 同构的多输出面合成 ──
    // 相机帧（nativeFaceDetect 直写的直立 RGBA）+ overlay 键控/bloom 位图作纹理输入，
    // 单应网格/背板/柔光 GPU 合成；输出到 ar_surface 显示，录像编码器面直挂输出。
    private var frontGl: ArFrontGl? = null
    private var frontGlAttached: Surface? = null
    // 相机帧三缓冲轮转：det 线程写一块、交 GL 上传，避免 GL 读取期间被下一帧覆写
    private val camBufLock = Any()
    private var camBufs = arrayOfNulls<ByteBuffer>(3)
    private var camBufIdx = 0
    private var camBufBytes = 0
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

    @Volatile private var faceOv = arrayOf<FloatArray>()       // 每脸 17 值：四角 8 + 背板 8 + 距离 cm
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
    private var detLogTick = 0L
    // 检测后端：默认 CPU FP32。真机取证（vivo Mali）：ncnn Vulkan 检测与 overlay
    // GL 烘焙并发 ~3.5s 后 Vulkan 队列楔死（det 线程 native 空转、锚点停更、相机
    // 画面冻结）；后摄无检测故不受影响。debug.gscp.detgpu=1 切回 Vulkan FP16 A/B。
    private val backendIdx: Int = systemPropInt("debug.gscp.detgpu", 0)

    private fun systemPropInt(key: String, def: Int): Int = try {
        val sp = Class.forName("android.os.SystemProperties")
        val get = sp.getMethod("get", String::class.java, String::class.java)
        (get.invoke(null, key, def.toString()) as String).trim().toIntOrNull() ?: def
    } catch (_: Throwable) { def }
    private var playing = false
    private var openSent = false
    private var dbgFrames = 0

    // ── 以下 Paint/Canvas 均为 overlay CPU 特效烘焙(convertOverlayFrame)专用；
    //    合成上屏的画笔(相机压暗/内容提亮/背板/柔光/连接提示)已随 Canvas 路径移入
    //    ArFrontGl shader，相关 Paint 与渲染线程已删除。

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

    // overlay 发光（均匀化）烘焙：与旧实现同构（1/4 + 1/16 内容降采样），
    // 剪影统一着色（不透明像素平均色 + alpha 饱和，任意字符贡献相同光源）+ 金字塔
    // 多级降/升采样烘焙平滑（纯双线性缩放，必定生效）。三张产物位图（内容 / 1/4 光晕 /
    // 1/16 光晕）由 ArFrontGl 作为纹理上传，合成阶段 GPU 完成 ADD 光晕 + SRC_OVER 内容。
    private val glowCm = android.graphics.ColorMatrix()
    private val glowAlphaPaint = Paint(Paint.FILTER_BITMAP_FLAG)   // 剪影绘制（每帧设平均色 + alpha 饱和）
    private val glowSmoothPaint = Paint(Paint.FILTER_BITMAP_FLAG)  // 金字塔缩放（纯双线性低通）
    private var glowBmp: Bitmap? = null       // ob 1/4 降采样（中距光晕，烘焙平滑后）
    private var glowCanvas: Canvas? = null
    private var bloomFarBmp: Bitmap? = null   // ob 1/16 降采样（远距光晕，烘焙平滑后）
    private var bloomFarCanvas: Canvas? = null
    private var glowHalfBmp: Bitmap? = null   // ob 1/8 中间层（金字塔低通用）
    private var glowHalfCanvas: Canvas? = null
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
                val g = ensureRearGl(w, ht)
                val s = h.surface
                if (glAttached !== s) {
                    glAttached?.let { g.detachOutputSurface(it) }
                    glAttached = s
                    g.attachOutputSurface(s)
                }
                startRearPlayerIfReady()
            }

            override fun surfaceDestroyed(h: SurfaceHolder) {
                glAttached?.let { rearGl?.detachOutputSurface(it) }
                glAttached = null
            }
        })

        loadSettings()   // 与第一页同一组合成参数（同名 prefs key）

        cameraView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(h: SurfaceHolder) {
                android.util.Log.i("ar-ui", "surfaceCreated")
                surfaceReady = true; tryStart()
            }
            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {
                android.util.Log.i("ar-ui", "surfaceChanged ${w}x$ht")
                // 前摄 GL 合成输出面：与后摄 glSurface 同样的挂载/摘除模式
                val g = ensureFrontGl(w, ht)
                val s = h.surface
                if (frontGlAttached !== s) {
                    frontGlAttached?.let { g.detachOutputSurface(it) }
                    frontGlAttached = s
                    g.attachOutputSurface(s)
                }
                g.setActive(frontCamera)
                maybeStartFrontPlayer()   // 后摄同款：GL 面就绪 → 起会话
            }

            override fun surfaceDestroyed(h: SurfaceHolder) {
                android.util.Log.i("ar-ui", "surfaceDestroyed")
                surfaceReady = false
                stopCamera()
                frontGlAttached?.let { frontGl?.detachOutputSurface(it) }
                frontGlAttached = null
            }
        })

        findViewById<Button>(R.id.button_connect).setOnClickListener { startAr() }
        recOverlay = findViewById(R.id.rec_overlay)
        val lastThumb = findViewById<ImageView>(R.id.last_thumb)
        // 相机 App 缩略图键惯例：bitmap 裁圆（outline 来自 bg_thumb 正圆），
        // 白色圆环由 foreground 叠加不被裁
        lastThumb.clipToOutline = true
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

    /** 与 MainActivity 同一套参数下发到后摄合成器；仅黑边抠像特殊：
     *  未启用抠像（keyHigh ≤ keyLow）时回退 AR 原校准值（暗色底即抠透）。 */
    private fun applyComposerSettings(c: RearComposer) {
        c.setOverlayScale(overlayScalePct / 100f)
        var keyLow = keyLowPct / 100f
        var keyHigh = keyHighPct / 100f
        var featherPower = featherPowerPct / 100f
        var featherRadius = featherRadiusPx
        if (keyHigh <= keyLow) {
            keyLow = 0.08f; keyHigh = 0.28f; featherPower = 1.2f; featherRadius = 15
        }
        c.setOverlayParams(
            overlayAlphaPct / 100f,
            overlayBrightnessPct / 100f,
            overlaySaturationPct / 100f,
            dimStrengthPct / 100f,
            keyLow,
            keyHigh,
            featherPower,
            featherRadius,
        )
        c.setBaseBrightness(baseBrightnessPct / 100f)
        c.setBottomRotation(bottomRotationDeg.toFloat(), bottomMirror)
        c.setTopRotation(topRotationDeg, topMirror)
    }

    /** 后摄首次切入时建 GL 合成器（尺寸取 GL 面实际大小，保证 1:1 输出）。
     *  ArRearGl = 复用前摄 CPU 烘焙位图（convertOverlayFrame：黑键抠像 + 光晕剪影），
     *  contain 完整显示；overlays 快照与 ensureFrontGl 同一份 overlayLock 数据。 */
    private fun ensureRearGl(w: Int, h: Int): ArRearGl {
        var g = rearGl
        if (g == null) {
            g = ArRearGl(w, h, overlays = { action ->
                synchronized(overlayLock) {
                    action(overlayBmp, glowBmp, bloomFarBmp, overlayVersion)
                }
            })
            rearGl = g
            applyComposerSettings(g)
        }
        return g
    }

    /** 前摄首次显示时建 GL 合成器（尺寸取显示面实际大小，保证 1:1 输出）。
     *  overlay 快照经 overlayLock 提供给 GL 线程，位图在回调持锁期间有效。 */
    private fun ensureFrontGl(w: Int, h: Int): ArFrontGl {
        var g = frontGl
        if (g == null) {
            g = ArFrontGl(
                resources.displayMetrics.density, w, h,
                overlays = { action ->
                    synchronized(overlayLock) {
                        action(overlayBmp, glowBmp, bloomFarBmp, overlayVersion)
                    }
                },
            )
            // 每拍唤醒录像编码线程取包（编码帧率≈GL 帧率，与旧 Canvas 路径的 tick 等价）
            g.frameCallback = { recorder?.tick() }
            frontGl = g
        }
        return g
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
            if (!ok) android.util.Log.w(TAG, "det net open failed: $j")
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
        try {
            val okCopy = copyPlanes(img)
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
            val disp = windowManager.defaultDisplay
            val displayRot = (disp?.rotation ?: 0) * 90
            val rot = (sensorOrientation - displayRot + 360) % 360
            val rw = if (rot == 90 || rot == 270) frameH else frameW
            val rh = if (rot == 90 || rot == 270) frameW else frameH

            val rb: ByteBuffer = synchronized(camBufLock) {
                val need = rw * rh * 4
                if (camBufBytes != need) {
                    for (i in camBufs.indices) {
                        camBufs[i] = ByteBuffer.allocateDirect(need).order(ByteOrder.nativeOrder())
                    }
                    camBufBytes = need
                }
                val b = camBufs[camBufIdx]!!
                camBufIdx = (camBufIdx + 1) % camBufs.size
                b
            }

            val detT0 = android.os.SystemClock.uptimeMillis()
            val j = ArNative.nativeFaceDetect(
                yBuf!!, uBuf!!, vBuf!!, frameW, frameH,
                yStride, uStride, vStride, uPix, vPix, rot, detectOn, rb)
            val detMs = android.os.SystemClock.uptimeMillis() - detT0
            if (detMs > 3000) {
                // 卡死取证：单次检测超 3s（正常 5~60ms）——每 3s 记一条直到返回
                android.util.Log.w(TAG, "det STUCK ${detMs}ms (Vulkan/GL 并发楔死?)")
            } else if (++detLogTick % 150L == 1L) {
                android.util.Log.i(TAG, "det ${detMs}ms/frame")
            }

            if (++dbgFrames <= 3)
                android.util.Log.i(TAG, "frame#$dbgFrames rot=$rot upright=${rw}x$rh json=" +
                    (if (j.length > 160) j.substring(0, 160) else j))

            parseResult(j)

            // GL 合成：相机帧纹理上传 + 人脸锚点状态（检测直写的 RGBA 原样上屏，所见即所测；
            // 旧 copyPixelsFromBuffer + Canvas 软件绘制路径已整体移除）
            frontGl?.let { g ->
                rb.rewind()
                g.postCameraFrame(rb, rw, rh)
                g.setFace(faceOv.firstOrNull(), rw, rh)
            }
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "帧处理异常", t)
        } finally {
            busy.set(false)
        }
    }

    /** 提取 face0 overlay 四角锚点；人脸丢失时 faceOv 为空。
     *  （旧调试面板文本/fps 统计已随 Canvas 路径废弃移除，native 异常走日志） */
    private fun parseResult(j: String) {
        val ovList = ArrayList<FloatArray>()
        try {
            val o = JSONObject(j)
            if (o.optBoolean("ok", true)) {
                val fs = o.optJSONArray("faces")
                val f = if (fs != null && fs.length() > 0) fs.optJSONObject(0) else null
                val ovArr = f?.optJSONArray("overlay")
                val bkArr = f?.optJSONArray("back")
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
                    ov[16] = f!!.optDouble("ovD", 30.0).toFloat()
                    ovList.add(ov)
                }
            } else {
                android.util.Log.w(TAG, "detect err: " + o.optString("err", "?"))
            }
        } catch (e: Exception) {
            android.util.Log.w(TAG, "JSON 解析失败", e)
        }
        faceOv = ovList.toTypedArray()
    }

    /** 前摄录像尺寸 = camera 有效区域（GL 面上 letterbox 内容矩形，偶数对齐）。
     *  相机/显示面未就绪返回 null。与 ArFrontGl 裁剪映射同一矩形公式。 */
    private fun frontRecordSize(): IntArray? {
        val fw = frameW; val fh = frameH
        val vw = cameraView.width; val vh = cameraView.height
        if (fw <= 0 || fh <= 0 || vw <= 0 || vh <= 0) return null
        val dispRot = (windowManager.defaultDisplay?.rotation ?: 0) * 90
        val rot = (sensorOrientation - dispRot + 360) % 360
        val cw = if (rot == 90 || rot == 270) fh else fw
        val ch = if (rot == 90 || rot == 270) fw else fh
        val sc = minOf(vw.toFloat() / cw, vh.toFloat() / ch)
        fun even(v: Float): Int { val i = v.toInt(); return if (i and 1 == 1) i - 1 else i }
        val w = even(cw * sc); val h = even(ch * sc)
        return if (w > 0 && h > 0) intArrayOf(w, h) else null
    }

    private fun startRec() {
        android.util.Log.i(TAG, "rec: startRec called, cameraView=${cameraView.width}x${cameraView.height}")
        // 录像分辨率：前摄=camera 有效区域（letterbox 内容矩形，无黑边）；
        // 后摄=3:4 GL 面（与合成器输出一致）
        val sw: Int; val sh: Int
        if (frontCamera) {
            val s = frontRecordSize()
            if (s == null) { showStatusText("画面未就绪，无法录像"); return }
            sw = s[0]; sh = s[1]
        } else { sw = glSurface.width; sh = glSurface.height }
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
                // GL 合成：编码器面直接挂当前模式合成器输出（前摄=frontGl 裁剪 camera
                // 有效区域，后摄=rearGl 全画面），合成结果 GPU 直喂编码器，免 CPU 重画
                val inSurf = r.inputSurface
                if (ok && inSurf != null) {
                    if (frontCamera) frontGl?.attachOutputSurface(inSurf, crop = true)
                    else rearGl?.attachOutputSurface(inSurf)
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
        // 预热尺寸与录像面一致：前摄=camera 有效区域；后摄=3:4 GL 面
        val s = if (frontCamera) frontRecordSize() else null
        val ws = s?.get(0) ?: glSurface.width
        val hs = s?.get(1) ?: glSurface.height
        Thread {
            var name: String? = null
            try {
                // GL 模式：先把编码器面从当前模式合成器摘掉（同步、不 release），再停编码器，
                // 避免 GL 线程向已失效的编码器面 swap
                val inSurf = r.inputSurface
                if (inSurf != null) {
                    if (frontCamera) frontGl?.detachOutputSurface(inSurf, releaseSurface = false)
                    else rearGl?.detachOutputSurface(inSurf, releaseSurface = false)
                }
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
                // 与 ArRecorder.start 同参（高质量：VBR 20Mbps + 30fps，分辨率不缩放），
                // 否则预热编码器被复用时参数不一致
                fmt.setInteger(MediaFormat.KEY_BIT_RATE, 20_000_000)
                fmt.setInteger(MediaFormat.KEY_BITRATE_MODE,
                    MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
                fmt.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
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

    /** 底部相机控件条可见性：进入预览（playing）即显示、退出预览隐藏；
     *  预览中无触屏 10s 自动下滑隐藏，任意触摸上滑恢复（见 onScreenInteraction）。
     *  录像按钮/缩略图/切换钮在同一个 bottom_controls 容器里，一起显示/隐藏。主线程安全。 */
    private fun updateRecBtnVisible() {
        val show = playing
        if (show == lastRecBtnShow) return
        lastRecBtnShow = show
        recTimerHandler.post {
            val v = findViewById<android.view.View>(R.id.bottom_controls)
            recTimerHandler.removeCallbacks(idleHideControls)
            if (show) {
                // 进预览：复位为显示态并启动空闲计时
                controlsShown = true
                v.animate().cancel()
                v.translationY = 0f; v.alpha = 1f
                v.visibility = android.view.View.VISIBLE
                recTimerHandler.postDelayed(idleHideControls, CONTROLS_IDLE_MS)
            } else {
                // 退预览：取消计时、复位动画状态（下次进预览从显示态开始）
                controlsShown = true
                v.animate().cancel()
                v.translationY = 0f; v.alpha = 1f
                v.visibility = android.view.View.GONE
            }
        }
    }

    // ── 控件条空闲自动隐藏/恢复 ──
    private var controlsShown = true
    private val idleHideControls = Runnable { hideControls() }

    /** 任意触屏（含按钮，dispatchTouchEvent 统一入口）：恢复显示并重置 10s 空闲计时。 */
    private fun onScreenInteraction() {
        if (!playing) return
        recTimerHandler.removeCallbacks(idleHideControls)
        if (!controlsShown) showControls()
        recTimerHandler.postDelayed(idleHideControls, CONTROLS_IDLE_MS)
    }

    private fun hideControls() {
        if (!controlsShown) return
        controlsShown = false
        val v = findViewById<android.view.View>(R.id.bottom_controls)
        // 下滑出屏（高度 + 底边距）并淡出；结束后 INVISIBLE 使按钮不可误触
        val margin = (v.layoutParams as? android.widget.FrameLayout.LayoutParams)?.bottomMargin ?: 0
        v.animate().translationY((v.height + margin).toFloat()).alpha(0f).setDuration(250)
            .withEndAction { if (!controlsShown) v.visibility = android.view.View.INVISIBLE }
    }

    private fun showControls() {
        if (controlsShown) return
        controlsShown = true
        val v = findViewById<android.view.View>(R.id.bottom_controls)
        v.visibility = android.view.View.VISIBLE
        v.animate().translationY(0f).alpha(1f).setDuration(250)
            .withEndAction { if (controlsShown) v.translationY = 0f }
    }

    override fun dispatchTouchEvent(ev: android.view.MotionEvent): Boolean {
        if (ev.action == android.view.MotionEvent.ACTION_DOWN) onScreenInteraction()
        return super.dispatchTouchEvent(ev)
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
        private var encThread: Thread? = null
        private val tickLock = Object()

        // ---- 音轨（仅手机麦克风 → AAC；眼镜外放由麦克风自然拾音，不混解码 PCM）----
        // 主视频编码线程在 rec-encoder；音频编码在 audio-mix 线程，二者共享 muxer。
        private val muxerLock = Object()   // mediaMuxer 非线程安全，写样本需互斥
        private var wantsAudio = false       // 本路是否拥有音轨（麦克风权限 OK 且 AAC 就绪）
        private var audioTrack = -1           // 音频轨
        private var mic: AudioRecord? = null
        private var aac: MediaCodec? = null
        private var audioThread: Thread? = null
        private var audioPtsUs = 0L

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
                // 高质量：VBR 20Mbps + 30fps；分辨率=录像面原始尺寸（前摄=全屏 GL 面，
                // 后摄=3:4 GL 面），不缩放
                fmt.setInteger(MediaFormat.KEY_BIT_RATE, 20_000_000)
                fmt.setInteger(MediaFormat.KEY_BITRATE_MODE,
                    MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
                fmt.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
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
            lastPtsUs = Long.MIN_VALUE
            startMs = SystemClock.elapsedRealtime()
            recording = true
            android.util.Log.i(TAG, "rec: start e(${w}x${h}) name=$name")
            // 麦克风权限 OK 则开音轨（仅手机麦克风；眼镜外放由麦克风自然拾音）
            wantsAudio = micGranted() && initMicAndAac()
            pt("audio", t0)
            encThread = Thread({ encLoop() }, "rec-encoder").apply { start() }
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

        /** 每帧由 GL 合成器 frameCallback 调用：唤醒编码线程取包（自身卡顿不会阻塞 GL 线程） */
        fun tick() {
            if (!recording) return
            synchronized(tickLock) { tickLock.notifyAll() }
        }

        /** 编码线程：前后摄合成结果均由 GL 合成器（前摄 frontGl / 后摄 rearGl）直接推到
         *  编码器输入面，本线程只做取包 + PTS；GL 每拍 frameCallback 会唤醒本线程提前取包。 */
        private fun encLoop() {
            while (recording) {
                if (inSurf == null) { sleepTick(); continue }
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

        /** 音轨线程：仅手机麦克风(单声道 ×MIC_GAIN) → 复制为双声道 → AAC → muxer。
         *  眼镜端声音不进音轨：外放会被麦克风自然拾音，混入解码 PCM 会双重采录。 */
        private fun mixLoop() {
            val frameSamples = 960          // 20ms @48k
            val ac = aac ?: return
            val micRec = mic
            val micShort = ShortArray(frameSamples)
            val outBi = ByteBuffer.allocateDirect(frameSamples * 4).order(ByteOrder.LITTLE_ENDIAN)
            val bf = MediaCodec.BufferInfo()
            while (recording) {
                val samplesGot = if (micRec != null) {
                    val got = micRec.read(micShort, 0, frameSamples); if (got > 0) got else 0
                } else 0
                outBi.clear()
                for (i in 0 until frameSamples) {
                    // 统一固定增益（无分段/曲线）；饱和截断到 i16，单声道复制到 L/R
                    val s = (if (i < samplesGot) micShort[i].toInt() * MIC_GAIN else 0)
                        .coerceIn(-32768, 32767).toShort()
                    outBi.putShort(s)
                    outBi.putShort(s)
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
                var sumR = 0; var sumG = 0; var sumB = 0; var nOp = 0   // 不透明内容平均色（光晕统一色调）
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
                        if (a >= 64) { sumR += r; sumG += g; sumB += b; nOp++ }
                        o++
                    }
                }
                overlayBmp!!.setPixels(pix, 0, w, 0, 0, w, h)
                // 发光底图：内容剪影（不透明像素平均色 + alpha 饱和）→ 金字塔低通烘焙平滑。
                // 剪影让任意字符贡献相同的发光源；多级降/升采样（纯双线性缩放）等效大半径
                // 低通，不依赖 maskFilter——光晕均匀、无块状明暗边界
                val gw = maxOf(8, w shr 2); val gh = maxOf(8, h shr 2)
                // 低通中间层降到 1/16（旧 1/8）、远距层降到 1/32（旧 1/16）：
                // 光晕扩散面积更大，单位面积亮度更低
                val hw = maxOf(4, w shr 4); val hh = maxOf(4, h shr 4)
                val fw = maxOf(2, w shr 5); val fh = maxOf(2, h shr 5)
                if (glowBmp == null || glowBmp!!.width != gw || glowBmp!!.height != gh) {
                    glowBmp?.recycle()
                    glowBmp = Bitmap.createBitmap(gw, gh, Bitmap.Config.ARGB_8888)
                    glowCanvas = Canvas(glowBmp!!)
                }
                if (glowHalfBmp == null || glowHalfBmp!!.width != hw || glowHalfBmp!!.height != hh) {
                    glowHalfBmp?.recycle()
                    glowHalfBmp = Bitmap.createBitmap(hw, hh, Bitmap.Config.ARGB_8888)
                    glowHalfCanvas = Canvas(glowHalfBmp!!)
                }
                if (bloomFarBmp == null || bloomFarBmp!!.width != fw || bloomFarBmp!!.height != fh) {
                    bloomFarBmp?.recycle()
                    bloomFarBmp = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
                    bloomFarCanvas = Canvas(bloomFarBmp!!)
                }
                glowSrc.set(0, 0, w, h); glowDst.set(0, 0, gw, gh); farDst.set(0, 0, fw, fh)
                if (nOp > 0) {
                    // 内容平均色 → 光晕统一色调；alpha ×8 饱和 → 剪影（键控羽化边缘计入轮廓）
                    val tR = sumR.toFloat() / nOp
                    val tG = sumG.toFloat() / nOp
                    val tB = sumB.toFloat() / nOp
                    glowCm.set(floatArrayOf(
                        0f, 0f, 0f, 0f, tR,
                        0f, 0f, 0f, 0f, tG,
                        0f, 0f, 0f, 0f, tB,
                        0f, 0f, 0f, 8f, 0f,
                    ))
                    glowAlphaPaint.colorFilter = android.graphics.ColorMatrixColorFilter(glowCm)
                    // 剪影 → 1/4
                    glowCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                    glowCanvas!!.drawBitmap(overlayBmp!!, glowSrc, glowDst, glowAlphaPaint)
                    // 金字塔低通：1/4 → 1/8 → 1/4 双线性往返 = 大核平滑，两轮更柔
                    for (round in 0 until 2) {
                        glowHalfCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                        glowHalfCanvas!!.drawBitmap(glowBmp!!, null,
                            android.graphics.RectF(0f, 0f, hw.toFloat(), hh.toFloat()),
                            glowSmoothPaint)
                        glowCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                        glowCanvas!!.drawBitmap(glowHalfBmp!!, null,
                            android.graphics.RectF(0f, 0f, gw.toFloat(), gh.toFloat()),
                            glowSmoothPaint)
                    }
                    // 远距 = 平滑后 1/4 再降 1/16
                    bloomFarCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                    bloomFarCanvas!!.drawBitmap(glowBmp!!, glowDst, farDst, glowSmoothPaint)
                } else {
                    glowCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                    bloomFarCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                }
                overlayNew = true
                overlayVersion++   // 通知 GL 合成器上传新的内容/光晕纹理
                // debug.gscp.dumpoverlay=1:在 100/200/300 包时各转储一张解码位图
                // (首帧为空帧,需等 UI 渲染后的包)
                if (diagOverlayPkg >= 100L * (dumpCount + 1) && dumpCount < 3) {
                    dumpCount++
                    try {
                        val f = java.io.File(getExternalFilesDir(null), "overlay_dump_${dumpCount}.png")
                        java.io.FileOutputStream(f).use { overlayBmp!!.compress(
                            Bitmap.CompressFormat.PNG, 100, it) }
                        android.util.Log.i(TAG, "overlay dump#${dumpCount}: ${f.absolutePath}")
                    } catch (e: Exception) {
                        android.util.Log.w(TAG, "overlay dump fail", e)
                    }
                }
            }
        } catch (t: Throwable) {
            // Throwable：JNI 方法不匹配等 Error 也不得杀死解码线程
            android.util.Log.w(TAG, "overlay frame convert fail", t)
        } finally {
            try { img.close() } catch (_: Exception) {}
        }
    }

    private fun showStatus(s: String?) { showStatusText(s ?: statusTextFor()) }

    /**
     * overlay 烘焙路径开关：debug.gscp.ovgl（默认 0 = CPU 烘焙。
     * ⚠ 前摄 GL 加速尚不可用：GPU 烘焙路径在前摄连续 overlay 流下仍有问题，
     * 后摄 GL 烘焙 + GL 原生 SR 也待真机复验——真机调试时可 setprop 1 试验，
     * 试验完记得恢复）。置 1 = GPU 烘焙 OverlayBakeCore（解码直写 GL）。
     * 读取时机 = 建流时（onOverlayPrepare / startRearPlayerIfReady），
     * setprop 运行中切换下次连接生效。
     */
    private fun glBakeEnabled(): Boolean {
        return try {
            val sp = Class.forName("android.os.SystemProperties")
            val get = sp.getMethod("get", String::class.java, String::class.java)
            val v = (get.invoke(null, "debug.gscp.ovgl", "0") as String).trim()
            v != "0" && !v.equals("false", true)
        } catch (_: Throwable) {
            true
        }
    }

    /**
     * overlay 超分开关：debug.gscp.ovsr（默认 1 = GPU 烘焙前先过 GL 原生 ESPCN ×2
     * 灰度超分，480² 亮度 → 960²，文字/图标显著增锐；置 0 关闭）。建流时读取，
     * 运行中切换下次连接生效。权重缺失/浮点渲染目标不可用时核心自动退回非 SR。
     */
    private fun glSrEnabled(): Boolean {
        return try {
            val sp = Class.forName("android.os.SystemProperties")
            val get = sp.getMethod("get", String::class.java, String::class.java)
            val v = (get.invoke(null, "debug.gscp.ovsr", "1") as String).trim()
            v != "0" && !v.equals("false", true)
        } catch (_: Throwable) {
            true
        }
    }

    /** ESPCN fp32 权重（espcn_x2.f32，81KB，懒加载缓存）。 */
    @Volatile private var srWeights: ByteArray? = null
    private fun srWeights(): ByteArray? {
        srWeights?.let { return it }
        return try {
            assets.open("espcn_x2.f32").use { it.readBytes() }.also { srWeights = it }
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "sr weights load fail", t)
            null
        }
    }

    // ── scrcpy overlay 连接 ───────────────────────────────────


    private fun startAr() {
        android.util.Log.i("ar-ui", "startAr")
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
        frontActive = true      // 会话活动标记：maybeStartFrontPlayer 的启动门槛
        attachFrontGlIfLive()   // 同 Activity 重连必须补挂 GL 输出（见函数注释）
        updateRecBtnVisible()   // 进入预览 → 显示底部三键（缩略图-录像-切换）
        maybeStartFrontPlayer()
        showStatusText(statusTextFor())
        // 预览就绪后后台预热录像编码器，避免首次点录像时偶发 1~5s 初始化
        camHandler?.postDelayed({
            // 预热尺寸与录像面一致：前摄=camera 有效区域；后摄=3:4 GL 面
            val s = if (frontCamera) frontRecordSize() else null
            val ws = s?.get(0) ?: glSurface.width
            val hs = s?.get(1) ?: glSurface.height
            if (ws > 0 && hs > 0) armWarmVideo(ws, hs)
        }, 1800)
    }

    /** 补挂前摄 GL 输出面（幂等）。teardown 会同步摘除输出面；实测部分机型
     *  ar_panel GONE 时 surfaceDestroyed/Create/Changed 全部不重发，holder
     *  surface 保持 invalid —— 仅靠 surface 回调永远挂不回去，相机与眼镜流
     *  照常运行却无输出，画面卡死在最后一帧。
     *  surface 有效 → 直接挂；已失效 → 强制 SurfaceView 重建表面（GONE→VISIBLE）
     *  触发完整回调，并短重试兜底（挂载幂等，与回调路径互斥）。 */
    private fun attachFrontGlIfLive() {
        val g = frontGl
        val s = cameraView.holder.surface
        android.util.Log.i("ar-ui", "attachFrontGlIfLive: gl=${g != null} attached=${frontGlAttached != null} surf=${s != null} valid=${s?.isValid}")
        if (g == null) return
        if (frontGlAttached != null) return
        if (s != null && s.isValid) {
            frontGlAttached = s
            g.attachOutputSurface(s)
            g.setActive(frontCamera)
            maybeStartFrontPlayer()   // surface 有效直挂路径同样要起会话
            return
        }
        // 表面已失效且回调不会重发：强制重建
        android.util.Log.i("ar-ui", "surface invalid -> force recreate")
        cameraView.visibility = android.view.View.GONE
        cameraView.post { cameraView.visibility = android.view.View.VISIBLE }
        // 重建回调若仍未到，短重试兜底（每次都幂等检查）
        for (delay in longArrayOf(400, 1000, 2000)) {
            recTimerHandler.postDelayed({ attachFrontGlIfLive() }, delay)
        }
    }

    // ── 子系统切换：前摄（overlay-only + Canvas AR）与后摄（普通模式管线 + GL）互相独立 ──

    // scrcpy 会话异步收尾线程（载体 = killall app_process）。新旧会话切换必须串行：
    // 新连接若不等旧收尾完成，旧 killall 会误杀刚拉起的新 server——流永不到达、
    // 也无错误回调，界面卡死在"连接中"（返回→再连接 100% 复现的根因）。
    @Volatile private var scrcpyShutdown: Thread? = null

    /** 起前摄会话：后摄同款 GlassesPlayer 管线 + overlayOnly 连接（手机画面由
     *  Camera2 提供，不消费眼镜视频流）。前置条件 = frontGl 已建（cameraView
     *  surfaceChanged / attachFrontGlIfLive 重建后都会再调本函数）；connect 挪到
     *  工作线程，先等旧会话收尾完成（防 killall 误杀新 server）。 */
    private fun maybeStartFrontPlayer() {
        if (!frontActive || frontPlayer != null) return
        val g = frontGl ?: return   // GL 面未就绪：surfaceChanged/attach 回调会再触发
        android.util.Log.i("ar-ui", "maybeStartFrontPlayer")
        val ip = prefs.getString("ip", null) ?: ipEdit.text.toString().trim()
        if (ip.isEmpty()) { showStatusText("未配置眼镜 IP"); return }
        glassesState = 1
        frontOverlaySeen = false   // 流看门狗按每次连接独立判定
        resetOverlay()             // 与后摄 g.reset() 同位：新会话清残留（位图 + 烘焙核心）
        val useGlBake = glBakeEnabled()
        val useSr = useGlBake && glSrEnabled()
        val player = GlassesPlayer(
            this, g, audioEnabled, bottomRotationDeg, bottomMirror,
            applySettings = { /* 前摄不消费 RearComposer 的视频参数 */ },
            events = frontEvents,
            resetMixerOnStop = false,   // 合成器跨会话复用（与后摄同理由）
            overlayImageCallback = if (useGlBake) null else { img -> convertOverlayFrame(img) },
            overlaySr = useSr,
            overlaySrWeights = if (useSr) srWeights() else null,
            overlayOnly = true,
        )
        frontPlayer = player
        val waiter = scrcpyShutdown
        Thread {
            try { waiter?.join(5000) } catch (_: InterruptedException) {}
            player.connect(ip)
        }.start()
    }

    /** 切前摄：起新的 overlay-only 连接；旧后摄管线异步收尾并登记到
     *  [scrcpyShutdown]，新连接会等它完成（避免 killall 误杀新 server）。 */
    private fun enterFrontMode() {
        frontCamera = true
        frontActive = true
        glassesState = 1
        frontOverlaySeen = false
        camHandler?.post { closeCameraNow(); startCamera() }
        glSurface.visibility = android.view.View.GONE
        frontGl?.setActive(true)   // 前摄 GL 合成恢复上屏
        // 先登记后摄收尾再起新连接：新连接等 killall 完成才发起（防误杀）
        val old = rearPlayer
        rearPlayer = null
        rearActive = false
        if (old != null) {
            scrcpyShutdown = Thread { try { old.stop() } catch (_: Exception) {} }.also { it.start() }
        }
        maybeStartFrontPlayer()
        showStatusText(statusTextFor())
    }

    /** 切后摄：停前摄子系统（登记收尾），起完全复用普通模式的 GlassesPlayer
     *  （全流硬解 → GL 合成）。后摄连接同样等旧收尾完成再发起。 */
    private fun enterRearMode() {
        rearActive = true
        glassesState = 1
        camHandler?.post { closeCameraNow() }   // 后摄基底=眼镜视频，手机相机整体停掉省电
        glSurface.visibility = android.view.View.VISIBLE   // surfaceChanged → ensureRearGl → startRearPlayerIfReady
        frontGl?.setActive(false)   // 前摄 GL 合成停画（只清黑，垫在 3:4 GL 面之下）
        showStatusText("后摄：连接眼镜画面…")
        val oldFront = frontPlayer
        frontPlayer = null
        frontActive = false
        if (oldFront != null) {
            scrcpyShutdown = Thread { try { oldFront.stop() } catch (_: Exception) {} }.also { it.start() }
        }
        startRearPlayerIfReady()
        resetOverlay()
    }

    /** 后摄管线就绪即连接（等 GL 面尺寸出来建好 mixer）。
     *  connect 挪到工作线程，先等旧会话收尾完成。 */
    private fun startRearPlayerIfReady() {
        if (!rearActive || rearPlayer != null) return
        val g = rearGl ?: return
        val ip = prefs.getString("ip", null) ?: ipEdit.text.toString().trim()
        if (ip.isEmpty()) { showStatusText("未配置眼镜 IP"); return }
        g.reset()   // 重建两路输入面，清掉上一会话残留帧
        val useGlBake = glBakeEnabled()
        val useSr = useGlBake && glSrEnabled()
        val player = GlassesPlayer(
            this, g, audioEnabled, bottomRotationDeg, bottomMirror,
            applySettings = { applyComposerSettings(it) },
            events = rearEvents,
            resetMixerOnStop = false,   // 合成器跨会话复用：reset 统一在新会话开始时做，防异步 stop 竞态
            // debug.gscp.ovgl=1（默认）：overlay 解码直写 ArRearGl 内烘焙核心输入面
            // （RearComposer.openOverlayStream），CPU 烘焙不再参与；置 0 回退
            // convertOverlayFrame CPU 烘焙位图路径（下方回调）。
            overlayImageCallback = if (useGlBake) null else { img -> convertOverlayFrame(img) },
            overlaySr = useSr,
            overlaySrWeights = if (useSr) srWeights() else null,
        )
        rearPlayer = player
        val waiter = scrcpyShutdown
        Thread {
            try { waiter?.join(5000) } catch (_: InterruptedException) {}
            player.connect(ip)
        }.start()
    }

    /** 停后摄子系统（幂等）：断开管线、藏 GL 面。异步释放，不阻塞调用线程；
     *  收尾线程登记到 [scrcpyShutdown] 供后续连接串行等待。
     *  GL 输出面同步摘除（不等 surfaceDestroyed），防收尾过渡帧串画面。 */
    private fun exitRearMode() {
        rearActive = false
        val old = rearPlayer
        rearPlayer = null
        if (old != null) {
            scrcpyShutdown = Thread { try { old.stop() } catch (_: Exception) {} }.also { it.start() }
        }
        glAttached?.let { rearGl?.detachOutputSurface(it) }
        glAttached = null
        glSurface.visibility = android.view.View.GONE
    }

    /** 断开/出错后收尾：停录像、复位两个子系统、回到设置面板。可在任意线程调用。 */
    private fun teardownToSettings(msg: String) {
        android.util.Log.i("ar-ui", "teardown: $msg")
        runOnUiThread {
            watchdogRunnable?.let { recTimerHandler.removeCallbacks(it) }
            if (recorder?.recording == true) stopRec()
            val fp = frontPlayer
            frontPlayer = null
            if (fp != null) {
                scrcpyShutdown = Thread { try { fp.stop() } catch (_: Exception) {} }.also { it.start() }
            }
            resetOverlay()
            // 同步摘除前摄 GL 输出（不等 surfaceDestroyed 的滞后回调）：
            // 否则过渡帧里旧 AR 画面/连接提示会叠在已显示的设置页上
            frontGlAttached?.let { frontGl?.detachOutputSurface(it) }
            frontGlAttached = null
            frontGl?.setActive(false)
            exitRearMode()
            arPanel.visibility = android.view.View.GONE
            settingsPanel.visibility = android.view.View.VISIBLE
            progressView.visibility = android.view.View.GONE
            findViewById<android.view.View>(R.id.bottom_controls).visibility = android.view.View.GONE
            lastRecBtnShow = false
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
            overlayVersion++   // 通知 GL 合成器清空 overlay/光晕纹理
        }
        // GPU 烘焙路径：同步关闭输入面并清空产物纹理（下次建流重开）
        frontGl?.closeOverlayStream()
    }

    private fun exitAr() {
        android.util.Log.i("ar-ui", "exitAr")
        playing = false
        frontActive = false
        // 前摄 player 收尾登记在 teardownToSettings（scrcpyShutdown 串行，
        // 下次连接先等 killall 完成——防误杀新 server 卡死在连接中）
        teardownToSettings("")
    }

    /** 恢复前摄 AR 基准态：隐藏 GL 合成面、重开手机前摄。退出预览/断连时调用。 */
    private fun restoreFrontMode() {
        frontCamera = true
        glSurface.visibility = android.view.View.GONE
        frontGl?.setActive(true)
        camHandler?.post {
            closeCameraNow()
            startCamera()
        }
    }

    // ── 前摄子系统事件（GlassesPlayer overlayOnly 管线；解码/音频接线在 player 内，
    //  这里只消费状态：看门狗布防、overlay 流到达标记、诊断计数与 GL 产物转储）──
    @Volatile private var frontOverlaySeen = false   // 流看门狗：onOverlayPrepare 已到
    private var watchdogRunnable: Runnable? = null   // 看门狗句柄：换连接/收尾时必须撤销
    private val frontEvents = object : GlassesPlayer.Events {
        override fun onConnect() {
            android.util.Log.i("ar-ui", "front onConnect")
            glassesState = 2
            runOnUiThread {
                progressView.visibility = android.view.View.GONE
                showStatusText(statusTextFor())
            }
            // 流看门狗：连接成功但 server 被杀/挂死时不会有任何错误回调，
            // 界面会永久卡在"已连接无画面"。8s 内没等到 overlay 流就按失败收尾。
            // （runnable 存句柄：teardown/新连接时撤销，否则旧定时器会误杀下一次连接）
            watchdogRunnable?.let { recTimerHandler.removeCallbacks(it) }
            val wd = Runnable {
                if (playing && frontActive && !frontOverlaySeen) {
                    android.util.Log.i("ar-ui", "watchdog fire")
                    playing = false
                    teardownToSettings("连接超时，未收到眼镜画面")
                }
            }
            watchdogRunnable = wd
            recTimerHandler.postDelayed(wd, 8000)
        }

        override fun onOverlayPrepare(width: Int, height: Int) {
            frontOverlaySeen = true   // 流看门狗：overlay 流已到达
            runOnUiThread {
                showStatusText("overlay ${width}×${height}，请正对手机摄像头")
            }
        }

        override fun onOverlayPackage(seq: Long) {
            diagOverlayPkg = seq
            if (seq % 150L == 1L) android.util.Log.i("ar-ui", "overlayPkg #$seq")
            // GL 烘焙路径的产物转储（与 CPU 路径 convertOverlayFrame 内的转储同节拍，
            // debug.gscp.dumpoverlay 语义：100/200/300 包各转储一批 content/glow/far）
            if (glBakeEnabled() && seq >= 100L * (dumpCount + 1) && dumpCount < 3) {
                dumpCount++
                val dir = getExternalFilesDir(null)?.absolutePath
                if (dir != null) frontGl?.dumpOverlay("$dir/ovgl_dump_${dumpCount}")
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
                // 与 onDisconnect 同路：连接失败不滞留 AR 屏（playing=false 会让
                // 控件条交互全部失效——含空闲隐藏后的单击恢复）。
                // 注意：onError 在 adb 线程回调，只能经 teardownToSettings 的
                // runOnUiThread 触 UI；严禁在此线程直接 Toast/碰视图。
                playing = false
                teardownToSettings("连接失败，请检查眼镜 IP 与网络")
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
                // teardownToSettings 内含 exitRearMode + restoreFrontMode + 控件条/
                // 进度条隐藏；onError 在 adb 线程回调，UI 一律经 teardown 的
                // runOnUiThread 触发，严禁直接碰视图/Toast
                playing = false
                teardownToSettings("连接失败，请检查眼镜 IP 与网络")
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
        stopCamera()
        val rp = rearPlayer
        rearPlayer = null
        if (rp != null) Thread { try { rp.stop() } catch (_: Exception) {} }.start()
        val fp = frontPlayer
        frontPlayer = null
        if (fp != null) Thread { try { fp.stop() } catch (_: Exception) {} }.start()
        det.shutdown()
        openExec.shutdown()
        resetOverlay()
        synchronized (warmLock) {
            try { warmSurf?.let { if (it.isValid) it.release() } } catch (_: Exception) {}
            try { warmCodec?.release() } catch (_: Exception) {}
            warmCodec = null; warmSurf = null
        }
        try { rearGl?.release() } catch (_: Exception) {}
        rearGl = null
        frontGl?.release()
        frontGl = null
        super.onDestroy()
    }

    companion object {
        private const val TAG = "ar-java"
        private const val REQ_CAM = 1
        private const val REQ_MIC = 2
        // det_10g 图内写死了 448 输入的 FPN 上采样尺寸，实时检测同样固定 448
        private const val LIVE_INPUT = 448   // det_10g 图内写死 448 输入的 FPN 上采样尺寸
        private const val CONTROLS_IDLE_MS = 10_000L   // 无触屏 10s 后控件条下滑隐藏
        // 麦克风混音：统一固定增益（无分段/曲线处理），和值饱和截断到 i16
        private const val MIC_GAIN = 10
    }
}
