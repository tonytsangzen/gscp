package com.gscp.desktop

import android.annotation.SuppressLint
import android.app.Activity
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
import android.media.Image
import android.media.ImageReader
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
    private var overlayDecoder: VideoDecoder? = null
    private val overlayLock = Any()
    private var overlayBmp: Bitmap? = null
    private var overlayPix: IntArray? = null
    private var yScr: ByteArray? = null
    private var uScr: ByteArray? = null
    private var vScr: ByteArray? = null
    private var overlayNew = false

    private val ui = Handler(Looper.getMainLooper())
    private var camThread: HandlerThread? = null
    private var camHandler: Handler? = null
    private val det: ExecutorService = Executors.newSingleThreadExecutor()
    private val openExec: ExecutorService = Executors.newSingleThreadExecutor()

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

    // 眼镜连接状态：0=未连接 1=连接中 2=已连接
    @Volatile private var glassesState = 0

    // 相机参数
    private var cameraId: String? = null
    private var sensorOrientation = 90
    private val frontCamera = true
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
    private val camMatrix = Matrix()
    private val camDimPaint = Paint().apply {
        // 相机整体亮度 90%（ColorMatrix 缩放 RGB 分量）
        colorFilter = android.graphics.ColorMatrixColorFilter(
            android.graphics.ColorMatrix(floatArrayOf(
                0.90f, 0f, 0f, 0f, 0f,
                0f, 0.90f, 0f, 0f, 0f,
                0f, 0f, 0.90f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f,
            )))
    }
    // lazy：构造期 resources 未挂载，不能在字段初始化器里取密度
    private val glowPaint: Paint by lazy {
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 6 * dp0()
            color = 0x1490EE90.toInt()
            maskFilter = android.graphics.BlurMaskFilter(
                5 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
        }
    }
    private val platePath = Path()
    private val vertsBuf = FloatArray(8)   // 保留：背板/诊断备用
    private val perspVerts = FloatArray((MESH_N + 1) * (MESH_N + 1) * 2)
    private lateinit var dimPaint: Paint
    private var dbgRender = 0

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

    // overlay 发光：ob 的 1/4 降采样副本 + 模糊画笔，画在内容之下形成柔和光晕
    // （普通合成 SRC_OVER，不再用 PLUS 叠亮 → 不会把文字叠出重影）。
    private val contentGlowPaint: Paint by lazy {
        Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            maskFilter = android.graphics.BlurMaskFilter(
                2 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
        }
    }
    private val glowAlphaPaint = Paint().apply { alpha = 190 }   // 光晕强度 ≈ 74%
    private var glowBmp: Bitmap? = null
    private var glowCanvas: Canvas? = null
    private var glowSrc = Rect(0, 0, 0, 0)
    private var glowDst = Rect(0, 0, 0, 0)

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

        cameraView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(h: SurfaceHolder) { surfaceReady = true; tryStart() }
            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}
            override fun surfaceDestroyed(h: SurfaceHolder) { surfaceReady = false; stopCamera() }
        })

        findViewById<Button>(R.id.button_connect).setOnClickListener { startAr() }
        findViewById<Button>(R.id.button_exit).setOnClickListener {
            if (playing) exitAr() else finish()
        }
        ipEdit.setText(prefs.getString("ip", ""))

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
        } else {
            statusText.text = "未授予相机权限，无法追踪人脸"
        }
    }

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
            // overlay 内容已去除全部特效，直接普通合成贴到四角（键控透明区不覆盖相机）
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                blendMode = android.graphics.BlendMode.SRC_OVER
            } else {
                @Suppress("DEPRECATION")
                xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.SRC_OVER)
            }
        }
        // overlay 内容普通压缩；背景板（投影背板）作为投影区衬底
        dimPaint = Paint().apply { color = 0x1490EE90.toInt() }
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
        ch.post {
            try { session?.close() } catch (_: Exception) {}
            try { cameraDevice?.close() } catch (_: Exception) {}
            session = null
            cameraDevice = null
            closeStream()
        }
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
            ui.post { render() }
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
    }

    // ── 绘制 ──────────────────────────────────────────────────

    private fun render() {
        if (!surfaceReady) return
        val td = System.nanoTime()
        var c: Canvas? = null
        try { c = cameraView.holder.lockCanvas() } catch (_: Exception) {}
        if (c == null) return
        try {
            c.drawColor(Color.BLACK)
            val b: Bitmap? = synchronized(frameLock) { bmp }
            val vw = cameraView.width.toFloat()
            val vh = cameraView.height.toFloat()
            var sc = 1f; var dx = 0f; var dy = 0f
            if (b != null) {
                // 等比例缩放完整显示相机画面（letterbox 居中，不裁剪）；
                // 标注（框/关键点/法线/overlay）共用同一变换，自动跟随
                sc = minOf(vw / b.width, vh / b.height)
                dx = (vw - b.width * sc) / 2f
                dy = (vh - b.height * sc) / 2f
                val m = camMatrix
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
                // 四角诊断：content quad 屏幕尺寸（量纵横比）
                if (++dbgRender % 30 == 1) {
                    val qw = maxOf(qx[0], qx[1], qx[2], qx[3]) - minOf(qx[0], qx[1], qx[2], qx[3])
                    val qh = maxOf(qy[0], qy[1], qy[2], qy[3]) - minOf(qy[0], qy[1], qy[2], qy[3])
                    android.util.Log.i(TAG,
                        "quad screen=${qw.toInt()}x${qh.toInt()} " +
                        "c0=(${qx[0].toInt()},${qy[0].toInt()}) c1=(${qx[1].toInt()},${qy[1].toInt()}) " +
                        "c2=(${qx[2].toInt()},${qy[2].toInt()}) c3=(${qx[3].toInt()},${qy[3].toInt()})")
                }
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
                    val p = platePath
                    p.rewind()
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
                    val vn = perspVerts
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
                    // overlay 发光：ob 的 1/4 降采样副本（模糊画笔），画在内容之下 → 柔和光晕。
                    // 同一 mesh 投影，与内容完全重合；SRC_OVER 普通合成，不会把文字叠亮出重影。
                    val gb = synchronized(overlayLock) { glowBmp }
                    if (gb != null) {
                        c.drawBitmapMesh(gb, MESH_N, MESH_N, perspVerts, 0, null, 0, contentGlowPaint)
                    }
                    // overlay 内容：普通合成贴到四角（键控透明区透出相机）
                    c.drawBitmapMesh(ob, MESH_N, MESH_N, perspVerts, 0, null, 0, meshPaint)
                    // 渲染侧诊断转储（与显示同一时刻）
                    if (++dbgRender % 240 == 1) {
                        try {
                            val dir = getExternalFilesDir(null)
                            ob.compress(Bitmap.CompressFormat.PNG, 100,
                                java.io.File(dir, "overlay_dump.png").outputStream())
                        } catch (_: Exception) {}
                    }
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
            // 调试信息面板
            val lines = debugLines
            if (lines.isNotEmpty()) {
                val lh = textPaint.textSize * 1.5f
                val pad = 8 * resources.displayMetrics.density
                var pw = 0f
                for (l in lines) pw = maxOf(pw, textPaint.measureText(l))
                val panel = RectF(pad / 2, pad / 2, pw + pad * 2, lines.size * lh + pad * 1.5f)
                c.drawRoundRect(panel, 8f, 8f, panelPaint)
                for (i in lines.indices)
                    c.drawText(lines[i], pad, panel.top + pad / 2 + (i + 1) * lh - lh * 0.25f, textPaint)
            }
        } finally {
            try { cameraView.holder.unlockCanvasAndPost(c) } catch (_: Exception) {}
            lastDrawMs = ((System.nanoTime() - td) / 1e6).toFloat()
        }
    }

    private fun dp0(): Float = resources.displayMetrics.density

    private fun statusTextFor(): String = when (glassesState) {
        0 -> "未连接眼镜"
        1 -> "连接眼镜中…"
        else -> "已连接，等待画面…"
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
                // 发光底图：ob 降采样 1/4（模糊画笔 + 降采样双重柔化 → 柔和光晕）
                val gw = maxOf(8, w shr 2); val gh = maxOf(8, h shr 2)
                if (glowBmp == null || glowBmp!!.width != gw || glowBmp!!.height != gh) {
                    glowBmp?.recycle()
                    glowBmp = Bitmap.createBitmap(gw, gh, Bitmap.Config.ARGB_8888)
                    glowCanvas = Canvas(glowBmp!!)
                }
                glowSrc.set(0, 0, w, h); glowDst.set(0, 0, gw, gh)
                glowCanvas!!.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                glowCanvas!!.drawBitmap(overlayBmp!!, glowSrc, glowDst, glowAlphaPaint)
                overlayNew = true
            }
        } catch (t: Throwable) {
            // Throwable：JNI 方法不匹配等 Error 也不得杀死解码线程
            android.util.Log.w(TAG, "overlay frame convert fail", t)
        } finally {
            try { img.close() } catch (_: Exception) {}
        }
    }

    private fun showStatus(s: String?) {
        runOnUiThread { statusText.text = s ?: statusTextFor() }
    }

    // ── scrcpy overlay 连接 ───────────────────────────────────

    private fun startAr() {
        val ip = ipEdit.text.toString().trim()
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            Toast.makeText(this, "请输入有效的 IP 地址", Toast.LENGTH_SHORT).show()
            return
        }
        prefs.edit().putString("ip", ip).apply()

        overlayDecoder = VideoDecoder()
        connection = ScrcpyConnection(this, audioEnabled = false, overlayOnly = true)
        glassesState = 1
        connection!!.connectAsync(ip, 5555, callback)
        settingsPanel.visibility = android.view.View.GONE
        arPanel.visibility = android.view.View.VISIBLE
        progressView.visibility = android.view.View.VISIBLE
        statusText.text = statusTextFor()
        playing = true
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
        glassesState = 0
        connection?.disconnect()
        runOnUiThread {
            progressView.visibility = android.view.View.GONE
            overlayDecoder?.stop()
            overlayDecoder = null
            resetOverlay()
            arPanel.visibility = android.view.View.GONE
            settingsPanel.visibility = android.view.View.VISIBLE
        }
    }

    private val callback = object : ScrcpyConnection.EventCallback {
        override fun onConnect() {
            glassesState = 2
            runOnUiThread {
                progressView.visibility = android.view.View.GONE
                statusText.text = statusTextFor()
            }
        }

        override fun onVideoPrepare(codec: String, width: Int, height: Int) {}

        override fun onVideoPackage(buffer: ByteArray, offset: Int, length: Int) {}

        override fun onAudioPrepare(codec: String, frameRate: Int, channel: Int) {}

        override fun onAudioPackage(buffer: ByteArray, offset: Int, length: Int) {}

        override fun onOverlayPrepare(codec: String, width: Int, height: Int) {
            runOnUiThread {
                statusText.text = "overlay ${width}×${height}，请正对手机摄像头"
                try {
                    resetOverlay()
                    overlayDecoder?.frameCallback = { img -> convertOverlayFrame(img) }
                    // 无 Surface 解码：getOutputImage 给出 CPU 可读 I420（绕开厂商平铺格式）
                    overlayDecoder?.start(width, height, null)
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
            if (playing) {
                playing = false
                glassesState = 0
                runOnUiThread {
                    overlayDecoder?.stop()
                    overlayDecoder = null
                    resetOverlay()
                    arPanel.visibility = android.view.View.GONE
                    settingsPanel.visibility = android.view.View.VISIBLE
                    statusText.text = "连接已断开"
                }
            }
        }

        override fun onError() {
            playing = false
            glassesState = 0
            runOnUiThread {
                progressView.visibility = android.view.View.GONE
                statusText.text = "连接失败，请检查眼镜 IP 与网络"
                Toast.makeText(this@ArActivity, "连接失败", Toast.LENGTH_SHORT).show()
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
        connection?.disconnect()
        stopCamera()
        det.shutdown()
        openExec.shutdown()
        resetOverlay()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "ar-java"
        private const val REQ_CAM = 1
        // det_10g 图内写死了 448 输入的 FPN 上采样尺寸，实时检测同样固定 448
        private const val LIVE_INPUT = 448
        private const val SCORE_THRESH = 0.50f
    }
}
