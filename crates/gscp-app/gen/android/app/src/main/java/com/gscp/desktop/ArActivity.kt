package com.gscp.desktop

import android.annotation.SuppressLint
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
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
    private lateinit var statusText: TextView
    private lateinit var progressView: android.view.View
    private lateinit var ipEdit: EditText

    private var connection: ScrcpyConnection? = null
    private var overlayDecoder: VideoDecoder? = null
    private val overlayLock = Any()
    private var overlayBmp: Bitmap? = null
    private var overlayPix: IntArray? = null
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
    private lateinit var meshPaint: Paint
    private lateinit var glowMeshPaint: Paint
    private lateinit var darkBloomPaint: Paint
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
        meshPaint = Paint(Paint.FILTER_BITMAP_FLAG)
        glowMeshPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
            // 加色彩色发光：bloom 模糊层按内容自身颜色加色溢出（文字发光）
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                blendMode = android.graphics.BlendMode.PLUS
            } else {
                @Suppress("DEPRECATION")
                xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.ADD)
            }
            alpha = 150   // 发光强度（降采样模糊后能量分散）
        }
        darkBloomPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
            // 黑色染色 bloom：内容边缘暗晕（SRC_IN 染黑后正常叠加 = 局部压暗）
            colorFilter = android.graphics.PorterDuffColorFilter(
                0xFF000000.toInt(), android.graphics.PorterDuff.Mode.SRC_IN)
            alpha = 110
        }
        // 叠加方式 = RGB 相加（PC 合成器同款加色混合；键控后的暗像素加成 ≈0）
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            meshPaint.blendMode = android.graphics.BlendMode.PLUS
        } else {
            @Suppress("DEPRECATION")
            meshPaint.xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.ADD)
        }
        // overlay 区域半透明背景层：先压暗相机，再叠加内容（提升可读性）
        dimPaint = Paint().apply { color = 0x59000000.toInt() }
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

    /** 只提取 overlay 四角锚点（屏幕投影位置）；人脸丢失时为空 → 显示连接状态。 */
    private fun parseResult(j: String, rw: Int, rh: Int) {
        val lines = ArrayList<String>()
        val boxes = ArrayList<FloatArray>()
        val scores = ArrayList<Float>()
        val kpsList = ArrayList<FloatArray>()
        val normalList = ArrayList<FloatArray>()
        val ovList = ArrayList<FloatArray>()
        val yprList = ArrayList<FloatArray>()
        try {
            val o = JSONObject(j)
            lastConvMs = o.optDouble("convMs", 0.0).toFloat()
            lastInferMs = o.optDouble("ms", 0.0).toFloat()
            val total = lastCopyMs + lastProcMs + lastDrawMs
            lines.add(String.format(Locale.ROOT, "FPS %.1f · 帧耗时 %.1fms", fpsEma[0], total))
            lines.add(String.format(Locale.ROOT,
                "耗时: 拷贝 %.1f · 转换 %.1f · 推理 %.1f · 关键点 %.1f · 绘制 %.1f ms",
                lastCopyMs, lastConvMs, lastInferMs, 0f, lastDrawMs))
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
                for (i in 0 until n) {
                    val f = fs?.optJSONObject(i) ?: continue
                    val b = f.optJSONArray("box") ?: continue
                    val k = f.optJSONArray("kps") ?: continue
                    if (b.length() < 4 || k.length() < 10) continue
                    boxes.add(FloatArray(4) { q -> b.optDouble(q, 0.0).toFloat() })
                    scores.add(f.optDouble("score", 0.0).toFloat())
                    kpsList.add(FloatArray(10) { q -> k.optDouble(q, 0.0).toFloat() })
                    val po = f.optJSONArray("pose")
                    val nm = f.optJSONArray("normal")
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
                    if (po != null && po.length() == 3) {
                        yprList.add(FloatArray(3) { q -> po.optDouble(q, 0.0).toFloat() })
                    }
                    if (nm != null && nm.length() == 2) {
                        val nrm = FloatArray(4)
                        for (q in 0 until 2) {
                            val pt = nm.optJSONArray(q)
                            nrm[2 * q] = pt?.optDouble(0, 0.0)?.toFloat() ?: 0f
                            nrm[2 * q + 1] = pt?.optDouble(1, 0.0)?.toFloat() ?: 0f
                        }
                        normalList.add(nrm)
                    }
                }
                if (n > 0 && kpsList.isNotEmpty()) {
                    val k0 = kpsList[0]
                    val sb = StringBuilder("face0 点: ")
                    val names = arrayOf("眼", "眼", "鼻", "嘴角", "嘴角")
                    for (q in 0 until 5)
                        sb.append(names[q]).append(q).append("(")
                            .append(Math.round(k0[2 * q])).append(",")
                            .append(Math.round(k0[2 * q + 1])).append(") ")
                    lines.add(sb.toString())
                    if (yprList.isNotEmpty()) {
                        val p0 = yprList[0]
                        lines.add(String.format(Locale.ROOT, "face0 姿态: yaw %.0f° pitch %.0f° roll %.0f°",
                            p0[1], p0[0], p0[2]))
                    }
                }
                lines.add("算法:" + o.optString("algo", "fusion"))
            }
        } catch (e: Exception) {
            lines.add("JSON 解析失败: $e")
        }
        if (openErr != null) lines.add("网络加载失败: $openErr")
        debugLines = lines.toTypedArray()
        faceOv = ovList.toTypedArray()
        if (ovList.isNotEmpty() && ++cornerDbg % 40 == 1) {
            val o = ovList[0]
            android.util.Log.i(TAG, String.format(Locale.ROOT,
                "corners c0=(%.0f,%.0f) c1=(%.0f,%.0f) c2=(%.0f,%.0f) c3=(%.0f,%.0f)",
                o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]))
        }
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
                val m = Matrix()
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
                // 相机整体亮度 90%（ColorMatrix 缩放 RGB 分量）
                val dim = Paint().apply {
                    colorFilter = android.graphics.ColorMatrixColorFilter(
                        android.graphics.ColorMatrix(floatArrayOf(
                            0.90f, 0f, 0f, 0f, 0f,
                            0f, 0.90f, 0f, 0f, 0f,
                            0f, 0f, 0.90f, 0f, 0f,
                            0f, 0f, 0f, 1f, 0f,
                        )))
                }
                c.drawBitmap(b, m, dim)
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
                // 背景平面：沿 overlay 透视四角描出（法线/旋转与人脸同步），
                // 圆角用顶点内插实现（四角向形心收缩 rounded 比例后二次贝塞尔连接）。
                val qx = floatArrayOf(mapX(ov[0]), mapX(ov[2]), mapX(ov[4]), mapX(ov[6]))
                val qy = floatArrayOf(mapY(ov[1]), mapY(ov[3]), mapY(ov[5]), mapY(ov[7]))
                var ccx2 = 0f; var ccy2 = 0f
                for (q in 0 until 4) { ccx2 += qx[q] / 4f; ccy2 += qy[q] / 4f }
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
                        ox[q] = (qx[kp] + (qx[prevIdx] - qx[kp]) * 0.10f)
                        oy[q] = (qy[kp] + (qy[prevIdx] - qy[kp]) * 0.10f)
                        ix[q] = (qx[kp] + (qx[nextIdx] - qx[kp]) * 0.10f)
                        iy[q] = (qy[kp] + (qy[nextIdx] - qy[kp]) * 0.10f)
                    }
                    p.moveTo(ox[0], oy[0])
                    for (q in 0 until 4) {
                        val kp = ord[q]
                        // 圆角 q：入点 A_q --(控制=角点)--> 出点 B_q
                        p.quadTo(qx[kp], qy[kp], ix[q], iy[q])
                        // 边：出点 B_q --直线--> 下一角入点 A_{q+1}
                        val n = (q + 1) % 4
                        p.lineTo(ox[n], oy[n])
                    }
                    p.close()
                    return p
                }
                val platePath = roundedQuad()
                // 发光（无边线）：模糊光晕画在背板之下，沿轮廓向外溢出形成柔光
                val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    style = Paint.Style.STROKE
                    strokeWidth = 6 * dp0()
                    color = 0x1490EE90.toInt()
                    maskFilter = android.graphics.BlurMaskFilter(
                        5 * dp0(), android.graphics.BlurMaskFilter.Blur.NORMAL)
                }
                c.drawPath(platePath, glow)
                // 半透明背板（与 overlay 平面同姿态、同圆角轮廓）：浅绿低透明
                dimPaint.color = 0x1490EE90.toInt()
                c.drawPath(platePath, dimPaint)
                if (ob != null) {
                    // drawBitmapMesh 顶点序：(0,0),(w,0),(0,h),(w,h) → 四角 TL,TR,BL,BR；
                    // meshPaint = 加色混合（RGB 相加），内容贴在四角上
                    // 实测角点布局：c0=左下 c1=右下 c2=右上 c3=左上
                    // mesh 槽位 TL,TR,BL,BR ← c3,c2,c0,c1
                    val verts = floatArrayOf(
                        mapX(ov[6]), mapY(ov[7]),
                        mapX(ov[4]), mapY(ov[5]),
                        mapX(ov[0]), mapY(ov[1]),
                        mapX(ov[2]), mapY(ov[3]),
                    )
                    // 内容 bloom（1/4 降采样模糊）：
                    // ① 黑色染色 bloom = 内容边缘的柔和暗晕（变暗）
                    // ② 加色彩色 bloom = 文字/亮内容的彩色发光（恢复）
                    // ③ 清晰主体
                    val sw = (ob.width / 4).coerceAtLeast(8)
                    val sh = (ob.height / 4).coerceAtLeast(8)
                    val small = Bitmap.createScaledBitmap(ob, sw, sh, true)
                    c.drawBitmapMesh(small, 1, 1, verts, 0, null, 0, darkBloomPaint)
                    c.drawBitmapMesh(small, 1, 1, verts, 0, null, 0, glowMeshPaint)
                    if (small !== ob) small.recycle()
                    c.drawBitmapMesh(ob, 1, 1, verts, 0, null, 0, meshPaint)
                }
                // 屏幕画面未就绪 → 锚点处显示连接状态
                if (ob == null) {
                    drawConnectionStatus(c, ccx2, ccy2)
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

    /** 解码线程回调：overlay I420 Image → RGBA Bitmap（BT.601 limited，兼容 planar/semiplanar）。 */
    private fun convertOverlayFrame(img: Image) {
        try {
            val w = img.width
            val h = img.height
            val pl = img.planes
            val yB = pl[0].buffer
            val uB = pl[1].buffer
            val vB = pl[2].buffer
            val yRs = pl[0].rowStride
            val uRs = pl[1].rowStride
            val vRs = pl[2].rowStride
            val uPs = pl[1].pixelStride
            val vPs = pl[2].pixelStride
            synchronized(overlayLock) {
                var pix = overlayPix
                if (pix == null || pix.size != w * h) { pix = IntArray(w * h); overlayPix = pix }
                if (overlayBmp == null || overlayBmp!!.width != w || overlayBmp!!.height != h) {
                    overlayBmp?.recycle()
                    overlayBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                }
                // BT.601 limited（与 native yuvToBgr 同式）；兼容 planar/semiplanar UV。
                // 逐像素做 PC 合成器同款处理：黑键抠像（褐/暗背景→透明）+ 饱和度 ×1.12。
                var o = 0
                for (j in 0 until h) {
                    val yRow = j * yRs
                    val uRow = (j shr 1) * uRs
                    val vRow = (j shr 1) * vRs
                    for (i in 0 until w) {
                        // 水平镜像：观察者看到的眼镜屏与内容左右相反（旧 GL 路径 u→1-u 同理）
                        val si = w - 1 - i
                        val y = (yB.get(yRow + si).toInt() and 0xFF) - 16
                        val u = (uB.get(uRow + (si shr 1) * uPs).toInt() and 0xFF) - 128
                        val v = (vB.get(vRow + (si shr 1) * vPs).toInt() and 0xFF) - 128
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
                        val l2 = luma
                        r = (l2 + (r - l2) * 28 / 25).coerceIn(0, 255)
                        g = (l2 + (g - l2) * 28 / 25).coerceIn(0, 255)
                        b = (l2 + (b - l2) * 28 / 25).coerceIn(0, 255)
                        pix[o] = (a shl 24) or (r shl 16) or (g shl 8) or b
                        o++
                    }
                }
                overlayBmp!!.setPixels(pix, 0, w, 0, 0, w, h)
                overlayNew = true
            }
        } catch (e: Exception) {
            android.util.Log.w(TAG, "overlay frame convert fail", e)
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
