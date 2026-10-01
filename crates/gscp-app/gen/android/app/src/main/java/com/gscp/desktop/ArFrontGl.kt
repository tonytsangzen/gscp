package com.gscp.desktop

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.opengl.GLUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * AR 前摄 GL 合成器（GLES2；EGL 脚手架与 [SurfaceMixer] 同构，支持多输出面）：
 * 把旧 CPU Canvas 上屏路径整体搬到 GPU，上屏与录像共用同一份合成结果
 * （编码器面直挂输出，与后摄 SurfaceMixer 一致）：
 *  - 相机层：检测线程产出的直立 RGBA（Camera2 → nativeFaceDetect 直写，所见即所测）
 *    glTexSubImage2D 上传，前置镜像 + letterbox 居中，RGB×0.70 压暗（同旧 camDimPaint）；
 *  - 背板层：overlay quad 上 SDF 圆角矩形衬底 + 外缘柔光（颜色/透明度可 setprop 调参）；
 *  - overlay 层：只渲染眼镜画面本身——内容单应 5×5 网格（同旧 drawBitmapMesh 顶点算法）
 *    + 其派生的两级 bloom 光晕（远 ADD α90 → 中 ADD α120 → 内容 SRC_OVER，RGB×1.4+12）。
 *    不叠加任何非眼镜来源的 UI（"连接中..."提示文字已移除）：无内容时只有背板。
 * 33ms 定时推帧到全部挂载输出；active=false（后摄模式）只输出黑场。
 */
class ArFrontGl(
    private val density: Float,
    val width: Int,
    val height: Int,
    /** 每拍取 overlay 内容快照（GL 线程调用；实现方持锁回调，位图仅在回调内有效）。 */
    private val overlays: WithOverlays,
) : RearComposer {
    // ── RearComposer（后摄同款 GlassesPlayer 管线的合成器面）────────────
    // 前摄连接为 overlayOnly（无视频流）：bottom/top 输入面不被消费，给 1×1 占位；
    // overlay 走覆写的 openOverlayStream（GPU 烘焙核心）。视频参数 setter 均不消费。
    private var placeholderTex = 0
    private var placeholderSt: SurfaceTexture? = null
    private var placeholderSurface: Surface? = null

    override fun getBottomSurface(): Surface = placeholderSurface
        ?: throw IllegalStateException("frontGl placeholder surface 未初始化（GL 线程未就绪）")

    override fun getTopSurface(): Surface = getBottomSurface()

    override fun setBottomAspectRatio(ratio: Float) {}
    override fun setBottomRotation(rotation: Float, mirror: Boolean) {}
    override fun setTopAspectRatio(ratio: Float) {}
    override fun setOverlayScale(scale: Float) {}
    override fun setOverlayParams(
        alpha: Float, brightness: Float, saturation: Float, dim: Float,
        keyLow: Float, keyHigh: Float, featherPower: Float, featherRadiusPx: Int,
    ) {}
    override fun setBaseBrightness(value: Float) {}
    override fun setTopRotation(rotationDeg: Int, mirror: Boolean) {}

    /** 会话收尾：关 GPU 烘焙核心（CPU 烘焙位图/版本由 ArActivity.resetOverlay 复位）。 */
    override fun reset() {
        closeOverlayStream()
    }

    /** 1×1 占位输入面（GL 线程创建； GlassesPlayer overlayOnly 流程不会消费）。 */
    private fun createPlaceholderSurface() {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, 1, 1, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        placeholderTex = t[0]
        val st = SurfaceTexture(t[0])
        st.setDefaultBufferSize(1, 1)
        placeholderSt = st
        placeholderSurface = Surface(st)
    }
    fun interface WithOverlays {
        fun snapshot(action: (content: Bitmap?, glow: Bitmap?, bloom: Bitmap?, version: Int) -> Unit)
    }

    /** 每拍渲染完成后回调（GL 线程；驱动录像编码线程 drain，≈GL 帧率）。 */
    var frameCallback: (() -> Unit)? = null

    @Volatile private var active = true
    @Volatile private var released = false
    // face0 17 值（overlay 四角 8 + 背板 8 + 距离），直立图像坐标；camW/H = 直立尺寸
    @Volatile private var face: FloatArray? = null
    // 锚点渲染侧指数平滑（一阶低通，τ≈70ms）：锚点率(~20fps，Vulkan 退化时更低)
    // 低于渲染率(30fps)时阶梯跟随 + 速度外推在噪声锚点上会抖——改为渲染帧指数
    // 趋近最新锚点：无过冲、天然消检测噪声，代价是固定 ~70ms 平滑延迟。
    // native 的 One-Euro 已滤锚点噪声，此级只做时间维度的连续化。
    private val faceRender = FloatArray(17)
    private var faceRenderValid = false
    private var lastRenderAt = 0L
    // overlay 流活性：由宿主注入检查（3s 无 overlay 包 = 暂停 → 隐藏 overlay+背板）
    @Volatile var streamAliveCheck: (() -> Boolean)? = null
    @Volatile private var streamAlive = true
    private var streamAliveLogged = true
    private var spdS = -1f                           // 锚点运动速度的指数平滑（px/帧）
    private var tauScale = 1f                        // debug.gscp.tauscale：τ 缩放（>1 更平滑/更迟滞，<1 更跟手）
    @Volatile private var camW = 0
    @Volatile private var camH = 0

    // 背板颜色/透明度（系统属性轮询，真机 setprop 免编译调参）：
    //   debug.gscp.platecolor  RRGGBB / #RRGGBB（默认 004000 暗绿）
    //   debug.gscp.platealpha  衬底不透明度 0..1（默认 0.1）
    private val plateColor = floatArrayOf(0f, 0x40 / 255f, 0f)
    private var plateAlpha = 0.1f
    private var propTick = 0

    // overlay 内容镜像开关（debug.gscp.ovmirror，默认开）：
    // convertOverlayFrame 烘焙时已按观察者视角水平镜像；本开关开=维持该观感，
    // 关=GL 网格 u 翻转抵消烘焙镜像（还原眼镜流原始朝向，正文字/号牌用）
    @Volatile private var ovMirror = true


    private val renderSurfaces = mutableListOf<RenderSurface>()

    // EGL
    private val eglDisplay: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private lateinit var eglContext: EGLContext
    private lateinit var eglConfig: EGLConfig

    private val handlerThread: HandlerThread = HandlerThread("ar-front-gl")
    private val handler: Handler

    private inner class RenderSurface(val surface: Surface, val crop: Boolean) {
        val eglSurface: EGLSurface
        val surfW: Int
        val surfH: Int

        init {
            val surfaceAttributes = intArrayOf(EGL14.EGL_NONE)
            eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, surface, surfaceAttributes, 0)
            val v = IntArray(1)
            EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_WIDTH, v, 0)
            surfW = v[0]
            EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_HEIGHT, v, 0)
            surfH = v[0]
        }

        fun release() {
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
        }
    }

    // GL：三个 pass（相机纹理 / 背板 SDF / 网格贴图）
    private var pCamera = 0
    private var pPlate = 0
    private var pMesh = 0
    private var uCamTex = 0
    private var uCamDim = 0
    private var uPlateSize = 0
    private var uPlateRadius = 0
    private var uPlateGlowSigma = 0
    private var uPlateColor = 0
    private var uPlateAlpha = 0
    private var uMeshTex = 0
    private var uMeshAlpha = 0
    private var uMeshMul = 0
    private var uMeshAdd = 0
    private var uMeshFlip = 0

    private var camTex = 0
    private var camTexW = 0
    private var camTexH = 0
    private var contentTex = 0
    private var glowTex = 0
    private var bloomTex = 0
    private var overlayVersionSeen = Int.MIN_VALUE
    private var hasContent = false

    // GPU 烘焙核心（debug.gscp.ovgl=1 时替代 CPU 烘焙位图快照）：解码直写其输入面，
    // GL 线程 bakeIfPending 后产物纹理直接供 drawScene（同一 context，零拷贝）。
    @Volatile private var baker: OverlayBakeCore? = null

    private lateinit var quadBuf: FloatBuffer
    private lateinit var meshBuf: FloatBuffer
    private val quadArr = FloatArray(6 * 4)
    private val meshArr = FloatArray(MESH_N * MESH_N * 6 * 4)
    private val meshPos = FloatArray((MESH_N + 1) * (MESH_N + 1) * 2)   // 单应网格（canvas px）

    init {
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        handler.post {
            val version = IntArray(2)
            EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

            val configAttributes = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                // RECORDABLE_ANDROID：允许把合成结果挂到 MediaCodec 编码器输入 Surface（录像）
                android.opengl.EGLExt.EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(eglDisplay, configAttributes, 0, configs, 0, 1, numConfigs, 0)
            eglConfig = configs[0]!!

            val contextAttributes = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,   // ES3：overlay GL 原生 ESPCN 超分需要；GLSL 100 旧 shader 兼容
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttributes, 0)

            val pbSurface = EGL14.eglCreatePbufferSurface(
                eglDisplay, eglConfig,
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(eglDisplay, pbSurface, pbSurface, eglContext)
            initGl()
            createPlaceholderSurface()

            renderFrame()
        }
    }

    private fun initGl() {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
        pCamera = buildProgram(FRAG_CAMERA)
        pPlate = buildProgram(FRAG_PLATE)
        pMesh = buildProgram(FRAG_MESH)
        uCamTex = GLES20.glGetUniformLocation(pCamera, "uTex")
        uCamDim = GLES20.glGetUniformLocation(pCamera, "uDim")
        uPlateSize = GLES20.glGetUniformLocation(pPlate, "uSize")
        uPlateRadius = GLES20.glGetUniformLocation(pPlate, "uRadius")
        uPlateGlowSigma = GLES20.glGetUniformLocation(pPlate, "uGlowSigma")
        uPlateColor = GLES20.glGetUniformLocation(pPlate, "uPlateColor")
        uPlateAlpha = GLES20.glGetUniformLocation(pPlate, "uPlateAlpha")
        uMeshTex = GLES20.glGetUniformLocation(pMesh, "uTex")
        uMeshAlpha = GLES20.glGetUniformLocation(pMesh, "uAlpha")
        uMeshMul = GLES20.glGetUniformLocation(pMesh, "uMul")
        uMeshAdd = GLES20.glGetUniformLocation(pMesh, "uAdd")
        uMeshFlip = GLES20.glGetUniformLocation(pMesh, "uFlip")

        quadBuf = ByteBuffer.allocateDirect(quadArr.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        meshBuf = ByteBuffer.allocateDirect(meshArr.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        // 三个 program 的唯一 attribute 均绑定到 location 0
        GLES20.glEnableVertexAttribArray(0)
    }

    // ── 对外接口（线程安全）────────────────────────────────────

    fun setActive(a: Boolean) {
        active = a
    }

    /** 相机帧（nativeFaceDetect 直写的直立 RGBA，rw×rh）：GL 线程上传为纹理。 */
    fun postCameraFrame(buf: ByteBuffer, w: Int, h: Int) {
        camW = w
        camH = h
        handler.post {
            if (released) return@post
            try {
                ensureCameraTex(w, h)
                buf.rewind()
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTex)
                GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, w, h,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            } catch (t: Throwable) {
                Log.w(TAG, "camera tex upload fail", t)
            }
        }
    }

    /** 人脸锚点状态（face0 17 值或 null）+ 直立相机尺寸。 */
    fun setFace(ov17: FloatArray?, w: Int, h: Int) {
        face = ov17
        if (ov17 == null || w != camW || h != camH) faceRenderValid = false   // 复位：下一帧直接吸附
        camW = w
        camH = h
    }

    /**
     * 开 overlay 流的 GPU 烘焙路径：GL 线程建 OverlayBakeCore（同 context，产物
     * 纹理直接可用），同步返回解码输入面。与 CPU 烘焙互斥（见 debug.gscp.ovgl）。
     * sr=true：核心内部先过 GL 原生 ESPCN ×2 超分（失败自动退回非 SR）。
     */
    override fun openOverlayStream(streamW: Int, streamH: Int, sr: Boolean, srWeights: ByteArray?): Surface {
        val latch = java.util.concurrent.CountDownLatch(1)
        var surface: Surface? = null
        handler.post {
            if (released) {
                latch.countDown()
                return@post
            }
            try {
                val old = baker
                baker = null
                old?.let { runCatching { it.release() } }
                val b = OverlayBakeCore()
                surface = b.open(streamW, streamH, handler, sr, srWeights)
                baker = b
                ArNative.nativeSetOverlaySize(b.contentW(), b.contentH())
                Log.i(TAG, "overlay gl-bake open ${streamW}x${streamH} content=${b.contentW()}x${b.contentH()}")
            } catch (t: Throwable) {
                Log.w(TAG, "overlay gl-bake open fail", t)
                runCatching { baker?.release() }
                baker = null
            }
            latch.countDown()
        }
        try { latch.await(2, java.util.concurrent.TimeUnit.SECONDS) } catch (_: InterruptedException) {}
        return surface ?: throw IllegalStateException("overlay gl-bake open timeout")
    }

    /** 关闭 GPU 烘焙路径并清空产物纹理（断流/回退 CPU 路径前调用）。 */
    fun closeOverlayStream() {
        handler.post {
            val b = baker ?: return@post
            baker = null
            try { b.release() } catch (t: Throwable) { Log.w(TAG, "overlay gl-bake close fail", t) }
        }
    }

    /** GPU 烘焙产物转储（真机 A/B 取证；GL 线程执行）。 */
    fun dumpOverlay(pathPrefix: String) {
        handler.post { baker?.dump(pathPrefix) }
    }

    /** 挂输出面。crop=true：只输出 camera 有效区域（letterbox 内容矩形，
     *  拉伸铺满该面）——录像编码器用，画面不含屏幕黑边。 */
    fun attachOutputSurface(surface: Surface, crop: Boolean = false) {
        android.util.Log.i("ar-front-gl", "attachOutputSurface crop=$crop count=${renderSurfaces.size + 1}")
        handler.post {
            if (released) return@post
            synchronized(renderSurfaces) {
                renderSurfaces.add(RenderSurface(surface, crop))
            }
        }
    }

    /** 摘除一个输出面。同步等待 GL 线程完成（防后续仍向已失效面 swap）；
     *  releaseSurface=false 用于编码器输入 Surface（生命周期归编码器管）。 */
    fun detachOutputSurface(surface: Surface, releaseSurface: Boolean = true) {
        android.util.Log.i("ar-front-gl", "detachOutputSurface")
        val latch = java.util.concurrent.CountDownLatch(1)
        handler.post {
            synchronized(renderSurfaces) {
                val it = renderSurfaces.iterator()
                while (it.hasNext()) {
                    val rs = it.next()
                    if (rs.surface === surface) {
                        it.remove()
                        rs.release()
                    }
                }
            }
            if (releaseSurface) surface.release()
            latch.countDown()
        }
        try { latch.await(1, java.util.concurrent.TimeUnit.SECONDS) } catch (_: InterruptedException) {}
    }

    fun release() {
        released = true
        handler.post {
            synchronized(renderSurfaces) {
                for (rs in renderSurfaces) rs.release()
                renderSurfaces.clear()
            }
            baker?.let { runCatching { it.release() } }
            baker = null
            placeholderSurface?.release()
            placeholderSurface = null
            placeholderSt?.release()
            placeholderSt = null
            if (placeholderTex != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(placeholderTex), 0)
                placeholderTex = 0
            }
            val texs = intArrayOf(camTex, contentTex, glowTex, bloomTex)
            GLES20.glDeleteTextures(texs.size, texs, 0)
            camTex = 0; contentTex = 0; glowTex = 0; bloomTex = 0
            EGL14.eglMakeCurrent(eglDisplay,
                EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)
            handlerThread.quitSafely()
        }
    }

    // ── 渲染循环 ───────────────────────────────────────────────

    private fun renderFrame() {
        if (!released) {
            try {
                tick()
            } catch (t: Throwable) {
                Log.w(TAG, "front gl tick fail", t)
            }
            frameCallback?.invoke()
            handler.postDelayed({ renderFrame() }, 33)
        }
    }

    private fun tick() {
        pollPlateProps()
        val alive = streamAliveCheck?.invoke() ?: true
        if (alive != streamAlive) {
            streamAlive = alive
            streamAliveLogged = false
            Log.i(TAG, "overlay stream " + if (alive) "resumed" else "paused >3s -> hide overlay/plate")
        }
        val b = baker
        if (b != null) {
            // GPU 烘焙：消费解码新帧（产物纹理原地更新，无需上传/版本比对）
            try { b.bakeIfPending() } catch (t: Throwable) { Log.w(TAG, "ov bake fail", t) }
        } else {
            // overlay 内容快照 → 纹理（版本号变化才重新上传；位图在 provider 持锁期间有效）
            overlays.snapshot { content, glow, bloom, ver ->
                if (ver != overlayVersionSeen) {
                    overlayVersionSeen = ver
                    contentTex = uploadBitmap(contentTex, content)
                    glowTex = uploadBitmap(glowTex, glow)
                    bloomTex = uploadBitmap(bloomTex, bloom)
                    hasContent = content != null && !content.isRecycled
                }
            }
        }
        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                EGL14.eglMakeCurrent(eglDisplay, rs.eglSurface, rs.eglSurface, eglContext)
                if (rs.crop && camW > 0 && camH > 0) {
                    // 裁剪映射（视口技巧，几何不变）：把 canvas 上的 camera letterbox
                    // 矩形 (x0,y0,dw,dh) 映射到整个编码器面。顶点仍是 canvas px→clip，
                    // 视口选 (vp0,vp) 使 window = k*(p - rectOffset)：
                    //   x: window = kx*(px-x0)        → vpw=W*kx, vp0x=-kx*x0
                    //   y: window = Eh-ky*(py-y0)（GL y 向上）→ vph=H*ky, vp0y=Eh+ky*y0-H*ky
                    // 相机未就绪（camW=0）保持黑场，避免首帧前全屏画面被压进编码器。
                    val sc = minOf(width.toFloat() / camW, height.toFloat() / camH)
                    val dw = camW * sc
                    val dh = camH * sc
                    val x0 = (width - dw) / 2f
                    val y0 = (height - dh) / 2f
                    val kx = rs.surfW / dw
                    val ky = rs.surfH / dh
                    GLES20.glViewport((-kx * x0).toInt(), (rs.surfH + ky * y0 - height * ky).toInt(),
                        (width * kx).toInt(), (height * ky).toInt())
                } else {
                    GLES20.glViewport(0, 0, width, height)
                }
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                if (active && !(rs.crop && camW <= 0)) drawScene()
                EGL14.eglSwapBuffers(eglDisplay, rs.eglSurface)
            }
        }
    }

    private fun drawScene() {
        val w = camW
        val h = camH
        // overlay 纹理源：GPU 烘焙产物优先，回退 CPU 烘焙位图上传
        val b = baker
        val contentTex: Int
        val glowTex: Int
        val bloomTex: Int
        val hasContent: Boolean
        if (b != null && b.hasBaked()) {
            contentTex = b.contentTex(); glowTex = b.glowTex(); bloomTex = b.bloomTex()
            hasContent = true
        } else {
            contentTex = this.contentTex; glowTex = this.glowTex; bloomTex = this.bloomTex
            hasContent = this.hasContent
        }
        // ── 相机层：前置镜像 + letterbox，RGB×0.70 压暗 ──
        if (camTex != 0 && w > 0 && h > 0) {
            val sc = minOf(width.toFloat() / w, height.toFloat() / h)
            val dw = w * sc
            val dh = h * sc
            val x0 = (width - dw) / 2f
            val y0 = (height - dh) / 2f
            GLES20.glDisable(GLES20.GL_BLEND)
            GLES20.glUseProgram(pCamera)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, camTex)
            GLES20.glUniform1i(uCamTex, 0)
            GLES20.glUniform1f(uCamDim, CAM_DIM)
            putQuad(x0, y0, x0 + dw, y0 + dh, 1f, 0f, 0f, 1f)   // 镜像：u 左右翻转
            GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        }

        var ov = face ?: return
        if (!streamAlive) return   // 流暂停 >3s：隐藏 overlay 与背板（相机层照常）
        // 锚点指数平滑：每渲染帧向最新锚点趋近 1-e^(-dt/τ)。锚点率低于渲染率时
        // 填平间隔（不再阶梯），检测噪声被低通（不再抖）；锚点停滞时输出冻结
        // 在最后位置平滑停住（无过冲无回跳）。
        val now = android.os.SystemClock.uptimeMillis()
        val dt = (now - lastRenderAt).coerceIn(1L, 100L).toFloat()
        lastRenderAt = now
        if (!faceRenderValid || faceRender.size != ov.size) {
            System.arraycopy(ov, 0, faceRender, 0, ov.size)
            faceRenderValid = true
            spdS = -1f
        } else {
            // 自适应 τ（1€ 思路）：锚点位移大（转头）→ τ≈30ms 低延迟；静止/微动 →
            // τ≈110ms 强平滑。move = 四角平均位移（canvas px），spdS 指数平滑防切档抖动。
            var move = 0f
            for (i in 0 until 8) { val d = ov[i] - faceRender[i]; move += d * d }
            move = kotlin.math.sqrt(move / 8f)
            spdS = if (spdS < 0f) move else spdS + (move - spdS) * (1f - kotlin.math.exp(-dt / 120f))
            val f = ((spdS - 6f) / 30f).coerceIn(0f, 1f)
            val tau = ((110f - 80f * f) * tauScale).coerceAtLeast(25f)
            val k = 1f - kotlin.math.exp(-dt / tau)
            for (i in ov.indices) faceRender[i] += (ov[i] - faceRender[i]) * k
        }
        ov = faceRender
        if (w <= 0 || h <= 0) return
        // overlay 内容四角（canvas px；与旧 mapX/mapY 一致的前置镜像）
        val sc = minOf(width.toFloat() / w, height.toFloat() / h)
        val x0 = (width - w * sc) / 2f
        val y0 = (height - h * sc) / 2f
        val tlX = (w - ov[6]) * sc + x0; val tlY = ov[7] * sc + y0
        val trX = (w - ov[4]) * sc + x0; val trY = ov[5] * sc + y0
        val brX = (w - ov[2]) * sc + x0; val brY = ov[3] * sc + y0
        val blX = (w - ov[0]) * sc + x0; val blY = ov[1] * sc + y0

        // ── 背板：圆角矩形衬底 + 外缘柔光（SDF；仅人脸锚定即画，属显示背景）──
        val ex = (dist(trX, trY, tlX, tlY) + dist(brX, brY, blX, blY)) / 2f
        val ey = (dist(blX, blY, tlX, tlY) + dist(brX, brY, trX, trY)) / 2f
        GLES20.glUseProgram(pPlate)
        blendPremult()
        GLES20.glUniform2f(uPlateSize, ex, ey)
        GLES20.glUniform1f(uPlateRadius, PLATE_CORNER * minOf(ex, ey))
        GLES20.glUniform1f(uPlateGlowSigma, GLOW_SIGMA_DP * density)
        GLES20.glUniform3f(uPlateColor, plateColor[0], plateColor[1], plateColor[2])
        GLES20.glUniform1f(uPlateAlpha, plateAlpha)
        putQuad4(tlX, tlY, trX, trY, brX, brY, blX, blY)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)

        // ── overlay：只渲染眼镜画面（单应网格投影 + 派生 bloom）；
        //    无内容时不画内容（"连接中..."提示等自有 UI 已移除）──
        if (hasContent) {
            computeHomography(tlX, tlY, trX, trY, brX, brY, blX, blY)
            packMesh()
            drawPackedMesh(bloomTex, additive = true, alpha = BLOOM_FAR_ALPHA, mul = MUL_BLOOM, add = 0f)
            drawPackedMesh(glowTex, additive = true, alpha = BLOOM_MID_ALPHA, mul = MUL_BLOOM, add = 0f)
            drawPackedMesh(contentTex, additive = false, alpha = 1f, mul = MUL_CONTENT, add = 12f / 255f)
        }
    }

    // ── 几何与绘制 ─────────────────────────────────────────────

    /** Heckbert square→quad 单应：网格顶点取单应像（与旧 drawBitmapMesh 顶点算法逐行一致）。 */
    private fun computeHomography(
        tlX: Float, tlY: Float, trX: Float, trY: Float,
        brX: Float, brY: Float, blX: Float, blY: Float,
    ) {
        val d1x = trX - brX; val d1y = trY - brY
        val d2x = blX - brX; val d2y = blY - brY
        val sxq = tlX - trX + brX - blX
        val syq = tlY - trY + brY - blY
        val den = d1x * d2y - d1y * d2x
        var k = 0
        if (abs(den) > 1e-6f) {
            val gh = (sxq * d2y - syq * d2x) / den
            val hv = (d1x * syq - d1y * sxq) / den
            val av = trX - tlX + gh * trX
            val bv = blX - tlX + hv * blX
            val dv = trY - tlY + gh * trY
            val ev = blY - tlY + hv * blY
            for (j in 0..MESH_N) {
                val vv = j.toFloat() / MESH_N
                for (i in 0..MESH_N) {
                    val uu = i.toFloat() / MESH_N
                    val wq = gh * uu + hv * vv + 1f
                    meshPos[k++] = (av * uu + bv * vv + tlX) / wq
                    meshPos[k++] = (dv * uu + ev * vv + tlY) / wq
                }
            }
        } else {
            // 退化（近似仿射）：回退四角双线性（与旧实现一致）
            for (j in 0..MESH_N) {
                val vv = j.toFloat() / MESH_N
                val ly0 = blX + (brX - blX) * vv; val ly1 = tlX + (trX - tlX) * vv
                val lx0 = blY + (brY - blY) * vv; val lx1 = tlY + (trY - tlY) * vv
                for (i in 0..MESH_N) {
                    val uu = i.toFloat() / MESH_N
                    meshPos[k++] = ly0 + (ly1 - ly0) * uu
                    meshPos[k++] = lx0 + (lx1 - lx0) * uu
                }
            }
        }
    }

    /** 单应网格 → 三角形顶点（canvas px → clip，uv 随网格），写入 [meshBuf]。 */
    private fun packMesh() {
        val a = meshArr
        val cols = MESH_N + 1
        var k = 0
        for (j in 0 until MESH_N) {
            for (i in 0 until MESH_N) {
                val i0 = j * cols + i
                val u0 = i.toFloat() / MESH_N; val v0 = j.toFloat() / MESH_N
                val u1 = (i + 1).toFloat() / MESH_N; val v1 = (j + 1).toFloat() / MESH_N
                k = pushV(a, k, i0, u0, v0); k = pushV(a, k, i0 + cols, u0, v1); k = pushV(a, k, i0 + 1, u1, v0)
                k = pushV(a, k, i0 + 1, u1, v0); k = pushV(a, k, i0 + cols, u0, v1); k = pushV(a, k, i0 + cols + 1, u1, v1)
            }
        }
        meshBuf.position(0)
        meshBuf.put(a, 0, k)
        meshBuf.position(0)
    }

    private fun pushV(a: FloatArray, k: Int, idx: Int, u: Float, v: Float): Int {
        a[k] = meshPos[2 * idx] / width.toFloat() * 2f - 1f
        a[k + 1] = 1f - meshPos[2 * idx + 1] / height.toFloat() * 2f
        a[k + 2] = u
        a[k + 3] = v
        return k + 4
    }

    private fun drawPackedMesh(tex: Int, additive: Boolean, alpha: Float, mul: FloatArray, add: Float) {
        if (tex == 0) return
        if (additive) blendAdd() else blendPremult()
        GLES20.glUseProgram(pMesh)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(uMeshTex, 0)
        GLES20.glUniform1f(uMeshAlpha, alpha)
        GLES20.glUniform3f(uMeshMul, mul[0], mul[1], mul[2])
        GLES20.glUniform1f(uMeshAdd, add)
        GLES20.glUniform1f(uMeshFlip, if (ovMirror) 0f else 1f)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, meshBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, MESH_N * MESH_N * 6)
    }

    /** 矩形（canvas px）+ uv 矩形 → 两三角形写入 [quadBuf]。 */
    private fun putQuad(x0: Float, y0: Float, x1: Float, y1: Float, u0: Float, v0: Float, u1: Float, v1: Float) {
        val a = quadArr
        var k = 0
        fun put(px: Float, py: Float, u: Float, v: Float) {
            a[k++] = px / width.toFloat() * 2f - 1f
            a[k++] = 1f - py / height.toFloat() * 2f
            a[k++] = u
            a[k++] = v
        }
        put(x0, y0, u0, v0); put(x1, y0, u1, v0); put(x0, y1, u0, v1)
        put(x1, y0, u1, v0); put(x1, y1, u1, v1); put(x0, y1, u0, v1)
        quadBuf.position(0)
        quadBuf.put(a)
        quadBuf.position(0)
    }

    /** 四角任意的 quad（canvas px，序 TL,TR,BR,BL）→ 两三角形写入 [quadBuf]。 */
    private fun putQuad4(
        tlX: Float, tlY: Float, trX: Float, trY: Float,
        brX: Float, brY: Float, blX: Float, blY: Float,
    ) {
        val a = quadArr
        var k = 0
        fun put(px: Float, py: Float, u: Float, v: Float) {
            a[k++] = px / width.toFloat() * 2f - 1f
            a[k++] = 1f - py / height.toFloat() * 2f
            a[k++] = u
            a[k++] = v
        }
        put(tlX, tlY, 0f, 0f); put(trX, trY, 1f, 0f); put(blX, blY, 0f, 1f)
        put(trX, trY, 1f, 0f); put(brX, brY, 1f, 1f); put(blX, blY, 0f, 1f)
        quadBuf.position(0)
        quadBuf.put(a)
        quadBuf.position(0)
    }

    private fun dist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = (x2 - x1).toDouble()
        val dy = (y2 - y1).toDouble()
        return kotlin.math.sqrt(dx * dx + dy * dy).toFloat()
    }

    // ── 纹理与 GL 状态 ─────────────────────────────────────────

    /** 背板颜色/透明度 + overlay 镜像开关属性轮询（GL 线程，每 30 拍一次，
     *  SystemProperties 反射读取）。 */
    private fun pollPlateProps() {
        if (++propTick % 30 != 1) return
        try {
            val sp = Class.forName("android.os.SystemProperties")
            val get = sp.getMethod("get", String::class.java, String::class.java)
            val m = (get.invoke(null, "debug.gscp.ovmirror", "") as String).trim()
            if (m.isNotEmpty()) ovMirror = m != "0" && !m.equals("false", true)
            val hex = (get.invoke(null, "debug.gscp.platecolor", "004000") as String)
                .trim().removePrefix("#").removePrefix("0x")
            if (hex.length >= 6) {
                val rgb = hex.takeLast(6).toLongOrNull(16)
                if (rgb != null) {
                    plateColor[0] = ((rgb shr 16) and 0xFF) / 255f
                    plateColor[1] = ((rgb shr 8) and 0xFF) / 255f
                    plateColor[2] = (rgb and 0xFF) / 255f
                }
            }
            val a = (get.invoke(null, "debug.gscp.platealpha", "") as String).trim()
            if (a.isNotEmpty()) a.toFloatOrNull()?.let { plateAlpha = it.coerceIn(0f, 1f) }
            val ts = (get.invoke(null, "debug.gscp.tauscale", "") as String).trim().toFloatOrNull()
            if (ts != null && ts >= 0.25f) tauScale = ts.coerceAtMost(4f)
        } catch (_: Throwable) {
        }
    }

    private fun ensureCameraTex(w: Int, h: Int) {
        if (camTex != 0 && camTexW == w && camTexH == h) return
        if (camTex != 0) GLES20.glDeleteTextures(1, intArrayOf(camTex), 0)
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        texParams2D()
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        camTex = t[0]
        camTexW = w
        camTexH = h
    }

    /** 位图 → GL_TEXTURE_2D（GLUtils：位图首行 → v=0，与本文 uv 约定一致）；
     *  bm 为 null/已回收时清除旧纹理并返回 0。 */
    private fun uploadBitmap(old: Int, bm: Bitmap?): Int {
        if (old != 0) GLES20.glDeleteTextures(1, intArrayOf(old), 0)
        if (bm == null || bm.isRecycled) return 0
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        texParams2D()
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bm, 0)
        return t[0]
    }

    private fun texParams2D() {
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
    }

    private fun blendPremult() {
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFuncSeparate(
            GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA,
            GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
    }

    private fun blendAdd() {
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE)
    }

    private fun buildProgram(fragSrc: String): Int {
        val vs = loadShader(GLES20.GL_VERTEX_SHADER, VERT_SRC)
        val fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragSrc)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glBindAttribLocation(program, 0, "aPosUv")
        GLES20.glLinkProgram(program)
        val status = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0)
        if (status[0] != GLES20.GL_TRUE) {
            Log.e(TAG, "program link error: " + GLES20.glGetProgramInfoLog(program))
            throw RuntimeException("program link error")
        }
        return program
    }

    private fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, shaderCode)
        GLES20.glCompileShader(shader)
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] != GLES20.GL_TRUE) {
            Log.e(TAG, "Shader compile error: " + GLES20.glGetShaderInfoLog(shader))
            throw RuntimeException("Shader compile error")
        }
        return shader
    }

    companion object {
        private const val TAG = "ar-front-gl"
        private const val MESH_N = 4              // 内容透视网格密度（与旧 drawBitmapMesh 一致）
        private const val CAM_DIM = 0.70f         // 相机压暗（同旧 camDimPaint）
        private const val PLATE_CORNER = 0.10f    // 背板圆角（边长比例，同旧 10% 切角）
        private const val GLOW_SIGMA_DP = 5f      // 背板柔光 σ（同旧 BlurMaskFilter 5dp）
        // 光晕（bloom）参数：透明度较旧值（150/200）下调 —— 减轻加色光晕渗入文字笔画
        // 间隙造成的发雾；扩散后单位面积能量下降，提亮倍数回落到 1.0（需要更亮再调）
        private const val BLOOM_FAR_ALPHA = 90f / 255f    // 远距光晕层透明度
        private const val BLOOM_MID_ALPHA = 120f / 255f   // 中距光晕层透明度
        private const val BLOOM_BRIGHTNESS = 1.0f         // 光晕颜色提亮倍数
        private val MUL_UNIT = floatArrayOf(1f, 1f, 1f)
        private val MUL_CONTENT = floatArrayOf(1.40f, 1.40f, 1.40f)   // 内容提亮（同旧 meshPaint）
        private val MUL_BLOOM = floatArrayOf(
            BLOOM_BRIGHTNESS, BLOOM_BRIGHTNESS, BLOOM_BRIGHTNESS)

        /** 顶点：xy = clip 坐标，zw = uv（纹理 v=0 为位图首行/顶部）。 */
        private const val VERT_SRC = """
            attribute vec4 aPosUv;
            varying vec2 vUv;
            void main() {
                vUv = aPosUv.zw;
                gl_Position = vec4(aPosUv.xy, 0.0, 1.0);
            }
        """

        private const val FRAG_CAMERA = """
            precision mediump float;
            varying vec2 vUv;
            uniform sampler2D uTex;
            uniform float uDim;
            void main() {
                gl_FragColor = vec4(texture2D(uTex, vUv).rgb * uDim, 1.0);
            }
        """

        /** 背板：局部 px 空间圆角矩形 SDF；内部 uPlateColor×uPlateAlpha 衬底 +
         *  外缘同色高斯柔光（premultiplied 输出，blend 为 ONE/ONE_MINUS_SRC_ALPHA）。 */
        private const val FRAG_PLATE = """
            precision mediump float;
            varying vec2 vUv;
            uniform vec2 uSize;
            uniform float uRadius;
            uniform float uGlowSigma;
            uniform vec3 uPlateColor;
            uniform float uPlateAlpha;
            void main() {
                vec2 p = (vUv - 0.5) * uSize;
                vec2 q = abs(p) - (uSize * 0.5 - uRadius);
                float d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - uRadius;
                float fill = clamp(0.5 - d, 0.0, 1.0) * uPlateAlpha;
                float glow = (10.0 / 255.0) * exp(-pow(max(d, 0.0) / uGlowSigma, 2.0));
                float a = clamp(fill + glow, 0.0, 1.0);
                gl_FragColor = vec4(uPlateColor * a, a);
            }
        """

        /** 网格贴图：premultiplied 采样；uMul/uAdd 在非预乘域做内容提亮（1.4/+12），
         *  uAlpha 整层透明度；uFlip=1 时水平翻转（镜像关档抵消烘焙镜像）；
         *  加色/普通合成由 blend 状态区分。 */
        private const val FRAG_MESH = """
            precision mediump float;
            varying vec2 vUv;
            uniform sampler2D uTex;
            uniform float uAlpha;
            uniform vec3 uMul;
            uniform float uAdd;
            uniform float uFlip;
            void main() {
                vec2 uv = vec2(mix(vUv.x, 1.0 - vUv.x, uFlip), vUv.y);
                vec4 s = texture2D(uTex, uv);
                vec3 rgb = vec3(0.0);
                if (s.a > 0.004) {
                    rgb = clamp((s.rgb / s.a) * uMul + uAdd, 0.0, 1.0) * s.a;
                }
                gl_FragColor = vec4(rgb * uAlpha, s.a * uAlpha);
            }
        """
    }
}
