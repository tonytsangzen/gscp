package com.gscp.desktop

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * AR 后摄 GL 合成器：底层 = 眼镜相机码流（decoder → OES SurfaceTexture），
 * contain letterbox 完整显示（前摄相机层同款 min(vw/tw, vh/th) 数学，结构上
 * 不可能裁切），镜像/亮度可配；顶层 = 复用前摄 CPU 烘焙位图
 * （ArActivity.convertOverlayFrame：keyLut 黑键抠像 + 饱和度 ×1.12 + 内容
 * 平均色剪影光晕，纹理上下空带已在烘焙时裁掉），GL 侧仅上传纹理并按前摄
 * 同款三层合成（远 ADD α90 → 中 ADD α120 → 内容 SRC_OVER，un-premult ×1.4+12）。
 *
 * 为什么不走 GLSL 现场键控（旧实现）：luma 羽化区间的半透明像素以
 * "暗色源 RGB × alpha" 预乘合成，字形边缘/压缩噪声会压暗底层 = 抠图残留
 * 黑晕；CPU 烘焙把黑底置为全透明、位图按 premultiplied 经 GLUtils 上传，
 * FRAG_MESH 先 un-premult 还原真实色再做内容提亮，边缘不再发黑，观感与
 * 前摄完全一致。烘焙含观察者视角镜像（前摄投影语义），后摄 = 眼镜自身画面
 * 直出，默认 uFlip=1 抵消（与普通模式 SurfaceMixer 不镜像一致）；
 * debug.gscp.ovmirror=1 可保留烘焙镜像（与前摄同名属性同语义）。
 */
class ArRearGl(
    val width: Int,
    val height: Int,
    /** 每拍取 overlay 烘焙快照（GL 线程调用；实现方持锁回调，位图仅在回调内有效）。 */
    private val overlays: WithOverlays,
) : RearComposer {
    fun interface WithOverlays {
        fun snapshot(action: (content: Bitmap?, glow: Bitmap?, bloom: Bitmap?, version: Int) -> Unit)
    }

    private var bottomRotation = 90f
    private var bottomMirror = false
    private var bottomAspect = 4f / 3f          // 纹理 H/W（GlassesPlayer 口径）
    private var baseBrightness = 1.0f
    private var topRotationDeg = 0
    private var topMirror = false
    private var topAspect = 1f                  // 烘焙位图 W/H（空带已裁，恒 1:1）
    private var overlayScale = 1.0f
    // 后摄默认抵消烘焙镜像；debug.gscp.ovmirror=1 保留（同前摄属性语义）
    @Volatile private var ovMirror = false
    private var propTick = 0

    private val lock = object {}
    private val renderSurfaces = mutableListOf<RenderSurface>()
    @Volatile private var released = false
    // GL 初始化成功才为 true：初始化失败（个别老驱动/模拟器桥 shader 必编译失败）
    // 时禁用渲染而不是让 GL 线程带未捕获异常杀死整个应用。
    @Volatile private var glOk = false

    // EGL
    private val eglDisplay: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private lateinit var eglContext: EGLContext
    private lateinit var eglConfig: EGLConfig

    private val handlerThread: HandlerThread = HandlerThread("ar-rear-gl")
    private val handler: Handler

    private inner class RenderSurface(val surface: Surface) {
        val eglSurface: EGLSurface
        init {
            eglSurface = EGL14.eglCreateWindowSurface(
                eglDisplay, eglConfig, surface, intArrayOf(EGL14.EGL_NONE), 0)
        }
        fun release() = EGL14.eglDestroySurface(eglDisplay, eglSurface)
    }

    private var pBottom = 0
    private var pMesh = 0
    private var uBTex = 0
    private var uBMirror = 0
    private var uBBright = 0
    private var uMeshTex = 0
    private var uMeshAlpha = 0
    private var uMeshMul = 0
    private var uMeshAdd = 0
    private var uMeshFlip = 0

    // 前摄同款烘焙产物纹理：内容 / 1/4 中距光晕 / 1/16 远距光晕
    private var contentTex = 0
    private var glowTex = 0
    private var bloomTex = 0
    private var overlayVersionSeen = Int.MIN_VALUE
    private var hasContent = false
    private var diagTick = 0                    // 诊断节流(每 ~2s 一条)
    // overlay 流活性：由宿主注入检查（3s 无 overlay 包 = 暂停 → 隐藏顶层 overlay）
    @Volatile var streamAliveCheck: (() -> Boolean)? = null
    @Volatile private var streamAlive = true

    // GPU 烘焙核心（debug.gscp.ovgl=1 时替代 CPU 烘焙位图快照）；
    // openOverlayStream 建，reset() 随会话重建，产物纹理同 context 直用。
    @Volatile private var baker: OverlayBakeCore? = null
    private var bakeStreamW = 0
    private var bakeStreamH = 0
    private var bakeStreamSr = false
    private var bakeStreamSrWeights: ByteArray? = null

    private var bottomTexture = 0
    private var topTexture = 0                  // 接口占位（见 createStreams）
    private lateinit var bottomSurfaceTexture: SurfaceTexture
    private lateinit var topSurfaceTexture: SurfaceTexture
    private lateinit var bottomSurface: Surface
    private lateinit var topSurface: Surface
    private var bottomTs = 0L
    private var diagBottom = 0L

    private lateinit var quadBuf: FloatBuffer
    private val quadArr = FloatArray(6 * 4)

    init {
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        handler.post {
            try {
                val version = IntArray(2)
                EGL14.eglInitialize(eglDisplay, version, 0, version, 1)
                val configAttributes = intArrayOf(
                    EGL14.EGL_RED_SIZE, 8,
                    EGL14.EGL_GREEN_SIZE, 8,
                    EGL14.EGL_BLUE_SIZE, 8,
                    EGL14.EGL_ALPHA_SIZE, 8,
                    EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                    android.opengl.EGLExt.EGL_RECORDABLE_ANDROID, 1,
                    EGL14.EGL_NONE
                )
                val configs = arrayOfNulls<EGLConfig>(1)
                val numConfigs = IntArray(1)
                EGL14.eglChooseConfig(eglDisplay, configAttributes, 0, configs, 0, 1, numConfigs, 0)
                if (numConfigs[0] == 0) throw IllegalStateException("no matching EGLConfig")
                eglConfig = configs[0]!!
                // ES3 优先（overlay GL 原生 ESPCN 超分需要；GLSL 100 旧 shader 兼容）；
                // 只有 ES2 的老驱动拿不到 ES3 上下文时降级 ES2（超分退化为关闭）。
                eglContext = createContext(3)
                if (eglContext == EGL14.EGL_NO_CONTEXT) {
                    Log.w("ar-rear-gl", "ES3 context unavailable, falling back to ES2")
                    eglContext = createContext(2)
                }
                val pb = EGL14.eglCreatePbufferSurface(
                    eglDisplay, eglConfig,
                    intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
                EGL14.eglMakeCurrent(eglDisplay, pb, pb, eglContext)
                initGl()
                glOk = true
            } catch (t: Throwable) {
                // 兼容性兜底：shader/上下文初始化失败只降级（画面黑、无 GL 特效），不崩进程。
                if (!::eglContext.isInitialized) eglContext = EGL14.EGL_NO_CONTEXT
                glOk = false
                Log.e("ar-rear-gl", "rear GL init failed - GL rendering disabled", t)
            }
            renderFrame()
        }
    }

    private fun createContext(clientVersion: Int): EGLContext =
        EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, clientVersion, EGL14.EGL_NONE), 0)

    private fun initGl() {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
        pBottom = buildProgram(FRAG_BOTTOM)
        pMesh = buildProgram(FRAG_MESH)
        uBTex = GLES20.glGetUniformLocation(pBottom, "uTex")
        uBMirror = GLES20.glGetUniformLocation(pBottom, "uMirror")
        uBBright = GLES20.glGetUniformLocation(pBottom, "uBright")
        uMeshTex = GLES20.glGetUniformLocation(pMesh, "uTex")
        uMeshAlpha = GLES20.glGetUniformLocation(pMesh, "uAlpha")
        uMeshMul = GLES20.glGetUniformLocation(pMesh, "uMul")
        uMeshAdd = GLES20.glGetUniformLocation(pMesh, "uAdd")
        uMeshFlip = GLES20.glGetUniformLocation(pMesh, "uFlip")
        quadBuf = ByteBuffer.allocateDirect(quadArr.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        GLES20.glEnableVertexAttribArray(0)
        createStreams()
    }

    /** 底层 OES 输入面（decoder 直渲染），frame-available 更新时间戳。
     *  top 仅为 RearComposer 接口占位（1×1）：后摄 overlay 改走 CPU 烘焙位图，
     *  解码端（GlassesPlayer overlayImageCallback）不再消费 top Surface。 */
    private fun createStreams() {
        bottomTexture = createOesTexture()
        bottomSurfaceTexture = SurfaceTexture(bottomTexture)
        bottomSurfaceTexture.setDefaultBufferSize(width, height)
        bottomSurface = Surface(bottomSurfaceTexture)
        bottomSurfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                bottomSurfaceTexture.updateTexImage()
                bottomTs = System.currentTimeMillis()
                if (++diagBottom % 150L == 1L) Log.i("ar-ui", "rear bottom frame #$diagBottom")
            }
        }
        topTexture = createOesTexture()
        topSurfaceTexture = SurfaceTexture(topTexture)
        topSurfaceTexture.setDefaultBufferSize(1, 1)
        topSurface = Surface(topSurfaceTexture)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    // ── RearComposer ───────────────────────────────────────────

    override fun getBottomSurface(): Surface = bottomSurface
    override fun getTopSurface(): Surface = topSurface   // 占位，见 createStreams

    /** GPU 烘焙输入面：GL 线程建/重建烘焙核心（会话级，reset() 重建）。
     *  同步返回解码输入面；CPU 烘焙回退路径（overlayImageCallback）不走此接口。
     *  sr=true：核心内部先过 GL 原生 ESPCN ×2 超分（失败自动退回非 SR）。 */
    override fun openOverlayStream(streamW: Int, streamH: Int, sr: Boolean, srWeights: ByteArray?): Surface {
        val latch = java.util.concurrent.CountDownLatch(1)
        var surface: Surface? = null
        handler.post {
            if (released) {
                latch.countDown()
                return@post
            }
            try {
                closeBakerLocked()
                val b = OverlayBakeCore()
                surface = b.open(streamW, streamH, handler, sr, srWeights)
                bakeStreamSrWeights = srWeights
                baker = b
                bakeStreamW = streamW
                bakeStreamH = streamH
                bakeStreamSr = sr
                ArNative.nativeSetOverlaySize(b.contentW(), b.contentH())
                Log.i("ar-rear-gl", "overlay gl-bake open ${streamW}x${streamH} content=${b.contentW()}x${b.contentH()}")
            } catch (t: Throwable) {
                Log.w("ar-rear-gl", "overlay gl-bake open fail", t)
                surface = null
            }
            latch.countDown()
        }
        try { latch.await(2, java.util.concurrent.TimeUnit.SECONDS) } catch (_: InterruptedException) {}
        return surface ?: throw IllegalStateException("overlay gl-bake open timeout")
    }

    private fun closeBakerLocked() {
        val b = baker ?: return
        baker = null
        try { b.release() } catch (t: Throwable) { Log.w("ar-rear-gl", "baker release fail", t) }
    }

    override fun setBottomAspectRatio(ratio: Float) { bottomAspect = ratio }
    override fun setBottomRotation(rotation: Float, mirror: Boolean) {
        bottomRotation = rotation; bottomMirror = mirror
    }
    override fun setTopAspectRatio(ratio: Float) {
        // 烘焙位图自带真实宽高比（上传时覆盖本值），此回调仅接口兼容
    }
    override fun setTopRotation(rotationDeg: Int, mirror: Boolean) {
        topRotationDeg = rotationDeg; topMirror = mirror
    }
    override fun setOverlayScale(scale: Float) { overlayScale = scale.coerceIn(0.1f, 8f) }
    override fun setBaseBrightness(value: Float) { baseBrightness = value.coerceIn(0f, 4f) }
    override fun setOverlayParams(
        alpha: Float, brightness: Float, saturation: Float, dim: Float,
        keyLow: Float, keyHigh: Float, featherPower: Float, featherRadiusPx: Int,
    ) {
        // 键控/饱和度已在 CPU 烘焙（convertOverlayFrame）完成；alpha/brightness/
        // sat/dim 为普通模式 SurfaceMixer 的参数，前摄同款渲染不消费。
    }

    /** 断开后重建输入面，清掉上一会话残留帧。 */
    override fun reset() {
        Log.i("ar-ui", "rear reset")
        diagBottom = 0
        handler.post {
            if (released) return@post
            GLES20.glDeleteTextures(2, intArrayOf(bottomTexture, topTexture), 0)
            bottomSurfaceTexture.release(); bottomSurface.release()
            topSurfaceTexture.release(); topSurface.release()
            // 烘焙核心随会话重建（清掉旧会话帧；流尺寸沿用上次建流参数）
            hasContent = false
            val b = baker
            baker = null
            if (b != null && bakeStreamW > 0) {
                runCatching { b.release() }
                try {
                    val nb = OverlayBakeCore()
                    nb.open(bakeStreamW, bakeStreamH, handler, bakeStreamSr, bakeStreamSrWeights)
                    baker = nb
                    ArNative.nativeSetOverlaySize(nb.contentW(), nb.contentH())
                } catch (t: Throwable) {
                    Log.w("ar-rear-gl", "baker reopen fail", t)
                }
            }
            createStreams()
        }
    }

    // ── 输出面（同 ArFrontGl：多输出 + 同步摘除）────────────────

    fun attachOutputSurface(surface: Surface) {
        handler.post {
            if (released || !glOk) return@post
            synchronized(renderSurfaces) { renderSurfaces.add(RenderSurface(surface)) }
        }
    }

    fun detachOutputSurface(surface: Surface, releaseSurface: Boolean = true) {
        val latch = java.util.concurrent.CountDownLatch(1)
        handler.post {
            synchronized(renderSurfaces) {
                val it = renderSurfaces.iterator()
                while (it.hasNext()) {
                    val rs = it.next()
                    if (rs.surface === surface) { it.remove(); rs.release() }
                }
            }
            if (releaseSurface) surface.release()
            latch.countDown()
        }
        try { latch.await(1, java.util.concurrent.TimeUnit.SECONDS) } catch (_: InterruptedException) {}
    }

    /** 每拍渲染完成后回调（GL 线程；驱动录像编码线程 drain）。 */
    var frameCallback: (() -> Unit)? = null

    /** GPU 烘焙产物转储（真机 A/B 取证；GL 线程执行）。 */
    fun dumpOverlay(pathPrefix: String) {
        handler.post { baker?.dump(pathPrefix) }
    }

    fun release() {
        released = true
        handler.post {
            synchronized(renderSurfaces) {
                for (rs in renderSurfaces) rs.release()
                renderSurfaces.clear()
            }
            closeBakerLocked()
            if (glOk) {
                GLES20.glDeleteTextures(2, intArrayOf(bottomTexture, topTexture), 0)
                GLES20.glDeleteTextures(3, intArrayOf(contentTex, glowTex, bloomTex), 0)
            }
            EGL14.eglMakeCurrent(eglDisplay,
                EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)
            handlerThread.quitSafely()
        }
    }

    // ── 渲染 ───────────────────────────────────────────────────

    private fun renderFrame() {
        if (!released) {
            try { tick() } catch (t: Throwable) { Log.w("ar-rear-gl", "tick fail", t) }
            frameCallback?.invoke()
            handler.postDelayed({ renderFrame() }, 33)
        }
    }

    private fun pollProps() {
        if (++propTick % 30 != 1) return
        try {
            val sp = Class.forName("android.os.SystemProperties")
            val get = sp.getMethod("get", String::class.java, String::class.java)
            val m = (get.invoke(null, "debug.gscp.ovmirror", "") as String).trim()
            if (m.isNotEmpty()) ovMirror = m != "0" && !m.equals("false", true)
        } catch (_: Throwable) {
        }
    }

    private fun tick() {
        if (!glOk) return
        pollProps()
        val alive = streamAliveCheck?.invoke() ?: true
        if (alive != streamAlive) {
            streamAlive = alive
            Log.i("ar-rear-gl", "overlay stream " + if (alive) "resumed" else "paused >3s -> hide overlay")
        }
        val b = baker
        if (b != null) {
            // GPU 烘焙：消费解码新帧（产物纹理原地更新）。overlay 流按需发帧，
            // UI 静默期保留最后内容。
            try { b.bakeIfPending() } catch (t: Throwable) { Log.w("ar-rear-gl", "ov bake fail", t) }
            if (!hasContent && b.hasBaked()) {
                hasContent = true
                topAspect = if (b.contentH() > 0) b.contentW().toFloat() / b.contentH() else 1f
            }
        } else {
            // 前摄同款：overlay 烘焙快照 → 纹理（版本号变化才重新上传；位图在
            // provider 持锁期间有效）
            overlays.snapshot { content, glow, bloom, ver ->
                if (ver != overlayVersionSeen) {
                    overlayVersionSeen = ver
                    contentTex = uploadBitmap(contentTex, content)
                    glowTex = uploadBitmap(glowTex, glow)
                    bloomTex = uploadBitmap(bloomTex, bloom)
                    hasContent = content != null && !content.isRecycled
                    content?.let { topAspect = it.width.toFloat() / it.height.toFloat() }
                }
            }
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                EGL14.eglMakeCurrent(eglDisplay, rs.eglSurface, rs.eglSurface, eglContext)
                GLES20.glViewport(0, 0, width, height)
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                drawScene()
                EGL14.eglSwapBuffers(eglDisplay, rs.eglSurface)
            }
        }
    }

    private fun drawScene() {
        // ── 底层：眼镜视频 contain letterbox（前摄相机层同款数学，完整不裁切）──
        val cw = width.toFloat(); val ch = height.toFloat()
        val rot90 = (bottomRotation.toInt() / 90) % 2 == 1
        // 显示宽高比（w/h）：rot90 时显示宽 = 纹理高 → D = H/W = bottomAspect
        val dispW = if (rot90) bottomAspect else 1f / bottomAspect
        var dw: Float; var dh: Float
        if (dispW >= cw / ch) { dw = cw; dh = cw / dispW } else { dh = ch; dw = ch * dispW }
        val bx0 = (cw - dw) / 2f; val by0 = (ch - dh) / 2f
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(pBottom)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, bottomTexture)
        GLES20.glUniform1i(uBTex, 0)
        GLES20.glUniform1i(uBMirror, if (bottomMirror) 1 else 0)
        GLES20.glUniform1f(uBBright, baseBrightness)
        // 纹理采样按 bottomRotation 旋转（与 mix.frag rotQuad 同向；镜像在 shader 内）
        putQuad(bx0, by0, bx0 + dw, by0 + dh, 0f, 0f, 1f, 1f,
            rotQ = (bottomRotation.toInt() / 90) % 4)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)

        // ── 顶层：前摄同款发光（远 ADD α90 → 中 ADD α120 → 内容）──
        // overlay 流暂停 >3s：隐藏顶层（底层眼镜视频照常）
        if (!streamAlive) return
        if (!hasContent) return
        // 纹理源：GPU 烘焙产物优先，回退 CPU 烘焙位图上传
        val bk = baker
        val cTex: Int; val gTex: Int; val fTex: Int
        if (bk != null && bk.hasBaked()) {
            cTex = bk.contentTex(); gTex = bk.glowTex(); fTex = bk.bloomTex()
        } else {
            cTex = contentTex; gTex = glowTex; fTex = bloomTex
        }
        drawTopLayer(fTex, additive = true, alpha = BLOOM_FAR_ALPHA, mul = MUL_BLOOM, add = 0f)
        drawTopLayer(gTex, additive = true, alpha = BLOOM_MID_ALPHA, mul = MUL_BLOOM, add = 0f)
        drawTopLayer(cTex, additive = false, alpha = 1f, mul = MUL_CONTENT, add = 12f / 255f)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /** 烘焙纹理单层绘制（前摄 drawPackedMesh 的 quad 版）：加色/普通合成由
     *  blend 状态区分；uFlip 抵消（或保留）烘焙镜像。 */
    private fun drawTopLayer(tex: Int, additive: Boolean, alpha: Float, mul: FloatArray, add: Float) {
        if (tex == 0) return
        GLES20.glEnable(GLES20.GL_BLEND)
        if (additive) {
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE)
        } else {
            GLES20.glBlendFuncSeparate(
                GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA,
                GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        }
        GLES20.glUseProgram(pMesh)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glUniform1i(uMeshTex, 0)
        GLES20.glUniform1f(uMeshAlpha, alpha)
        GLES20.glUniform3f(uMeshMul, mul[0], mul[1], mul[2])
        GLES20.glUniform1f(uMeshAdd, add)
        // uFlip=1 抵消烘焙镜像（后摄=眼镜自身画面直出，同普通模式不镜像）；
        // debug.gscp.ovmirror=1 保留烘焙镜像，与 setTopRotation(mirror) 取异或
        val flip = (if (ovMirror) 0 else 1) xor (if (topMirror) 1 else 0)
        GLES20.glUniform1f(uMeshFlip, flip.toFloat())
        drawTopRectQuad()
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
    }

    /** 顶层 overlay 矩形（烘焙位图真实宽高比，居中 contain ×scale）。 */
    private fun drawTopRectQuad() {
        val cw = width.toFloat(); val ch = height.toFloat()
        val tRot90 = (topRotationDeg / 90) % 2 == 1
        val disp = if (tRot90) 1f / topAspect else topAspect
        val canvasAspect = cw / ch
        var rw: Float; var rh: Float
        if (disp >= canvasAspect) { rw = cw; rh = cw / disp } else { rh = ch; rw = ch * disp }
        rw *= overlayScale; rh *= overlayScale
        val tx0 = (cw - rw) / 2f; val ty0 = (ch - rh) / 2f
        if (++diagTick % 60 == 1) Log.i("ar-ui", "rear topRect topAspect=$topAspect rw=$rw rh=$rh scale=$overlayScale")
        val q = (topRotationDeg / 90) % 4
        putQuad(tx0, ty0, tx0 + rw, ty0 + rh, 0f, 0f, 1f, 1f, q)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
    }

    /** 矩形（px）→ 两三角形，带 uv 旋转象限（前摄同口径）。 */
    private fun putQuad(
        x0: Float, y0: Float, x1: Float, y1: Float,
        u0: Float, v0: Float, u1: Float, v1: Float,
        rotQ: Int = 0,
    ) {
        val a = quadArr
        var k = 0
        fun put(px: Float, py: Float, u: Float, v: Float) {
            a[k++] = px / width.toFloat() * 2f - 1f
            a[k++] = 1f - py / height.toFloat() * 2f
            var uu = u
            var vv = v
            if (rotQ == 1) { val t = uu; uu = 1f - vv; vv = t }
            else if (rotQ == 2) { uu = 1f - uu; vv = 1f - vv }
            else if (rotQ == 3) { val t = uu; uu = vv; vv = 1f - t }
            a[k++] = uu; a[k++] = vv
        }
        put(x0, y0, u0, v0); put(x1, y0, u1, v0); put(x0, y1, u0, v1)
        put(x1, y0, u1, v0); put(x1, y1, u1, v1); put(x0, y1, u0, v1)
        quadBuf.position(0)
        quadBuf.put(a)
        quadBuf.position(0)
    }

    private fun createOesTexture(): Int {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, t[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return t[0]
    }

    /** 位图 → GL_TEXTURE_2D（GLUtils：按 premultiplied 上传，位图首行 → v=0，
     *  与本文 uv 约定一致）；bm 为 null/已回收时清除旧纹理并返回 0。 */
    private fun uploadBitmap(old: Int, bm: Bitmap?): Int {
        if (old != 0) GLES20.glDeleteTextures(1, intArrayOf(old), 0)
        if (bm == null || bm.isRecycled) return 0
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bm, 0)
        return t[0]
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
            Log.e("ar-rear-gl", "link error: " + GLES20.glGetProgramInfoLog(program))
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
            Log.e("ar-rear-gl", "compile error: " + GLES20.glGetShaderInfoLog(shader))
            throw RuntimeException("Shader compile error")
        }
        return shader
    }

    companion object {
        private const val BLOOM_FAR_ALPHA = 90f / 255f    // 远距光晕层透明度(同前摄)
        private const val BLOOM_MID_ALPHA = 120f / 255f   // 中距光晕层透明度(同前摄)
        private val MUL_CONTENT = floatArrayOf(1.40f, 1.40f, 1.40f)   // 内容提亮(同前摄)
        private val MUL_BLOOM = floatArrayOf(1f, 1f, 1f)
        private const val VERT_SRC = """
            attribute vec4 aPosUv;
            varying vec2 vUv;
            void main() {
                vUv = aPosUv.zw;
                gl_Position = vec4(aPosUv.xy, 0.0, 1.0);
            }
        """

        /** 底层：OES 直采 + 镜像 + 亮度。 */
        private const val FRAG_BOTTOM = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vUv;
            uniform samplerExternalOES uTex;
            uniform int uMirror;
            uniform float uBright;
            void main() {
                vec2 uv = vUv;
                if (uMirror == 1) uv.x = 1.0 - uv.x;
                gl_FragColor = vec4(texture2D(uTex, uv).rgb * uBright, 1.0);
            }
        """

        /** 烘焙纹理贴图（前摄 FRAG_MESH 同款）：premultiplied 采样；uMul/uAdd 在
         *  非预乘域做内容提亮（1.4/+12），uAlpha 整层透明度；uFlip=1 时
         *  水平翻转（抵消烘焙镜像）；加色/普通合成由 blend 状态区分。 */
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
