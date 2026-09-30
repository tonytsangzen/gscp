package com.gscp.desktop

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * AR 后摄 GL 合成器：完整移植前摄 [ArFrontGl] 的处理架构 —— 同一套 EGL 脚手架
 * （独立 GL 线程、RECORDABLE、多输出面 + 同步摘除、33ms 推帧、录像面裁剪挂载），
 * 几何全部用"显示矩形 + 纹理 UV"直算，不走矩阵缩放：
 *  - 底层 = 眼镜相机码流（decoder → OES SurfaceTexture），contain letterbox
 *    完整显示（前摄相机层同款 min(vw/tw, vh/th) 数学，结构上不可能裁切），
 *    镜像/亮度可配；
 *  - 顶层 = 眼镜 overlay 码流，居中 contain 矩形 ×overlay_scale，UV 裁掉上下
 *    80px 空带（= 前摄 CPU 烘焙的同款裁带口径），luma 黑键抠像 + 饱和度 ×1.12
 *    + 亮度增益（mix.frag 效果链的忠实子集）；2 秒无帧自动隐匿。
 * 前摄模式的无内容语义在此不适用：后摄没有自有 UI，只画两路码流。
 */
class ArRearGl(val width: Int, val height: Int) : RearComposer {

    private var bottomRotation = 90f
    private var bottomMirror = false
    private var bottomAspect = 4f / 3f          // 纹理 H/W（GlassesPlayer 口径）
    private var baseBrightness = 1.0f
    private var topRotationDeg = 0
    private var topMirror = false
    private var topAspect = 3f / 4f             // 纹理 W/H
    private var overlayScale = 1.0f
    private var overlayAlpha = 1.0f
    private var overlayBrightness = 1.0f
    private var overlaySaturation = 1.0f
    private var keyLow = 0.08f
    private var keyHigh = 0.28f
    private var featherPower = 1.2f
    private var featherRadius = 15f / 255f

    private val lock = object {}
    private val renderSurfaces = mutableListOf<RenderSurface>()
    @Volatile private var released = false

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
    private var pBlitOes = 0
    private var pBloom = 0
    private var pBlit = 0
    private var pTopDirect = 0
    private var uBTex = 0
    private var uBMirror = 0
    private var uBBright = 0
    private var uBlitOesTex = 0
    private var uKeyLow = 0
    private var uKeyHigh = 0
    private var uKeyFeather = 0
    private var uTopDirectTex = 0
    private var uTopDirectKeyLow = 0
    private var uTopDirectKeyHigh = 0
    private var uTopDirectFeather = 0
    private var uBlitTex = 0
    private var uBloomTex = 0
    private var uBloomGain = 0
    private var uBloomKeyLow = 0
    private var uBloomKeyHigh = 0
    private var uBloomFeather = 0

    // 发光金字塔 FBO(前摄 bloom 的 GPU 等价):1/4(中距,金字塔低通)与 1/16(远距)
    private class Fbo(val w: Int, val h: Int) {
        val tex = IntArray(1); val fbo = IntArray(1)
        fun ensure() {
            if (fbo[0] != 0) return
            GLES20.glGenTextures(1, tex, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[0])
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
            GLES20.glGenFramebuffers(1, fbo, 0)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0])
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, tex[0], 0)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        }
    }
    private lateinit var fbQuarter: Fbo   // 1/4:键控结果 + 金字塔低通(中距光晕源)
    private lateinit var fbEighth: Fbo    // 1/8:低通中间层
    private lateinit var fbFar: Fbo       // 1/16:远距光晕
    private var lastKeyTs = 0L            // 顶层无新帧则跳过 FBO 重建
    private var diagTick = 0              // 诊断节流(每 ~1s 一条)

    private var bottomTexture = 0
    private var topTexture = 0
    private lateinit var bottomSurfaceTexture: SurfaceTexture
    private lateinit var topSurfaceTexture: SurfaceTexture
    private lateinit var bottomSurface: Surface
    private lateinit var topSurface: Surface
    private var bottomTs = 0L
    private var topTs = 0L
    private var diagBottom = 0L
    private var diagTop = 0L
    // debug.gscp.reardirect=1:顶层跳过 FBO/发光,OES 直采键控上屏(链路对比开关)
    @Volatile private var directTop = false
    private var propTick = 0

    private fun pollProps() {
        if (++propTick % 30 != 1) return
        try {
            val sp = Class.forName("android.os.SystemProperties")
            val get = sp.getMethod("get", String::class.java, String::class.java)
            val rd = (get.invoke(null, "debug.gscp.reardirect", "") as String).trim()
            if (rd.isNotEmpty()) directTop = rd == "1"
        } catch (_: Throwable) {
        }
    }

    private lateinit var quadBuf: FloatBuffer
    private val quadArr = FloatArray(6 * 4)

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
                android.opengl.EGLExt.EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(eglDisplay, configAttributes, 0, configs, 0, 1, numConfigs, 0)
            eglConfig = configs[0]!!
            eglContext = EGL14.eglCreateContext(
                eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
            val pb = EGL14.eglCreatePbufferSurface(
                eglDisplay, eglConfig,
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(eglDisplay, pb, pb, eglContext)
            initGl()
            renderFrame()
        }
    }

    private fun initGl() {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
        pBottom = buildProgram(FRAG_BOTTOM)
        pBlitOes = buildProgram(FRAG_BLIT_OES)
        pBloom = buildProgram(FRAG_BLOOM)
        pBlit = buildProgram(FRAG_BLIT)
        pTopDirect = buildProgram(FRAG_TOP_DIRECT)
        uBTex = GLES20.glGetUniformLocation(pBottom, "uTex")
        uBMirror = GLES20.glGetUniformLocation(pBottom, "uMirror")
        uBBright = GLES20.glGetUniformLocation(pBottom, "uBright")
        uBlitOesTex = GLES20.glGetUniformLocation(pBlitOes, "uTex")
        uKeyLow = GLES20.glGetUniformLocation(pTopDirect, "uKeyLow")
        uKeyHigh = GLES20.glGetUniformLocation(pTopDirect, "uKeyHigh")
        uKeyFeather = GLES20.glGetUniformLocation(pTopDirect, "uFeather")
        quadBuf = ByteBuffer.allocateDirect(quadArr.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        GLES20.glEnableVertexAttribArray(0)
        createStreams()
    }

    /** 两路 OES 输入面（decoder 直渲染），frame-available 更新时间戳。 */
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
        topSurfaceTexture.setDefaultBufferSize(width, height)
        topSurface = Surface(topSurfaceTexture)
        topSurfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                topSurfaceTexture.updateTexImage()
                topTs = System.currentTimeMillis()
                if (++diagTop % 150L == 1L) Log.i("ar-ui", "rear top frame #$diagTop")
            }
        }
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    // ── RearComposer ───────────────────────────────────────────

    override fun getBottomSurface(): Surface = bottomSurface
    override fun getTopSurface(): Surface = topSurface

    override fun setBottomAspectRatio(ratio: Float) { bottomAspect = ratio }
    override fun setBottomRotation(rotation: Float, mirror: Boolean) {
        bottomRotation = rotation; bottomMirror = mirror
    }
    override fun setTopAspectRatio(ratio: Float) { topAspect = ratio }
    override fun setTopRotation(rotationDeg: Int, mirror: Boolean) {
        topRotationDeg = rotationDeg; topMirror = mirror
    }
    override fun setOverlayScale(scale: Float) { overlayScale = scale.coerceIn(0.1f, 8f) }
    override fun setBaseBrightness(value: Float) { baseBrightness = value.coerceIn(0f, 4f) }
    override fun setOverlayParams(
        alpha: Float, brightness: Float, saturation: Float, dim: Float,
        keyLow: Float, keyHigh: Float, featherPower: Float, featherRadiusPx: Int,
    ) {
        overlayAlpha = alpha.coerceIn(0f, 1f)
        overlayBrightness = brightness.coerceAtLeast(0f)
        overlaySaturation = saturation.coerceIn(0f, 4f)
        this.keyLow = keyLow.coerceIn(0f, 1f)
        this.keyHigh = keyHigh.coerceIn(0f, 1f)
        this.featherPower = featherPower.coerceIn(0.1f, 8f)
        featherRadius = featherRadiusPx / 255f
    }

    /** 断开后重建两路输入面，清掉上一会话残留帧。 */
    override fun reset() {
        Log.i("ar-ui", "rear reset")
        diagBottom = 0; diagTop = 0
        handler.post {
            if (released) return@post
            GLES20.glDeleteTextures(2, intArrayOf(bottomTexture, topTexture), 0)
            bottomSurfaceTexture.release(); bottomSurface.release()
            topSurfaceTexture.release(); topSurface.release()
            createStreams()
        }
    }

    // ── 输出面（同 ArFrontGl：多输出 + 同步摘除）────────────────

    fun attachOutputSurface(surface: Surface) {
        handler.post {
            if (released) return@post
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

    fun release() {
        released = true
        handler.post {
            synchronized(renderSurfaces) {
                for (rs in renderSurfaces) rs.release()
                renderSurfaces.clear()
            }
            GLES20.glDeleteTextures(2, intArrayOf(bottomTexture, topTexture), 0)
            for (f in listOf(fbQuarter, fbEighth, fbFar)) {
                GLES20.glDeleteFramebuffers(1, f.fbo, 0)
                GLES20.glDeleteTextures(1, f.tex, 0)
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

    private fun tick() {
        pollProps()
        // 顶层不按 2s 无帧隐匿(前摄语义:保留最后一帧)——overlay 流按需发帧,
        // UI 无变化时静默,收到首帧后持续显示即可。首帧为空时等后续 UI 帧覆盖。
        if (!directTop && topTs > 0 && topTs != lastKeyTs) { renderTopFbos(); lastKeyTs = topTs }
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

    /** 键控 → 1/4 FBO → 金字塔低通(1/4→1/8→1/4 ×2,前摄同款口径)→ 远距 1/16。 */
    private fun renderTopFbos() {
        fbQuarter.ensure(); fbEighth.ensure(); fbFar.ensure()
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbQuarter.fbo[0])
        GLES20.glViewport(0, 0, fbQuarter.w, fbQuarter.h)
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        // 先整帧平拷进 FBO(OES→FBO 在本机 GPU 上不可用,这里只做 2D→2D;
        // 键控延后到内容/光晕采样阶段)
        GLES20.glUseProgram(pBlitOes)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, topTexture)
        GLES20.glUniform1i(uBlitOesTex, 0)
        putQuad(0f, 0f, width.toFloat(), height.toFloat(), 0f, 0f, 1f, 1f)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        for (round in 0 until 2) {
            blit(fbQuarter, fbEighth)
            blit(fbEighth, fbQuarter)
        }
        blit(fbQuarter, fbFar)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        // 诊断:回读 1/4 FBO 的 alpha 统计(键控结果非空 → 内容/bloom 才有得画)
        if (++diagTick % 30 == 1) {
            val w = fbQuarter.w; val h = fbQuarter.h
            val buf = ByteBuffer.allocateDirect(w * h * 4).order(ByteOrder.nativeOrder())
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbQuarter.fbo[0])
            GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            buf.rewind()
            var maxA = 0; var nonZero = 0
            while (buf.hasRemaining()) {
                val a = buf.getInt() shr 24 and 0xFF
                if (a > maxA) maxA = a
                if (a > 16) nonZero++
            }
            Log.i("ar-ui", "rear fboA maxA=$maxA nonZero=$nonZero/${w * h}")
        }
    }

    private fun blit(src: Fbo, dst: Fbo) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, dst.fbo[0])
        GLES20.glViewport(0, 0, dst.w, dst.h)
        GLES20.glUseProgram(pBlit)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, src.tex[0])
        GLES20.glUniform1i(uBlitTex, 0)
        putQuad(0f, 0f, width.toFloat(), height.toFloat(), 0f, 0f, 1f, 1f)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
    }

    /** 顶层 overlay 矩形(裁带后内容真实宽高比,居中 contain ×scale)。 */
    private fun drawTopRectQuad() {
        val cw = width.toFloat(); val ch = height.toFloat()
        val band = if (topAspect < 1f) (1f - topAspect) / 2f else 0f
        // 裁掉上下空带后的内容宽高比(竖向流恰为 1:1,同前摄)
        val contentAspect = topAspect / (1f - 2f * band)
        val tRot90 = (topRotationDeg / 90) % 2 == 1
        val disp = if (tRot90) 1f / contentAspect else contentAspect
        val canvasAspect = cw / ch
        var rw: Float; var rh: Float
        if (disp >= canvasAspect) { rw = cw; rh = cw / disp } else { rh = ch; rw = ch * disp }
        rw *= overlayScale; rh *= overlayScale
        val tx0 = (cw - rw) / 2f; val ty0 = (ch - rh) / 2f
        if (++diagTick % 60 == 1) Log.i("ar-ui", "rear topRect topAspect=$topAspect band=$band rw=$rw rh=$rh scale=$overlayScale key=$keyLow/$keyHigh/$featherPower")
        val q = (topRotationDeg / 90) % 4
        putQuad(tx0, ty0, tx0 + rw, ty0 + rh, 0f, 0f, 1f, 1f, q, topMirror, band)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
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

        // ── 顶层：眼镜 overlay,前摄同款发光(远 ADD α90 → 中 ADD α120 → 内容 ×1.4+12)──
        if (topTs <= 0L) return   // 尚未收到任何 overlay 帧
        if (directTop) {
            // 对比模式:OES 直采 + luma 键控,无 FBO/发光
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFuncSeparate(
                GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA,
                GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUseProgram(pTopDirect)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, topTexture)
            GLES20.glUniform1i(uTopDirectTex, 0)
            GLES20.glUniform1f(uTopDirectKeyLow, keyLow)
            GLES20.glUniform1f(uTopDirectKeyHigh, keyHigh)
            GLES20.glUniform1f(uTopDirectFeather, featherPower)
            drawTopRectQuad()
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
            GLES20.glDisable(GLES20.GL_BLEND)
            return
        }
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE)   // 加色:premult 光晕
        GLES20.glUseProgram(pBloom)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(uBloomTex, 0)
        GLES20.glUniform1f(uBloomGain, BLOOM_FAR_ALPHA)
        GLES20.glUniform1f(uBloomKeyLow, keyLow)
        GLES20.glUniform1f(uBloomKeyHigh, keyHigh)
        GLES20.glUniform1f(uBloomFeather, featherPower)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fbFar.tex[0])
        putQuad(0f, 0f, width.toFloat(), height.toFloat(), 0f, 0f, 1f, 1f)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glUniform1f(uBloomGain, BLOOM_MID_ALPHA)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fbQuarter.tex[0])
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)

        // 内容:SRC_OVER,OES 直采键控 + RGB ×1.4 + 12 提亮(前摄 MUL_CONTENT 口径)
        GLES20.glBlendFuncSeparate(
            GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA,
            GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(pTopDirect)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, topTexture)
        GLES20.glUniform1i(uTopDirectTex, 0)
        GLES20.glUniform1f(uTopDirectKeyLow, keyLow)
        GLES20.glUniform1f(uTopDirectKeyHigh, keyHigh)
        GLES20.glUniform1f(uTopDirectFeather, featherPower)
        drawTopRectQuad()
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /** 矩形（px）→ 两三角形，带 uv 旋转象限/镜像/竖向空带裁剪（前摄同口径）。 */
    private fun putQuad(
        x0: Float, y0: Float, x1: Float, y1: Float,
        u0: Float, v0: Float, u1: Float, v1: Float,
        rotQ: Int = 0, mirror: Boolean = false, vBand: Float = 0f,
    ) {
        val a = quadArr
        var k = 0
        fun put(px: Float, py: Float, u: Float, v: Float) {
            a[k++] = px / width.toFloat() * 2f - 1f
            a[k++] = 1f - py / height.toFloat() * 2f
            var uu = if (mirror) 1f - u else u
            var vv = v
            if (rotQ == 1) { val t = uu; uu = 1f - vv; vv = t }
            else if (rotQ == 2) { uu = 1f - uu; vv = 1f - vv }
            else if (rotQ == 3) { val t = uu; uu = vv; vv = 1f - t }
            vv = vBand + vv * (1f - 2f * vBand)   // 空带裁剪（纹理 v 空间）
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


        /** 直通顶层:OES 直采 + luma 键控(对比模式,无 FBO)。 */
        private const val FRAG_TOP_DIRECT = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vUv;
            uniform samplerExternalOES uTex;
            uniform float uKeyLow;
            uniform float uKeyHigh;
            uniform float uFeather;
            void main() {
                vec4 s = texture2D(uTex, vUv);
                float luma = dot(s.rgb, vec3(0.299, 0.587, 0.114));
                float lo = max(uKeyLow - uFeather, 0.0);
                float hi = min(uKeyHigh + uFeather, 1.0);
                float t = clamp((luma - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
                float a = pow(t, uFeather);
                vec3 straight = clamp(s.rgb * 1.4 + 12.0 / 255.0, 0.0, 1.0);
                gl_FragColor = vec4(straight * a, a);
            }
        """

        /** OES 平拷进 FBO(仅 2D→FBO;OES→FBO 在本机 GPU 上不可用)。 */
        private const val FRAG_BLIT_OES = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vUv;
            uniform samplerExternalOES uTex;
            void main() {
                gl_FragColor = texture2D(uTex, vUv);
            }
        """

        /** 金字塔低通 blit:纯双线性缩放(下/上采样即大核低通,前摄同款)。 */
        private const val FRAG_BLIT = """
            precision mediump float;
            varying vec2 vUv;
            uniform sampler2D uTex;
            void main() {
                gl_FragColor = texture2D(uTex, vUv);
            }
        """

        /** 光晕加色层:premult 采样 × 整层透明度(远 90/255、中 120/255),blend ONE/ONE。 */
        private const val FRAG_BLOOM = """
            precision mediump float;
            varying vec2 vUv;
            uniform sampler2D uTex;
            uniform float uGain;
            uniform float uKeyLow;
            uniform float uKeyHigh;
            uniform float uFeather;
            void main() {
                vec4 s = texture2D(uTex, vUv);
                float luma = dot(s.rgb, vec3(0.299, 0.587, 0.114));
                float lo = max(uKeyLow - uFeather, 0.0);
                float hi = min(uKeyHigh + uFeather, 1.0);
                float t = clamp((luma - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
                gl_FragColor = s * pow(t, uFeather) * uGain;
            }
        """

    }
}
