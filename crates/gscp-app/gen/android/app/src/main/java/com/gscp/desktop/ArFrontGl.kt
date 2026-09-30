package com.gscp.desktop

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
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
 * 把旧 CPU Canvas 上屏路径（全屏 drawBitmap + drawBitmapMesh×3 + BlurMaskFilter 光晕）
 * 整体搬到 GPU，上屏与录像共用同一份合成结果（编码器面直挂输出，与后摄 SurfaceMixer 一致）：
 *  - 相机层：检测线程产出的直立 RGBA（Camera2 → nativeFaceDetect 直写，所见即所测）
 *    glTexSubImage2D 上传，前置镜像 + letterbox 居中，RGB×0.70 压暗（同旧 camDimPaint）；
 *  - 背板层：overlay quad 上 SDF 圆角矩形（0x08 绿衬底）+ 外缘高斯柔光（同旧 glowPaint）；
 *  - overlay 层：Heckbert square→quad 单应 5×5 网格（同旧 drawBitmapMesh 顶点算法），
 *    远距光晕（ADD α150）→ 中距光晕（ADD α200）→ 内容（SRC_OVER，RGB×1.4+12 提亮）；
 *  - 未有内容时在 quad 形心处绘制"连接中..."位图（同旧 connPaint）。
 * 33ms 定时推帧到全部挂载输出；active=false（后摄模式）只输出黑场。
 */
class ArFrontGl(
    private val density: Float,
    val width: Int,
    val height: Int,
    /** 每拍取 overlay 内容快照（GL 线程调用；实现方持锁回调，位图仅在回调内有效）。 */
    private val overlays: WithOverlays,
) {
    fun interface WithOverlays {
        fun snapshot(action: (content: Bitmap?, glow: Bitmap?, bloom: Bitmap?, version: Int) -> Unit)
    }

    /** 每拍渲染完成后回调（GL 线程；驱动录像编码线程 drain，≈GL 帧率）。 */
    var frameCallback: (() -> Unit)? = null

    @Volatile private var active = true
    @Volatile private var released = false
    // face0 17 值（overlay 四角 8 + 背板 8 + 距离），直立图像坐标；camW/H = 直立尺寸
    @Volatile private var face: FloatArray? = null
    @Volatile private var camW = 0
    @Volatile private var camH = 0

    private val renderSurfaces = mutableListOf<RenderSurface>()

    // EGL
    private val eglDisplay: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private lateinit var eglContext: EGLContext
    private lateinit var eglConfig: EGLConfig

    private val handlerThread: HandlerThread = HandlerThread("ar-front-gl")
    private val handler: Handler

    private inner class RenderSurface(val surface: Surface) {
        val eglSurface: EGLSurface

        init {
            val surfaceAttributes = intArrayOf(EGL14.EGL_NONE)
            eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, surface, surfaceAttributes, 0)
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
    private var uMeshTex = 0
    private var uMeshAlpha = 0
    private var uMeshMul = 0
    private var uMeshAdd = 0

    private var camTex = 0
    private var camTexW = 0
    private var camTexH = 0
    private var contentTex = 0
    private var glowTex = 0
    private var bloomTex = 0
    private var overlayVersionSeen = Int.MIN_VALUE
    private var hasContent = false
    private var hintTex = 0
    private var hintW = 0
    private var hintH = 0

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
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttributes, 0)

            val pbSurface = EGL14.eglCreatePbufferSurface(
                eglDisplay, eglConfig,
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(eglDisplay, pbSurface, pbSurface, eglContext)
            initGl()

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
        uMeshTex = GLES20.glGetUniformLocation(pMesh, "uTex")
        uMeshAlpha = GLES20.glGetUniformLocation(pMesh, "uAlpha")
        uMeshMul = GLES20.glGetUniformLocation(pMesh, "uMul")
        uMeshAdd = GLES20.glGetUniformLocation(pMesh, "uAdd")

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
        camW = w
        camH = h
    }

    fun attachOutputSurface(surface: Surface) {
        handler.post {
            if (released) return@post
            synchronized(renderSurfaces) {
                renderSurfaces.add(RenderSurface(surface))
            }
        }
    }

    /** 摘除一个输出面。同步等待 GL 线程完成（防后续仍向已失效面 swap）；
     *  releaseSurface=false 用于编码器输入 Surface（生命周期归编码器管）。 */
    fun detachOutputSurface(surface: Surface, releaseSurface: Boolean = true) {
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
            val texs = intArrayOf(camTex, contentTex, glowTex, bloomTex, hintTex)
            GLES20.glDeleteTextures(texs.size, texs, 0)
            camTex = 0; contentTex = 0; glowTex = 0; bloomTex = 0; hintTex = 0
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
        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                EGL14.eglMakeCurrent(eglDisplay, rs.eglSurface, rs.eglSurface, eglContext)
                GLES20.glViewport(0, 0, width, height)
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                if (active) drawScene()
                EGL14.eglSwapBuffers(eglDisplay, rs.eglSurface)
            }
        }
    }

    private fun drawScene() {
        val w = camW
        val h = camH
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

        val ov = face ?: return
        if (w <= 0 || h <= 0) return
        // overlay 内容四角（canvas px；与旧 mapX/mapY 一致的前置镜像）
        val sc = minOf(width.toFloat() / w, height.toFloat() / h)
        val x0 = (width - w * sc) / 2f
        val y0 = (height - h * sc) / 2f
        val tlX = (w - ov[6]) * sc + x0; val tlY = ov[7] * sc + y0
        val trX = (w - ov[4]) * sc + x0; val trY = ov[5] * sc + y0
        val brX = (w - ov[2]) * sc + x0; val brY = ov[3] * sc + y0
        val blX = (w - ov[0]) * sc + x0; val blY = ov[1] * sc + y0

        // ── 背板：圆角矩形衬底 + 外缘柔光（SDF，同旧 glowPaint + dimPaint）──
        val ex = (dist(trX, trY, tlX, tlY) + dist(brX, brY, blX, blY)) / 2f
        val ey = (dist(blX, blY, tlX, tlY) + dist(brX, brY, trX, trY)) / 2f
        GLES20.glUseProgram(pPlate)
        blendPremult()
        GLES20.glUniform2f(uPlateSize, ex, ey)
        GLES20.glUniform1f(uPlateRadius, PLATE_CORNER * minOf(ex, ey))
        GLES20.glUniform1f(uPlateGlowSigma, GLOW_SIGMA_DP * density)
        putQuad4(tlX, tlY, trX, trY, brX, brY, blX, blY)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)

        // ── overlay + bloom：单应网格投影（金字塔 ADD 光晕 → 内容 SRC_OVER）──
        computeHomography(tlX, tlY, trX, trY, brX, brY, blX, blY)
        if (hasContent) {
            packMesh()
            drawPackedMesh(bloomTex, additive = true, alpha = 150f / 255f, mul = MUL_UNIT, add = 0f)
            drawPackedMesh(glowTex, additive = true, alpha = 200f / 255f, mul = MUL_UNIT, add = 0f)
            drawPackedMesh(contentTex, additive = false, alpha = 1f, mul = MUL_CONTENT, add = 12f / 255f)
        } else {
            drawHint((tlX + trX + brX + blX) / 4f, (tlY + trY + brY + blY) / 4f)
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

    /** 形心处"连接中..."（位图一次烘焙，同旧 connPaint 文案与样式）。 */
    private fun drawHint(cx: Float, cy: Float) {
        if (hintTex == 0 && !ensureHintTex()) return
        blendPremult()
        GLES20.glUseProgram(pMesh)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, hintTex)
        GLES20.glUniform1i(uMeshTex, 0)
        GLES20.glUniform1f(uMeshAlpha, 1f)
        GLES20.glUniform3f(uMeshMul, 1f, 1f, 1f)
        GLES20.glUniform1f(uMeshAdd, 0f)
        putQuad(cx - hintW / 2f, cy - hintH / 2f, cx + hintW / 2f, cy + hintH / 2f, 0f, 0f, 1f, 1f)
        GLES20.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, quadBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
    }

    private fun ensureHintTex(): Boolean {
        if (hintTex != 0) return true
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF90EE90.toInt()
            typeface = Typeface.MONOSPACE
            textSize = 16 * density
        }
        val t = "连接中..."
        val pad = 2f * density
        val bw = (p.measureText(t) + pad * 2).toInt().coerceAtLeast(1)
        val bh = (p.textSize + pad * 2).toInt().coerceAtLeast(1)
        val bm = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        Canvas(bm).drawText(t, pad, pad - p.ascent(), p)
        hintTex = uploadBitmap(0, bm)
        if (hintTex != 0) {
            hintW = bw
            hintH = bh
            return true
        }
        return false
    }

    // ── 纹理与 GL 状态 ─────────────────────────────────────────

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

    private fun dist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = (x2 - x1).toDouble()
        val dy = (y2 - y1).toDouble()
        return kotlin.math.sqrt(dx * dx + dy * dy).toFloat()
    }

    companion object {
        private const val TAG = "ar-front-gl"
        private const val MESH_N = 4              // 内容透视网格密度（与旧 drawBitmapMesh 一致）
        private const val CAM_DIM = 0.70f         // 相机压暗（同旧 camDimPaint）
        private const val PLATE_CORNER = 0.10f    // 背板圆角（边长比例，同旧 10% 切角）
        private const val GLOW_SIGMA_DP = 5f      // 背板柔光 σ（同旧 BlurMaskFilter 5dp）
        private val MUL_UNIT = floatArrayOf(1f, 1f, 1f)
        private val MUL_CONTENT = floatArrayOf(1.40f, 1.40f, 1.40f)   // 内容提亮（同旧 meshPaint）

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

        /** 背板：局部 px 空间圆角矩形 SDF；内部 0x08 绿衬底 + 外缘 0x0A 高斯柔光。 */
        private const val FRAG_PLATE = """
            precision mediump float;
            varying vec2 vUv;
            uniform vec2 uSize;
            uniform float uRadius;
            uniform float uGlowSigma;
            void main() {
                vec2 p = (vUv - 0.5) * uSize;
                vec2 q = abs(p) - (uSize * 0.5 - uRadius);
                float d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - uRadius;
                float fill = clamp(0.5 - d, 0.0, 1.0) * (8.0 / 255.0);
                float glow = (10.0 / 255.0) * exp(-pow(max(d, 0.0) / uGlowSigma, 2.0));
                float a = clamp(fill + glow, 0.0, 1.0);
                gl_FragColor = vec4(0.0, a, 0.0, a);
            }
        """

        /** 网格贴图：premultiplied 采样；uMul/uAdd 在非预乘域做内容提亮（1.4/+12），
         *  uAlpha 整层透明度（光晕 150/255、200/255）；加色/普通合成由 blend 状态区分。 */
        private const val FRAG_MESH = """
            precision mediump float;
            varying vec2 vUv;
            uniform sampler2D uTex;
            uniform float uAlpha;
            uniform vec3 uMul;
            uniform float uAdd;
            void main() {
                vec4 s = texture2D(uTex, vUv);
                vec3 rgb = vec3(0.0);
                if (s.a > 0.004) {
                    rgb = clamp((s.rgb / s.a) * uMul + uAdd, 0.0, 1.0) * s.a;
                }
                gl_FragColor = vec4(rgb * uAlpha, s.a * uAlpha);
            }
        """
    }
}
