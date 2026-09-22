package com.gscp.desktop

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.tan

/**
 * AR 试验渲染器（单 GLSurfaceView）：
 * - 背景 = 相机 Preview 流（SurfaceTexture，GPU 零拷贝）。顶点 uv 先经
 *   「顺时针 90° 旋转的逆映射」到传感器坐标，着色器内再应用 SurfaceTexture
 *   变换矩阵——与分析帧喂给 YuNet 前的顺时针 90° 旋转严格互逆，
 *   检测/锚定与画面像素级对齐；
 * - 前景 = 眼镜 overlay 虚拟平面（MediaCodec → EXTERNAL_OES），锚定在
 *   第一张稳定追踪人脸的正前方：位于「相机→人脸」方向、深度 = 距离滑条
 *   （厘米）处，平面法线与人脸法线重合，固定物理宽度基准 14cm；
 * - 检测结果可视化：5 特征点（品红）+ 人脸框（绿）；
 * - 无眼镜时可切半透明测试图层。
 */
class ArOverlayRenderer : GLSurfaceView.Renderer {
    /** 最新人脸变换矩阵（列主序 4×4，正立显示空间），null = 未锁定。 */
    @Volatile
    var faceMatrix: FloatArray? = null

    /** 平面目标深度（厘米），距离滑条。 */
    @Volatile
    var planeDistance = 40f

    /** 平面大小系数（1.0 = 基准 14cm 宽）。 */
    @Volatile
    var planeScale = 1.0f

    /** overlay 画面宽高比（宽/高），由流 meta 更新。 */
    @Volatile
    var overlayAspect = 4f / 3f

    /** 无眼镜测试模式：绘制内置半透明测试图层。 */
    @Volatile
    var useTestPattern = false

    /** 法线翻转（绕 Y 轴 180°，平面背面朝向相机）。 */
    @Volatile
    var flipNormal = false

    /** 检测结果可视化数据（正立空间归一化）：[5 点 x,y | bbox x,y,w,h]，null = 无。 */
    @Volatile
    var detection: FloatArray? = null

    /** 相机就绪回调（GL 线程初始化完成后触发）。 */
    var onCameraSurfaceReady: (Surface) -> Unit = {}

    private var overlaySurfaceTexture: SurfaceTexture? = null
    private var overlayTextureId = 0
    private var cameraSurfaceTexture: SurfaceTexture? = null
    private var cameraTextureId = 0

    private var oesProgram = 0
    private var oesAPosition = 0
    private var oesATexCoord = 0
    private var oesUTexture = 0
    private var oesUMvp = 0
    private var oesUAlpha = 0
    private var bgProgram = 0
    private var bgAPosition = 0
    private var bgATexCoord = 0
    private var bgUTexture = 0
    private var bgUMvp = 0
    private var bgUSTMat = 0
    private var bgUAlpha = 0
    private var testProgram = 0
    private var testTextureId = 0
    private var testAPosition = 0
    private var testATexCoord = 0
    private var testUTexture = 0
    private var testUMvp = 0
    private var testUAlpha = 0
    private var markerProgram = 0
    private var markerAPosition = 0
    private var markerUColor = 0
    private val markerVertexBuffer: FloatBuffer = run {
        val b = ByteBuffer.allocateDirect(120 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        b.position(0)
        b
    }
    private var viewportW = 1
    private var viewportH = 1
    private var cameraBufferWidth = 1280f
    private var cameraBufferHeight = 720f

    @Volatile
    private var cameraSurface: Surface? = null

    private val quadVertexBuffer: FloatBuffer = floatBufferOf(
        // x, y, u, v —— 单位平面（中心原点）
        -0.5f, 0.5f, 0f, 0f,
        -0.5f, -0.5f, 0f, 1f,
        0.5f, -0.5f, 1f, 1f,
        -0.5f, 0.5f, 0f, 0f,
        0.5f, -0.5f, 1f, 1f,
        0.5f, 0.5f, 1f, 0f,
    )
    private val fullscreenVertexBuffer: FloatBuffer = run {
        val b = ByteBuffer.allocateDirect(24 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        b.position(0)
        b
    }
        private val identity = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )
    private val stMat = FloatArray(16)

    private val mvp = FloatArray(16)
    private val proj = FloatArray(16)
    private val model = FloatArray(16)

    /** overlay 解码目标 Surface（MediaCodec 直渲染到我们的纹理）。 */
    fun getOverlaySurface(): Surface {
        val st = overlaySurfaceTexture ?: throw IllegalStateException("GL 未初始化")
        return Surface(st)
    }

    fun cameraSurfaceOrNull(): Surface? = cameraSurface

    /** 预览缓冲尺寸跟随 CameraX 选择（letterbox 按此等比）。 */
    fun setCameraResolution(w: Int, h: Int) {
        try {
            cameraSurfaceTexture?.setDefaultBufferSize(w, h)
        } catch (_: Exception) {
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        overlayTextureId = createOesTexture()
        overlaySurfaceTexture = SurfaceTexture(overlayTextureId).apply {
            setDefaultBufferSize(1024, 768)
        }
        cameraTextureId = createOesTexture()
        cameraSurfaceTexture = SurfaceTexture(cameraTextureId).apply {
            setDefaultBufferSize(1280, 720)
        }
        cameraSurface = Surface(cameraSurfaceTexture)

        oesProgram = buildProgram(OES_FRAGMENT, DEFAULT_VERTEX)
        oesAPosition = GLES20.glGetAttribLocation(oesProgram, "aPosition")
        oesATexCoord = GLES20.glGetAttribLocation(oesProgram, "aTexCoord")
        oesUTexture = GLES20.glGetUniformLocation(oesProgram, "uTexture")
        oesUMvp = GLES20.glGetUniformLocation(oesProgram, "uMvp")
        oesUAlpha = GLES20.glGetUniformLocation(oesProgram, "uAlpha")

        bgProgram = buildProgram(BG_FRAGMENT, BG_VERTEX)
        bgAPosition = GLES20.glGetAttribLocation(bgProgram, "aPosition")
        bgATexCoord = GLES20.glGetAttribLocation(bgProgram, "aTexCoord")
        bgUTexture = GLES20.glGetUniformLocation(bgProgram, "uTexture")
        bgUMvp = GLES20.glGetUniformLocation(bgProgram, "uMvp")
        bgUSTMat = GLES20.glGetUniformLocation(bgProgram, "uSTMat")

        testProgram = buildProgram(TEST_FRAGMENT, DEFAULT_VERTEX)
        testAPosition = GLES20.glGetAttribLocation(testProgram, "aPosition")
        testATexCoord = GLES20.glGetAttribLocation(testProgram, "aTexCoord")
        testUTexture = GLES20.glGetUniformLocation(testProgram, "uTexture")
        testUMvp = GLES20.glGetUniformLocation(testProgram, "uMvp")
        testUAlpha = GLES20.glGetUniformLocation(testProgram, "uAlpha")

        markerProgram = buildProgram(SOLID_FRAGMENT, DEFAULT_VERTEX)
        markerAPosition = GLES20.glGetAttribLocation(markerProgram, "aPosition")
        markerUColor = GLES20.glGetUniformLocation(markerProgram, "uColor")
        GLES20.glUseProgram(markerProgram)
        GLES20.glUniformMatrix4fv(
            GLES20.glGetUniformLocation(markerProgram, "uMvp"), 1, false, identity, 0,
        )

        testTextureId = createTestTexture()
        android.util.Log.i("gscp-ar", "renderer ready build=20260922.7")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportW = width
        viewportH = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        // 背景：相机 Preview 流（零拷贝）。顶点 uv 先做「顺时针 90° 旋转的逆」
        // 映射到传感器坐标，着色器里再经 SurfaceTexture 变换矩阵——与分析帧
        // 喂给 YuNet 前的顺时针 90° 旋转严格互逆，两条流像素级对齐。
        try {
            cameraSurfaceTexture?.updateTexImage()
        } catch (_: Exception) {
        }
        drawFullscreen(cameraTextureId)

        // 前景：overlay 平面（未锁定人脸时不画）
        val face = faceMatrix ?: return
        try {
            overlaySurfaceTexture?.updateTexImage()
        } catch (_: Exception) {
        }

        // 投影：FOV 以正立画面（720×1280）垂直方向为基准，完整视野可见
        val imgW = 720f
        val imgH = 1280f
        val near = 5f
        val far = 1000f
        val fovY = Math.toRadians(50.0)
        val t = (tan(fovY / 2) * near).toFloat()
        val r = t * (imgW / imgH)
        run {
            var i = 0
            while (i < 16) mvp[i++] = 0f
        }
        proj[0] = near / r; proj[5] = near / t; proj[10] = -(far + near) / (far - near)
        proj[11] = -1f; proj[14] = -2f * far * near / (far - near)

        // 平面位于「相机→人脸」方向、深度 = planeDistance（厘米）处；
        // 旋转列取人脸朝向（法线重合）；固定物理宽度基准 14cm。
        val faceDepth = kotlin.math.abs(face[14]).coerceAtLeast(1f)
        val k = planeDistance / faceDepth
        for (row in face.indices) {
            model[row] = face[row]
        }
        model[12] = face[12] * k
        model[13] = face[13] * k
        model[14] = face[14] * k
        val size = 14f * planeScale
        val ar = overlayAspect.coerceIn(0.5f, 3f)
        model[0] = face[0] * size
        model[1] = face[1] * size
        model[2] = face[2] * size
        model[4] = face[4] * size / ar
        model[5] = face[5] * size / ar
        model[6] = face[6] * size / ar
        model[3] = 0f; model[7] = 0f; model[15] = 1f
        if (flipNormal) {
            model[0] = -model[0]; model[1] = -model[1]; model[2] = -model[2]
            model[8] = -model[8]; model[9] = -model[9]; model[10] = -model[10]
        }

        multiply(proj, model, mvp)
        // letterbox 缩放：正立画面 NDC 映射到屏幕居中区域
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val ndcX = imgW * fit / viewportW
        val ndcY = imgH * fit / viewportH
        mvp[0] *= ndcX; mvp[4] *= ndcX; mvp[8] *= ndcX; mvp[12] *= ndcX
        mvp[1] *= ndcY; mvp[5] *= ndcY; mvp[9] *= ndcY; mvp[13] *= ndcY
        drawOverlayQuad()

        // 检测结果可视化：5 特征点（品红）+ 人脸框（绿）
        detection?.let { d ->
            if (d.size >= 14) drawMarkers(d, ndcX, ndcY)
        }
    }

    private fun drawOverlayQuad() {
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        if (useTestPattern) {
            GLES20.glUseProgram(testProgram)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, testTextureId)
            GLES20.glUniform1i(testUTexture, 0)
            GLES20.glUniformMatrix4fv(testUMvp, 1, false, mvp, 0)
            GLES20.glUniform1f(testUAlpha, 1f)
            quadVertexBuffer.position(0)
            GLES20.glEnableVertexAttribArray(testAPosition)
            GLES20.glVertexAttribPointer(testAPosition, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
            GLES20.glEnableVertexAttribArray(testATexCoord)
            quadVertexBuffer.position(2)
            GLES20.glVertexAttribPointer(testATexCoord, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
            GLES20.glDisableVertexAttribArray(testAPosition)
            GLES20.glDisableVertexAttribArray(testATexCoord)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        } else {
            GLES20.glUseProgram(oesProgram)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, overlayTextureId)
            GLES20.glUniform1i(oesUTexture, 0)
            GLES20.glUniformMatrix4fv(oesUMvp, 1, false, mvp, 0)
            GLES20.glUniform1f(oesUAlpha, 1f)
            quadVertexBuffer.position(0)
            GLES20.glEnableVertexAttribArray(oesAPosition)
            GLES20.glVertexAttribPointer(oesAPosition, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
            GLES20.glEnableVertexAttribArray(oesATexCoord)
            quadVertexBuffer.position(2)
            GLES20.glVertexAttribPointer(oesATexCoord, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
            GLES20.glDisableVertexAttribArray(oesAPosition)
            GLES20.glDisableVertexAttribArray(oesATexCoord)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
        }
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /**
     * 背景：等比缩放居中（letterbox）。顶点 uv = 屏幕坐标经「顺时针 90° 旋转」
     * 映射到传感器坐标（与模型输入旋转互逆），着色器内再经 SurfaceTexture
     * 变换矩阵采样——最终显示与分析帧方向严格一致。
     */
    private fun drawFullscreen(tex: Int) {
        val imgW = cameraBufferHeight
        val imgH = cameraBufferWidth
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val hw = imgW * fit / viewportW
        val hh = imgH * fit / viewportH
        // 每顶点：屏幕 uv（0..1 over letterbox rect）→ 传感器 uv = (1 - v, u)
        val scr = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 0f, 1f, 1f, 1f, 0f)
        val pos = floatArrayOf(-hw, hh, -hw, -hh, hw, -hh, -hw, hh, hw, -hh, hw, hh)
        val verts = FloatArray(24)
        for (i in 0 until 6) {
            verts[i * 4] = pos[i * 2]
            verts[i * 4 + 1] = pos[i * 2 + 1]
            verts[i * 4 + 2] = 1f - scr[i * 2 + 1]
            verts[i * 4 + 3] = scr[i * 2]
        }
        fullscreenVertexBuffer.clear()
        fullscreenVertexBuffer.put(verts).position(0)
        GLES20.glUseProgram(bgProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex)
        GLES20.glUniform1i(bgUTexture, 0)
        GLES20.glUniformMatrix4fv(bgUMvp, 1, false, identity, 0)
        GLES20.glUniformMatrix4fv(bgUSTMat, 1, false, stMat, 0)
        GLES20.glUniform1f(bgUAlpha, 1f)
        fullscreenVertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(bgAPosition)
        GLES20.glVertexAttribPointer(bgAPosition, 2, GLES20.GL_FLOAT, false, 16, fullscreenVertexBuffer)
        GLES20.glEnableVertexAttribArray(bgATexCoord)
        fullscreenVertexBuffer.position(2)
        GLES20.glVertexAttribPointer(bgATexCoord, 2, GLES20.GL_FLOAT, false, 16, fullscreenVertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisableVertexAttribArray(bgAPosition)
        GLES20.glDisableVertexAttribArray(bgATexCoord)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    private fun floatBufferOf(vararg values: Float): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(values)
            .apply { position(0) }

    private fun createOesTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return textures[0]
    }

    /** 无眼镜测试图层：半透明青绿底 + 边框 + TOP 标记 + 准星（确认朝向与锚定）。 */
    private fun createTestTexture(): Int {
        val s = 512
        val bmp = Bitmap.createBitmap(s, s, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG)
        c.drawColor(0xD930E0A0.toInt()) // 高不透明度青绿底（试验期便于观察）
        paint.color = 0xFFEFFF00.toInt()
        paint.style = android.graphics.Paint.Style.STROKE
        paint.strokeWidth = s * 0.02f
        c.drawRect(s * 0.02f, s * 0.02f, s * 0.98f, s * 0.98f, paint)
        paint.style = android.graphics.Paint.Style.FILL
        paint.color = 0xFFFFFFFF.toInt()
        paint.textSize = s * 0.12f
        paint.textAlign = android.graphics.Paint.Align.CENTER
        c.drawText("TOP", s * 0.5f, s * 0.16f, paint)
        c.drawLine(s * 0.5f, s * 0.30f, s * 0.5f, s * 0.70f, paint)
        c.drawLine(s * 0.30f, s * 0.5f, s * 0.70f, s * 0.5f, paint)
        c.drawCircle(s * 0.5f, s * 0.5f, s * 0.06f, paint)

        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        bmp.recycle()
        return textures[0]
    }

    private fun buildProgram(fragmentSrc: String, vertexSrc: String): Int {
        fun compile(type: Int, src: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, src)
            GLES20.glCompileShader(shader)
            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            if (status[0] != GLES20.GL_TRUE) {
                throw RuntimeException("AR shader 编译失败: " + GLES20.glGetShaderInfoLog(shader))
            }
            return shader
        }
        val vs = compile(GLES20.GL_VERTEX_SHADER, vertexSrc.trimIndent())
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSrc.trimIndent())
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glLinkProgram(program)
        return program
    }

    /** 在背景区域内绘制 5 特征点与人脸框（输入为正立空间归一化坐标）。 */
    private fun drawMarkers(d: FloatArray, ndcX: Float, ndcY: Float) {
        GLES20.glUseProgram(markerProgram)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        val vb = markerVertexBuffer
        vb.clear()
        fun quad(cx: Float, cy: Float, hw: Float, hh: Float) {
            val x0 = cx - hw; val x1 = cx + hw
            val y0 = cy - hh; val y1 = cy + hh
            vb.put(x0).put(y1); vb.put(x0).put(y0); vb.put(x1).put(y0)
            vb.put(x0).put(y1); vb.put(x1).put(y0); vb.put(x1).put(y1)
        }
        for (i in 0 until 5) {
            val nx = (d[i * 2] - 0.5f) * 2f * ndcX
            val ny = (0.5f - d[i * 2 + 1]) * 2f * ndcY
            quad(nx, ny, 0.012f, 0.012f)
        }
        vb.position(0)
        GLES20.glVertexAttribPointer(markerAPosition, 2, GLES20.GL_FLOAT, false, 0, vb)
        GLES20.glEnableVertexAttribArray(markerAPosition)
        GLES20.glUniform4f(markerUColor, 1f, 0.2f, 1f, 0.95f)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 30)
        GLES20.glDisableVertexAttribArray(markerAPosition)
        vb.clear()
        val bx0 = (d[10] - 0.5f) * 2f * ndcX
        val by1 = (0.5f - d[11]) * 2f * ndcY
        val bx1 = (d[10] + d[12] - 0.5f) * 2f * ndcX
        val by0 = (0.5f - d[11] - d[13]) * 2f * ndcY
        val th = 0.004f
        quad((bx0 + bx1) / 2, by1, (bx1 - bx0) / 2, th)
        quad((bx0 + bx1) / 2, by0, (bx1 - bx0) / 2, th)
        quad(bx0, (by0 + by1) / 2, th, (by1 - by0) / 2)
        quad(bx1, (by0 + by1) / 2, th, (by1 - by0) / 2)
        vb.position(0)
        GLES20.glVertexAttribPointer(markerAPosition, 2, GLES20.GL_FLOAT, false, 0, vb)
        GLES20.glEnableVertexAttribArray(markerAPosition)
        GLES20.glUniform4f(markerUColor, 0.2f, 1f, 0.3f, 0.9f)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 24)
        GLES20.glDisableVertexAttribArray(markerAPosition)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun multiply(a: FloatArray, b: FloatArray, out: FloatArray) {
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                out[col * 4 + row] = sum
            }
        }
    }
}

private val DEFAULT_VERTEX = """
    precision mediump float;
    attribute vec4 aPosition;
    attribute vec2 aTexCoord;
    uniform mat4 uMvp;
    varying vec2 vTexCoord;
    void main() {
        gl_Position = uMvp * vec4(aPosition.xy, 0.0, 1.0);
        vTexCoord = aTexCoord;
    }
"""

private val BG_VERTEX = """
    precision mediump float;
    attribute vec4 aPosition;
    attribute vec2 aTexCoord;
    uniform mat4 uMvp;
    uniform mat4 uSTMat;
    varying vec2 vTexCoord;
    void main() {
        gl_Position = uMvp * vec4(aPosition.xy, 0.0, 1.0);
        vTexCoord = (uSTMat * vec4(aTexCoord, 0.0, 1.0)).xy;
    }
"""

private val BG_FRAGMENT = """
    #extension GL_OES_EGL_image_external : require
    precision mediump float;
    uniform samplerExternalOES uTexture;
    uniform float uAlpha;
    varying vec2 vTexCoord;
    void main() {
        vec4 c = texture2D(uTexture, vTexCoord);
        gl_FragColor = vec4(c.rgb, c.a * uAlpha);
    }
"""

private val SOLID_FRAGMENT = """
    precision mediump float;
    uniform vec4 uColor;
    void main() {
        gl_FragColor = uColor;
    }
"""

private val OES_FRAGMENT = """
    #extension GL_OES_EGL_image_external : require
    precision mediump float;
    uniform samplerExternalOES uTexture;
    uniform float uAlpha;
    varying vec2 vTexCoord;
    void main() {
        vec4 c = texture2D(uTexture, vTexCoord);
        gl_FragColor = vec4(c.rgb, c.a * uAlpha);
    }
"""

private val TEST_FRAGMENT = """
    precision mediump float;
    uniform sampler2D uTexture;
    uniform float uAlpha;
    varying vec2 vTexCoord;
    void main() {
        vec4 c = texture2D(uTexture, vTexCoord);
        gl_FragColor = vec4(c.rgb, c.a * uAlpha);
    }
"""
