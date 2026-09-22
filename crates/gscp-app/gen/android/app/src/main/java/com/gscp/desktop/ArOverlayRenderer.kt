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
 * - 背景 = 手机相机实时画面，等比缩放居中显示（不裁剪，最大 FOV，黑边补底）；
 * - 前景 = 眼镜 overlay 虚拟平面（MediaCodec → EXTERNAL_OES），锚定在
 *   第一张稳定追踪人脸的正前方：位于「相机→人脸」方向、深度 = 距离滑条
 *   （厘米）处，平面法线与人脸法线重合，固定物理宽度基准 14cm；
 * - 无眼镜时可切半透明测试图层，验证追踪与锚定。
 *
 * MediaPipe Face Landmarker 的 4×4 变换矩阵为列主序仿射、单位厘米
 * （实测人脸 ~60cm 时 t.z ≈ -63）。
 */
class ArOverlayRenderer : GLSurfaceView.Renderer {
    /** 最新人脸变换矩阵（列主序 4×4），null = 未锁定。 */
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

    /** 相机就绪回调（GL 线程初始化完成后触发）。 */
    var onCameraSurfaceReady: (Surface) -> Unit = {}

    /** 相机实际缓冲尺寸（由 CameraX SurfaceRequest 提供，等比缩放依据）。 */
    @Volatile
    var cameraBufferWidth = 1280f

    @Volatile
    var cameraBufferHeight = 720f

    fun setCameraResolution(w: Int, h: Int) {
        cameraBufferWidth = w.coerceAtLeast(1).toFloat()
        cameraBufferHeight = h.coerceAtLeast(1).toFloat()
        // 缓冲尺寸改为相机真实分辨率（等比例），CameraX 后续帧按此尺寸交付
        try {
            cameraSurfaceTexture?.setDefaultBufferSize(w, h)
        } catch (_: Exception) {
        }
    }

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
    private var testProgram = 0
    private var testTextureId = 0
    private var testAPosition = 0
    private var testATexCoord = 0
    private var testUTexture = 0
    private var testUMvp = 0
    private var testUAlpha = 0
    private var viewportW = 1
    private var viewportH = 1

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

    private val mvp = FloatArray(16)
    private val proj = FloatArray(16)
    private val model = FloatArray(16)

    // 位姿指数平滑状态（抑制 MediaPipe 每帧抖动）
    // 系数：越小越稳（延迟越大）；小脸时位姿噪声大，取偏稳值
    private val SMOOTH_ALPHA_T = 0.35f
    private val SMOOTH_ALPHA_R = 0.22f
    private val smoothT = FloatArray(3)
    private val smoothR = FloatArray(16)
    private var smoothValid = false

    /** 屏幕显示 = 分析帧绕视线轴顺时针旋转 90°（相机空间 Rz(-90)，实测校准）。 */
    private val screenRotation = floatArrayOf(
        0f, -1f, 0f, 0f,
        1f, 0f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )

    /** overlay 解码目标 Surface（MediaCodec 直渲染到我们的纹理）。 */
    fun getOverlaySurface(): Surface {
        val st = overlaySurfaceTexture ?: throw IllegalStateException("GL 未初始化")
        return Surface(st)
    }

    fun cameraSurfaceOrNull(): Surface? = cameraSurface

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

        oesProgram = buildProgram(OES_FRAGMENT)
        oesAPosition = GLES20.glGetAttribLocation(oesProgram, "aPosition")
        oesATexCoord = GLES20.glGetAttribLocation(oesProgram, "aTexCoord")
        oesUTexture = GLES20.glGetUniformLocation(oesProgram, "uTexture")
        oesUMvp = GLES20.glGetUniformLocation(oesProgram, "uMvp")
        oesUAlpha = GLES20.glGetUniformLocation(oesProgram, "uAlpha")

        testProgram = buildProgram(TEST_FRAGMENT)
        testAPosition = GLES20.glGetAttribLocation(testProgram, "aPosition")
        testATexCoord = GLES20.glGetAttribLocation(testProgram, "aTexCoord")
        testUTexture = GLES20.glGetUniformLocation(testProgram, "uTexture")
        testUMvp = GLES20.glGetUniformLocation(testProgram, "uMvp")
        testUAlpha = GLES20.glGetUniformLocation(testProgram, "uAlpha")

        testTextureId = createTestTexture()
        android.util.Log.i("gscp-ar", "renderer ready build=20260922.3 viewport=${viewportW}x$viewportH")
        onCameraSurfaceReady(cameraSurface!!)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportW = width
        viewportH = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        // 背景：相机实时画面，等比缩放居中显示（不裁剪，最大 FOV，黑边补底）
        try {
            cameraSurfaceTexture?.updateTexImage()
        } catch (_: Exception) {
        }
        drawFullscreen(cameraTextureId)

        // 前景：overlay 平面（未锁定人脸时不画）。
        // 人脸矩阵在分析帧坐标系（未旋转），先旋转到屏幕显示方向。
        val faceRaw = faceMatrix
        if (faceRaw == null) {
            smoothValid = false
            return
        }
        val faceRot = FloatArray(16)
        multiply(screenRotation, faceRaw, faceRot)
        val face = FloatArray(16)
        smoothFace(faceRot, face)
        try {
            overlaySurfaceTexture?.updateTexImage()
        } catch (_: Exception) {
        }

        // 投影：FOV 以相机画面（旋转 90° 后的竖幅 720×1280）垂直方向为基准，
        // 相机完整视野全部可见（最大 FOV，不裁剪）。
        // 旋转 90° 后的显示尺寸 = 相机真实缓冲的宽高互换（等比例）
        val imgW = cameraBufferHeight
        val imgH = cameraBufferWidth
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

        // overlay 与背景同向旋转（背景顺时针 90° = 相机空间 Rz(+90)），
        // 加上背景基准累计为 Rz(180)：绕视线轴转 180°。
        val rz = floatArrayOf(
            -1f, 0f, 0f, 0f,
            0f, -1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f,
        )
        val tmp = proj.copyOf()
        multiply(rz, tmp, proj)

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
        // letterbox 缩放：相机画面 NDC 映射到屏幕居中区域
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val ndcX = imgW * fit / viewportW
        val ndcY = imgH * fit / viewportH
        mvp[0] *= ndcX; mvp[4] *= ndcX; mvp[8] *= ndcX; mvp[12] *= ndcX
        mvp[1] *= ndcY; mvp[5] *= ndcY; mvp[9] *= ndcY; mvp[13] *= ndcY
        drawOverlayQuad()
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

    /** 背景：等比缩放居中（letterbox），画面顺时针旋转 90°，完整显示。 */
    private fun drawFullscreen(tex: Int) {
        val imgW = cameraBufferHeight
        val imgH = cameraBufferWidth
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val hw = imgW * fit / viewportW
        val hh = imgH * fit / viewportH
        // 画面顺时针旋转 90°：屏幕四角采样自旋转后的纹理位置
        val verts = floatArrayOf(
            -hw, hh, 1f, 0f,
            -hw, -hh, 0f, 0f,
            hw, -hh, 0f, 1f,
            -hw, hh, 1f, 0f,
            hw, -hh, 0f, 1f,
            hw, hh, 1f, 1f,
        )
        fullscreenVertexBuffer.clear()
        fullscreenVertexBuffer.put(verts).position(0)
        GLES20.glUseProgram(oesProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex)
        GLES20.glUniform1i(oesUTexture, 0)
        GLES20.glUniformMatrix4fv(oesUMvp, 1, false, identity, 0)
        GLES20.glUniform1f(oesUAlpha, 1f)
        fullscreenVertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(oesAPosition)
        GLES20.glVertexAttribPointer(oesAPosition, 2, GLES20.GL_FLOAT, false, 16, fullscreenVertexBuffer)
        GLES20.glEnableVertexAttribArray(oesATexCoord)
        fullscreenVertexBuffer.position(2)
        GLES20.glVertexAttribPointer(oesATexCoord, 2, GLES20.GL_FLOAT, false, 16, fullscreenVertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisableVertexAttribArray(oesAPosition)
        GLES20.glDisableVertexAttribArray(oesATexCoord)
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

    private fun buildProgram(fragmentSrc: String): Int {
        val vertex = """
            precision mediump float;
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            uniform mat4 uMvp;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = uMvp * vec4(aPosition.xy, 0.0, 1.0);
                vTexCoord = aTexCoord;
            }
        """.trimIndent()

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
        val vs = compile(GLES20.GL_VERTEX_SHADER, vertex)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSrc.trimIndent())
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glLinkProgram(program)
        return program
    }

    /**
     * 位姿指数平滑：平移线性插值，旋转列线性插值后 Gram-Schmidt 正交化。
     * 人脸丢失（smoothValid 复位）后下一帧直接贴合，避免回跳。
     */
    private fun smoothFace(faceRot: FloatArray, out: FloatArray) {
        if (!smoothValid) {
            for (i in 0..2) smoothT[i] = faceRot[12 + i]
            for (i in 0..15) smoothR[i] = faceRot[i]
            smoothValid = true
        } else {
            for (i in 0..2) smoothT[i] += (faceRot[12 + i] - smoothT[i]) * SMOOTH_ALPHA_T
            for (i in 0..15) smoothR[i] += (faceRot[i] - smoothR[i]) * SMOOTH_ALPHA_R
        }
        for (i in 0..15) out[i] = smoothR[i]
        out[12] = smoothT[0]; out[13] = smoothT[1]; out[14] = smoothT[2]
        // Gram-Schmidt 正交化旋转列，防止插值漂移导致剪切
        normalizeCol(out, 0)
        // 列 1 减去其在列 0 上的投影
        var d = out[0] * out[4] + out[1] * out[5] + out[2] * out[6]
        for (i in 0..2) out[4 + i] -= d * out[i]
        normalizeCol(out, 1)
        // 列 2 = 列 0 × 列 1
        out[8] = out[1] * out[6] - out[2] * out[5]
        out[9] = out[2] * out[4] - out[0] * out[6]
        out[10] = out[0] * out[5] - out[1] * out[4]
        out[3] = 0f; out[7] = 0f; out[11] = 0f; out[15] = 1f
    }

    private fun normalizeCol(m: FloatArray, col: Int) {
        val b = col * 4
        val len = kotlin.math.sqrt(m[b] * m[b] + m[b + 1] * m[b + 1] + m[b + 2] * m[b + 2])
        if (len > 1e-6f) {
            m[b] /= len; m[b + 1] /= len; m[b + 2] /= len
        }
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
