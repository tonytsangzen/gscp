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
 * - 背景 = 手机相机实时画面（CameraX → SurfaceTexture → EXTERNAL_OES）；
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

    /** 法线方向翻转（试验参数）。 */
    @Volatile
    var flipNormal = false

    /** overlay 画面宽高比（宽/高），由流 meta 更新。 */
    @Volatile
    var overlayAspect = 4f / 3f

    /** 无眼镜测试模式：绘制内置半透明测试图层。 */
    @Volatile
    var useTestPattern = false

    /** 输出旋转象限（0..3，相机传感器与竖屏显示的旋转差）。 */
    @Volatile
    var outputRotationQuadrant = 1

    /** 前摄：输出画面水平镜像（与传感器方向相反才符合直觉）。 */
    @Volatile
    var outputMirror = true

    // 背景等比缩放的采样比例（fx, fy ≤ 1 = 裁剪放大），overlay 投影同步补偿
    @Volatile
    var cropFx = 1f

    @Volatile
    var cropFy = 1f

    /** 相机就绪回调（GL 线程初始化完成后触发）。 */
    var onCameraSurfaceReady: (Surface) -> Unit = {}

    /** 相机实际缓冲尺寸（由 CameraX SurfaceRequest 提供，背景等比缩放依据）。 */
    @Volatile
    var cameraBufferWidth = 1280f

    @Volatile
    var cameraBufferHeight = 720f

    fun setCameraResolution(w: Int, h: Int) {
        cameraBufferWidth = w.coerceAtLeast(1).toFloat()
        cameraBufferHeight = h.coerceAtLeast(1).toFloat()
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
    private val fullscreenVertexBuffer: FloatBuffer = floatBufferOf(
        -1f, -1f, 0f, 0f,
        1f, -1f, 1f, 0f,
        -1f, 1f, 0f, 1f,
        -1f, 1f, 0f, 1f,
        1f, -1f, 1f, 0f,
        1f, 1f, 1f, 1f,
    )
    private val identity = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )

    private val mvp = FloatArray(16)
    private val proj = FloatArray(16)
    private val model = FloatArray(16)

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
        android.util.Log.i("gscp-ar", "renderer ready (build 20260922.2, ${viewportW}x$viewportH)")
        onCameraSurfaceReady(cameraSurface!!)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportW = width
        viewportH = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        // 背景：相机实时画面（铺满全屏）
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

        val useTest = useTestPattern
        val fovY = Math.toRadians(50.0)
        val near = 5f
        val far = 1000f
        val aspect = viewportW.toFloat() / viewportH.toFloat()
        val t = (tan(fovY / 2) * near).toFloat()
        val r = t * aspect
        run {
            var i = 0
            while (i < 16) mvp[i++] = 0f
        }
        proj[0] = near / r; proj[5] = near / t; proj[10] = -(far + near) / (far - near)
        proj[11] = -1f; proj[14] = -2f * far * near / (far - near)
        // 输出旋转/镜像与背景一致（proj 绕视线轴旋转 + 水平镜像），
        // 保证 overlay 平面与背景画面方向统一、锚定不错位。
        run {
            val q = outputRotationQuadrant % 4
            if (q != 0) {
                val a = -(q * 90f).toDouble()
                val c = kotlin.math.cos(Math.toRadians(a)).toFloat()
                val s = kotlin.math.sin(Math.toRadians(a)).toFloat()
                val rz = floatArrayOf(
                    c, s, 0f, 0f,
                    -s, c, 0f, 0f,
                    0f, 0f, 1f, 0f,
                    0f, 0f, 0f, 1f,
                )
                val tmp = proj.copyOf()
                multiply(rz, tmp, proj)
            }
            if (outputMirror) {
                proj[0] = -proj[0]
                proj[8] = -proj[8]
            }
            // 等比裁剪补偿（作用在旋转后的显示轴上）：FOV 随裁剪同步收窄，
            // overlay 平面与背景画面的位置/大小保持一致。
            proj[0] *= cropFx
            proj[5] *= cropFy
        }

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

        multiply(proj, model, mvp)

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        if (useTest) {
            GLES20.glUseProgram(testProgram)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, testTextureId)
            GLES20.glUniform1i(testUTexture, 0)
            GLES20.glUniformMatrix4fv(testUMvp, 1, false, mvp, 0)
            GLES20.glUniform1f(testUAlpha, 1f)
            drawQuad(testAPosition, testATexCoord)
        } else {
            GLES20.glUseProgram(oesProgram)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, overlayTextureId)
            GLES20.glUniform1i(oesUTexture, 0)
            GLES20.glUniformMatrix4fv(oesUMvp, 1, false, mvp, 0)
            GLES20.glUniform1f(oesUAlpha, 1f)
            drawQuad(oesAPosition, oesATexCoord)
        }
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun drawFullscreen(tex: Int) {
        // 背景等比缩放（cover 裁剪）：按相机实际缓冲尺寸（旋转后）→ 视口覆盖
        val imgW = if (outputRotationQuadrant % 2 == 1) cameraBufferHeight else cameraBufferWidth
        val imgH = if (outputRotationQuadrant % 2 == 1) cameraBufferWidth else cameraBufferHeight
        val cover = maxOf(viewportW / imgW, viewportH / imgH)
        cropFx = (viewportW / cover / imgW).coerceIn(0f, 1f)
        cropFy = (viewportH / cover / imgH).coerceIn(0f, 1f)

        GLES20.glUseProgram(oesProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex)
        GLES20.glUniform1i(oesUTexture, 0)
        GLES20.glUniformMatrix4fv(oesUMvp, 1, false, identity, 0)
        GLES20.glUniform1f(oesUAlpha, 1f)
        // 背景 UV 按输出旋转象限 + 前摄镜像 + 等比裁剪变换
        val q = outputRotationQuadrant % 4
        val mirror = outputMirror
        val pos = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, -1f, 1f, 1f, -1f, 1f, 1f)
        val baseUv = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f, 1f)
        val out = FloatArray(24)
        for (i in 0 until 6) {
            var u = baseUv[i * 2] - 0.5f
            var v = baseUv[i * 2 + 1] - 0.5f
            if (mirror) u = -u
            // 逆旋转（象限 q 的显示旋转的逆），再按裁剪比例采样
            val un: FloatArray = when (q) {
                1 -> floatArrayOf(v, -u)
                2 -> floatArrayOf(-u, -v)
                3 -> floatArrayOf(-v, u)
                else -> floatArrayOf(u, v)
            }
            out[i * 4] = pos[i * 2]
            out[i * 4 + 1] = pos[i * 2 + 1]
            out[i * 4 + 2] = 0.5f + un[0] * cropFx
            out[i * 4 + 3] = 0.5f + un[1] * cropFy
        }
        fullscreenVertexBuffer.clear()
        fullscreenVertexBuffer.put(out).position(0)
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

    private fun drawQuad(aPos: Int, aUv: Int) {
        quadVertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
        GLES20.glEnableVertexAttribArray(aUv)
        quadVertexBuffer.position(2)
        GLES20.glVertexAttribPointer(aUv, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aUv)
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
        c.drawColor(0x5930E0A0.toInt()) // 半透明青绿底
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
