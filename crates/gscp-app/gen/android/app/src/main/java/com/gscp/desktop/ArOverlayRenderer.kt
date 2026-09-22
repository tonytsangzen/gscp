package com.gscp.desktop

import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.tan

/**
 * AR 试验渲染器：透明背景 GL 层，只画 overlay 虚拟平面。
 *
 * 平面锚定在第一张稳定追踪的人脸正前方：
 * - 位置 = 人脸平移 + 人脸法线 × 距离；
 * - 朝向 = 人脸旋转（画面法线与人脸法线重合）；
 * - 人脸位姿来自 MediaPipe Face Landmarker 的 4×4 变换矩阵（列主序），
 *   覆盖 [0,1] 归一化图像空间，按相机竖屏画布映射到视锥。
 * - 投影按假设的相机垂直 FOV 近似（试验性，未做真实内参标定）。
 *
 * 稳定判定由 ArActivity 提供：连续追踪足够帧数才开始上报位姿，
 * 丢失后保留最后位姿一段时间再隐藏。
 */
class ArOverlayRenderer : GLSurfaceView.Renderer {
    /** 最新人脸变换矩阵（列主序 4×4，覆盖 [-1,1] 归一化空间），null = 未锁定。 */
    @Volatile
    var faceMatrix: FloatArray? = null

    /** 平面距离（归一化相机空间的系数，由距离滑条换算）。 */
    @Volatile
    var planeDistance = 0.35f

    /** 平面大小系数（1.0 = 基准，随距离近大远小）。 */
    @Volatile
    var planeScale = 1.0f

    /** 法线方向翻转（试验参数：若平面跑到头后侧则打开）。 */
    @Volatile
    var flipNormal = false

    /** overlay 画面宽高比（宽/高），由流 meta 更新。 */
    @Volatile
    var overlayAspect = 4f / 3f

    private var surfaceTexture: SurfaceTexture? = null
    private var textureId = 0
    private var program = 0
    private var aPosition = 0
    private var aTexCoord = 0
    private var uTexture = 0
    private var uMvp = 0
    private var uAlpha = 0
    private var viewportW = 1
    private var viewportH = 1

    private val vertexBuffer: FloatBuffer = floatBufferOf(
        // x, y, u, v —— 平面单位四边形（中心原点，宽 1 高 1/ar，实际尺寸在 shader 里缩放）
        -0.5f, 0.5f, 0f, 0f,
        -0.5f, -0.5f, 0f, 1f,
        0.5f, -0.5f, 1f, 1f,
        -0.5f, 0.5f, 0f, 0f,
        0.5f, -0.5f, 1f, 1f,
        0.5f, 0.5f, 1f, 0f,
    )

    private val mvp = FloatArray(16)
    private val proj = FloatArray(16)
    private val model = FloatArray(16)

    /** overlay 解码目标 Surface（MediaCodec 直渲染到我们的纹理）。 */
    fun getOverlaySurface(): Surface {
        val st = surfaceTexture ?: throw IllegalStateException("GL 未初始化")
        return Surface(st)
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        textureId = createOesTexture()
        surfaceTexture = SurfaceTexture(textureId).apply {
            setDefaultBufferSize(1024, 768)
        }
        program = buildProgram()
        aPosition = GLES20.glGetAttribLocation(program, "aPosition")
        aTexCoord = GLES20.glGetAttribLocation(program, "aTexCoord")
        uTexture = GLES20.glGetUniformLocation(program, "uTexture")
        uMvp = GLES20.glGetUniformLocation(program, "uMvp")
        uAlpha = GLES20.glGetUniformLocation(program, "uAlpha")
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportW = width
        viewportH = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        surfaceTexture?.updateTexImageIfAvailable()

        val faceRaw = faceMatrix ?: return
        val face = normalizeToFrustum(faceRaw)

        // 透视投影：假设垂直 FOV 50°（试验近似）
        val fovY = Math.toRadians(50.0)
        val near = 0.05f
        val far = 20f
        val aspect = viewportW.toFloat() / viewportH.toFloat()
        val t = (tan(fovY / 2) * near).toFloat()
        val r = t * aspect
        run {
            var i = 0
            while (i < 16) mvp[i++] = 0f
        }
        proj[0] = near / r; proj[5] = near / t; proj[10] = -(far + near) / (far - near)
        proj[11] = -1f; proj[14] = -2f * far * near / (far - near)

        // model：直接使用人脸变换（列主序），再沿人脸法线推出去 distance，
        // 平面尺寸 = distance × scale（近大远小，符合空间中固定物理宽度的观感）。
        val nrm = floatArrayOf(face[8], face[9], face[10])
        val sign = if (flipNormal) -1f else 1f
        val d = planeDistance * sign
        for (row in face.indices) {
            model[row] = face[row]
        }
        // 平移列（第 4 列，列主序索引 12..14）沿法线偏移
        model[12] = face[12] + nrm[0] * d
        model[13] = face[13] + nrm[1] * d
        model[14] = face[14] + nrm[2] * d
        // 缩放列（第 1/2 列）按 distance×scale 定平面大小，Y 再按宽高比
        val size = planeDistance * planeScale
        val ar = overlayAspect.coerceIn(0.5f, 3f)
        model[0] = face[0] * size
        model[1] = face[1] * size
        model[2] = face[2] * size
        model[4] = face[4] * size / ar
        model[5] = face[5] * size / ar
        model[6] = face[6] * size / ar
        model[3] = 0f; model[7] = 0f; model[15] = 1f

        multiply(proj, model, mvp)

        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(uTexture, 0)
        GLES20.glUniformMatrix4fv(uMvp, 1, false, mvp, 0)
        GLES20.glUniform1f(uAlpha, 1f)

        vertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(aPosition)
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer)
        GLES20.glEnableVertexAttribArray(aTexCoord)
        vertexBuffer.position(2)
        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisableVertexAttribArray(aPosition)
        GLES20.glDisableVertexAttribArray(aTexCoord)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    /** 人脸矩阵覆盖 [-1,1] 归一化空间；竖屏画布按比例放大到视锥空间。 */
    private fun normalizeToFrustum(face: FloatArray): FloatArray {
        // MediaPipe 变换矩阵平移在 [-1,1] 归一化坐标；把 X/Y 放大到近似米级
        // （试验近似：×2.2），旋转列保持不变。
        val out = face.copyOf()
        val s = 2.2f
        out[12] = face[12] * s
        out[13] = face[13] * s
        return out
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

    private fun buildProgram(): Int {
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
        val fragment = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES uTexture;
            uniform float uAlpha;
            varying vec2 vTexCoord;
            void main() {
                vec4 c = texture2D(uTexture, vTexCoord);
                gl_FragColor = vec4(c.rgb, c.a * uAlpha);
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
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fragment)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glLinkProgram(program)
        return program
    }
}

/** updateTexImage 的可用性辅助：无新帧时保持上一帧内容，不报错。 */
fun SurfaceTexture.updateTexImageIfAvailable() {
    try {
        updateTexImage()
    } catch (_: Exception) {
    }
}
