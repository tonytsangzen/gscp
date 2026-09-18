package com.gscp.desktop

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.Matrix
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer


/**
 * 双层合成渲染（GLES2），效果链与桌面端 WGSL 管线对齐：
 * - 底层 = 眼镜 camera 码流（MediaCodec → EXTERNAL_OES），旋转/镜像/宽高比 cover；
 * - 顶层 = overlay 码流，轴对齐矩形 contain 铺放 × overlay_scale，效果链：
 *   5 点 AA + 边缘锐化 → 黑边抠像(luma 阈值+羽化) → 饱和度增强 → 加色混合，
 *   overlay 区域内底图按 dim_strength 圆角蒙版压暗，2 秒无帧自动隐藏；
 * - 输出可同时挂多个 Surface（屏幕 + 录制）。
 */
class SurfaceMixer(val context: Context, val width: Int, val height: Int) {
    // 底层配置
    private var bottomRotation = 90f
    private var bottomMirror = false
    private var bottomAspectRatio = 4f / 3f
    private var baseBrightness = 1.0f

    // 顶层配置
    private var topRotationDeg = 0
    private var topMirror = false
    private var topAspectRatio = 4f / 3f
    private var overlayScale = 1.0f

    // overlay 效果参数（与桌面 EffectParams 同名同语义）
    private var overlayAlpha = 1.0f
    private var overlayBrightness = 1.0f
    private var overlaySaturation = 1.0f
    private var dimStrength = 0.0f
    private var keyLow = 0.0f
    private var keyHigh = 0.0f
    private var featherPower = 1.35f
    private var featherRadius = 0.0f

    private val lock = object {}
    private val renderSurfaces = mutableListOf<RenderSurface>()

    // EGL
    private val eglDisplay: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private lateinit var eglContext: EGLContext
    private lateinit var eglConfig: EGLConfig

    // GL
    private lateinit var topSurface: Surface
    private lateinit var bottomSurface: Surface
    private lateinit var topSurfaceTexture: SurfaceTexture
    private lateinit var bottomSurfaceTexture: SurfaceTexture
    private lateinit var vertexBuffer: FloatBuffer
    private lateinit var texBuffer: FloatBuffer
    private lateinit var bottomViewMatrix: FloatArray

    private var program = 0
    private var topTexture = 0
    private var bottomTexture = 0
    private var aPosition = 0
    private var aTexCoordinator = 0
    private var uBottomTexture = 0
    private var uTopTexture = 0
    private var uBottomViewMatrix = 0
    private var uBottomMirror = 0
    private var uBaseBrightness = 0
    private var uTopRect = 0
    private var uTopRotation = 0
    private var uTopMirror = 0
    private var uTopEnable = 0
    private var uTopTexel = 0
    private var uOverlayAlpha = 0
    private var uOverlayBrightness = 0
    private var uOverlaySaturation = 0
    private var uDimStrength = 0
    private var uKeyLow = 0
    private var uKeyHigh = 0
    private var uFeatherPower = 0
    private var uFeatherRadius = 0
    private var topViewTimestamp: Long = 0
    private var bottomViewTimestamp: Long = 0

    private val handlerThread: HandlerThread = HandlerThread("gscp-gl-thread")
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

    init {
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        handler.post {
            val version = IntArray(2)
            EGL14.eglInitialize(eglDisplay, version, 0, version, 1)
            Log.d(TAG, "EGL Version ${version[0]} ${version[1]}")

            val configAttributes = intArrayOf(
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            val ret = EGL14.eglChooseConfig(eglDisplay, configAttributes, 0, configs, 0, 1, numConfigs, 0)
            Log.d(TAG, "EGL Config $ret ${configs[0]}")
            eglConfig = configs[0]!!

            val contextAttributes = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
            )
            eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttributes, 0)

            val pbufferAttribs = intArrayOf(
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
            )
            val pbSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs, 0)
            EGL14.eglMakeCurrent(eglDisplay, pbSurface, pbSurface, eglContext)
            initGl()
            renderFrame()
        }
    }

    private fun releaseSurface() {
        GLES20.glDeleteTextures(2, intArrayOf(topTexture, bottomTexture), 0)
        bottomSurfaceTexture.release()
        bottomSurface.release()
        topSurfaceTexture.release()
        topSurface.release()
    }

    private fun createSurface() {
        bottomTexture = createTexture()
        bottomSurfaceTexture = SurfaceTexture(bottomTexture)
        bottomSurfaceTexture.setDefaultBufferSize(width, height)
        bottomSurface = Surface(bottomSurfaceTexture)
        bottomSurfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                bottomSurfaceTexture.updateTexImage()
                bottomViewTimestamp = System.currentTimeMillis()
            }
        }

        topTexture = createTexture()
        topSurfaceTexture = SurfaceTexture(topTexture)
        topSurfaceTexture.setDefaultBufferSize(width, height)
        topSurface = Surface(topSurfaceTexture)
        topSurfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                topSurfaceTexture.updateTexImage()
                topViewTimestamp = System.currentTimeMillis()
            }
        }
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    private fun initGl() {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
        program = createProgram()
        aPosition = GLES20.glGetAttribLocation(program, "aPosition")
        aTexCoordinator = GLES20.glGetAttribLocation(program, "aTexCoord")
        uBottomTexture = GLES20.glGetUniformLocation(program, "uBottomTexture")
        uTopTexture = GLES20.glGetUniformLocation(program, "uTopTexture")
        uBottomViewMatrix = GLES20.glGetUniformLocation(program, "uBottomViewMatrix")
        uBottomMirror = GLES20.glGetUniformLocation(program, "uBottomMirror")
        uBaseBrightness = GLES20.glGetUniformLocation(program, "uBaseBrightness")
        uTopRect = GLES20.glGetUniformLocation(program, "uTopRect")
        uTopRotation = GLES20.glGetUniformLocation(program, "uTopRotation")
        uTopMirror = GLES20.glGetUniformLocation(program, "uTopMirror")
        uTopEnable = GLES20.glGetUniformLocation(program, "uTopEnable")
        uTopTexel = GLES20.glGetUniformLocation(program, "uTopTexel")
        uOverlayAlpha = GLES20.glGetUniformLocation(program, "uOverlayAlpha")
        uOverlayBrightness = GLES20.glGetUniformLocation(program, "uOverlayBrightness")
        uOverlaySaturation = GLES20.glGetUniformLocation(program, "uOverlaySaturation")
        uDimStrength = GLES20.glGetUniformLocation(program, "uDimStrength")
        uKeyLow = GLES20.glGetUniformLocation(program, "uKeyLow")
        uKeyHigh = GLES20.glGetUniformLocation(program, "uKeyHigh")
        uFeatherPower = GLES20.glGetUniformLocation(program, "uFeatherPower")
        uFeatherRadius = GLES20.glGetUniformLocation(program, "uFeatherRadius")

        val vertices = floatArrayOf(
            -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
            -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f
        )
        vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        vertexBuffer.put(vertices).position(0)

        val textureCoordinates = floatArrayOf(
            0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        )
        texBuffer = ByteBuffer.allocateDirect(textureCoordinates.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        texBuffer.put(textureCoordinates).position(0)

        bottomViewMatrix = FloatArray(16)
        createSurface()
        refreshMatrices()
    }

    fun attachOutputSurface(surface: Surface) {
        handler.post {
            synchronized(renderSurfaces) {
                renderSurfaces.add(RenderSurface(surface))
            }
        }
    }

    fun detachOutputSurface(surface: Surface) {
        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                if (rs.surface == surface) {
                    surface.release()
                    renderSurfaces.remove(rs)
                    return
                }
            }
        }
    }

    fun getTopSurface(): Surface = topSurface

    fun getBottomSurface(): Surface = bottomSurface

    private fun renderFrame() {
        GLES20.glUseProgram(program)

        GLES20.glEnableVertexAttribArray(aPosition)
        vertexBuffer.position(0)
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)

        GLES20.glEnableVertexAttribArray(aTexCoordinator)
        texBuffer.position(0)
        GLES20.glVertexAttribPointer(aTexCoordinator, 2, GLES20.GL_FLOAT, false, 0, texBuffer)

        GLES20.glUniformMatrix4fv(uBottomViewMatrix, 1, false, bottomViewMatrix, 0)
        GLES20.glUniform1i(uBottomMirror, if (bottomMirror) 1 else 0)
        GLES20.glUniform1f(uBaseBrightness, baseBrightness)

        // overlay 几何：内容旋转 90/270 时宽高比取倒数后 contain 铺放 × overlay_scale
        val contentAspect = if ((topRotationDeg / 90) % 2 == 1) 1f / topAspectRatio else topAspectRatio
        val canvasAspect = width.toFloat() / height
        var rw: Float
        var rh: Float
        if (contentAspect / canvasAspect > 1f) {
            rh = 1f; rw = canvasAspect / contentAspect
        } else {
            rw = 1f; rh = contentAspect / canvasAspect
        }
        rw *= overlayScale; rh *= overlayScale
        GLES20.glUniform4f(uTopRect, 0.5f, 0.5f, rw / 2f, rh / 2f)
        GLES20.glUniform1i(uTopRotation, (topRotationDeg / 90) % 4)
        GLES20.glUniform1i(uTopMirror, if (topMirror) 1 else 0)
        GLES20.glUniform2f(uTopTexel, 1f / width, 1f / height)

        GLES20.glUniform1f(uOverlayAlpha, overlayAlpha)
        GLES20.glUniform1f(uOverlayBrightness, overlayBrightness)
        GLES20.glUniform1f(uOverlaySaturation, overlaySaturation)
        GLES20.glUniform1f(uDimStrength, dimStrength)
        GLES20.glUniform1f(uKeyLow, keyLow)
        GLES20.glUniform1f(uKeyHigh, keyHigh)
        GLES20.glUniform1f(uFeatherPower, featherPower)
        GLES20.glUniform1f(uFeatherRadius, featherRadius)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, bottomTexture)
        GLES20.glUniform1i(uBottomTexture, 0)

        // overlay 2 秒无帧自动隐藏（与桌面端静默语义一致）
        val topAlive = System.currentTimeMillis() - topViewTimestamp < 2000
        if (topAlive) {
            GLES20.glUniform1i(uTopEnable, 1)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, topTexture)
            GLES20.glUniform1i(uTopTexture, 1)
        } else {
            GLES20.glUniform1i(uTopEnable, 0)
        }

        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                EGL14.eglMakeCurrent(eglDisplay, rs.eglSurface, rs.eglSurface, eglContext)
                GLES20.glViewport(0, 0, width, height)
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
                EGL14.eglSwapBuffers(eglDisplay, rs.eglSurface)
            }
        }

        GLES20.glDisableVertexAttribArray(aPosition)
        GLES20.glDisableVertexAttribArray(aTexCoordinator)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
        handler.postDelayed({ renderFrame() }, 33)
    }

    /** 断开后重建两路输入 Surface（旧 SurfaceTexture 随会话失效）。 */
    fun reset() {
        handler.post {
            releaseSurface()
            createSurface()
        }
    }

    fun release() {
        for (rs in renderSurfaces) {
            rs.release()
        }
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)
        handlerThread.quitSafely()
    }

    private fun createTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        return textures[0]
    }

    private fun createProgram(): Int {
        val vertex = context.assets.open("shader/mix.vert").reader().readText()
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertex)
        val fragment = context.assets.open("shader/mix.frag").reader().readText()
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragment)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
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

    private fun updateMatrix(matrix: FloatArray, aspectRatio: Float, rotation: Float) {
        Matrix.setIdentityM(matrix, 0)

        // cover：铺满画布（宽高比不匹配时短边对齐，长边裁剪）
        val canvasAspect = width.toFloat() / height
        var scaleX = 1.0f
        var scaleY = 1.0f
        if (aspectRatio > canvasAspect) {
            scaleX = canvasAspect / aspectRatio
        } else {
            scaleY = aspectRatio / canvasAspect
        }

        // 旋转
        Matrix.translateM(matrix, 0, 0.5f, 0.5f, 0f)
        Matrix.rotateM(matrix, 0, rotation, 0f, 0f, 1f)
        Matrix.translateM(matrix, 0, -0.5f, -0.5f, 0f)

        // 缩放
        Matrix.scaleM(matrix, 0, scaleX, scaleY, 1f)

        // 居中
        val offsetX = ((1.0 - scaleX) / 2.0) / scaleX
        val offsetY = ((1.0 - scaleY) / 2.0) / scaleY
        Matrix.translateM(matrix, 0, offsetX.toFloat(), offsetY.toFloat(), 0f)
    }

    fun setBottomRotation(rotation: Float, mirror: Boolean) {
        bottomRotation = rotation
        bottomMirror = mirror
        refreshMatrices()
    }

    fun setBottomAspectRatio(ratio: Float) {
        bottomAspectRatio = ratio
        refreshMatrices()
    }

    fun setTopRotation(rotationDeg: Int, mirror: Boolean) {
        topRotationDeg = rotationDeg
        topMirror = mirror
    }

    fun setTopAspectRatio(ratio: Float) {
        topAspectRatio = ratio
    }

    /** overlay 相对 contain 铺放的缩放（1.0 = 铺满 contain 矩形）。 */
    fun setOverlayScale(scale: Float) {
        overlayScale = scale.coerceIn(0.1f, 8.0f)
    }

    /** 效果参数批量下发（与桌面 EffectParams 同名同语义）。 */
    fun setOverlayParams(
        alpha: Float,
        brightness: Float,
        saturation: Float,
        dim: Float,
        keyLow: Float,
        keyHigh: Float,
        featherPower: Float,
        featherRadiusPx: Int,
    ) {
        overlayAlpha = alpha.coerceIn(0f, 1f)
        overlayBrightness = brightness.coerceAtLeast(0f)
        overlaySaturation = saturation.coerceIn(0f, 4f)
        dimStrength = dim.coerceIn(0f, 1f)
        this.keyLow = keyLow.coerceIn(0f, 1f)
        this.keyHigh = keyHigh.coerceIn(0f, 1f)
        this.featherPower = featherPower.coerceIn(0.1f, 8f)
        this.featherRadius = featherRadiusPx / 255f
    }

    /** 底图亮度（与桌面 base_brightness 同语义）。 */
    fun setBaseBrightness(value: Float) {
        baseBrightness = value.coerceIn(0f, 4f)
    }

    private fun refreshMatrices() {
        updateMatrix(bottomViewMatrix, bottomAspectRatio, bottomRotation)
    }

    companion object {
        private const val TAG = "gscp-mixer"
    }
}
