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
 * 双层合成渲染（GLES2）：
 * - 底层 = 眼镜 camera 码流（MediaCodec 输出 Surface → EXTERNAL_OES），
 *   支持旋转/镜像/宽高比（cover 铺满）；
 * - 顶层 = overlay 码流，支持缩放/宽高比（contain 居中），2 秒无帧自动隐藏
 *   （与桌面端 overlay 静默语义一致）。
 * 输出可同时挂多个 Surface（屏幕 + 录制）。
 */
class SurfaceMixer(val context: Context, val width: Int, val height: Int) {
    // 配置
    private var bottomRotation = 90f
    private var bottomScale = 1.0f
    private var bottomMirror = false
    private var topRotation = 0.0f
    private var topScale = 0.7f
    private var topMirror = false
    private var topAspectRatio = 4f / 3f
    private var bottomAspectRatio = 4f / 3f

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
    private lateinit var topViewMatrix: FloatArray

    private var program = 0
    private var topTexture = 0
    private var bottomTexture = 0
    private var aPosition = 0
    private var aTexCoordinator = 0
    private var uBottomTexture = 0
    private var uTopTexture = 0
    private var uBottomTransform = 0
    private var uTopTransform = 0
    private var uBottomViewMatrix = 0
    private var uTopViewMatrix = 0
    private var uBottomMirror = 0
    private var uTopMirror = 0
    private var uTopEnable = 0
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
        uBottomTransform = GLES20.glGetUniformLocation(program, "uBottomTransform")
        uTopTransform = GLES20.glGetUniformLocation(program, "uTopTransform")
        uBottomViewMatrix = GLES20.glGetUniformLocation(program, "uBottomViewMatrix")
        uTopViewMatrix = GLES20.glGetUniformLocation(program, "uTopViewMatrix")
        uBottomMirror = GLES20.glGetUniformLocation(program, "uBottomMirror")
        uTopMirror = GLES20.glGetUniformLocation(program, "uTopMirror")
        uTopEnable = GLES20.glGetUniformLocation(program, "uTopEnable")

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
        topViewMatrix = FloatArray(16)
        createSurface()

        updateMatrix(bottomViewMatrix, bottomAspectRatio, bottomScale, bottomRotation, true)
        updateMatrix(topViewMatrix, topAspectRatio, topScale, topRotation, false)
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
        GLES20.glUniformMatrix4fv(uTopViewMatrix, 1, false, topViewMatrix, 0)

        GLES20.glUniform1i(uBottomMirror, if (bottomMirror) 1 else 0)
        GLES20.glUniform1i(uTopMirror, if (topMirror) 1 else 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, bottomTexture)
        GLES20.glUniform1i(uBottomTexture, 0)

        if (System.currentTimeMillis() - topViewTimestamp < 2000) {
            GLES20.glUniform1i(uTopEnable, 1)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, topTexture)
            GLES20.glUniform1i(uTopTexture, 1)
        } else {
            GLES20.glUniform1i(uTopEnable, 0)
        }

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        synchronized(renderSurfaces) {
            for (rs in renderSurfaces) {
                EGL14.eglMakeCurrent(eglDisplay, rs.eglSurface, rs.eglSurface, eglContext)
                GLES20.glViewport(0, 0, width, height)
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
                GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
                EGL14.eglSwapBuffers(eglDisplay, rs.eglSurface)
            }
        }

        GLES20.glDisable(GLES20.GL_BLEND)
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

    private fun updateMatrix(matrix: FloatArray, aspectRatio: Float, scale: Float, rotation: Float, fill: Boolean) {
        Matrix.setIdentityM(matrix, 0)

        var scaleX = 1.0f / scale
        var scaleY = 1.0f / scale
        val viewAspect = width.toFloat() / height
        if (viewAspect != aspectRatio) {
            if (fill) {
                if (aspectRatio > 1.0) scaleX = scaleY / aspectRatio
                else scaleY = scaleX * aspectRatio
            } else {
                scaleY = scaleX * aspectRatio / viewAspect
            }
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

    /** value ∈ 0..100 → overlay 缩放 0.4..1.0 */
    fun setTopScale(value: Int) {
        topScale = 0.4f + 0.6f * value / 100.0f
        refreshMatrices()
    }

    fun setBottomRotation(rotation: Float, mirror: Boolean) {
        bottomRotation = rotation
        bottomMirror = mirror
        refreshMatrices()
    }

    fun setTopRotation(rotation: Float, mirror: Boolean) {
        topRotation = rotation
        topMirror = mirror
        refreshMatrices()
    }

    fun setBottomAspectRatio(ratio: Float) {
        bottomAspectRatio = ratio
        refreshMatrices()
    }

    fun setTopAspectRatio(ratio: Float) {
        topAspectRatio = ratio
        refreshMatrices()
    }

    private fun refreshMatrices() {
        updateMatrix(bottomViewMatrix, bottomAspectRatio, bottomScale, bottomRotation, true)
        updateMatrix(topViewMatrix, topAspectRatio, topScale, topRotation, false)
    }

    companion object {
        private const val TAG = "gscp-mixer"
    }
}
