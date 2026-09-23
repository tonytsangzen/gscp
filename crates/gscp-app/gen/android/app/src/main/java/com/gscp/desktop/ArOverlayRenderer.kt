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
 * AR 试验渲染器（单 GLSurfaceView）：
 * - 背景 = 相机 Preview 流（SurfaceTexture，GPU 零拷贝）。顶点 uv 先经
 *   「顺时针 90° 旋转的逆映射」到传感器坐标（与本机验证过的检测输入旋转
 *   严格一致），检测/锚定与画面像素级对齐；
 * - 前景 = 眼镜 overlay 虚拟平面（MediaCodec → EXTERNAL_OES），锚定在
 *   第一张稳定追踪人脸的正前方：位于「相机→人脸」方向、深度 = 距离滑条
 *   （厘米）处，平面平行于画面（正对相机），仅随 roll 转动贴合眼线，
 *   宽度由检测眼距动态推算（overlayWidthCm × planeScale）；
 * - 无眼镜时可切半透明测试图层。
 */
class ArOverlayRenderer : GLSurfaceView.Renderer {
    /** 最新人脸变换矩阵（列主序 4×4，正立显示空间），null = 未锁定。 */
    @Volatile
    var faceMatrix: FloatArray? = null

    /** 面部 roll（弧度，绕视轴/平面法线）：仅无 3D 解算时，平面平行于画面只跟此 roll。 */
    @Volatile
    var faceRoll = 0f

    /**
     * 3D 面部姿态（由 5 点重建的金字塔底面平面）：9 个 float = [rx,ry,rz, ux,uy,uz, nx,ny,nz]，
     * 即平面右轴、上轴、法线（均为渲染器世界系、单位向量，法线朝向相机）。
     * 非空时优先用此姿态倾斜平面；为空则退回「平行画面 + faceRoll」。
     */
    @Volatile
    var faceBasis: FloatArray? = null

    /** 平面目标深度（厘米），距离滑条。 */
    @Volatile
    var planeDistance = 40f

    /** 平面大小系数（1.0 = 基准 14cm 宽）。 */
    @Volatile
    var planeScale = 1.0f

    /** overlay 物理宽度（厘米），由检测眼距动态推算；最终显示宽度再乘 planeScale。 */
    @Volatile
    var overlayWidthCm = 14f

    /** overlay 画面宽高比（宽/高），由流 meta 更新。 */
    @Volatile
    var overlayAspect = 4f / 3f

    /** 法线翻转（绕 Y 轴 180°，平面背面朝向相机）。 */
    @Volatile
    var flipNormal = false

    /** 相机就绪回调（GL 线程初始化完成后触发）。 */
    var onCameraSurfaceReady: (Surface) -> Unit = {}

    private var overlaySurfaceTexture: SurfaceTexture? = null
    private var overlayTextureId = 0
    /** overlay 纹理缓冲尺寸（SurfaceTexture.setDefaultBufferSize(480,480)）。 */
    private var overlayTexW = 480
    private var overlayTexH = 480
    private var cameraSurfaceTexture: SurfaceTexture? = null
    private var cameraTextureId = 0

    private var oesProgram = 0
    private var oesAPosition = 0
    private var oesATexCoord = 0
    private var oesUTexture = 0
    private var oesUMvp = 0
    private var oesUAlpha = 0
    // overlay 专用程序：采样眼镜流并做「黑底→透明」键控（沿用普通模式 blackKeyAlpha）
    private var overlayProgram = 0
    private var ovAPosition = 0
    private var ovATexCoord = 0
    private var ovUTexture = 0
    private var ovUMvp = 0
    private var ovUAlpha = 0
    private var ovUKeyLow = 0
    private var ovUKeyHigh = 0
    private var ovUFeatherPower = 0
    private var ovUFeatherRadius = 0
    private var ovUBrightness = 0
    private var ovUSaturation = 0
    private var ovUTexel = 0
    private var ovUCropTop = 0
    private var ovUBgColor = 0
    private var ovUBgAlpha = 0
    private var ovUCornerRadius = 0
    private var ovUBgInset = 0

    /** overlay 背面圆角矩形灰色背景（亮度 ≈5%，比之前绿色 10% 再降一半）。 */
    private val bgColor = floatArrayOf(0.5f, 0.5f, 0.5f)
    private val bgAlpha = 0.04f
    private val bgCornerRadius = 0.07f
    private val bgInset = 0.012f

    /** 黑键阈值（luma）：低于 keyLow 转透明、高于 keyHigh 保留，中间羽化。 */
    private var overlayKeyLow = 0.08f
    private var overlayKeyHigh = 0.28f
    private var overlayFeatherPower = 1.2f
    private var overlayFeatherRadius = 0.06f
    /** 普通模式效果参数（alpha/brightness/saturation 默认 1.0，与 Mixer 一致）。 */
    private var overlayBrightness = 1.0f
    private var overlaySaturation = 1.0f
    /** 去掉 overlay 顶部 N 像素（不显示）：这里去掉顶部 160px。 */
    private var overlayTopCropPx = 160f
    /** overlay 画面上移高度 = 平面半高(size/ar) × 此系数；0.25 = 上移八分之一脸部高度。 */
    private var overlayOffsetY = 0.25f
    private var bgProgram = 0
    private var bgAPosition = 0
    private var bgATexCoord = 0
    private var bgUTexture = 0
    private var bgUMvp = 0
    private var bgUSTMat = 0
    private var bgUAlpha = 0
    // head-pose 3D 轴杆：画 GSCP 求解的头部朝向（right/up/normal）作为 3D 彩色线段
    private var poseLineProgram = 0
    private var poseAPosition = 0
    private var poseUMvp = 0
    private var poseUColor = 0
    private val poseLineVerts: FloatBuffer = ByteBuffer.allocateDirect(2 * 3 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    /** 是否在画面里画头部姿态轴杆（调试 HUD）。 */
    @Volatile
    var drawHeadPoseAxes = true
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

    /** 不抛异常的版本：GL 未就绪时返回 null（调用方跳过解码，避免崩溃）。 */
    fun getOverlaySurfaceOrNull(): Surface? =
        overlaySurfaceTexture?.let { Surface(it) }

    fun cameraSurfaceOrNull(): Surface? = cameraSurface

    /** 预览缓冲尺寸跟随 CameraX 选择（letterbox 按此等比）。 */
    fun setCameraResolution(w: Int, h: Int) {
        cameraBufferWidth = w.coerceAtLeast(1).toFloat()
        cameraBufferHeight = h.coerceAtLeast(1).toFloat()
        try {
            cameraSurfaceTexture?.setDefaultBufferSize(w, h)
        } catch (_: Exception) {
        }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        overlayTextureId = createOesTexture()
        overlaySurfaceTexture = SurfaceTexture(overlayTextureId).apply {
            setDefaultBufferSize(480, 480)
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

        overlayProgram = buildProgram(OVERLAY_KEYED_FRAGMENT, DEFAULT_VERTEX)
        ovAPosition = GLES20.glGetAttribLocation(overlayProgram, "aPosition")
        ovATexCoord = GLES20.glGetAttribLocation(overlayProgram, "aTexCoord")
        ovUTexture = GLES20.glGetUniformLocation(overlayProgram, "uTexture")
        ovUMvp = GLES20.glGetUniformLocation(overlayProgram, "uMvp")
        ovUAlpha = GLES20.glGetUniformLocation(overlayProgram, "uAlpha")
        ovUKeyLow = GLES20.glGetUniformLocation(overlayProgram, "uKeyLow")
        ovUKeyHigh = GLES20.glGetUniformLocation(overlayProgram, "uKeyHigh")
        ovUFeatherPower = GLES20.glGetUniformLocation(overlayProgram, "uFeatherPower")
        ovUFeatherRadius = GLES20.glGetUniformLocation(overlayProgram, "uFeatherRadius")
        ovUBrightness = GLES20.glGetUniformLocation(overlayProgram, "uBrightness")
        ovUSaturation = GLES20.glGetUniformLocation(overlayProgram, "uSaturation")
        ovUTexel = GLES20.glGetUniformLocation(overlayProgram, "uTexel")
        ovUCropTop = GLES20.glGetUniformLocation(overlayProgram, "uCropTop")
        ovUBgColor = GLES20.glGetUniformLocation(overlayProgram, "uBgColor")
        ovUBgAlpha = GLES20.glGetUniformLocation(overlayProgram, "uBgAlpha")
        ovUCornerRadius = GLES20.glGetUniformLocation(overlayProgram, "uCornerRadius")
        ovUBgInset = GLES20.glGetUniformLocation(overlayProgram, "uBgInset")

        bgProgram = buildProgram(BG_FRAGMENT, BG_VERTEX)
        bgAPosition = GLES20.glGetAttribLocation(bgProgram, "aPosition")
        bgATexCoord = GLES20.glGetAttribLocation(bgProgram, "aTexCoord")
        bgUTexture = GLES20.glGetUniformLocation(bgProgram, "uTexture")
        bgUMvp = GLES20.glGetUniformLocation(bgProgram, "uMvp")
        bgUSTMat = GLES20.glGetUniformLocation(bgProgram, "uSTMat")

        poseLineProgram = buildProgram(POSE_LINE_FRAGMENT, POSE_LINE_VERTEX)
        poseAPosition = GLES20.glGetAttribLocation(poseLineProgram, "aPosition")
        poseUMvp = GLES20.glGetUniformLocation(poseLineProgram, "uMvp")
        poseUColor = GLES20.glGetUniformLocation(poseLineProgram, "uColor")

        // 关键：通知 Activity 相机 Surface 已就绪，从而把 CameraX Preview 绑定进来。
        // 83eaa2f 重写时漏掉了这一行，相机从未启动，SurfaceTexture 一直无帧→黑屏。
        cameraSurfaceTexture?.let {
            cameraSurface = Surface(it)
            onCameraSurfaceReady(cameraSurface!!)
        }
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

        // 平面姿态一律由人脸 3D 决定，不再强制与屏幕平行：
        // 主路径用 5 点重建的金字塔底面姿态（faceBasis：右/上/法线，世界系，法线朝相机），
        // 使平面与底面平行且随头部 yaw/pitch 三维倾斜；无解算时退化为正对相机的平面兜底。
        val size = overlayWidthCm.coerceAtLeast(1f) * planeScale
        val ar = overlayAspect.coerceIn(0.5f, 3f)
        val hy = size / ar
        model[3] = 0f; model[7] = 0f; model[11] = 0f; model[15] = 1f
        val basis = faceBasis
        if (basis != null && basis.size >= 9) {
            model[0] = basis[0] * size; model[1] = basis[1] * size; model[2] = basis[2] * size
            model[4] = basis[3] * hy;  model[5] = basis[4] * hy;  model[6] = basis[5] * hy
            model[8] = basis[6];       model[9] = basis[7];       model[10] = basis[8]
            // 位置用 5 点重建的世界坐标（cm，已含 3 瞳距前移），并沿「上轴」抬 overlayOffsetY×半高
            model[12] = face[12] + basis[3] * hy * overlayOffsetY
            model[13] = face[13] + basis[4] * hy * overlayOffsetY
            model[14] = face[14] + basis[5] * hy * overlayOffsetY
        } else {
            // 兜底：正对相机平面（法线 +Z），位置直接用面坐标，不再强制贴到 planeDistance
            val cr = kotlin.math.cos(faceRoll.toDouble()).toFloat()
            val sr = kotlin.math.sin(faceRoll.toDouble()).toFloat()
            model[0] = cr * size; model[1] = sr * size; model[2] = 0f
            model[4] = -sr * hy; model[5] = cr * hy;  model[6] = 0f
            model[8] = 0f;          model[9] = 0f;          model[10] = 1f
            model[12] = face[12]
            model[13] = face[13] + hy * overlayOffsetY
            model[14] = face[14]
        }
        if (flipNormal) {
            model[8] = -model[8]; model[9] = -model[9]; model[10] = -model[10]
        }

        multiply(proj, model, mvp)
        // letterbox 缩放：正立画面 NDC 映射到屏幕居中区域
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val ndcX = imgW * fit / viewportW
        val ndcY = imgH * fit / viewportH
        mvp[0] *= ndcX; mvp[4] *= ndcX; mvp[8] *= ndcX; mvp[12] *= ndcX
        mvp[0] *= ndcX; mvp[4] *= ndcX; mvp[8] *= ndcX; mvp[12] *= ndcX
        mvp[1] *= ndcY; mvp[5] *= ndcY; mvp[9] *= ndcY; mvp[13] *= ndcY
        drawOverlayQuad()
        // 头部姿态 3D 轴杆（红=右，绿=上，蓝=法线/朝向相机）
        val pb = basis ?: floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        if (drawHeadPoseAxes) drawHeadPose(pb, face, ndcX, ndcY)
    }

    /** 画头部 3D 轴杆：以人脸位置为原点，沿 faceBasis 的右/上/法线各画一条彩线。 */
    private fun drawHeadPose(basis: FloatArray, face: FloatArray, ndcX: Float, ndcY: Float) {
        if (poseLineProgram == 0) return
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(poseLineProgram)
        // 世界坐标（cm）直接投影（model=identity），与 overlay 同一 proj + letterbox
        val pm = FloatArray(16)
        for (i in 0 until 16) pm[i] = proj[i]
        pm[0] *= ndcX; pm[4] *= ndcX; pm[8] *= ndcX; pm[12] *= ndcX
        pm[1] *= ndcY; pm[5] *= ndcY; pm[9] *= ndcY; pm[13] *= ndcY
        GLES20.glUniformMatrix4fv(poseUMvp, 1, false, pm, 0)
        val len = overlayWidthCm.coerceAtLeast(4f) * 0.7f
        val px = face[12]; val py = face[13]; val pz = face[14]
        drawAxisLine(px, py, pz, basis[0], basis[1], basis[2], len, 1f, 0.2f, 0.2f)       // 右红
        drawAxisLine(px, py, pz, basis[3], basis[4], basis[5], len, 0.2f, 1f, 0.2f)       // 上绿
        drawAxisLine(px, py, pz, basis[6], basis[7], basis[8], len, 0.3f, 0.5f, 1f)        // 法线蓝
    }

    private fun drawAxisLine(px: Float, py: Float, pz: Float,
                             dx: Float, dy: Float, dz: Float, len: Float,
                             r: Float, g: Float, b: Float) {
        poseLineVerts.position(0)
        poseLineVerts.put(px); poseLineVerts.put(py); poseLineVerts.put(pz)
        poseLineVerts.put(px + dx * len); poseLineVerts.put(py + dy * len); poseLineVerts.put(pz + dz * len)
        poseLineVerts.position(0)
        GLES20.glUniform4f(poseUColor, r, g, b, 1f)
        GLES20.glVertexAttribPointer(poseAPosition, 3, GLES20.GL_FLOAT, false, 0, poseLineVerts)
        GLES20.glEnableVertexAttribArray(poseAPosition)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
        GLES20.glDisableVertexAttribArray(poseAPosition)
    }

    private fun drawOverlayQuad() {
        GLES20.glEnable(GLES20.GL_BLEND)
        // 加色混合（与普通模式 color = base + overlayRGB 一致）；overlay 片段已预乘 alpha。
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE)
        // 真实眼镜 overlay：复刻普通模式滤镜（5 点 AA + 边缘锐化 + 黑键 + 饱和度 ×1.12）。
        GLES20.glUseProgram(overlayProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, overlayTextureId)
        GLES20.glUniform1i(ovUTexture, 0)
        GLES20.glUniformMatrix4fv(ovUMvp, 1, false, mvp, 0)
        GLES20.glUniform1f(ovUAlpha, 1f)
        GLES20.glUniform1f(ovUKeyLow, overlayKeyLow)
        GLES20.glUniform1f(ovUKeyHigh, overlayKeyHigh)
        GLES20.glUniform1f(ovUFeatherPower, overlayFeatherPower)
        GLES20.glUniform1f(ovUFeatherRadius, overlayFeatherRadius)
        GLES20.glUniform1f(ovUBrightness, overlayBrightness)
        GLES20.glUniform1f(ovUSaturation, overlaySaturation)
        GLES20.glUniform1f(ovUCropTop, overlayTopCropPx / overlayTexH)
        GLES20.glUniform2f(ovUTexel, 1f / overlayTexW.toFloat(), 1f / overlayTexH.toFloat())
        GLES20.glUniform3f(ovUBgColor, bgColor[0], bgColor[1], bgColor[2])
        GLES20.glUniform1f(ovUBgAlpha, bgAlpha)
        GLES20.glUniform1f(ovUCornerRadius, bgCornerRadius)
        GLES20.glUniform1f(ovUBgInset, bgInset)
        quadVertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(ovAPosition)
        GLES20.glVertexAttribPointer(ovAPosition, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
        GLES20.glEnableVertexAttribArray(ovATexCoord)
        quadVertexBuffer.position(2)
        GLES20.glVertexAttribPointer(ovATexCoord, 2, GLES20.GL_FLOAT, false, 16, quadVertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
        GLES20.glDisableVertexAttribArray(ovAPosition)
        GLES20.glDisableVertexAttribArray(ovATexCoord)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /**
     * 背景：等比缩放居中（letterbox）。顶点 uv = 屏幕坐标经「顺时针 90° 旋转」
     * 映射到传感器坐标（与模型输入旋转互逆）——本机已验证可用的渲染路径
     * （oesProgram + 手动 CW UV）。显示做了水平翻转（v→1-v）以去除左右镜像，
     * 与分析帧/YuNet/YuNet 的地标坐标系一致（否则 overlay 左右反）。
     */
    private fun drawFullscreen(tex: Int) {
        val imgW = cameraBufferHeight
        val imgH = cameraBufferWidth
        val fit = minOf(viewportW / imgW, viewportH / imgH)
        val hw = imgW * fit / viewportW
        val hh = imgH * fit / viewportH
        // 画面顺时针旋转 90°：纹理 u 映射屏幕竖直、v 映射屏幕水平。
        // 仅水平翻转 v（v→1-v）去除左右镜像（翻 u 会变成上下翻转）。
        val verts = floatArrayOf(
            -hw, hh, 1f, 1f,
            -hw, -hh, 0f, 1f,
            hw, -hh, 0f, 0f,
            -hw, hh, 1f, 1f,
            hw, -hh, 0f, 0f,
            hw, hh, 1f, 0f,
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

// head-pose 3D 轴杆：仅位姿顶点 → 线
private val POSE_LINE_VERTEX = """
    precision mediump float;
    attribute vec3 aPosition;
    uniform mat4 uMvp;
    void main() {
        gl_Position = uMvp * vec4(aPosition, 1.0);
    }
"""

private val POSE_LINE_FRAGMENT = """
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

// overlay 专用：与普通模式 mix.frag 的叠加一致——5 点 AA + 边缘锐化 + 黑底键控 +
// 饱和度增强 ×1.12，输出「预乘 alpha 的 overlayRGB」，配合加色混合 = 底图 + overlayRGB。
private val OVERLAY_KEYED_FRAGMENT = """
    #extension GL_OES_EGL_image_external : require
    precision mediump float;
    uniform samplerExternalOES uTexture;
    uniform float uAlpha;
    uniform float uBrightness;
    uniform float uSaturation;
    uniform float uKeyLow;
    uniform float uKeyHigh;
    uniform float uFeatherPower;
    uniform float uFeatherRadius;
    uniform float uCropTop;
    uniform vec2 uTexel;
    uniform vec3 uBgColor;
    uniform float uBgAlpha;
    uniform float uCornerRadius;
    uniform float uBgInset;
    varying vec2 vTexCoord;

    float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

    vec3 boostSaturation(vec3 color, float gain) {
        float l = luma(color);
        vec3 gray = vec3(l);
        return clamp(gray + (color - gray) * gain, 0.0, 1.0);
    }

    // 黑边抠像（= 普通模式 blackKeyAlpha）：返回 0..1 的存在度（黑→0、内容→1）
    float blackKeyAlpha(vec3 rgb, float srcAlpha) {
        if (uKeyHigh <= uKeyLow) return srcAlpha;
        float lo = max(uKeyLow - uFeatherRadius, 0.0);
        float hi = min(uKeyHigh + uFeatherRadius, 1.0);
        float t = clamp((luma(rgb) - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
        return srcAlpha * pow(t, uFeatherPower);
    }

    // 5 点 AA + 边缘锐化（= 普通模式 sampleTopFiltered）
    vec4 sampleTopFiltered(vec2 t) {
        vec2 aa = uTexel * 0.65;
        vec4 center = texture2D(uTexture, t);
        vec4 aaColor = center * 0.4
            + texture2D(uTexture, t + vec2(-aa.x, -aa.y)) * 0.15
            + texture2D(uTexture, t + vec2( aa.x, -aa.y)) * 0.15
            + texture2D(uTexture, t + vec2(-aa.x,  aa.y)) * 0.15
            + texture2D(uTexture, t + vec2( aa.x,  aa.y)) * 0.15;
        vec4 neighbors = (
            texture2D(uTexture, t + vec2( uTexel.x, 0.0)) +
            texture2D(uTexture, t + vec2(-uTexel.x, 0.0)) +
            texture2D(uTexture, t + vec2(0.0,  uTexel.y)) +
            texture2D(uTexture, t + vec2(0.0, -uTexel.y))
        ) * 0.25;
        float edgeAlpha = abs(center.a - neighbors.a);
        float edgeLuma = abs(luma(center.rgb - neighbors.rgb));
        float edgeMask = smoothstep(0.02, 0.12, max(edgeAlpha, edgeLuma * 1.5));
        vec3 sharpened = clamp(aaColor.rgb + (center.rgb - neighbors.rgb) * 0.8 * edgeMask, 0.0, 1.0);
        return vec4(sharpened, aaColor.a);
    }

    void main() {
        // overlay 内容水平镜像（u→1-u），修正眼镜左右反。v=0 在画面顶部：
        // uCropTop=160/480，裁剪掉顶部 160px，只采样其下方内容。
        vec2 tc = vec2(1.0 - vTexCoord.x, vTexCoord.y);
        vec2 t = vec2(tc.x, uCropTop + tc.y * (1.0 - uCropTop));
        vec4 o = sampleTopFiltered(t);
        float alphaOut = blackKeyAlpha(o.rgb, o.a) * uAlpha;
        vec3 enhanced = clamp(boostSaturation(o.rgb, uSaturation) * 1.12, 0.0, 1.0);
        float gain = uAlpha * uBrightness;

        // 圆角矩形绿色背景：覆盖可见区域 [0,1]×[uCropTop,1]，眼镜内容叠加在其上
        vec2 c = vec2(0.5, (uCropTop + 1.0) * 0.5);
        vec2 b = vec2(0.5 - uBgInset, (1.0 - uCropTop) * 0.5 - uBgInset);
        vec2 q = abs(vTexCoord - c) - b + uCornerRadius;
        float dBg = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - uCornerRadius;
        float bgMask = 1.0 - smoothstep(0.0, 0.012, dBg);
        vec3 bg = uBgColor * (uBgAlpha * bgMask);

        // 预乘 alpha 的 bg·(1-眼镜存在度) + overlayRGB：加色混合后 = 底图 + 绿底 + 眼镜
        gl_FragColor = vec4(bg * (1.0 - alphaOut) + enhanced * alphaOut * gain, 1.0);
    }
"""
