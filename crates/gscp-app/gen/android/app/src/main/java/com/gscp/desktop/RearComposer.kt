package com.gscp.desktop

import android.view.Surface

/**
 * 后摄流合成器抽象：GlassesPlayer 只依赖此接口取输入面/下发参数。
 * 实现一：[SurfaceMixer]（普通播放页，矩阵缩放管线）；
 * 实现二：[ArRearGl]（AR 后摄，复用前摄 CPU 烘焙位图，contain 完整显示；
 * GlassesPlayer 传 overlayImageCallback 时 getTopSurface 不被消费）。
 */
interface RearComposer {
    fun getBottomSurface(): Surface
    fun getTopSurface(): Surface

    /**
     * overlay 流建流：按流尺寸开一个解码输入面。默认回落 [getTopSurface]
     * （普通模式 SurfaceMixer 的 GLSL 现场键控管线不变）；ArRearGl 覆写为
     * GPU 烘焙核心（OverlayBakeCore）的输入面，替代 CPU 烘焙。
     * sr=true 时烘焙核心启用 GL 原生 ESPCN ×2 超分（debug.gscp.ovsr）。
     */
    fun openOverlayStream(
        width: Int,
        height: Int,
        sr: Boolean = false,
        srWeights: ByteArray? = null,
    ): Surface = getTopSurface()

    fun setBottomAspectRatio(ratio: Float)
    fun setBottomRotation(rotation: Float, mirror: Boolean)
    fun setTopAspectRatio(ratio: Float)
    fun setOverlayScale(scale: Float)
    fun setOverlayParams(
        alpha: Float,
        brightness: Float,
        saturation: Float,
        dim: Float,
        keyLow: Float,
        keyHigh: Float,
        featherPower: Float,
        featherRadiusPx: Int,
    )
    fun setBaseBrightness(value: Float)
    fun setTopRotation(rotationDeg: Int, mirror: Boolean)
    fun reset()
}
