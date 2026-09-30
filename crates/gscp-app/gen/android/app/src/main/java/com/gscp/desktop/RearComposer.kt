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
