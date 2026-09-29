package com.gscp.desktop

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View

/**
 * 录像启动/停止中的忙碌指示：与录像按钮**同圆心、同外径**的旋转进度环。
 * 一条亮弧沿圆周匀速旋转（自带持续重绘，可见即转，GONE 即停），叠加在按钮上。
 */
class RingSpinner @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val density = resources.displayMetrics.density

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f * density
        color = Color.parseColor("#33000000")
        strokeCap = Paint.Cap.ROUND
    }
    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f * density
        color = Color.parseColor("#FF00E5FF")   // 青：对红点/白方块都有对比
        strokeCap = Paint.Cap.ROUND
    }
    private val rect = RectF()
    private var running = false

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f
        // 外径贴近按钮外缘（本 View 与按钮同 76dp 同心）：“外径相同”
        val r = (minOf(width, height) / 2f) - 2f * density - arcPaint.strokeWidth / 2f
        rect.set(cx - r, cy - r, cx + r, cy + r)
        canvas.drawArc(rect, 0f, 360f, false, trackPaint)          // 底环
        val deg = (SystemClock.uptimeMillis() % 900) / 900f * 360f // 每圈 0.9s
        canvas.drawArc(rect, deg, 150f, false, arcPaint)           // 亮弧随角度转
        if (running && visibility == VISIBLE) postInvalidateOnAnimation()
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        running = true
        invalidate()
    }

    override fun onDetachedFromWindow() {
        running = false
        super.onDetachedFromWindow()
    }
}