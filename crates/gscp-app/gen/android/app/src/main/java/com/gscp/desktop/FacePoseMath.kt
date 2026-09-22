package com.gscp.desktop

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 5 点 DLT-PnP 头部位姿解算（纯 Kotlin，无 OpenCV 依赖）。
 *
 * 输入：3D 人脸模型点（厘米，x 右 / y 下 / z 远离相机）与对应归一化图像点
 * （focal=1，u 右 / v 下），输出 4×4 相机空间位姿（列主序，GL 约定：
 * x 右 / y 上 / z 朝向相机）。
 */
object FacePoseMath {

    /** 人脸 3D 模型点（厘米，近似通用人脸）：YuNet 五点顺序。 */
    val MODEL_POINTS = listOf(
        Triple(-2.9, 1.2, 0.0),   // 0 右眼（图像左侧）
        Triple(2.9, 1.2, 0.0),    // 1 左眼（图像右侧）
        Triple(0.0, 0.4, -1.6),   // 2 鼻尖（朝向相机凸出）
        Triple(-1.7, 3.4, -0.3),  // 3 右嘴角
        Triple(1.7, 3.4, -0.3),   // 4 左嘴角
    )

    /**
     * DLT-PnP。modelPts 厘米；imgPts 为 focal=1 归一化坐标。
     * 返回列主序 4×4（GL 约定），失败返回 null。
     */
    fun solvePnp(
        modelPts: List<Triple<Double, Double, Double>>,
        imgPts: List<Pair<Double, Double>>,
    ): FloatArray? {
        val n = minOf(modelPts.size, imgPts.size)
        if (n < 4) return null
        val rows = 2 * n
        val cols = 12
        val a = Array(rows) { DoubleArray(cols) }
        for (i in 0 until n) {
            val (px, py, pz) = modelPts[i]
            val (u, v) = imgPts[i]
            val r1 = i * 2
            // u·(r3·P + t3) - (r1·P + t1) = 0
            a[r1][0] = u * px - px; a[r1][1] = u * py - py; a[r1][2] = u * pz - pz
            a[r1][3] = 0.0; a[r1][4] = 0.0; a[r1][5] = 0.0
            a[r1][6] = u * px; a[r1][7] = u * py; a[r1][8] = u * pz
            a[r1][9] = -1.0; a[r1][10] = 0.0; a[r1][11] = u
            val r2 = i * 2 + 1
            // v·(r3·P + t3) - (r2·P + t2) = 0
            a[r2][0] = 0.0; a[r2][1] = 0.0; a[r2][2] = 0.0
            a[r2][3] = v * px - px; a[r2][4] = v * py - py; a[r2][5] = v * pz - pz
            a[r2][6] = v * px; a[r2][7] = v * py; a[r2][8] = v * pz
            a[r2][9] = 0.0; a[r2][10] = -1.0; a[r2][11] = v
        }
        val x = nullspaceVector(a, rows, cols) ?: return null

        // 行范数归一（旋转行尺度 ≈ 1）
        val factor = 1.0 / maxOf(1e-9, (norm3(x, 0) + norm3(x, 3) + norm3(x, 6)) / 3.0)
        for (i in 0 until 12) x[i] *= factor

        // polar 正交化旋转 3×3
        val r3 = arrayOf(
            doubleArrayOf(x[0], x[1], x[2]),
            doubleArrayOf(x[3], x[4], x[5]),
            doubleArrayOf(x[6], x[7], x[8]),
        )
        val ro = polarRotation(r3) ?: return null

        // 图像约定（y 下 / z 远离）→ GL 约定（y 上 / z 朝向相机）：D·R·D，D=diag(1,-1,-1)
        val rg = arrayOf(
            doubleArrayOf(ro[0][0], -ro[0][1], -ro[0][2]),
            doubleArrayOf(-ro[1][0], ro[1][1], ro[1][2]),
            doubleArrayOf(-ro[2][0], ro[2][1], ro[2][2]),
        )
        val out = FloatArray(16)
        out[0] = rg[0][0].toFloat(); out[1] = rg[1][0].toFloat(); out[2] = rg[2][0].toFloat(); out[3] = 0f
        out[4] = rg[0][1].toFloat(); out[5] = rg[1][1].toFloat(); out[6] = rg[2][1].toFloat(); out[7] = 0f
        out[8] = rg[0][2].toFloat(); out[9] = rg[1][2].toFloat(); out[10] = rg[2][2].toFloat(); out[11] = 0f
        out[12] = x[9].toFloat()
        out[13] = (-x[10]).toFloat()
        out[14] = (-x[11]).toFloat()
        out[15] = 1f
        return out
    }

    /** 高斯消元求齐次方程组 Ax=0 的解（自由变量置 1）。 */
    private fun nullspaceVector(a: Array<DoubleArray>, rows: Int, cols: Int): DoubleArray? {
        val m = Array(rows) { a[it].copyOf() }
        val pivotCol = IntArray(cols) { -1 }
        var rank = 0
        for (col in 0 until cols) {
            var best = rank
            for (r in rank until rows) {
                if (abs(m[r][col]) > abs(m[best][col])) best = r
            }
            if (abs(m[best][col]) < 1e-10) continue
            val tmp = m[rank]; m[rank] = m[best]; m[best] = tmp
            val pv = m[rank][col]
            for (c in col until cols) m[rank][c] /= pv
            for (r in 0 until rows) {
                if (r != rank && abs(m[r][col]) > 1e-12) {
                    val f = m[r][col]
                    for (c in col until cols) m[r][c] -= f * m[rank][c]
                }
            }
            pivotCol[rank] = col
            rank++
            if (rank == cols) break
        }
        if (rank >= cols) return null // 满秩：无非平凡解
        // 找自由变量（无主元的列）
        val free = (0 until cols).first { c -> pivotCol.indexOf(c) < 0 }
        val x = DoubleArray(cols)
        x[free] = 1.0
        for (r in rank - 1 downTo 0) {
            val pc = pivotCol[r]
            var sum = 0.0
            for (c in pc + 1 until cols) sum += m[r][c] * x[c]
            x[pc] = -sum
        }
        // 方向归一（t.z 为负：人脸在相机前方）
        if (x[11] > 0) for (i in 0 until cols) x[i] = -x[i]
        return x
    }

    private fun norm3(x: DoubleArray, offset: Int): Double =
        sqrt(x[offset] * x[offset] + x[offset + 1] * x[offset + 1] + x[offset + 2] * x[offset + 2])

    /** polar 分解：R = R3·(R3ᵀR3)^(-1/2)，返回最近正交旋转阵。 */
    private fun polarRotation(r: Array<DoubleArray>): Array<DoubleArray>? {
        // H = RᵀR（3×3 对称）
        val h = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var sum = 0.0
                for (k in 0 until 3) sum += r[k][i] * r[k][j]
                h[i][j] = sum
            }
        }
        val (lambda, vec) = jacobiEigen3(h) ?: return null
        // W = V·diag(1/√λ)·Vᵀ
        val w = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var sum = 0.0
                for (k in 0 until 3) {
                    val inv = if (lambda[k] > 1e-12) 1.0 / sqrt(lambda[k]) else 0.0
                    sum += vec[i][k] * inv * vec[j][k]
                }
                w[i][j] = sum
            }
        }
        // R = R3 × W
        val ro = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                var sum = 0.0
                for (k in 0 until 3) sum += r[i][k] * w[k][j]
                ro[i][j] = sum
            }
        }
        // 行列式修正（保证是旋转而非反射）
        val det = ro[0][0] * (ro[1][1] * ro[2][2] - ro[1][2] * ro[2][1]) -
            ro[0][1] * (ro[1][0] * ro[2][2] - ro[1][2] * ro[2][0]) +
            ro[0][2] * (ro[1][0] * ro[2][1] - ro[1][1] * ro[2][0])
        if (det < 0) {
            for (i in 0 until 3) ro[i][2] = -ro[i][2]
        }
        return ro
    }

    /** 3×3 对称矩阵 Jacobi 特征分解：返回 (特征值, 特征向量列)。 */
    private fun jacobiEigen3(h: Array<DoubleArray>): Pair<DoubleArray, Array<DoubleArray>>? {
        val a = Array(3) { h[it].copyOf() }
        val v = Array(3) { i -> DoubleArray(3).also { it[i] = 1.0 } }
        for (iter in 0 until 64) {
            var p = 0; var q = 1; var off = abs(a[0][1])
            if (abs(a[0][2]) > off) { p = 0; q = 2; off = abs(a[0][2]) }
            if (abs(a[1][2]) > off) { p = 1; q = 2; off = abs(a[1][2]) }
            if (off < 1e-12) break
            val app = a[p][p]; val aqq = a[q][q]; val apq = a[p][q]
            val theta = (aqq - app) / (2 * apq)
            val tg = (if (theta >= 0) 1.0 else -1.0) /
                (abs(theta) + sqrt(theta * theta + 1.0))
            val c = 1.0 / sqrt(tg * tg + 1.0)
            val s = tg * c
            for (k in 0 until 3) {
                val akp = a[k][p]; val akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }
            for (k in 0 until 3) {
                val apk = a[p][k]; val aqk = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }
            for (k in 0 until 3) {
                val vkp = v[k][p]; val vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }
        val lambda = DoubleArray(3) { a[it][it] }
        return Pair(lambda, v)
    }
}
