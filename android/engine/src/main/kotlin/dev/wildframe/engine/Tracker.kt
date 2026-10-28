package dev.wildframe.engine

import kotlin.math.sqrt

// Port of wildframe/tracking.py: constant-velocity Kalman filter per track over
// [cx, cy, s, r, vcx, vcy, vs] plus optimal IoU assignment (scipy's linear_sum_assignment is
// replaced by a Hungarian solver; both return a minimum-cost assignment).

private fun bboxToZ(b: DoubleArray): DoubleArray {
    val w = b[2] - b[0]
    val h = b[3] - b[1]
    return doubleArrayOf(b[0] + w / 2.0, b[1] + h / 2.0, maxOf(w * h, 1e-6), w / maxOf(h, 1e-6))
}

private fun stateToBbox(x: DoubleArray): DoubleArray {
    val s = maxOf(x[2], 1e-6)
    val w = sqrt(s * x[3])
    val h = s / maxOf(w, 1e-6)
    return doubleArrayOf(x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0)
}

private fun boxIou(a: DoubleArray, b: DoubleArray): Double {
    val inter = maxOf(0.0, minOf(a[2], b[2]) - maxOf(a[0], b[0])) * maxOf(0.0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
    val areaA = maxOf(0.0, a[2] - a[0]) * maxOf(0.0, a[3] - a[1])
    val areaB = maxOf(0.0, b[2] - b[0]) * maxOf(0.0, b[3] - b[1])
    val union = areaA + areaB - inter
    return if (union > 1e-9) inter / union else 0.0
}

/** Small dense matrices as Array<DoubleArray>. */
private object Mat {
    fun eye(n: Int, v: Double = 1.0) = Array(n) { i -> DoubleArray(n) { j -> if (i == j) v else 0.0 } }
    fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(b[0].size) { j -> var s = 0.0; for (k in b.indices) s += a[i][k] * b[k][j]; s } }
    fun mulVec(a: Array<DoubleArray>, v: DoubleArray) = DoubleArray(a.size) { i -> var s = 0.0; for (k in v.indices) s += a[i][k] * v[k]; s }
    fun t(a: Array<DoubleArray>) = Array(a[0].size) { i -> DoubleArray(a.size) { j -> a[j][i] } }
    fun add(a: Array<DoubleArray>, b: Array<DoubleArray>) = Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] + b[i][j] } }
    fun sub(a: Array<DoubleArray>, b: Array<DoubleArray>) = Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] - b[i][j] } }

    /** Gauss-Jordan inverse with partial pivoting. */
    fun inv(m: Array<DoubleArray>): Array<DoubleArray> {
        val n = m.size
        val a = Array(n) { i -> DoubleArray(2 * n) { j -> if (j < n) m[i][j] else if (j - n == i) 1.0 else 0.0 } }
        for (c in 0 until n) {
            var p = c
            for (r in c + 1 until n) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r
            val tmp = a[c]; a[c] = a[p]; a[p] = tmp
            val d = a[c][c]
            for (j in 0 until 2 * n) a[c][j] /= d
            for (r in 0 until n) if (r != c) {
                val f = a[r][c]
                if (f != 0.0) for (j in 0 until 2 * n) a[r][j] -= f * a[c][j]
            }
        }
        return Array(n) { i -> DoubleArray(n) { j -> a[i][j + n] } }
    }
}

private val F: Array<DoubleArray> = Mat.eye(7).also { for (i in 0 until 3) it[i][i + 4] = 1.0 }
private val H: Array<DoubleArray> = Array(4) { i -> DoubleArray(7) { j -> if (i == j) 1.0 else 0.0 } }

internal class KalmanBoxTracker(val trackId: Int, bbox: DoubleArray) {
    var mean = DoubleArray(7).also { bboxToZ(bbox).copyInto(it) }
    var cov = Mat.eye(7, 10.0).also { for (i in 4 until 7) it[i][i] *= 100.0 }
    val q = Mat.eye(7).also { for (i in 4 until 7) it[i][i] *= 0.01 }
    val r = Mat.eye(4)
    var timeSinceUpdate = 0

    fun predict() {
        mean = Mat.mulVec(F, mean)
        cov = Mat.add(Mat.mul(Mat.mul(F, cov), Mat.t(F)), q)
        timeSinceUpdate += 1
    }

    fun update(bbox: DoubleArray) {
        val z = bboxToZ(bbox)
        val hx = Mat.mulVec(H, mean)
        val y = DoubleArray(4) { z[it] - hx[it] }
        val s = Mat.add(Mat.mul(Mat.mul(H, cov), Mat.t(H)), r)
        val k = Mat.mul(Mat.mul(cov, Mat.t(H)), Mat.inv(s))
        val ky = Mat.mulVec(k, y)
        mean = DoubleArray(7) { mean[it] + ky[it] }
        cov = Mat.mul(Mat.sub(Mat.eye(7), Mat.mul(k, H)), cov)
        timeSinceUpdate = 0
    }

    val bbox: DoubleArray get() = stateToBbox(mean)
}

/** Minimum-cost assignment for an n x m cost matrix; returns (row, col) pairs, min(n, m) of them. */
internal fun hungarian(cost: Array<DoubleArray>): List<Pair<Int, Int>> {
    val n0 = cost.size
    val m0 = if (n0 > 0) cost[0].size else 0
    if (n0 == 0 || m0 == 0) return emptyList()
    val transposed = n0 > m0
    val a = if (transposed) Array(m0) { i -> DoubleArray(n0) { j -> cost[j][i] } } else cost
    val n = a.size
    val m = a[0].size
    // e-maxx potentials formulation, 1-indexed, rows <= cols.
    val u = DoubleArray(n + 1)
    val v = DoubleArray(m + 1)
    val p = IntArray(m + 1)
    val way = IntArray(m + 1)
    for (i in 1..n) {
        p[0] = i
        var j0 = 0
        val minv = DoubleArray(m + 1) { Double.POSITIVE_INFINITY }
        val used = BooleanArray(m + 1)
        do {
            used[j0] = true
            val i0 = p[j0]
            var delta = Double.POSITIVE_INFINITY
            var j1 = 0
            for (j in 1..m) if (!used[j]) {
                val cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0 }
                if (minv[j] < delta) { delta = minv[j]; j1 = j }
            }
            for (j in 0..m) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta } else minv[j] -= delta
            }
            j0 = j1
        } while (p[j0] != 0)
        do {
            val j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        } while (j0 != 0)
    }
    val pairs = ArrayList<Pair<Int, Int>>()
    for (j in 1..m) if (p[j] != 0) {
        val row = p[j] - 1
        val col = j - 1
        pairs.add(if (transposed) col to row else row to col)
    }
    return pairs.sortedBy { it.first }
}

class ByteTracker(config: Config) {
    private val iouThreshold = config.double("track_iou_threshold", 0.3)
    private val maxAge = config.int("track_max_age", 30)
    private var tracks = ArrayList<KalmanBoxTracker>()
    private var nextId = 1

    /** One track id per box, same order; unmatched boxes get new ids. */
    fun update(boxes: List<FloatArray>): List<Int> {
        val dboxes = boxes.map { b -> DoubleArray(4) { b[it].toDouble() } }
        for (t in tracks) t.predict()
        val ids = arrayOfNulls<Int>(dboxes.size)
        val matched = HashSet<Int>()
        if (tracks.isNotEmpty() && dboxes.isNotEmpty()) {
            val cost = Array(tracks.size) { i -> DoubleArray(dboxes.size) { j -> 1.0 - boxIou(tracks[i].bbox, dboxes[j]) } }
            for ((r, c) in hungarian(cost)) {
                if (1.0 - cost[r][c] >= iouThreshold) {
                    tracks[r].update(dboxes[c])
                    ids[c] = tracks[r].trackId
                    matched.add(c)
                }
            }
        }
        for (j in dboxes.indices) if (j !in matched) {
            val t = KalmanBoxTracker(nextId++, dboxes[j])
            tracks.add(t)
            ids[j] = t.trackId
        }
        tracks = ArrayList(tracks.filter { it.timeSinceUpdate <= maxAge })
        return ids.map { it!! }
    }

    fun reset() {
        tracks.clear()
        nextId = 1
    }
}
