package dev.wildframe.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Random

class UnitTest {
    private fun bruteForceMin(cost: Array<DoubleArray>): Double {
        val n = cost.size
        val m = cost[0].size
        var best = Double.POSITIVE_INFINITY
        fun go(row: Int, usedCols: Set<Int>, acc: Double, assigned: Int) {
            if (row == n) {
                if (assigned == minOf(n, m)) best = minOf(best, acc)
                return
            }
            // Rows may stay unassigned only when there are more rows than columns.
            if (n - row > m - assigned) go(row + 1, usedCols, acc, assigned)
            for (c in 0 until m) if (c !in usedCols) go(row + 1, usedCols + c, acc + cost[row][c], assigned + 1)
        }
        go(0, emptySet(), 0.0, 0)
        return best
    }

    @Test
    fun hungarianMatchesBruteForce() {
        val rnd = Random(0)
        repeat(200) {
            val n = 1 + rnd.nextInt(5)
            val m = 1 + rnd.nextInt(5)
            val cost = Array(n) { DoubleArray(m) { rnd.nextDouble() } }
            val pairs = hungarian(cost)
            assertEquals(minOf(n, m), pairs.size)
            assertEquals(pairs.map { it.first }.toSet().size, pairs.size)
            assertEquals(pairs.map { it.second }.toSet().size, pairs.size)
            assertEquals(bruteForceMin(cost), pairs.sumOf { (r, c) -> cost[r][c] }, 1e-9)
        }
    }

    @Test
    fun jsonParsesNestedValues() {
        val v = Json.parse("""{"a": [1, 2.5, -3e2, null, true], "b": {"c": "x\"yé"}, "d": NaN}""") as Map<*, *>
        assertEquals(listOf(1.0, 2.5, -300.0, null, true), v["a"])
        assertEquals("x\"yé", (v["b"] as Map<*, *>)["c"])
        assertEquals(Double.NaN, v["d"] as Double, 0.0)
    }

    @Test
    fun letterboxMatchesPythonRounding() {
        val lb = Letterbox(1280, 720, 640)
        assertEquals(0.5f, lb.scale)
        assertEquals(640, lb.newW)
        assertEquals(360, lb.newH)
        assertEquals(0, lb.padX)
        assertEquals(140, lb.padY)
        assertEquals(100f, lb.toFrameY(190f))
    }

    @Test
    fun decoderKeepsBestAndSuppressesOverlaps() {
        val anchors = 3
        val k = 17
        val out = FloatArray((5 + 3 * k) * anchors)
        fun set(c: Int, a: Int, v: Float) { out[c * anchors + a] = v }
        // anchor 0 and 1 overlap heavily; anchor 2 is separate and below threshold.
        listOf(floatArrayOf(100f, 100f, 50f, 50f, 0.9f), floatArrayOf(102f, 101f, 50f, 50f, 0.8f), floatArrayOf(400f, 400f, 50f, 50f, 0.1f))
            .forEachIndexed { a, v -> for (c in 0 until 5) set(c, a, v[c]) }
        val dets = PoseDecoder.decode(out, anchors, k, Letterbox(640, 640, 640), 0.25f, 0.5f, 10)
        assertEquals(1, dets.size)
        assertEquals(0.9, dets[0].detConf, 1e-6)
        assertEquals(75f, dets[0].bbox[0])
        assertNull(dets[0].trackId)
    }
}
