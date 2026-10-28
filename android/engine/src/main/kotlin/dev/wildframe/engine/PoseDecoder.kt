package dev.wildframe.engine

import kotlin.math.min

// Port of the pre/post-processing in wildframe/pose/backend.py (letterbox geometry, box/keypoint
// decode, greedy NMS). The image resize itself is platform code (Android Bitmap); this file only
// holds the arithmetic, so it stays testable on the JVM.

/** Letterbox geometry for fitting a (w, h) frame into a square `size` canvas. */
class Letterbox(val frameW: Int, val frameH: Int, val size: Int) {
    val scale: Float = min(size.toFloat() / frameW, size.toFloat() / frameH)
    // Python's round() is round-half-even; Math.rint matches it.
    val newW: Int = Math.rint((frameW * scale).toDouble()).toInt()
    val newH: Int = Math.rint((frameH * scale).toDouble()).toInt()
    val padX: Int = (size - newW) / 2
    val padY: Int = (size - newH) / 2

    fun toFrameX(x: Float) = (x - padX) / scale
    fun toFrameY(y: Float) = (y - padY) / scale
}

object PoseDecoder {
    /**
     * Decode the raw graph output (1, 4 + 1 + 3K, anchors) laid out row-major as
     * [channel][anchor]: cx, cy, w, h, score, then K x (x, y, conf). Returns detections in
     * original-frame pixels without track ids, highest score first.
     */
    fun decode(
        output: FloatArray, anchors: Int, numKeypoints: Int, lb: Letterbox,
        confThreshold: Float, iouThreshold: Float, maxDet: Int,
    ): List<Detection> {
        val channels = 5 + 3 * numKeypoints
        require(output.size >= channels * anchors) { "output too small: ${output.size} < ${channels * anchors}" }
        fun at(c: Int, a: Int) = output[c * anchors + a]

        val candidates = ArrayList<Int>()
        for (a in 0 until anchors) if (at(4, a) >= confThreshold) candidates.add(a)
        if (candidates.isEmpty()) return emptyList()

        val boxes = HashMap<Int, FloatArray>(candidates.size * 2)
        for (a in candidates) {
            val cx = at(0, a); val cy = at(1, a); val w = at(2, a); val h = at(3, a)
            boxes[a] = floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        }

        val keep: List<Int> = if (maxDet == 1) {
            listOf(candidates.maxBy { at(4, it) })
        } else {
            greedyNms(candidates, { boxes.getValue(it) }, { at(4, it) }, iouThreshold, maxDet)
        }

        return keep.map { a ->
            val b = boxes.getValue(a)
            val bbox = floatArrayOf(lb.toFrameX(b[0]), lb.toFrameY(b[1]), lb.toFrameX(b[2]), lb.toFrameY(b[3]))
            val kpts = Array(numKeypoints) { k ->
                floatArrayOf(lb.toFrameX(at(5 + 3 * k, a)), lb.toFrameY(at(6 + 3 * k, a)), at(7 + 3 * k, a))
            }
            Detection(bbox, kpts, at(4, a).toDouble())
        }
    }

    /** backend._greedy_nms: score-descending, suppress IoU > threshold, stop at maxDet. */
    internal fun greedyNms(
        ids: List<Int>, box: (Int) -> FloatArray, score: (Int) -> Float, iouThreshold: Float, maxDet: Int,
    ): List<Int> {
        // Stable descending sort keeps equal scores in index order, like np.argsort(-scores).
        var order = ids.sortedByDescending { score(it) }
        val keep = ArrayList<Int>()
        while (order.isNotEmpty() && keep.size < maxDet) {
            val i = order[0]
            keep.add(i)
            if (order.size == 1) break
            val bi = box(i)
            order = order.drop(1).filter { iou(bi, box(it)) <= iouThreshold }
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix1 = maxOf(a[0], b[0]); val iy1 = maxOf(a[1], b[1])
        val ix2 = minOf(a[2], b[2]); val iy2 = minOf(a[3], b[3])
        val inter = maxOf(ix2 - ix1, 0f) * maxOf(iy2 - iy1, 0f)
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / maxOf(union, 1e-9f)
    }
}
