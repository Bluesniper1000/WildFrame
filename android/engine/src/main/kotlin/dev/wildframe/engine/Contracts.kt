package dev.wildframe.engine

// Port of wildframe/contracts.py and wildframe/pose/skeleton.py.
//
// Numeric convention for the whole port: keypoints and boxes are FloatArray (float32), and
// arithmetic stays in Float wherever the Python code operates on numpy float32 values (numpy 2
// keeps float32 when mixed with Python scalars), converting to Double where Python does
// (float(), math.*). That keeps thresholds and tie-breaks bit-compatible with the golden fixtures.

enum class RuleState(val value: String) {
    SATISFIED("satisfied"), VIOLATED("violated"), INDETERMINATE("indeterminate"),
}

enum class RuleName(val value: String) {
    LEAD_SPACE("lead_space"), EYE_THIRDS("eye_thirds"), LIMB_CLIP("limb_clip"),
    ODD_COUNT("odd_count"), CENTRALITY("centrality"), READY("ready");

    companion object {
        fun of(value: String): RuleName = values().first { it.value == value }
    }
}

enum class SubjectMode(val value: String) {
    NONE("none"), SINGLE("single"), DOMINANT("dominant"),
    GROUP_ALIGNED("group_aligned"), FACING_PAIR("facing_pair"), MIXED("mixed"),
}

/** One raw per-frame detection, original-frame pixel space. keypoints[k] = [x, y, conf]. */
class Detection(
    val bbox: FloatArray,
    val keypoints: Array<FloatArray>,
    val detConf: Double,
    val trackId: Int? = null,
) {
    fun withTrackId(id: Int?) = Detection(bbox, keypoints, detConf, id)
}

class Orientation(val facingVector: FloatArray?, val yawState: String, val confidence: Double)

/** Phase-A Subject == Phase-B member. */
class Subject(
    val bbox: FloatArray,
    val keypoints: Array<FloatArray>,
    val detConf: Double,
    val orientation: Orientation?,
    val trackId: Int?,
    val frameW: Int,
    val frameH: Int,
)

class SubjectSet(
    val mode: SubjectMode,
    val members: List<Subject>,
    val primary: Subject?,
    val count: Int,
    val countReliable: Boolean,
    val unionBbox: FloatArray?,
    val agreedOrientation: Orientation?,
    val modeReason: String,
    val frameW: Int,
    val frameH: Int,
)

data class RuleOutput(
    val rule: RuleName,
    val state: RuleState,
    val reason: String,
    val message: String? = null,
    val severity: Double = 0.0,
    val correction: Map<String, Any?>? = null,
    val referent: String? = null,
    val debug: Map<String, Any?> = emptyMap(),
)

object KP {
    const val LEFT_EYE = 0
    const val RIGHT_EYE = 1
    const val NOSE = 2
    const val NECK = 3
    const val ROOT_OF_TAIL = 4
    const val LEFT_SHOULDER = 5
    const val LEFT_ELBOW = 6
    const val LEFT_FRONT_PAW = 7
    const val RIGHT_SHOULDER = 8
    const val RIGHT_ELBOW = 9
    const val RIGHT_FRONT_PAW = 10
    const val LEFT_HIP = 11
    const val LEFT_KNEE = 12
    const val LEFT_BACK_PAW = 13
    const val RIGHT_HIP = 14
    const val RIGHT_KNEE = 15
    const val RIGHT_BACK_PAW = 16
    const val COUNT = 17
}

object Skeleton {
    /** name -> (proximal, mid, distal), proximal-to-distal. */
    val LIMBS: Map<String, IntArray> = mapOf(
        "left_front" to intArrayOf(KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_FRONT_PAW),
        "right_front" to intArrayOf(KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_FRONT_PAW),
        "left_back" to intArrayOf(KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_BACK_PAW),
        "right_back" to intArrayOf(KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_BACK_PAW),
    )

    /** Deterministic iteration order for tie-breaking (limb_clip). */
    val LIMB_ORDER = listOf("left_front", "right_front", "left_back", "right_back")

    val EDGES: List<Pair<Int, Int>> = listOf(
        KP.LEFT_EYE to KP.RIGHT_EYE, KP.LEFT_EYE to KP.NOSE, KP.RIGHT_EYE to KP.NOSE,
        KP.NOSE to KP.NECK, KP.NECK to KP.ROOT_OF_TAIL,
        KP.NECK to KP.LEFT_SHOULDER, KP.LEFT_SHOULDER to KP.LEFT_ELBOW, KP.LEFT_ELBOW to KP.LEFT_FRONT_PAW,
        KP.NECK to KP.RIGHT_SHOULDER, KP.RIGHT_SHOULDER to KP.RIGHT_ELBOW, KP.RIGHT_ELBOW to KP.RIGHT_FRONT_PAW,
        KP.ROOT_OF_TAIL to KP.LEFT_HIP, KP.LEFT_HIP to KP.LEFT_KNEE, KP.LEFT_KNEE to KP.LEFT_BACK_PAW,
        KP.ROOT_OF_TAIL to KP.RIGHT_HIP, KP.RIGHT_HIP to KP.RIGHT_KNEE, KP.RIGHT_KNEE to KP.RIGHT_BACK_PAW,
    )
}

/** numpy's float32 np.mean for a short 1-D array: pairwise_sum (8-way unrolled for 8 <= n <= 128,
 * sequential below 8), accumulated in float32, then divided by n in float32. */
internal fun numpyMeanF32(values: FloatArray): Float {
    val n = values.size
    if (n == 0) return Float.NaN
    val sum: Float
    if (n < 8) {
        var s = 0.0f  // numpy starts the sequential path from -0.0; identical for finite sums
        for (v in values) s += v
        sum = s
    } else {
        require(n <= 128) { "numpyMeanF32 only emulates the unrolled pairwise path (n <= 128)" }
        val r = FloatArray(8) { values[it] }
        var i = 8
        while (i < n - (n % 8)) {
            for (j in 0 until 8) r[j] += values[i + j]
            i += 8
        }
        var s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while (i < n) { s += values[i]; i++ }
        sum = s
    }
    return sum / n.toFloat()
}

/** Python's builtin max(a, b) / min(a, b): returns the FIRST argument unless the second is
 * strictly greater / smaller. Matters only for NaN, kept for fidelity. */
internal fun pyMax(a: Float, b: Float): Float = if (b > a) b else a
internal fun pyMin(a: Float, b: Float): Float = if (b < a) b else a
