package dev.wildframe.engine

import java.util.Locale
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.sqrt

// Port of wildframe/orientation.py and wildframe/scene.py.

object OrientationEstimator {
    private fun norm(x: Float, y: Float): Float = sqrt(x * x + y * y)

    private fun unit(x: Float, y: Float): FloatArray {
        val n = norm(x, y)
        return if (n > 1e-9f) floatArrayOf(x / n, y / n) else floatArrayOf(x, y)
    }

    /** config: the full rules config (uses its "orientation" block, or itself if absent). */
    fun compute(keypoints: Array<FloatArray>, config: Config): Orientation {
        val cfg = config.section("orientation") ?: config
        val minAxisConf = cfg.double("min_axis_conf", 0.3).toFloat()
        val frontalYawMaxDeg = cfg.double("frontal_yaw_max_deg", 25.0)

        val nose = keypoints[KP.NOSE]
        val neck = keypoints[KP.NECK]
        val tail = keypoints[KP.ROOT_OF_TAIL]

        var facing: FloatArray? = null
        var confidence = 0.0
        val headConf = (nose[2] + neck[2]) / 2.0f
        if (headConf >= minAxisConf) {
            facing = unit(nose[0] - neck[0], nose[1] - neck[1])
            confidence = headConf.toDouble()
        } else {
            val torsoConf = (tail[2] + neck[2]) / 2.0f
            if (torsoConf >= minAxisConf) {
                facing = unit(neck[0] - tail[0], neck[1] - tail[1])
                confidence = torsoConf.toDouble()
            }
        }
        if (facing == null) return Orientation(null, "unknown", 0.0)

        // Frontal check: both eyes confident and the nose close to the eye midpoint.
        val le = keypoints[KP.LEFT_EYE]
        val re = keypoints[KP.RIGHT_EYE]
        var frontalByEyes = false
        if (le[2] >= minAxisConf && re[2] >= minAxisConf) {
            val midX = (le[0] + re[0]) / 2.0f
            val interEye = norm(le[0] - re[0], le[1] - re[1])
            if (interEye > 1e-6f) {
                frontalByEyes = abs(nose[0] - midX) / interEye < 0.35f
            }
        }

        val angleFromVerticalDeg = Math.toDegrees(asin(minOf(1.0, abs(facing[0]).toDouble())))
        val frontalByHeading = angleFromVerticalDeg < frontalYawMaxDeg

        val yaw = when {
            frontalByEyes || frontalByHeading -> "frontal"
            facing[0] < 0f -> "left"
            else -> "right"
        }
        return Orientation(facing, yaw, confidence)
    }
}

class SceneState(
    var committedMode: SubjectMode = SubjectMode.NONE,
    var pendingMode: SubjectMode? = null,
    var pendingCount: Int = 0,
)

object Scene {
    internal fun bboxArea(b: FloatArray): Double =
        maxOf(0.0, (b[2] - b[0]).toDouble()) * maxOf(0.0, (b[3] - b[1]).toDouble())

    internal fun centerX(b: FloatArray): Double = ((b[0] + b[2]) / 2.0f).toDouble()

    /** scene._bbox_iou: the intersection is float32, and (python float union - float32 inter)
     * stays float32 under numpy's scalar promotion rules. */
    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix1 = pyMax(a[0], b[0]); val iy1 = pyMax(a[1], b[1])
        val ix2 = pyMin(a[2], b[2]); val iy2 = pyMin(a[3], b[3])
        val inter = pyMax(0.0f, ix2 - ix1) * pyMax(0.0f, iy2 - iy1)
        val union = (bboxArea(a) + bboxArea(b)).toFloat() - inter
        return if (union > 1e-9f) inter / union else 0.0f
    }

    private fun unionBbox(members: List<Subject>): FloatArray? {
        if (members.isEmpty()) return null
        return floatArrayOf(
            members.minOf { it.bbox[0] }, members.minOf { it.bbox[1] },
            members.maxOf { it.bbox[2] }, members.maxOf { it.bbox[3] },
        )
    }

    private fun selectPrimary(members: List<Subject>): Subject? {
        if (members.isEmpty()) return null
        fun score(m: Subject): Double {
            val confs = FloatArray(m.keypoints.size) { m.keypoints[it][2] }
            val meanConf = if (confs.isNotEmpty()) numpyMeanF32(confs).toDouble() else 0.0
            return bboxArea(m.bbox) * meanConf
        }
        val best = members.maxOf { score(it) }
        val tied = members.filter { score(it) == best }
        if (tied.size == 1) return tied[0]
        return tied.minWith(compareBy<Subject>({ it.trackId?.toDouble() ?: Double.POSITIVE_INFINITY }, { centerX(it.bbox) }))
    }

    private fun facesToward(m: Subject, otherCenterX: Double): Boolean {
        val o = m.orientation ?: return false
        if (o.yawState != "left" && o.yawState != "right") return false
        val selfX = centerX(m.bbox)
        return if (o.yawState == "left") otherCenterX < selfX else otherCenterX > selfX
    }

    private fun classifyMode(members: List<Subject>, primary: Subject?, cfg: Config): Pair<SubjectMode, String> {
        val n = members.size
        if (n == 0) return SubjectMode.NONE to "no members"
        if (n == 1) return SubjectMode.SINGLE to "1 of 1 member"

        val dominanceRatio = cfg.double("dominance_ratio", 2.5)
        val areas = members.map { bboxArea(it.bbox) }.sortedDescending()
        val primaryArea = bboxArea(primary!!.bbox)
        val secondArea = if (areas.size > 1) areas[1] else 0.0
        if (secondArea <= 0 || primaryArea >= dominanceRatio * secondArea) {
            return SubjectMode.DOMINANT to "primary ${String.format(Locale.ROOT, "%.1f", dominanceRatio)}x larger than next"
        }

        if (n == 2) {
            val (a, b) = members
            if (facesToward(a, centerX(b.bbox)) && facesToward(b, centerX(a.bbox))) {
                return SubjectMode.FACING_PAIR to "2 of 2 mutually facing"
            }
        }

        val agreementFraction = cfg.double("agreement_fraction", 0.75)
        val directional = members.filter { it.orientation?.yawState == "left" || it.orientation?.yawState == "right" }
        if (directional.size == n) {
            val left = directional.count { it.orientation!!.yawState == "left" }
            val right = n - left
            val majority = maxOf(left, right)
            val majorityDir = if (left >= right) "left" else "right"
            if (majority.toDouble() / n >= agreementFraction) {
                return SubjectMode.GROUP_ALIGNED to "$majority of $n facing $majorityDir"
            }
        }
        return SubjectMode.MIXED to "$n members, no shared alignment"
    }

    private fun countReliable(members: List<Subject>, cfg: Config): Boolean {
        val minDetConf = cfg.double("min_det_conf", 0.25)
        val confMargin = cfg.double("conf_margin", 0.05)
        val overlapIouMax = cfg.double("overlap_iou_max", 0.3).toFloat()
        if (members.any { it.detConf - minDetConf < confMargin }) return false
        for (i in members.indices) for (j in i + 1 until members.size) {
            if (iou(members[i].bbox, members[j].bbox) > overlapIouMax) return false
        }
        return true
    }

    private fun agreedOrientation(members: List<Subject>, mode: SubjectMode): Orientation? {
        if (mode != SubjectMode.GROUP_ALIGNED) return null
        val directional = members.filter { it.orientation?.yawState == "left" || it.orientation?.yawState == "right" }
        if (directional.isEmpty()) return null
        val left = directional.filter { it.orientation!!.yawState == "left" }
        val right = directional.filter { it.orientation!!.yawState == "right" }
        val majority = if (left.size >= right.size) left else right
        val confidence = majority.map { it.orientation!!.confidence }.average()
        val vx = numpyMeanF32(FloatArray(majority.size) { majority[it].orientation!!.facingVector!![0] })
        val vy = numpyMeanF32(FloatArray(majority.size) { majority[it].orientation!!.facingVector!![1] })
        return Orientation(floatArrayOf(vx, vy), majority[0].orientation!!.yawState, confidence)
    }

    /** scene.resolve_scene: (detections, frame dims, config, state) -> SubjectSet, mutating state. */
    fun resolve(detections: List<Detection>, frameW: Int, frameH: Int, config: Config, state: SceneState): SubjectSet {
        val sceneCfg = config.sectionOrEmpty("scene")
        val minDetConf = sceneCfg.double("min_det_conf", 0.25)
        val minBboxArea = sceneCfg.double("min_bbox_area_frac", 0.0) * frameW * frameH

        val members = detections
            .filter { it.detConf >= minDetConf && bboxArea(it.bbox) >= minBboxArea }
            .map {
                Subject(it.bbox, it.keypoints, it.detConf, OrientationEstimator.compute(it.keypoints, config), it.trackId, frameW, frameH)
            }
            .sortedBy { centerX(it.bbox) }

        val primary = selectPrimary(members)
        val (rawMode, modeReason) = classifyMode(members, primary, sceneCfg)
        val reliable = if (members.isNotEmpty()) countReliable(members, sceneCfg) else true

        val enterFrames = sceneCfg.int("mode_enter_frames", 1)
        when {
            rawMode == state.committedMode -> { state.pendingMode = null; state.pendingCount = 0 }
            state.committedMode == SubjectMode.NONE && rawMode != SubjectMode.NONE -> {
                // Acquiring a subject commits immediately (see scene.py).
                state.committedMode = rawMode; state.pendingMode = null; state.pendingCount = 0
            }
            rawMode == state.pendingMode -> {
                state.pendingCount += 1
                if (state.pendingCount >= enterFrames) {
                    state.committedMode = rawMode; state.pendingMode = null; state.pendingCount = 0
                }
            }
            else -> {
                state.pendingMode = rawMode; state.pendingCount = 1
                if (enterFrames <= 1) {
                    state.committedMode = rawMode; state.pendingMode = null; state.pendingCount = 0
                }
            }
        }

        val mode = state.committedMode
        return SubjectSet(
            mode = mode, members = members, primary = primary, count = members.size, countReliable = reliable,
            unionBbox = unionBbox(members), agreedOrientation = agreedOrientation(members, mode),
            modeReason = modeReason, frameW = frameW, frameH = frameH,
        )
    }

    /** The composite Subject lead_space/eye_thirds/centrality evaluate (scene.composite_subject). */
    fun compositeSubject(s: SubjectSet): Subject? {
        val primary = s.primary
        if (s.mode == SubjectMode.NONE || primary == null) return null
        if (s.mode == SubjectMode.SINGLE || s.mode == SubjectMode.DOMINANT) return primary
        val orientation = if (s.mode == SubjectMode.GROUP_ALIGNED) s.agreedOrientation else null
        return Subject(s.unionBbox!!, primary.keypoints, primary.detConf, orientation, primary.trackId, s.frameW, s.frameH)
    }

    fun membersForClipping(s: SubjectSet): List<Subject> =
        if (s.mode == SubjectMode.SINGLE || s.mode == SubjectMode.DOMINANT) listOfNotNull(s.primary) else s.members

    fun referentLabel(s: SubjectSet, memberIndex: Int?): String {
        val n = s.members.size
        return when {
            n == 0 -> "the frame"
            n == 1 -> "the animal"
            memberIndex == null -> "the group"
            memberIndex == 0 -> "the left animal"
            memberIndex == n - 1 -> "the right animal"
            else -> "one of the middle animals"
        }
    }
}
