package dev.wildframe.engine

// Port of wildframe/rules/*.py. Each rule is a pure function of the SubjectSet: gate() returns a
// reason to short-circuit to INDETERMINATE, evaluate() returns SATISFIED or VIOLATED (odd_count's
// "no clean fix" fallbacks are the one documented exception, as in Python).

abstract class Rule(val name: RuleName, protected val config: Config) {
    abstract fun gate(s: SubjectSet): String?
    abstract fun evaluate(s: SubjectSet): RuleOutput

    fun run(s: SubjectSet): RuleOutput {
        val reason = gate(s)
        return if (reason != null) RuleOutput(name, RuleState.INDETERMINATE, reason) else evaluate(s)
    }
}

private fun clamp01(x: Float): Float = pyMax(0.0f, pyMin(1.0f, x))
private fun clamp01(x: Double): Double = maxOf(0.0, minOf(1.0, x))

class LeadSpaceRule(config: Config) : Rule(RuleName.LEAD_SPACE, config) {
    override fun gate(s: SubjectSet): String? {
        val subject = Scene.compositeSubject(s) ?: return "no_subject"
        val o = subject.orientation
        if (o == null || o.facingVector == null) return "no_orientation"
        if (o.yawState == "frontal") return "frontal_yaw"
        return null
    }

    override fun evaluate(s: SubjectSet): RuleOutput {
        val subject = Scene.compositeSubject(s)!!
        val minLead = config.double("min_lead_fraction", 0.15).toFloat()
        val idealLead = config.double("ideal_lead_fraction", 0.30).toFloat()
        val frameW = subject.frameW
        val frontSide = if (subject.orientation!!.yawState == "left") "left" else "right"
        val leadPx: Float = if (frontSide == "left") subject.bbox[0] else frameW.toFloat() - subject.bbox[2]
        val leadFraction: Float = if (frameW > 0) leadPx / frameW.toFloat() else 0.0f

        val debug = mapOf("lead_fraction" to leadFraction.toDouble(), "front_side" to frontSide)
        if (leadFraction < minLead) {
            val severity = clamp01((minLead - leadFraction) / minLead)
            val correction = mapOf(
                "axis" to "pan",
                // Pan away from the front side to open space ahead of the subject.
                "direction" to if (frontSide == "left") "right" else "left",
                "magnitude_frac" to pyMax(0.0f, idealLead - leadFraction).toDouble(),
            )
            return RuleOutput(name, RuleState.VIOLATED, "insufficient_lead_space", severity = severity.toDouble(),
                correction = correction, debug = debug)
        }
        return RuleOutput(name, RuleState.SATISFIED, "ok", debug = debug)
    }
}

class EyeThirdsRule(config: Config) : Rule(RuleName.EYE_THIRDS, config) {
    override fun gate(s: SubjectSet): String? {
        val subject = Scene.compositeSubject(s) ?: return "no_subject"
        val threshold = config.double("kpt_conf_threshold", 0.0).toFloat()
        val le = subject.keypoints[KP.LEFT_EYE]
        val re = subject.keypoints[KP.RIGHT_EYE]
        if (le[2] <= threshold && re[2] <= threshold) return "eye_not_visible"
        return null
    }

    override fun evaluate(s: SubjectSet): RuleOutput {
        val subject = Scene.compositeSubject(s)!!
        val tolerance = config.double("thirds_tolerance_frac", 0.08)
        val w = subject.frameW
        val h = subject.frameH
        val le = subject.keypoints[KP.LEFT_EYE]
        val re = subject.keypoints[KP.RIGHT_EYE]
        val eyeUsed = if (le[2] >= re[2]) "left" else "right"
        val eye = if (eyeUsed == "left") le else re
        val ex = eye[0]
        val ey = eye[1]

        val points = listOf(
            w / 3.0 to h / 3.0, 2 * w / 3.0 to h / 3.0,
            w / 3.0 to 2 * h / 3.0, 2 * w / 3.0 to 2 * h / 3.0,
        )
        fun distTo(p: Pair<Double, Double>): Double =
            Math.hypot((ex - p.first.toFloat()).toDouble(), (ey - p.second.toFloat()).toDouble())
        val nearest = points.minBy { distTo(it) }
        val dist = distTo(nearest)
        val diag = Math.hypot(w.toDouble(), h.toDouble())
        val distFrac = if (diag > 0) dist / diag else 0.0

        val debug = mapOf(
            "eye_used" to eyeUsed,
            "nearest_intersection" to listOf(nearest.first, nearest.second),
            "dist_frac" to distFrac,
        )
        if (distFrac <= tolerance) return RuleOutput(name, RuleState.SATISFIED, "ok", debug = debug)

        val nx = nearest.first.toFloat()
        val ny = nearest.second.toFloat()
        val correction = mapOf(
            "axis" to "pan_or_tilt",
            "direction_x" to if (nx > ex) "right" else "left",
            "direction_y" to if (ny > ey) "down" else "up",
            "magnitude_px" to listOf((nx - ex).toDouble(), (ny - ey).toDouble()),
        )
        return RuleOutput(name, RuleState.VIOLATED, "eye_off_thirds", severity = clamp01(distFrac / tolerance - 1.0),
            correction = correction, debug = debug)
    }
}

class CentralityRule(config: Config) : Rule(RuleName.CENTRALITY, config) {
    override fun gate(s: SubjectSet): String? = if (Scene.compositeSubject(s) == null) "no_subject" else null

    override fun evaluate(s: SubjectSet): RuleOutput {
        val subject = Scene.compositeSubject(s)!!
        val tolerance = config.double("center_tolerance_frac", 0.12)
        val w = subject.frameW
        val h = subject.frameH
        val b = subject.bbox
        val sx = (b[0] + b[2]) / 2.0f
        val sy = (b[1] + b[3]) / 2.0f
        val fx = (w / 2.0).toFloat()
        val fy = (h / 2.0).toFloat()

        val dist = Math.hypot((sx - fx).toDouble(), (sy - fy).toDouble())
        val diag = Math.hypot(w.toDouble(), h.toDouble())
        val distFrac = if (diag > 0) dist / diag else 0.0

        val debug = mapOf("subject_center" to listOf(sx.toDouble(), sy.toDouble()), "dist_frac" to distFrac)
        if (distFrac <= tolerance) return RuleOutput(name, RuleState.SATISFIED, "ok", debug = debug)

        val correction = mapOf(
            "axis" to "pan_or_tilt",
            "direction_x" to if (fx > sx) "right" else "left",
            "direction_y" to if (fy > sy) "down" else "up",
            "magnitude_px" to listOf((fx - sx).toDouble(), (fy - sy).toDouble()),
        )
        return RuleOutput(name, RuleState.VIOLATED, "off_center", severity = clamp01(distFrac / tolerance - 1.0),
            correction = correction, debug = debug)
    }
}

class LimbClipRule(config: Config) : Rule(RuleName.LIMB_CLIP, config) {
    override fun gate(s: SubjectSet): String? {
        val members = Scene.membersForClipping(s)
        if (members.isEmpty()) return "no_subject"
        val minConf = config.double("min_kpt_conf", 0.25).toFloat()
        if (members.all { m -> m.keypoints.all { it[2] < minConf } }) return "no_visible_keypoints"
        return null
    }

    override fun evaluate(s: SubjectSet): RuleOutput {
        val members = Scene.membersForClipping(s)
        val minConf = config.double("min_kpt_conf", 0.25).toFloat()
        val edgeMarginFrac = config.double("edge_margin_frac", 0.003125)
        val w = s.frameW
        val h = s.frameH
        val marginX = edgeMarginFrac * w
        val marginY = edgeMarginFrac * h
        val left = marginX.toFloat()
        val right = (w - marginX).toFloat()
        val top = marginY.toFloat()
        val bottom = (h - marginY).toFloat()

        val checked = ArrayList<List<Any?>>()
        // Member position (left-to-right) major, limb order minor: same scene, same offender.
        for ((memberIndex, member) in members.withIndex()) {
            for (limb in Skeleton.LIMB_ORDER) {
                val kp = member.keypoints[Skeleton.LIMBS.getValue(limb)[2]]
                val x = kp[0]
                val y = kp[1]
                if (kp[2] < minConf) continue
                checked.add(listOf(memberIndex, limb))

                val edge = when {
                    x <= left -> "left"
                    x >= right -> "right"
                    y <= top -> "top"
                    y >= bottom -> "bottom"
                    else -> null
                }
                if (edge != null) {
                    val debug = mapOf("checked" to checked.toList(), "clipped_limb" to limb, "clipped_member_index" to memberIndex)
                    val correction = mapOf(
                        "axis" to "zoom", "direction" to "out", "limb" to limb, "edge" to edge,
                        "member_index" to memberIndex, "track_id" to member.trackId,
                    )
                    return RuleOutput(name, RuleState.VIOLATED, "limb_clipped", severity = 1.0,
                        correction = correction, debug = debug)
                }
            }
        }
        return RuleOutput(name, RuleState.SATISFIED, "ok", debug = mapOf("checked" to checked.toList(), "clipped_limb" to null))
    }
}

class OddCountRule(config: Config) : Rule(RuleName.ODD_COUNT, config) {
    private fun intersectArea(b: FloatArray, w: Int, h: Int): Float {
        val ix1 = pyMax(b[0], 0f); val iy1 = pyMax(b[1], 0f)
        val ix2 = pyMin(b[2], w.toFloat()); val iy2 = pyMin(b[3], h.toFloat())
        return pyMax(0.0f, ix2 - ix1) * pyMax(0.0f, iy2 - iy1)
    }

    private fun visibleFraction(b: FloatArray, w: Int, h: Int): Float {
        val total = Scene.bboxArea(b)
        if (total <= 0) return 0.0f
        return intersectArea(b, w, h) / total.toFloat()
    }

    /** (edge, frame-relative distance); ties resolve in left, right, top, bottom order. */
    private fun nearestEdge(b: FloatArray, w: Int, h: Int): Pair<String, Float> {
        val candidates = listOf(
            "left" to b[0] / w.toFloat(),
            "right" to (w.toFloat() - b[2]) / w.toFloat(),
            "top" to b[1] / h.toFloat(),
            "bottom" to (h.toFloat() - b[3]) / h.toFloat(),
        )
        var best = candidates[0]
        for (c in candidates.drop(1)) if (c.second < best.second) best = c
        return best
    }

    private fun isEdgeMember(m: Subject, w: Int, h: Int, edgeMarginFrac: Float): Boolean {
        val b = m.bbox
        val clipped = b[0] < 0f || b[1] < 0f || b[2] > w.toFloat() || b[3] > h.toFloat()
        return clipped || nearestEdge(b, w, h).second <= edgeMarginFrac
    }

    private fun exclusionCrop(t: FloatArray, edge: String, w: Int, h: Int): FloatArray = when (edge) {
        "left" -> floatArrayOf(t[2], 0f, w.toFloat(), h.toFloat())
        "right" -> floatArrayOf(0f, 0f, t[0], h.toFloat())
        "top" -> floatArrayOf(0f, t[3], w.toFloat(), h.toFloat())
        else -> floatArrayOf(0f, 0f, w.toFloat(), t[1])
    }

    private fun fullyInside(b: FloatArray, r: FloatArray): Boolean = b[0] >= r[0] && b[1] >= r[1] && b[2] <= r[2] && b[3] <= r[3]

    override fun gate(s: SubjectSet): String? {
        if (s.count == 0) return "no_subject"
        if (s.count == 1) return "single_animal"
        val ceiling = config.raw("ceiling")
        val ceilingValue = ((if (ceiling is Map<*, *>) ceiling["value"] else ceiling) as Number).toInt()
        if (s.count > ceilingValue) return "count_exceeds_ceiling"
        if (s.mode == SubjectMode.FACING_PAIR) return "facing_pair_mode"
        if (!s.countReliable) return "count_unreliable"
        return null
    }

    override fun evaluate(s: SubjectSet): RuleOutput {
        val count = s.count
        if (count % 2 == 1) return RuleOutput(name, RuleState.SATISFIED, "odd_count", debug = mapOf("count" to count))

        val edgeMarginFrac = config.double("edge_margin_frac", 0.03125).toFloat()
        val excludeBelowVisible = config.double("exclude_below_visible", 0.5).toFloat()
        val w = s.frameW
        val h = s.frameH

        val edgeMembers = s.members.withIndex().filter { isEdgeMember(it.value, w, h, edgeMarginFrac) }
        if (edgeMembers.isEmpty()) {
            return RuleOutput(name, RuleState.INDETERMINATE, "no_reframing_fixes_count", debug = mapOf("count" to count))
        }

        // Most-clipped (lowest visible fraction) first, tie-broken left-to-right.
        val target = edgeMembers.minWith(compareBy<IndexedValue<Subject>>({ visibleFraction(it.value.bbox, w, h) }, { it.index }))
        val targetIndex = target.index
        val visible = visibleFraction(target.value.bbox, w, h)
        val edge = nearestEdge(target.value.bbox, w, h).first
        val debug = mapOf(
            "count" to count, "target_member_index" to targetIndex, "target_track_id" to target.value.trackId,
            "visible_fraction" to visible.toDouble(), "edge" to edge,
        )

        val action: String
        val newCount: Int
        if (visible < excludeBelowVisible) {
            action = "exclude"
            val crop = exclusionCrop(target.value.bbox, edge, w, h)
            val others = s.members.filterIndexed { i, _ -> i != targetIndex }
            if (others.any { !fullyInside(it.bbox, crop) }) {
                return RuleOutput(name, RuleState.INDETERMINATE, "shift_would_clip_another_member", debug = debug)
            }
            newCount = others.size
        } else {
            // Including an already-counted member doesn't change the count (see odd_count.py).
            action = "include"
            newCount = count
        }

        if (newCount % 2 == 0) {
            return RuleOutput(name, RuleState.INDETERMINATE, "shift_does_not_fix_parity", debug = debug)
        }
        val correction = mapOf(
            "axis" to if (action == "exclude") "zoom" else "pan",
            "direction" to if (action == "exclude") "in" else "out",
            "action" to action, "edge" to edge, "member_index" to targetIndex, "track_id" to target.value.trackId,
        )
        return RuleOutput(name, RuleState.VIOLATED, "even_count_fixable_by_$action", severity = 1.0,
            correction = correction, debug = debug)
    }
}
