package dev.wildframe.engine

// Port of wildframe/smoothing.py, explain/priority.py, explain/templates.py and the
// deterministic part of pipeline.py (Pipeline.decide).

/** Per-(rule, track) hysteresis: a raw state is emitted only after N consecutive agreeing frames. */
class Smoother(rulesConfig: Config) {
    private class State(var committed: RuleState = RuleState.INDETERMINATE, var pending: RuleState? = null, var pendingCount: Int = 0)

    private val frames: Map<RuleName, Int> = RuleName.values().associateWith {
        rulesConfig.section(it.value)?.int("hysteresis_frames", 1) ?: 1
    }
    private val states = LinkedHashMap<Pair<RuleName, Int?>, State>()

    private fun get(rule: RuleName, trackId: Int?) = states.getOrPut(rule to trackId) { State() }

    fun update(rule: RuleName, raw: RuleState, trackId: Int?): RuleState {
        val st = get(rule, trackId)
        val threshold = frames.getValue(rule)
        when (raw) {
            st.committed -> { st.pending = null; st.pendingCount = 0 }
            st.pending -> {
                st.pendingCount += 1
                if (st.pendingCount >= threshold) { st.committed = raw; st.pending = null; st.pendingCount = 0 }
            }
            else -> {
                st.pending = raw; st.pendingCount = 1
                if (threshold <= 1) { st.committed = raw; st.pending = null; st.pendingCount = 0 }
            }
        }
        return st.committed
    }

    /** Commit immediately, bypassing hysteresis and clearing any pending flip. */
    fun commit(rule: RuleName, state: RuleState, trackId: Int?) {
        val st = get(rule, trackId)
        st.committed = state; st.pending = null; st.pendingCount = 0
    }

    fun resetTrack(trackId: Int?) {
        states.keys.removeAll { it.second == trackId }
    }

    fun resetAll() = states.clear()
}

/** A priority_order entry: one rule, or an OR-group where any satisfied member clears the slot. */
typealias PrioritySlot = List<RuleName>

object Priority {
    fun parse(raw: Any?): List<PrioritySlot> = (raw as List<*>).map { item ->
        if (item is List<*>) item.map { RuleName.of(it as String) } else listOf(RuleName.of(item as String))
    }

    private fun groupOutput(outputs: Map<RuleName, RuleOutput>, names: PrioritySlot): RuleOutput? {
        val present = names.mapNotNull { outputs[it] }
        if (present.isEmpty()) return null
        present.firstOrNull { it.state == RuleState.SATISFIED }?.let { return it }
        val violated = present.filter { it.state == RuleState.VIOLATED }
        if (violated.isNotEmpty()) return violated.minBy { it.severity }
        return present[0]
    }

    fun resolvedOutputs(outputs: Map<RuleName, RuleOutput>, order: List<PrioritySlot>): List<RuleOutput> =
        order.mapNotNull { groupOutput(outputs, it) }

    /** Highest-priority VIOLATED slot, else highest SATISFIED, else the first slot. */
    fun resolve(outputs: Map<RuleName, RuleOutput>, order: List<PrioritySlot>): RuleOutput? {
        if (outputs.isEmpty()) return null
        val ordered = resolvedOutputs(outputs, order)
        ordered.firstOrNull { it.state == RuleState.VIOLATED }?.let { return it }
        ordered.firstOrNull { it.state == RuleState.SATISFIED }?.let { return it }
        return ordered.firstOrNull()
    }
}

object Templates {
    private val TEMPLATES: Map<Pair<RuleName, RuleState>, String> = mapOf(
        (RuleName.LIMB_CLIP to RuleState.VIOLATED) to "Zoom out - the {limb} is clipped at the frame {edge}.",
        (RuleName.LIMB_CLIP to RuleState.SATISFIED) to "No limbs clipped.",
        (RuleName.LIMB_CLIP to RuleState.INDETERMINATE) to "Can't judge limb framing ({reason}).",
        (RuleName.LEAD_SPACE to RuleState.VIOLATED) to "Pan {direction} - give the animal more room to look/move into.",
        (RuleName.LEAD_SPACE to RuleState.SATISFIED) to "Good leading space.",
        (RuleName.LEAD_SPACE to RuleState.INDETERMINATE) to "Can't judge leading space ({reason}).",
        (RuleName.EYE_THIRDS to RuleState.VIOLATED) to "Reframe - move the eye toward a rule-of-thirds point.",
        (RuleName.EYE_THIRDS to RuleState.SATISFIED) to "Eye is well placed on the thirds grid.",
        (RuleName.EYE_THIRDS to RuleState.INDETERMINATE) to "Can't judge eye placement ({reason}).",
        (RuleName.CENTRALITY to RuleState.VIOLATED) to "Reframe - move the subject toward the center of the frame.",
        (RuleName.CENTRALITY to RuleState.SATISFIED) to "Subject is well centered.",
        (RuleName.CENTRALITY to RuleState.INDETERMINATE) to "Can't judge subject centering ({reason}).",
        (RuleName.ODD_COUNT to RuleState.VIOLATED) to "Zoom {direction} to {action} the animal near the {edge} edge and fix the count.",
        (RuleName.ODD_COUNT to RuleState.SATISFIED) to "Odd count of animals - balanced group.",
        (RuleName.ODD_COUNT to RuleState.INDETERMINATE) to "Can't judge group count ({reason}).",
        (RuleName.READY to RuleState.SATISFIED) to "Good composition - click now.",
        (RuleName.READY to RuleState.VIOLATED) to "Not ready yet - fix the issue above first.",
        (RuleName.READY to RuleState.INDETERMINATE) to "Still checking composition ({reason}).",
    )
    private val FIELD = Regex("\\{(\\w+)\\}")

    /** Fills RuleOutput.message; throws if the template or a referenced field is missing. */
    fun render(o: RuleOutput): RuleOutput {
        val template = TEMPLATES[o.rule to o.state] ?: error("no template registered for ${o.rule.value}/${o.state.value}")
        val fields = HashMap<String, Any?>()
        fields["reason"] = o.reason
        fields["referent"] = o.referent
        o.correction?.let { fields.putAll(it) }
        fields.putAll(o.debug)
        val message = FIELD.replace(template) { m ->
            val key = m.groupValues[1]
            require(fields.containsKey(key)) { "template field '$key' missing for ${o.rule.value}" }
            pyStr(fields[key])
        }
        return o.copy(message = message)
    }

    private fun pyStr(v: Any?): String = when (v) {
        null -> "None"
        is Boolean -> if (v) "True" else "False"
        else -> v.toString()
    }

    fun modeNotice(s: SubjectSet): String = "Now tracking: ${s.mode.value.replace("_", " ")} (${s.modeReason})."
}

class Decision(
    val subjectSet: SubjectSet,
    val chosen: RuleOutput?,
    val rawOutputs: Map<RuleName, RuleOutput>,
    val smoothedOutputs: Map<RuleName, RuleOutput>,
    val ready: RuleOutput?,
    val modeNotice: String?,
)

/** detections -> scene -> rules -> hysteresis -> ready/priority -> rendered instruction.
 * Stateful across frames (scene mode and per-rule hysteresis): feed frames in order. */
class Engine(val rulesConfig: Config) {
    private val rules: List<Rule> = listOf(
        LeadSpaceRule(rulesConfig.sectionOrEmpty("lead_space")),
        EyeThirdsRule(rulesConfig.sectionOrEmpty("eye_thirds")),
        LimbClipRule(rulesConfig.sectionOrEmpty("limb_clip")),
        OddCountRule(rulesConfig.sectionOrEmpty("odd_count")),
        CentralityRule(rulesConfig.sectionOrEmpty("centrality")),
    )
    private val smoother = Smoother(rulesConfig)
    private val priorityOrder = Priority.parse(rulesConfig.raw("priority_order"))
    private val lastCommitted = HashMap<Pair<RuleName, Int?>, RuleOutput>()
    private var lastPrimaryTrackId: Int? = null
    private val sceneState = SceneState()
    private var lastEmittedMode = SubjectMode.NONE

    fun decide(detections: List<Detection>, frameW: Int, frameH: Int): Decision {
        val s = Scene.resolve(detections, frameW, frameH, rulesConfig, sceneState)

        var notice: String? = null
        if (s.mode != lastEmittedMode) {
            notice = Templates.modeNotice(s)
            lastEmittedMode = s.mode
        }
        val primary = s.primary
        if (s.mode == SubjectMode.NONE || primary == null) {
            return Decision(s, null, emptyMap(), emptyMap(), null, notice)
        }

        // Hysteresis is per (rule, track); a departed primary's state must not linger.
        val trackId = primary.trackId
        if (trackId != lastPrimaryTrackId) {
            val old = lastPrimaryTrackId
            smoother.resetTrack(old)
            lastCommitted.keys.removeAll { it.second == old }
            lastPrimaryTrackId = trackId
        }

        val raw = LinkedHashMap<RuleName, RuleOutput>()
        for (rule in rules) {
            val out = rule.run(s)
            val memberIndex = (out.correction?.get("member_index") as? Number)?.toInt()
            raw[rule.name] = out.copy(referent = Scene.referentLabel(s, memberIndex))
        }
        val smoothed = LinkedHashMap<RuleName, RuleOutput>()
        for ((name, out) in raw) smoothed[name] = applySmoothing(name, out, trackId)

        val ready = resolveReady(smoothed, trackId)
        val chosen = if (ready.state == RuleState.SATISFIED) {
            Templates.render(ready)
        } else {
            Priority.resolve(smoothed, priorityOrder)?.let { Templates.render(it) }
        }
        return Decision(s, chosen, raw, smoothed, ready, notice)
    }

    /** Asymmetric: drops to not-ready at once on a committed blocking violation, needs
     * ready.hysteresis_frames all-clear frames to turn back on (see pipeline._resolve_ready). */
    private fun resolveReady(smoothed: Map<RuleName, RuleOutput>, trackId: Int?): RuleOutput {
        val severityMax = rulesConfig.sectionOrEmpty("ready").double("severity_max", 0.3)
        val slots = Priority.resolvedOutputs(smoothed, priorityOrder)
        val allClear = slots.all { it.state != RuleState.VIOLATED || it.severity <= severityMax }
        val raw = RuleOutput(RuleName.READY, if (allClear) RuleState.SATISFIED else RuleState.VIOLATED,
            if (allClear) "all_clear" else "has_blocking_violation")
        if (!allClear) smoother.commit(RuleName.READY, RuleState.VIOLATED, trackId)
        return applySmoothing(RuleName.READY, raw, trackId)
    }

    private fun applySmoothing(name: RuleName, raw: RuleOutput, trackId: Int?): RuleOutput {
        val key = name to trackId
        val state = smoother.update(name, raw.state, trackId)
        if (state == raw.state) {
            lastCommitted[key] = raw
            return raw
        }
        // Still showing the previously committed state: reuse that commit's own fields.
        val cached = lastCommitted[key]
        if (cached != null && cached.state == state) return cached
        return RuleOutput(name, state, "insufficient_history")
    }
}
