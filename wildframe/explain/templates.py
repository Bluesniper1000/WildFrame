from __future__ import annotations

from dataclasses import replace

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet

TEMPLATES: dict[tuple[RuleName, RuleState], str] = {
    (RuleName.LIMB_CLIP, RuleState.VIOLATED): "Zoom out - the {limb} is clipped at the frame {edge}.",
    (RuleName.LIMB_CLIP, RuleState.SATISFIED): "No limbs clipped.",
    (RuleName.LIMB_CLIP, RuleState.INDETERMINATE): "Can't judge limb framing ({reason}).",
    (RuleName.LEAD_SPACE, RuleState.VIOLATED): "Pan {direction} - give the animal more room to look/move into.",
    (RuleName.LEAD_SPACE, RuleState.SATISFIED): "Good leading space.",
    (RuleName.LEAD_SPACE, RuleState.INDETERMINATE): "Can't judge leading space ({reason}).",
    (RuleName.EYE_THIRDS, RuleState.VIOLATED): "Reframe - move the eye toward a rule-of-thirds point.",
    (RuleName.EYE_THIRDS, RuleState.SATISFIED): "Eye is well placed on the thirds grid.",
    (RuleName.EYE_THIRDS, RuleState.INDETERMINATE): "Can't judge eye placement ({reason}).",
    (RuleName.CENTRALITY, RuleState.VIOLATED): "Reframe - move the subject toward the center of the frame.",
    (RuleName.CENTRALITY, RuleState.SATISFIED): "Subject is well centered.",
    (RuleName.CENTRALITY, RuleState.INDETERMINATE): "Can't judge subject centering ({reason}).",
    (RuleName.ODD_COUNT, RuleState.VIOLATED): "Zoom {direction} to {action} the animal near the {edge} edge and fix the count.",
    (RuleName.ODD_COUNT, RuleState.SATISFIED): "Odd count of animals - balanced group.",
    (RuleName.ODD_COUNT, RuleState.INDETERMINATE): "Can't judge group count ({reason}).",
    (RuleName.READY, RuleState.SATISFIED): "Good composition - click now.",
    (RuleName.READY, RuleState.VIOLATED): "Not ready yet - fix the issue above first.",
    (RuleName.READY, RuleState.INDETERMINATE): "Still checking composition ({reason}).",
}


def render(output: RuleOutput) -> RuleOutput:
    """Returns a NEW RuleOutput with `message` filled in from TEMPLATES. Raises KeyError if no
    template matches, or if the template references a field not present on output.correction/
    output.debug - a missing message is a bug to surface loudly, not to pass through silently."""
    key = (output.rule, output.state)
    if key not in TEMPLATES:
        raise KeyError(f"no template registered for {key}")

    template = TEMPLATES[key]
    fields = {"reason": output.reason, "referent": output.referent}
    fields.update(output.correction or {})
    fields.update(output.debug or {})

    message = template.format(**fields)
    return replace(output, message=message)


def render_mode_notice(subject_set: SubjectSet) -> str:
    """One-line notice for a scene mode transition, emitted once per transition, before any
    rule instruction. Driven directly by scene.py's template-ready mode_reason."""
    mode_label = subject_set.mode.value.replace("_", " ")
    return f"Now tracking: {mode_label} ({subject_set.mode_reason})."
