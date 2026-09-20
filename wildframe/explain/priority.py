from __future__ import annotations

from wildframe.contracts import RuleName, RuleOutput, RuleState

PriorityItem = RuleName | tuple[RuleName, ...]

DEFAULT_PRIORITY_ORDER: list[PriorityItem] = [
    RuleName.LIMB_CLIP,
    RuleName.LEAD_SPACE,
    (RuleName.EYE_THIRDS, RuleName.CENTRALITY),
]


def _group_output(outputs: dict[RuleName, RuleOutput], names: tuple[RuleName, ...]) -> RuleOutput | None:
    """Collapse one priority-order slot (a single rule, or an OR-group of rules where
    satisfying any one of them is enough) into the single output that represents it.

    Any SATISFIED member is enough to call the whole slot satisfied - that's the point of an
    OR-group (e.g. eye_thirds vs. centrality: either valid composition is fine). Otherwise the
    least-severe VIOLATED member represents the slot (its message is the "closest to fixed" of
    the group). Otherwise, if nothing is present, this slot contributes nothing."""
    present = [outputs[n] for n in names if n in outputs]
    if not present:
        return None
    satisfied = [o for o in present if o.state == RuleState.SATISFIED]
    if satisfied:
        return satisfied[0]
    violated = [o for o in present if o.state == RuleState.VIOLATED]
    if violated:
        return min(violated, key=lambda o: o.severity)
    return present[0]


def resolved_outputs(
    outputs: dict[RuleName, RuleOutput], priority_order: list[PriorityItem] = DEFAULT_PRIORITY_ORDER
) -> list[RuleOutput]:
    """One representative RuleOutput per priority_order slot, in priority order - a plain
    RuleName resolves to its own output; a tuple (OR-group) collapses via `_group_output`."""
    reps = []
    for item in priority_order:
        names = item if isinstance(item, tuple) else (item,)
        rep = _group_output(outputs, names)
        if rep is not None:
            reps.append(rep)
    return reps


def resolve(
    outputs: dict[RuleName, RuleOutput], priority_order: list[PriorityItem] = DEFAULT_PRIORITY_ORDER
) -> RuleOutput | None:
    """Return the single output to surface to the photographer: the highest-priority slot
    currently VIOLATED. If none are VIOLATED, the highest-priority SATISFIED slot. If none
    are SATISFIED either, the highest-priority INDETERMINATE slot (so the UI still shows
    something). None only if `outputs` is empty."""
    if not outputs:
        return None

    ordered = resolved_outputs(outputs, priority_order)

    for out in ordered:
        if out.state == RuleState.VIOLATED:
            return out
    for out in ordered:
        if out.state == RuleState.SATISFIED:
            return out
    return ordered[0] if ordered else None
