from wildframe.contracts import RuleName, RuleOutput, RuleState
from wildframe.explain.priority import resolve, resolved_outputs

ORDER = [RuleName.LIMB_CLIP, RuleName.LEAD_SPACE, (RuleName.EYE_THIRDS, RuleName.CENTRALITY), RuleName.ODD_COUNT]


def _out(rule, state, severity=0.0):
    return RuleOutput(rule=rule, state=state, reason="test", severity=severity)


def test_or_group_satisfied_member_wins_even_if_partner_violated():
    outputs = {
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.SATISFIED),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.VIOLATED, severity=0.9),
    }
    chosen = resolve(outputs, ORDER)
    assert chosen.rule == RuleName.EYE_THIRDS
    assert chosen.state == RuleState.SATISFIED


def test_or_group_both_violated_picks_least_severe():
    outputs = {
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.VIOLATED, severity=0.8),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.VIOLATED, severity=0.2),
    }
    chosen = resolve(outputs, ORDER)
    assert chosen.rule == RuleName.CENTRALITY
    assert chosen.state == RuleState.VIOLATED


def test_or_group_higher_priority_violation_still_wins_over_group():
    outputs = {
        RuleName.LIMB_CLIP: _out(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=1.0),
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.SATISFIED),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.SATISFIED),
    }
    chosen = resolve(outputs, ORDER)
    assert chosen.rule == RuleName.LIMB_CLIP


def test_or_group_one_indeterminate_one_satisfied_is_satisfied():
    outputs = {
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.INDETERMINATE),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.SATISFIED),
    }
    chosen = resolve(outputs, ORDER)
    assert chosen.rule == RuleName.CENTRALITY
    assert chosen.state == RuleState.SATISFIED


def test_or_group_one_indeterminate_one_violated_surfaces_violation():
    outputs = {
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.INDETERMINATE),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.VIOLATED, severity=0.5),
    }
    chosen = resolve(outputs, ORDER)
    assert chosen.rule == RuleName.CENTRALITY
    assert chosen.state == RuleState.VIOLATED


def test_resolved_outputs_one_representative_per_slot():
    outputs = {
        RuleName.LIMB_CLIP: _out(RuleName.LIMB_CLIP, RuleState.SATISFIED),
        RuleName.LEAD_SPACE: _out(RuleName.LEAD_SPACE, RuleState.SATISFIED),
        RuleName.EYE_THIRDS: _out(RuleName.EYE_THIRDS, RuleState.SATISFIED),
        RuleName.CENTRALITY: _out(RuleName.CENTRALITY, RuleState.VIOLATED, severity=0.9),
        RuleName.ODD_COUNT: _out(RuleName.ODD_COUNT, RuleState.SATISFIED),
    }
    reps = resolved_outputs(outputs, ORDER)
    assert len(reps) == 4
    assert all(r.state == RuleState.SATISFIED for r in reps)
