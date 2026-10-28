"""Unit tests for the aggregate READY ("click now") signal in Pipeline._resolve_ready.
Builds a bare Pipeline via __new__ (skipping __init__, which would load the ONNX model) and
sets only the attributes _resolve_ready/_apply_smoothing actually touch."""

from __future__ import annotations

from wildframe.contracts import RuleName, RuleOutput, RuleState
from wildframe.pipeline import Pipeline
from wildframe.smoothing import Smoother

RULES_CONFIG = {"ready": {"severity_max": 0.3, "hysteresis_frames": 3}}


def _make_pipeline(rules_config: dict = RULES_CONFIG, priority_order=None) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.rules_config = rules_config
    pipeline.smoother = Smoother(rules_config)
    pipeline._last_committed_output = {}
    pipeline.priority_order = priority_order or [
        RuleName.LIMB_CLIP,
        RuleName.LEAD_SPACE,
        RuleName.EYE_THIRDS,
        RuleName.ODD_COUNT,
    ]
    return pipeline


def _output(rule: RuleName, state: RuleState, severity: float = 0.0) -> RuleOutput:
    return RuleOutput(rule=rule, state=state, reason="test", severity=severity)


def _all_satisfied() -> dict[RuleName, RuleOutput]:
    return {
        RuleName.LEAD_SPACE: _output(RuleName.LEAD_SPACE, RuleState.SATISFIED),
        RuleName.EYE_THIRDS: _output(RuleName.EYE_THIRDS, RuleState.SATISFIED),
        RuleName.LIMB_CLIP: _output(RuleName.LIMB_CLIP, RuleState.SATISFIED),
        RuleName.ODD_COUNT: _output(RuleName.ODD_COUNT, RuleState.SATISFIED),
    }


def test_all_satisfied_commits_to_ready_after_hysteresis():
    pipeline = _make_pipeline()
    outputs = _all_satisfied()
    results = [pipeline._resolve_ready(outputs, track_id=1).state for _ in range(3)]
    assert results[:2] == [RuleState.INDETERMINATE, RuleState.INDETERMINATE]
    assert results[2] == RuleState.SATISFIED


def test_high_severity_violation_blocks_ready():
    pipeline = _make_pipeline()
    outputs = _all_satisfied()
    outputs[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.9)
    result = None
    for _ in range(5):
        result = pipeline._resolve_ready(outputs, track_id=1)
    assert result.state == RuleState.VIOLATED
    assert result.reason == "has_blocking_violation"


def test_low_severity_violation_does_not_block_ready():
    pipeline = _make_pipeline()
    outputs = _all_satisfied()
    outputs[RuleName.ODD_COUNT] = _output(RuleName.ODD_COUNT, RuleState.VIOLATED, severity=0.1)
    result = None
    for _ in range(5):
        result = pipeline._resolve_ready(outputs, track_id=1)
    assert result.state == RuleState.SATISFIED


def test_indeterminate_rule_does_not_block_ready():
    pipeline = _make_pipeline()
    outputs = _all_satisfied()
    outputs[RuleName.EYE_THIRDS] = _output(RuleName.EYE_THIRDS, RuleState.INDETERMINATE)
    result = None
    for _ in range(5):
        result = pipeline._resolve_ready(outputs, track_id=1)
    assert result.state == RuleState.SATISFIED


def test_severity_exactly_at_threshold_does_not_block():
    pipeline = _make_pipeline()
    outputs = _all_satisfied()
    outputs[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.3)
    result = None
    for _ in range(5):
        result = pipeline._resolve_ready(outputs, track_id=1)
    assert result.state == RuleState.SATISFIED


def test_or_group_violated_partner_does_not_block_ready_when_other_satisfied():
    """The eye_thirds/centrality OR-group: eye_thirds satisfied should clear the slot for
    readiness purposes even though centrality alone is heavily violated - it must not take a
    raw per-rule severity_max pass over every RuleName independent of grouping."""
    order = [RuleName.LIMB_CLIP, RuleName.LEAD_SPACE, (RuleName.EYE_THIRDS, RuleName.CENTRALITY), RuleName.ODD_COUNT]
    pipeline = _make_pipeline(priority_order=order)
    outputs = _all_satisfied()
    outputs[RuleName.CENTRALITY] = _output(RuleName.CENTRALITY, RuleState.VIOLATED, severity=0.95)
    result = None
    for _ in range(5):
        result = pipeline._resolve_ready(outputs, track_id=1)
    assert result.state == RuleState.SATISFIED


def test_ready_tracked_independently_per_track_id():
    pipeline = _make_pipeline()
    satisfied = _all_satisfied()
    blocked = _all_satisfied()
    blocked[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.9)

    for _ in range(5):
        pipeline._resolve_ready(satisfied, track_id=1)
    for _ in range(5):
        result_blocked = pipeline._resolve_ready(blocked, track_id=2)

    assert pipeline._resolve_ready(satisfied, track_id=1).state == RuleState.SATISFIED
    assert result_blocked.state == RuleState.VIOLATED


def test_blocking_violation_drops_ready_immediately():
    """Turning ready OFF must not wait for READY's own hysteresis: the inputs are already-smoothed
    rule states, so "click now" must disappear on the first frame a blocking violation is committed."""
    pipeline = _make_pipeline()
    satisfied = _all_satisfied()
    for _ in range(5):
        ready = pipeline._resolve_ready(satisfied, track_id=1)
    assert ready.state == RuleState.SATISFIED

    blocked = _all_satisfied()
    blocked[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.9)
    result = pipeline._resolve_ready(blocked, track_id=1)
    assert result.state == RuleState.VIOLATED
    assert result.reason == "has_blocking_violation"


def test_ready_turns_back_on_only_after_hysteresis():
    """Turning ready ON stays smoothed: after a blocking violation clears, "click now" needs
    ready.hysteresis_frames (3 here) consecutive all-clear frames."""
    pipeline = _make_pipeline()
    blocked = _all_satisfied()
    blocked[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.9)
    pipeline._resolve_ready(blocked, track_id=1)

    satisfied = _all_satisfied()
    states = [pipeline._resolve_ready(satisfied, track_id=1).state for _ in range(3)]
    assert states == [RuleState.VIOLATED, RuleState.VIOLATED, RuleState.SATISFIED]


def test_one_frame_blocking_violation_interrupts_ready_streak():
    """A blocking violation resets the pending all-clear count, so ready can't commit on a streak
    that was interrupted."""
    pipeline = _make_pipeline()
    satisfied = _all_satisfied()
    blocked = _all_satisfied()
    blocked[RuleName.LIMB_CLIP] = _output(RuleName.LIMB_CLIP, RuleState.VIOLATED, severity=0.9)

    pipeline._resolve_ready(satisfied, track_id=1)
    pipeline._resolve_ready(satisfied, track_id=1)
    pipeline._resolve_ready(blocked, track_id=1)
    states = [pipeline._resolve_ready(satisfied, track_id=1).state for _ in range(3)]
    assert states == [RuleState.VIOLATED, RuleState.VIOLATED, RuleState.SATISFIED]
