import itertools

import pytest

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectMode
from wildframe.explain.templates import TEMPLATES, render, render_mode_notice
from tests.synthetic import make_subject, make_subject_set


def test_every_reachable_combination_has_a_template():
    for rule, state in itertools.product(RuleName, RuleState):
        assert (rule, state) in TEMPLATES, f"missing template for {(rule, state)}"


def test_render_produces_nonempty_message_for_every_combination():
    for rule, state in itertools.product(RuleName, RuleState):
        output = RuleOutput(
            rule=rule,
            state=state,
            reason="some_reason",
            correction={"limb": "left_front", "edge": "left", "direction": "left", "action": "exclude"},
            debug={},
            referent="the left animal",
        )
        rendered = render(output)
        assert rendered.message
        assert isinstance(rendered.message, str)
        # referent must survive rendering unchanged - it's carried on the struct, not consumed.
        assert rendered.referent == "the left animal"


def test_limb_clip_violated_exact_message():
    output = RuleOutput(
        rule=RuleName.LIMB_CLIP,
        state=RuleState.VIOLATED,
        reason="limb_clipped",
        correction={"limb": "left_front", "edge": "left"},
    )
    rendered = render(output)
    assert rendered.message == "Zoom out - the left_front is clipped at the frame left."


def test_lead_space_violated_exact_message():
    output = RuleOutput(
        rule=RuleName.LEAD_SPACE,
        state=RuleState.VIOLATED,
        reason="insufficient_lead_space",
        correction={"direction": "right", "magnitude_frac": 0.1},
    )
    rendered = render(output)
    assert rendered.message == "Pan right - give the animal more room to look/move into."


def test_indeterminate_reason_is_interpolated():
    output = RuleOutput(rule=RuleName.LEAD_SPACE, state=RuleState.INDETERMINATE, reason="frontal_yaw")
    rendered = render(output)
    assert rendered.message == "Can't judge leading space (frontal_yaw)."


def test_odd_count_violated_exact_message():
    output = RuleOutput(
        rule=RuleName.ODD_COUNT,
        state=RuleState.VIOLATED,
        reason="even_count_fixable_by_exclude",
        correction={"action": "exclude", "direction": "in", "edge": "left"},
    )
    rendered = render(output)
    assert rendered.message == "Zoom in to exclude the animal near the left edge and fix the count."


def test_render_mode_notice_uses_mode_and_reason():
    members = (make_subject(bbox=(0, 0, 100, 100)),)
    subject_set = make_subject_set(members, mode=SubjectMode.SINGLE)
    subject_set.mode_reason  # sanity: field exists
    notice = render_mode_notice(subject_set)
    assert "single" in notice
    assert subject_set.mode_reason in notice


def test_missing_correction_field_raises_loudly():
    output = RuleOutput(rule=RuleName.LIMB_CLIP, state=RuleState.VIOLATED, reason="limb_clipped", correction={})
    with pytest.raises(KeyError):
        render(output)
