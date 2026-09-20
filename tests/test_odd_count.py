import numpy as np

from wildframe.contracts import RuleState, SubjectMode
from wildframe.rules.odd_count import OddCountRule
from tests.synthetic import FRAME_W, FRAME_H, make_subject, make_subject_set

CONFIG = {
    "ceiling": {"value": 6, "source": "test"},
    "edge_margin_frac": 20 / 640,
    "exclude_below_visible": 0.5,
}


def _member(bbox):
    return make_subject(bbox=bbox)


def test_single_animal_is_indeterminate():
    rule = OddCountRule(CONFIG)
    members = (_member((100, 100, 200, 200)),)
    subject_set = make_subject_set(members, mode=SubjectMode.SINGLE)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "single_animal"


def test_facing_pair_is_indeterminate():
    rule = OddCountRule(CONFIG)
    members = (_member((100, 100, 200, 200)), _member((400, 100, 500, 200)))
    subject_set = make_subject_set(members, mode=SubjectMode.FACING_PAIR)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "facing_pair_mode"


def test_odd_count_is_satisfied():
    rule = OddCountRule(CONFIG)
    members = (
        _member((100, 100, 180, 220)),
        _member((300, 100, 380, 220)),
        _member((500, 100, 580, 220)),
    )
    subject_set = make_subject_set(members, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.SATISFIED
    assert out.reason == "odd_count"


def test_even_count_with_clipped_edge_member_is_violated_exclude():
    rule = OddCountRule(CONFIG)
    target = _member((-60, 100, 20, 220))  # visible_fraction = 0.25, clipped at left edge
    others = (
        _member((100, 100, 180, 220)),
        _member((300, 100, 380, 220)),
        _member((500, 100, 580, 220)),
    )
    subject_set = make_subject_set((target,) + others, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.VIOLATED
    assert out.correction["action"] == "exclude"
    assert out.correction["edge"] == "left"
    assert out.correction["member_index"] == 0


def test_even_count_with_mostly_visible_edge_member_is_indeterminate_include():
    rule = OddCountRule(CONFIG)
    target = _member((5, 100, 105, 220))  # near left edge (dist=5 <= margin=20), fully visible
    others = (
        _member((300, 100, 380, 220)),
        _member((500, 100, 580, 220)),
        _member((200, 300, 280, 400)),
    )
    subject_set = make_subject_set((target,) + others, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "shift_does_not_fix_parity"


def test_even_count_with_no_edge_member_is_indeterminate():
    rule = OddCountRule(CONFIG)
    members = (
        _member((100, 100, 180, 220)),
        _member((300, 100, 380, 220)),
        _member((500, 100, 560, 220)),
        _member((200, 300, 280, 400)),
    )
    subject_set = make_subject_set(members, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "no_reframing_fixes_count"


def test_even_count_exclusion_that_clips_another_member_is_indeterminate():
    rule = OddCountRule(CONFIG)
    target = _member((-60, 100, 20, 220))  # excluding this needs crop x >= 20
    other_also_near_left = _member((10, 300, 90, 400))  # x1=10 < 20, would be clipped by that crop
    rest = (
        _member((300, 100, 380, 220)),
        _member((500, 100, 580, 220)),
    )
    subject_set = make_subject_set((target, other_also_near_left) + rest, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "shift_would_clip_another_member"


def test_count_unreliable_is_indeterminate():
    rule = OddCountRule(CONFIG)
    members = (
        _member((100, 100, 180, 220)),
        _member((300, 100, 380, 220)),
        _member((500, 100, 560, 220)),
        _member((200, 300, 280, 400)),
    )
    subject_set = make_subject_set(members, mode=SubjectMode.MIXED, count_reliable=False)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "count_unreliable"


def test_count_exceeds_ceiling_is_indeterminate():
    rule = OddCountRule(CONFIG)
    members = tuple(_member((50 + i * 80, 100, 120 + i * 80, 200)) for i in range(7))
    subject_set = make_subject_set(members, mode=SubjectMode.MIXED)
    out = rule.run(subject_set)
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "count_exceeds_ceiling"
