from wildframe.contracts import RuleState
from wildframe.rules.centrality import CentralityRule
from tests.synthetic import FRAME_W, FRAME_H, make_subject, make_subject_set_single, make_subject_set_none

CONFIG = {"center_tolerance_frac": 0.12}


def test_centrality_satisfied_subject_near_center():
    rule = CentralityRule(CONFIG)
    cx, cy = FRAME_W / 2, FRAME_H / 2
    subject = make_subject(bbox=(cx - 40, cy - 40, cx + 40, cy + 40))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


def test_centrality_violated_subject_in_corner():
    rule = CentralityRule(CONFIG)
    subject = make_subject(bbox=(0, 0, 100, 100))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    assert out.correction["direction_x"] == "right"
    assert out.correction["direction_y"] == "down"


def test_centrality_gate_no_subject():
    rule = CentralityRule(CONFIG)
    out = rule.run(make_subject_set_none())
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "no_subject"
