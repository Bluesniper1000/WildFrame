import numpy as np
import pytest

from wildframe.contracts import Orientation, RuleState, Subject
from wildframe.pose.skeleton import KP
from wildframe.rules.eye_thirds import EyeThirdsRule
from wildframe.rules.lead_space import LeadSpaceRule
from wildframe.rules.limb_clip import LimbClipRule
from tests.synthetic import (
    FRAME_W,
    FRAME_H,
    make_subject,
    make_subject_set_single,
    clip_limb_at_edge,
    facing_left_pose,
)


# ---------------------------------------------------------------------------
# lead_space
# ---------------------------------------------------------------------------

LEAD_SPACE_CONFIG = {"min_lead_fraction": 0.15, "ideal_lead_fraction": 0.30}


def _orientation(yaw_state, vector=(-1.0, 0.0)):
    return Orientation(facing_vector=np.array(vector), yaw_state=yaw_state, confidence=0.9)


def test_lead_space_satisfied_facing_left_ample_space():
    rule = LeadSpaceRule(LEAD_SPACE_CONFIG)
    subject = make_subject(bbox=(0.5 * FRAME_W, 100, 0.9 * FRAME_W, 300), orientation=_orientation("left"))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


def test_lead_space_violated_flush_against_edge():
    rule = LeadSpaceRule(LEAD_SPACE_CONFIG)
    subject = make_subject(bbox=(0, 100, 0.3 * FRAME_W, 300), orientation=_orientation("left"))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    # Facing left, flush against the left edge: the camera must pan left (toward the gaze) so the
    # subject moves right in the frame and gains room ahead of it.
    assert out.correction["direction"] == "left"


def test_lead_space_indeterminate_frontal():
    rule = LeadSpaceRule(LEAD_SPACE_CONFIG)
    subject = make_subject(orientation=_orientation("frontal", vector=(0.0, 1.0)))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "frontal_yaw"


def test_lead_space_indeterminate_no_orientation():
    rule = LeadSpaceRule(LEAD_SPACE_CONFIG)
    subject = make_subject(orientation=None)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "no_orientation"


def test_lead_space_boundary_at_min_lead_fraction_is_satisfied():
    rule = LeadSpaceRule(LEAD_SPACE_CONFIG)
    x1 = 0.15 * FRAME_W
    subject = make_subject(bbox=(x1, 100, x1 + 200, 300), orientation=_orientation("left"))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


# ---------------------------------------------------------------------------
# eye_thirds
# ---------------------------------------------------------------------------

EYE_THIRDS_CONFIG = {"kpt_conf_threshold": 0.0, "thirds_tolerance_frac": 0.08}


def test_eye_thirds_satisfied_on_a_third():
    rule = EyeThirdsRule(EYE_THIRDS_CONFIG)
    tx, ty = FRAME_W / 3, FRAME_H / 3
    subject = make_subject({KP.LEFT_EYE: (tx, ty, 0.9), KP.RIGHT_EYE: (0, 0, 0.0)})
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


def test_eye_thirds_violated_at_center():
    rule = EyeThirdsRule(EYE_THIRDS_CONFIG)
    cx, cy = FRAME_W / 2, FRAME_H / 2
    subject = make_subject({KP.LEFT_EYE: (cx, cy, 0.9), KP.RIGHT_EYE: (0, 0, 0.0)})
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    assert out.debug["eye_used"] == "left"


def test_eye_thirds_indeterminate_both_eyes_invisible():
    rule = EyeThirdsRule(EYE_THIRDS_CONFIG)
    subject = make_subject({KP.LEFT_EYE: (100, 100, 0.0), KP.RIGHT_EYE: (100, 100, 0.0)})
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "eye_not_visible"


def test_eye_thirds_uses_visible_eye_in_profile():
    rule = EyeThirdsRule(EYE_THIRDS_CONFIG)
    tx, ty = 2 * FRAME_W / 3, 2 * FRAME_H / 3
    subject = make_subject({KP.LEFT_EYE: (0, 0, 0.0), KP.RIGHT_EYE: (tx, ty, 0.85)})
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED
    assert out.debug["eye_used"] == "right"


# ---------------------------------------------------------------------------
# limb_clip
# ---------------------------------------------------------------------------

LIMB_CLIP_CONFIG = {"edge_margin_frac": 2 / 640, "min_kpt_conf": 0.25}


def test_limb_clip_satisfied_no_limbs_near_edge():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    subject = make_subject(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


def test_limb_clip_violated_paw_at_edge():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    pose = clip_limb_at_edge(facing_left_pose(FRAME_W / 2, FRAME_H / 2), "left_front", "left")
    subject = make_subject(pose)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    assert out.correction["limb"] == "left_front"
    assert out.correction["edge"] == "left"


def test_limb_clip_violated_paw_off_frame_negative_coord():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    pose = facing_left_pose(FRAME_W / 2, FRAME_H / 2)
    pose[KP.LEFT_FRONT_PAW] = (-50, 100, 0.9)
    subject = make_subject(pose)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    assert out.correction["edge"] == "left"


def test_limb_clip_low_confidence_paw_is_skipped():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    pose = clip_limb_at_edge(facing_left_pose(FRAME_W / 2, FRAME_H / 2), "left_front", "left")
    pose[KP.LEFT_FRONT_PAW] = (pose[KP.LEFT_FRONT_PAW][0], pose[KP.LEFT_FRONT_PAW][1], 0.1)
    subject = make_subject(pose)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.SATISFIED


def test_limb_clip_deterministic_tie_break_order():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    pose = facing_left_pose(FRAME_W / 2, FRAME_H / 2)
    pose = clip_limb_at_edge(pose, "right_back", "bottom")
    pose = clip_limb_at_edge(pose, "left_front", "left")
    subject = make_subject(pose)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.VIOLATED
    assert out.correction["limb"] == "left_front"


def test_limb_clip_indeterminate_all_keypoints_zero_confidence():
    rule = LimbClipRule(LIMB_CLIP_CONFIG)
    pose = {kp: (0, 0, 0.0) for kp in KP}
    subject = make_subject(pose)
    out = rule.run(make_subject_set_single(subject))
    assert out.state == RuleState.INDETERMINATE
    assert out.reason == "no_visible_keypoints"
