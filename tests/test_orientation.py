import numpy as np
import pytest

from wildframe.orientation import compute_orientation
from wildframe.pose.skeleton import KP
from tests.synthetic import (
    FRAME_W,
    FRAME_H,
    make_detection,
    facing_left_pose,
    facing_right_pose,
    frontal_pose,
)

CONFIG = {"min_axis_conf": 0.3, "frontal_yaw_max_deg": 25}


def test_facing_left():
    det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    o = compute_orientation(det, CONFIG)
    assert o.facing_vector is not None
    assert o.yaw_state == "left"


def test_facing_right():
    det = make_detection(facing_right_pose(FRAME_W / 2, FRAME_H / 2))
    o = compute_orientation(det, CONFIG)
    assert o.facing_vector is not None
    assert o.yaw_state == "right"


def test_frontal():
    det = make_detection(frontal_pose(FRAME_W / 2, FRAME_H / 2))
    o = compute_orientation(det, CONFIG)
    assert o.yaw_state == "frontal"


def test_head_low_confidence_falls_back_to_torso_axis():
    det = make_detection(
        {
            KP.NOSE: (300, 200, 0.0),
            KP.NECK: (320, 200, 0.1),
            KP.ROOT_OF_TAIL: (400, 210, 0.9),
        }
    )
    o = compute_orientation(det, CONFIG)
    assert o.facing_vector is not None
    expected = (np.array([320, 200]) - np.array([400, 210]))
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(o.facing_vector, expected, atol=1e-5)


def test_all_axes_low_confidence_gives_unknown():
    det = make_detection(
        {
            KP.NOSE: (300, 200, 0.1),
            KP.NECK: (320, 200, 0.1),
            KP.ROOT_OF_TAIL: (400, 210, 0.1),
        }
    )
    o = compute_orientation(det, CONFIG)
    assert o.facing_vector is None
    assert o.yaw_state == "unknown"


def test_boundary_just_below_min_axis_conf_triggers_fallback():
    det = make_detection(
        {
            KP.NOSE: (300, 200, 0.29),
            KP.NECK: (320, 200, 0.29),
            KP.ROOT_OF_TAIL: (400, 210, 0.9),
        }
    )
    o = compute_orientation(det, CONFIG)
    # head_conf = 0.29 < 0.3 => falls back to torso axis (tail/neck), which succeeds.
    assert o.facing_vector is not None
    expected = (np.array([320, 200]) - np.array([400, 210]))
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(o.facing_vector, expected, atol=1e-5)


def test_confidence_is_actual_mean_of_keypoints_used():
    det = make_detection(
        {
            KP.NOSE: (250, 200, 0.6),
            KP.NECK: (320, 200, 0.8),
        }
    )
    o = compute_orientation(det, CONFIG)
    assert o.confidence == pytest.approx((0.6 + 0.8) / 2.0)
