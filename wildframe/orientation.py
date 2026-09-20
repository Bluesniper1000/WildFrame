# stage 1: facing vector, frontal-yaw gate

from __future__ import annotations

import math

import numpy as np

from wildframe.contracts import Detection, Orientation
from wildframe.pose.skeleton import KP


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_orientation(det: Detection, config: dict) -> Orientation:
    """Derive a 2D facing vector from keypoints and classify frontal/left/right/unknown.

    Primary heading axis: neck -> nose (the direction the head points, i.e. where the
    animal is looking/moving). Falls back to root_of_tail -> neck (torso axis) if the
    head keypoints are too low-confidence, then to "unknown" if neither is usable.
    """
    orient_cfg = config.get("orientation", config)
    min_axis_conf = float(orient_cfg.get("min_axis_conf", 0.3))
    frontal_yaw_max_deg = float(orient_cfg.get("frontal_yaw_max_deg", 25))

    kpts = det.keypoints  # (17, 3)
    nose, neck, tail = kpts[KP.NOSE], kpts[KP.NECK], kpts[KP.ROOT_OF_TAIL]

    facing_vector = None
    confidence = 0.0

    head_conf = (nose[2] + neck[2]) / 2.0
    if head_conf >= min_axis_conf:
        facing_vector = _unit(nose[:2] - neck[:2])
        confidence = float(head_conf)
    else:
        torso_conf = (tail[2] + neck[2]) / 2.0
        if torso_conf >= min_axis_conf:
            facing_vector = _unit(neck[:2] - tail[:2])
            confidence = float(torso_conf)

    if facing_vector is None:
        return Orientation(facing_vector=None, yaw_state="unknown", confidence=0.0)

    # Frontal check: with both eyes visible and confident, and the nose sitting close to
    # the eye-eye midpoint, the animal is facing (near) the camera rather than in profile.
    left_eye, right_eye = kpts[KP.LEFT_EYE], kpts[KP.RIGHT_EYE]
    is_frontal_by_eyes = False
    if left_eye[2] >= min_axis_conf and right_eye[2] >= min_axis_conf:
        eye_mid = (left_eye[:2] + right_eye[:2]) / 2.0
        inter_eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
        if inter_eye_dist > 1e-6:
            offset_frac = abs(nose[0] - eye_mid[0]) / inter_eye_dist
            is_frontal_by_eyes = offset_frac < 0.35

    # facing_vector.x near 0 (mostly vertical) => no strong lateral lean => ambiguous/frontal-ish.
    angle_from_vertical_deg = math.degrees(math.asin(min(1.0, abs(facing_vector[0]))))
    is_frontal_by_heading = angle_from_vertical_deg < frontal_yaw_max_deg

    if is_frontal_by_eyes or is_frontal_by_heading:
        yaw_state = "frontal"
    elif facing_vector[0] < 0:
        yaw_state = "left"
    else:
        yaw_state = "right"

    return Orientation(facing_vector=facing_vector, yaw_state=yaw_state, confidence=confidence)
