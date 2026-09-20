"""Synthetic Detection/Subject builders for deterministic rule/orientation/smoothing tests.
No model, no image I/O - pure numpy construction of keypoints in a fixed frame size."""

from __future__ import annotations

import numpy as np

from wildframe.contracts import Detection, Subject, SubjectMode, SubjectSet
from wildframe.pose.skeleton import KP, LIMBS, NUM_KEYPOINTS

FRAME_W, FRAME_H = 640, 480


def make_keypoints(overrides: dict[KP, tuple[float, float, float]], default_conf: float = 0.9) -> np.ndarray:
    """Builds a (17,3) array. Un-overridden keypoints default to a plausible standing-profile
    template centered in frame, at default_conf, so tests only need to override what they check."""
    cx, cy = FRAME_W / 2, FRAME_H / 2
    base = {
        KP.LEFT_EYE: (cx - 60, cy - 40, default_conf),
        KP.RIGHT_EYE: (cx - 55, cy - 38, default_conf),
        KP.NOSE: (cx - 80, cy - 30, default_conf),
        KP.NECK: (cx - 30, cy - 20, default_conf),
        KP.ROOT_OF_TAIL: (cx + 60, cy - 10, default_conf),
        KP.LEFT_SHOULDER: (cx - 20, cy, default_conf),
        KP.LEFT_ELBOW: (cx - 20, cy + 30, default_conf),
        KP.LEFT_FRONT_PAW: (cx - 20, cy + 60, default_conf),
        KP.RIGHT_SHOULDER: (cx - 15, cy + 2, default_conf),
        KP.RIGHT_ELBOW: (cx - 15, cy + 32, default_conf),
        KP.RIGHT_FRONT_PAW: (cx - 15, cy + 62, default_conf),
        KP.LEFT_HIP: (cx + 50, cy, default_conf),
        KP.LEFT_KNEE: (cx + 50, cy + 30, default_conf),
        KP.LEFT_BACK_PAW: (cx + 50, cy + 60, default_conf),
        KP.RIGHT_HIP: (cx + 55, cy + 2, default_conf),
        KP.RIGHT_KNEE: (cx + 55, cy + 32, default_conf),
        KP.RIGHT_BACK_PAW: (cx + 55, cy + 62, default_conf),
    }
    base.update(overrides)
    arr = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for kp, (x, y, c) in base.items():
        arr[int(kp)] = (x, y, c)
    return arr


def make_detection(
    keypoint_overrides: dict | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    det_conf: float = 0.9,
) -> Detection:
    kpts = make_keypoints(keypoint_overrides or {})
    if bbox is None:
        visible = kpts[kpts[:, 2] > 0]
        margin = 15
        if visible.shape[0] == 0:
            bbox = (0.0, 0.0, float(FRAME_W), float(FRAME_H))
        else:
            xs, ys = visible[:, 0], visible[:, 1]
            bbox = (xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin)
    return Detection(bbox_xyxy=np.array(bbox, dtype=np.float32), keypoints=kpts, det_conf=det_conf)


def make_subject(
    keypoint_overrides: dict | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    orientation=None,
    frame_dims: tuple[int, int] = (FRAME_W, FRAME_H),
) -> Subject:
    det = make_detection(keypoint_overrides, bbox)
    return Subject(
        bbox_xyxy=det.bbox_xyxy,
        keypoints=det.keypoints,
        det_conf=det.det_conf,
        orientation=orientation,
        frame_dims=frame_dims,
    )


def make_subject_set_single(subject: Subject) -> SubjectSet:
    """Wraps one Subject into the SINGLE-mode SubjectSet scene.resolve_scene would produce
    for a lone detection - the shape rules actually consume post Phase-B migration."""
    return SubjectSet(
        mode=SubjectMode.SINGLE,
        members=(subject,),
        primary=subject,
        count=1,
        count_reliable=True,
        union_bbox=subject.bbox_xyxy,
        agreed_orientation=None,
        mode_reason="1 of 1 member",
        frame_dims=subject.frame_dims,
    )


def make_subject_set(
    members: tuple,
    mode: SubjectMode = SubjectMode.MIXED,
    count_reliable: bool = True,
    frame_dims: tuple[int, int] = (FRAME_W, FRAME_H),
) -> SubjectSet:
    """Generic multi-member SubjectSet builder for rule tests that need full control over
    member positions/bboxes (e.g. odd_count) without going through the real scene resolver."""
    if members:
        boxes = np.stack([m.bbox_xyxy for m in members])
        union_bbox = np.array(
            [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], dtype=np.float32
        )
    else:
        union_bbox = None
    primary = members[0] if members else None
    return SubjectSet(
        mode=mode,
        members=tuple(members),
        primary=primary,
        count=len(members),
        count_reliable=count_reliable,
        union_bbox=union_bbox,
        agreed_orientation=None,
        mode_reason="test multi-member set",
        frame_dims=frame_dims,
    )


def make_subject_set_none() -> SubjectSet:
    return SubjectSet(
        mode=SubjectMode.NONE,
        members=(),
        primary=None,
        count=0,
        count_reliable=True,
        union_bbox=None,
        agreed_orientation=None,
        mode_reason="no members",
        frame_dims=(FRAME_W, FRAME_H),
    )


def facing_left_pose(cx: float, cy: float) -> dict:
    """Canonical profile pose with nose to the LEFT of neck."""
    return {
        KP.NOSE: (cx - 50, cy - 5, 0.9),
        KP.NECK: (cx, cy, 0.9),
        KP.ROOT_OF_TAIL: (cx + 80, cy, 0.9),
        KP.LEFT_EYE: (cx - 40, cy - 10, 0.85),
        KP.RIGHT_EYE: (cx - 38, cy - 12, 0.3),  # occluded side, low confidence
    }


def facing_right_pose(cx: float, cy: float) -> dict:
    """Canonical profile pose with nose to the RIGHT of neck."""
    return {
        KP.NOSE: (cx + 50, cy - 5, 0.9),
        KP.NECK: (cx, cy, 0.9),
        KP.ROOT_OF_TAIL: (cx - 80, cy, 0.9),
        KP.LEFT_EYE: (cx + 38, cy - 12, 0.3),  # occluded side, low confidence
        KP.RIGHT_EYE: (cx + 40, cy - 10, 0.85),
    }


def frontal_pose(cx: float, cy: float) -> dict:
    """Eyes symmetric around nose, both confident, nose below neck - facing the camera."""
    return {
        KP.NOSE: (cx, cy + 40, 0.9),
        KP.NECK: (cx, cy, 0.9),
        KP.ROOT_OF_TAIL: (cx, cy - 80, 0.9),
        KP.LEFT_EYE: (cx - 12, cy + 30, 0.9),
        KP.RIGHT_EYE: (cx + 12, cy + 30, 0.9),
    }


def clip_limb_at_edge(pose: dict, limb: str, edge: str, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> dict:
    """Takes a base pose dict and pushes the distal keypoint of the given limb name
    to sit exactly on the given frame edge ('left','right','top','bottom')."""
    pose = dict(pose)
    _, _, distal = LIMBS[limb]
    x, y, c = pose.get(distal, (frame_w / 2, frame_h / 2, 0.9))[:3]
    if edge == "left":
        x = 0
    elif edge == "right":
        x = frame_w
    elif edge == "top":
        y = 0
    elif edge == "bottom":
        y = frame_h
    pose[distal] = (x, y, c)
    return pose
