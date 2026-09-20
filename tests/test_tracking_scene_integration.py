"""Integration check for PLAN_multi_animal.md Step 4's verify condition: with a synthetic
sequence where two animals cross, the primary stays stable (tracked by identity, not by
position/order), and the single-animal case is unaffected."""

import numpy as np

from wildframe.contracts import Detection
from wildframe.scene import resolve_scene
from wildframe.tracking import ByteTracker
from tests.synthetic import FRAME_W, FRAME_H, make_detection, facing_left_pose, facing_right_pose

CONFIG = {
    "orientation": {"min_axis_conf": 0.3, "frontal_yaw_max_deg": 25},
    "scene": {
        "min_det_conf": 0.25,
        "min_bbox_area_frac": 0,
        "dominance_ratio": 2.5,
        "agreement_fraction": 0.75,
        "overlap_iou_max": 0.9,  # crossing boxes overlap heavily; keep count "reliable" for this test
        "conf_margin": 0.05,
        "mode_enter_frames": 1,
    },
}
TRACKER_CONFIG = {"track_iou_threshold": 0.3, "track_max_age": 5}


def _det_with_bbox(base_pose_fn, cx, cy, bbox):
    det = make_detection(base_pose_fn(cx, cy), bbox=bbox)
    return det


def test_primary_identity_survives_a_crossing():
    tracker = ByteTracker(TRACKER_CONFIG)

    # A starts left (bigger => primary), moving right. B starts right (smaller), moving left.
    # They cross around the midpoint; A must remain "primary" throughout by track identity,
    # not by which detection happens to come first / be positioned where in a given frame.
    # Step size is kept well below the box width so consecutive frames overlap enough for
    # IoU-based association (real video motion between frames is similarly small).
    a_positions = list(range(50, 460, 30))
    b_positions = list(range(450, 40, -30))

    primary_track_ids = []
    a_track_id = None

    for i, (ax, bx) in enumerate(zip(a_positions, b_positions)):
        det_a = _det_with_bbox(facing_right_pose, ax, 200, bbox=(ax - 40, 150, ax + 40, 280))
        det_b = _det_with_bbox(facing_left_pose, bx, 200, bbox=(bx - 40, 150, bx + 40, 280))

        boxes = [det_a.bbox_xyxy, det_b.bbox_xyxy]
        track_ids = tracker.update(boxes)

        det_a = Detection(bbox_xyxy=det_a.bbox_xyxy, keypoints=det_a.keypoints, det_conf=det_a.det_conf, track_id=track_ids[0])
        det_b = Detection(bbox_xyxy=det_b.bbox_xyxy, keypoints=det_b.keypoints, det_conf=det_b.det_conf, track_id=track_ids[1])

        if i == 0:
            a_track_id = det_a.track_id

        subject_set, _ = resolve_scene([det_a, det_b], (FRAME_W, FRAME_H), CONFIG)
        primary_track_ids.append(subject_set.primary.track_id)

    assert all(tid == a_track_id for tid in primary_track_ids)


def test_single_animal_track_id_is_stable_and_primary():
    tracker = ByteTracker(TRACKER_CONFIG)
    det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))

    ids_seen = set()
    for _ in range(5):
        track_ids = tracker.update([det.bbox_xyxy])
        d = Detection(bbox_xyxy=det.bbox_xyxy, keypoints=det.keypoints, det_conf=det.det_conf, track_id=track_ids[0])
        subject_set, _ = resolve_scene([d], (FRAME_W, FRAME_H), CONFIG)
        assert subject_set.primary.track_id == track_ids[0]
        ids_seen.add(track_ids[0])

    assert len(ids_seen) == 1
