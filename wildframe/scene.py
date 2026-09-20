"""Pure scene resolver: classifies a frame's detections into a SubjectMode and builds the
composite SubjectSet the rules consume. No learned components - filtering, primary selection,
mode classification, and reliability gating are all deterministic given config thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wildframe.contracts import Detection, Member, Orientation, Subject, SubjectMode, SubjectSet
from wildframe.orientation import compute_orientation


@dataclass
class SceneState:
    """Carried across frames for mode hysteresis - separate from, and stricter than, the
    per-rule hysteresis in smoothing.py."""

    committed_mode: SubjectMode = SubjectMode.NONE
    pending_mode: SubjectMode | None = None
    pending_count: int = 0


def _bbox_area(bbox: np.ndarray) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_center_x(bbox: np.ndarray) -> float:
    return float((bbox[0] + bbox[2]) / 2.0)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 1e-9 else 0.0


def _union_bbox(members: tuple[Member, ...]) -> np.ndarray | None:
    if not members:
        return None
    boxes = np.stack([m.bbox_xyxy for m in members])
    return np.array(
        [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], dtype=np.float32
    )


def _select_primary(members: tuple[Member, ...]) -> Member | None:
    if not members:
        return None

    def score(m: Member) -> float:
        mean_kpt_conf = float(np.mean(m.keypoints[:, 2])) if m.keypoints.size else 0.0
        return _bbox_area(m.bbox_xyxy) * mean_kpt_conf

    best_score = max(score(m) for m in members)
    tied = [m for m in members if score(m) == best_score]
    if len(tied) == 1:
        return tied[0]

    def tie_key(m: Member):
        track_key = m.track_id if m.track_id is not None else float("inf")
        return (track_key, _bbox_center_x(m.bbox_xyxy))

    return min(tied, key=tie_key)


def _classify_mode(members: tuple[Member, ...], primary: Member | None, config: dict) -> tuple[SubjectMode, str]:
    n = len(members)
    if n == 0:
        return SubjectMode.NONE, "no members"
    if n == 1:
        return SubjectMode.SINGLE, "1 of 1 member"

    dominance_ratio = float(config.get("dominance_ratio", 2.5))
    areas = sorted((_bbox_area(m.bbox_xyxy) for m in members), reverse=True)
    primary_area = _bbox_area(primary.bbox_xyxy)
    second_area = areas[1] if len(areas) > 1 else 0.0
    if second_area <= 0 or primary_area >= dominance_ratio * second_area:
        return SubjectMode.DOMINANT, f"primary {dominance_ratio:.1f}x larger than next"

    if n == 2:
        a, b = members
        a_center_x, b_center_x = _bbox_center_x(a.bbox_xyxy), _bbox_center_x(b.bbox_xyxy)
        a_faces_b = _faces_toward(a, b_center_x)
        b_faces_a = _faces_toward(b, a_center_x)
        if a_faces_b and b_faces_a:
            return SubjectMode.FACING_PAIR, "2 of 2 mutually facing"

    agreement_fraction = float(config.get("agreement_fraction", 0.75))
    directional = [m for m in members if m.orientation and m.orientation.yaw_state in ("left", "right")]
    if len(directional) == n:
        left_count = sum(1 for m in directional if m.orientation.yaw_state == "left")
        right_count = n - left_count
        majority_count = max(left_count, right_count)
        majority_dir = "left" if left_count >= right_count else "right"
        if n > 0 and majority_count / n >= agreement_fraction:
            return SubjectMode.GROUP_ALIGNED, f"{majority_count} of {n} facing {majority_dir}"

    return SubjectMode.MIXED, f"{n} members, no shared alignment"


def _faces_toward(member: Member, other_center_x: float) -> bool:
    orientation = member.orientation
    if orientation is None or orientation.yaw_state not in ("left", "right"):
        return False
    self_center_x = _bbox_center_x(member.bbox_xyxy)
    if orientation.yaw_state == "left":
        return other_center_x < self_center_x
    return other_center_x > self_center_x


def _count_reliable(members: tuple[Member, ...], config: dict) -> bool:
    min_det_conf = float(config.get("min_det_conf", 0.25))
    conf_margin = float(config.get("conf_margin", 0.05))
    overlap_iou_max = float(config.get("overlap_iou_max", 0.3))

    for m in members:
        if m.det_conf - min_det_conf < conf_margin:
            return False

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if _bbox_iou(members[i].bbox_xyxy, members[j].bbox_xyxy) > overlap_iou_max:
                return False
    return True


def _agreed_orientation(members: tuple[Member, ...], mode: SubjectMode) -> Orientation | None:
    if mode != SubjectMode.GROUP_ALIGNED:
        return None
    directional = [m for m in members if m.orientation and m.orientation.yaw_state in ("left", "right")]
    if not directional:
        return None
    left = [m for m in directional if m.orientation.yaw_state == "left"]
    right = [m for m in directional if m.orientation.yaw_state == "right"]
    majority = left if len(left) >= len(right) else right
    confidence = float(np.mean([m.orientation.confidence for m in majority]))
    yaw_state = majority[0].orientation.yaw_state
    vector = np.mean([m.orientation.facing_vector for m in majority], axis=0)
    return Orientation(facing_vector=vector, yaw_state=yaw_state, confidence=confidence)


def _build_subject_set(
    mode: SubjectMode,
    members: tuple[Member, ...],
    primary: Member | None,
    count_reliable: bool,
    mode_reason: str,
    frame_dims: tuple[int, int],
) -> SubjectSet:
    union_bbox = _union_bbox(members)
    agreed_orientation = _agreed_orientation(members, mode)
    return SubjectSet(
        mode=mode,
        members=members,
        primary=primary,
        count=len(members),
        count_reliable=count_reliable,
        union_bbox=union_bbox,
        agreed_orientation=agreed_orientation,
        mode_reason=mode_reason,
        frame_dims=frame_dims,
    )


def resolve_scene(
    detections: list[Detection],
    frame_dims: tuple[int, int],
    config: dict,
    prev_state: SceneState | None = None,
) -> tuple[SubjectSet, SceneState]:
    """(detections, frame_dims, config, prev_state) -> (SubjectSet, new_state).

    config: the full rules.yaml dict (uses config['scene'] plus config['orientation'] for
    per-member orientation computation).
    """
    scene_cfg = config.get("scene", {})
    prev_state = prev_state or SceneState()

    min_det_conf = float(scene_cfg.get("min_det_conf", 0.25))
    min_bbox_area_frac = float(scene_cfg.get("min_bbox_area_frac", 0))
    w, h = frame_dims
    min_bbox_area = min_bbox_area_frac * w * h

    filtered = [d for d in detections if d.det_conf >= min_det_conf and _bbox_area(d.bbox_xyxy) >= min_bbox_area]

    members = tuple(
        Subject(
            bbox_xyxy=d.bbox_xyxy,
            keypoints=d.keypoints,
            det_conf=d.det_conf,
            orientation=compute_orientation(d, config),
            track_id=d.track_id,
            frame_dims=frame_dims,
        )
        for d in filtered
    )
    members = tuple(sorted(members, key=lambda m: _bbox_center_x(m.bbox_xyxy)))

    primary = _select_primary(members)
    raw_mode, mode_reason = _classify_mode(members, primary, scene_cfg)
    count_reliable = _count_reliable(members, scene_cfg) if members else True

    mode_enter_frames = int(scene_cfg.get("mode_enter_frames", 1))
    new_state = SceneState(
        committed_mode=prev_state.committed_mode,
        pending_mode=prev_state.pending_mode,
        pending_count=prev_state.pending_count,
    )

    if raw_mode == new_state.committed_mode:
        new_state.pending_mode = None
        new_state.pending_count = 0
    elif new_state.committed_mode == SubjectMode.NONE and raw_mode != SubjectMode.NONE:
        # Acquiring a subject for the first time (or after a full loss) commits immediately -
        # hysteresis exists to stop mode LABELS from flickering during an ongoing scene, not to
        # delay showing any guidance at all once an animal is actually detected. This also keeps
        # a single-animal frame's output identical from the very first frame it's seen.
        new_state.committed_mode = raw_mode
        new_state.pending_mode = None
        new_state.pending_count = 0
    elif raw_mode == new_state.pending_mode:
        new_state.pending_count += 1
        if new_state.pending_count >= mode_enter_frames:
            new_state.committed_mode = raw_mode
            new_state.pending_mode = None
            new_state.pending_count = 0
    else:
        new_state.pending_mode = raw_mode
        new_state.pending_count = 1
        if mode_enter_frames <= 1:
            new_state.committed_mode = raw_mode
            new_state.pending_mode = None
            new_state.pending_count = 0

    emitted_mode = new_state.committed_mode
    subject_set = _build_subject_set(emitted_mode, members, primary, count_reliable, mode_reason, frame_dims)
    return subject_set, new_state


def composite_subject(subject_set: SubjectSet) -> Subject | None:
    """The single composite Subject that lead_space/eye_thirds evaluate, built per mode:
    SINGLE/DOMINANT -> the primary as-is. GROUP_ALIGNED/FACING_PAIR/MIXED -> union bbox,
    primary's keypoints (eye source is always the primary), and orientation that is the
    agreed direction for GROUP_ALIGNED or None otherwise (gates lead_space to indeterminate).
    None if there's no subject at all (NONE mode / no primary)."""
    if subject_set.mode == SubjectMode.NONE or subject_set.primary is None:
        return None
    if subject_set.mode in (SubjectMode.SINGLE, SubjectMode.DOMINANT):
        return subject_set.primary

    orientation = subject_set.agreed_orientation if subject_set.mode == SubjectMode.GROUP_ALIGNED else None
    return Subject(
        bbox_xyxy=subject_set.union_bbox,
        keypoints=subject_set.primary.keypoints,
        det_conf=subject_set.primary.det_conf,
        orientation=orientation,
        track_id=subject_set.primary.track_id,
        frame_dims=subject_set.frame_dims,
    )


def members_for_clipping(subject_set: SubjectSet) -> tuple[Member, ...]:
    """Which members limb_clip checks: just the primary in SINGLE/DOMINANT mode, all members
    once the frame is treated as a group (GROUP_ALIGNED/FACING_PAIR/MIXED)."""
    if subject_set.mode in (SubjectMode.SINGLE, SubjectMode.DOMINANT):
        return (subject_set.primary,) if subject_set.primary is not None else ()
    return subject_set.members


def referent_label(subject_set: SubjectSet, member_index: int | None = None) -> str:
    """Template-ready phrase naming what a rule output is about, e.g. "the animal", "the left
    animal", "the group" - derived purely from position in subject_set.members (already sorted
    left-to-right by scene.py), never from species/identity."""
    n = len(subject_set.members)
    if n == 0:
        return "the frame"
    if n == 1:
        return "the animal"

    if member_index is None:
        return "the group"

    if member_index == 0:
        return "the left animal"
    if member_index == n - 1:
        return "the right animal"
    return "one of the middle animals"
