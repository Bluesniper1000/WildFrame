# RuleOutput struct + RuleState enum

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RuleState(Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INDETERMINATE = "indeterminate"


class RuleName(str, Enum):
    LEAD_SPACE = "lead_space"
    EYE_THIRDS = "eye_thirds"
    LIMB_CLIP = "limb_clip"
    ODD_COUNT = "odd_count"
    CENTRALITY = "centrality"
    READY = "ready"   # not a composition rule - an aggregate "no blocking violations" signal


@dataclass(frozen=True)
class Detection:
    """One raw per-frame detection out of pose/backend.py, in ORIGINAL frame pixel space."""

    bbox_xyxy: np.ndarray   # shape (4,), float32
    keypoints: np.ndarray   # shape (17, 3): [x_px, y_px, kpt_conf], index-aligned to pose.skeleton.KP
    det_conf: float
    track_id: int | None = None   # assigned by wildframe.tracking.ByteTracker; None until tracking runs


@dataclass(frozen=True)
class Orientation:
    """Output of orientation.py for one Detection. None fields mean "could not be determined"."""

    facing_vector: np.ndarray | None   # shape (2,), unit vector, image-plane (x right+, y down+)
    yaw_state: str                     # one of "frontal", "left", "right", "unknown"
    confidence: float                  # mean confidence of the keypoints used to build the vector


@dataclass(frozen=True)
class Subject:
    """The composite thing rules evaluate against. In Phase A this always wraps exactly one
    Detection 1:1. Field names deliberately match what Phase B's `Member`/`SubjectSet.primary`
    will need, so rules can later consume a composite subject with minimal change."""

    bbox_xyxy: np.ndarray
    keypoints: np.ndarray
    det_conf: float
    orientation: Orientation | None
    track_id: int | None = None
    frame_dims: tuple[int, int] = (0, 0)   # (width, height) of the ORIGINAL frame


@dataclass(frozen=True)
class RuleOutput:
    rule: RuleName
    state: RuleState
    reason: str
    message: str | None = None
    severity: float = 0.0
    correction: dict | None = None
    referent: str | None = None   # reserved for Phase B (PLAN_multi_animal.md Step 7); always None in Phase A
    debug: dict = field(default_factory=dict)


class SubjectMode(Enum):
    NONE = "none"
    SINGLE = "single"
    DOMINANT = "dominant"
    GROUP_ALIGNED = "group_aligned"
    FACING_PAIR = "facing_pair"
    MIXED = "mixed"


# A Phase-B "member" is structurally identical to Phase A's Subject (see Subject's docstring) -
# no separate class needed, no rewrite required to move from single-animal to multi-animal.
Member = Subject


@dataclass(frozen=True)
class SubjectSet:
    """Composite view of the scene built by scene.py, consumed by the rules in place of a
    single Subject. In SINGLE/DOMINANT mode this degenerates to exactly what Phase A produced
    (one member, primary == that member, union_bbox == primary.bbox_xyxy)."""

    mode: SubjectMode
    members: tuple[Member, ...]        # post-filter, sorted left-to-right by bbox center x
    primary: Member | None
    count: int
    count_reliable: bool
    union_bbox: np.ndarray | None      # (4,) xyxy covering all members; None if count == 0
    agreed_orientation: Orientation | None
    mode_reason: str                   # template-ready, e.g. "3 of 3 facing left"
    frame_dims: tuple[int, int] = (0, 0)
