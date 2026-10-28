"""Generate the golden fixtures a port of the rule engine (e.g. the mobile app) must reproduce.

  python -m golden.generate          # rewrites golden/cases/*.json and golden/tracker/*.json

Each decision case is a scripted multi-frame sequence of detections (synthetic keypoints, explicit
track ids) run through Pipeline.decide() with the current configs/rules.yaml; every frame records
the scene resolution, raw and smoothed rule outputs, the ready signal and the final instruction.
Tracker cases feed box sequences to ByteTracker with configs/model.yaml and record the track ids.
See golden/README.md for the format and the comparison rules. tests/test_golden.py fails when the
Python behaviour drifts from the committed fixtures - regenerate deliberately, then review the diff.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from tests.synthetic import FRAME_H, FRAME_W, facing_left_pose, facing_right_pose, frontal_pose, make_keypoints
from wildframe.contracts import Detection, RuleOutput
from wildframe.pipeline import Decision, Pipeline
from wildframe.pose.backend import REPO_ROOT
from wildframe.pose.skeleton import KP
from wildframe.tracking import ByteTracker

GOLDEN_DIR = Path(__file__).resolve().parent
FORMAT_VERSION = 1
# The Android app ships JSON copies of configs/*.yaml; regenerated here with the fixtures.
ANDROID_ASSETS = REPO_ROOT / "android/app/src/main/assets"


# --- JSON conversion ----------------------------------------------------------------------------

def to_jsonable(x):
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        return x
    if isinstance(x, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if is_dataclass(x):
        return {f.name: to_jsonable(getattr(x, f.name)) for f in fields(x)}
    return x


# --- Synthetic scene builders -------------------------------------------------------------------

_POSES = {"left": facing_left_pose, "right": facing_right_pose, "frontal": frontal_pose}


def animal(cx: float, cy: float, facing: str, scale: float = 1.0, overrides: dict | None = None) -> np.ndarray:
    """Full (17, 3) keypoints for one animal: the tests/synthetic template built at frame centre,
    scaled about that centre, then moved to (cx, cy). `overrides` replaces keypoints afterwards,
    in final frame coordinates."""
    c0 = (FRAME_W / 2, FRAME_H / 2)
    kpts = make_keypoints(_POSES[facing](*c0)).astype(np.float64)
    kpts[:, 0] = (kpts[:, 0] - c0[0]) * scale + cx
    kpts[:, 1] = (kpts[:, 1] - c0[1]) * scale + cy
    for kp, value in (overrides or {}).items():
        kpts[int(kp)] = value
    return kpts.astype(np.float32)


def det(kpts: np.ndarray, track_id: int | None, det_conf: float = 0.9, margin: float = 15.0) -> Detection:
    xs, ys = kpts[:, 0], kpts[:, 1]
    bbox = np.array([xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin], dtype=np.float32)
    return Detection(bbox_xyxy=bbox, keypoints=kpts, det_conf=float(det_conf), track_id=track_id)


def repeat(frame: list[Detection], n: int) -> list[list[Detection]]:
    return [list(frame) for _ in range(n)]


def _paw_at_bottom(cx, cy, facing, scale=1.0):
    k = animal(cx, cy, facing, scale)
    k[KP.LEFT_BACK_PAW] = (k[KP.LEFT_BACK_PAW, 0], FRAME_H, 0.9)
    return k


def scenarios() -> dict[str, tuple[str, list[list[Detection]]]]:
    """name -> (description, frames). Detections carry explicit track ids, so these cases test
    the decision path independently of the tracker (which has its own fixtures)."""
    good = det(animal(380, 220, "left"), 1)
    clipped = det(_paw_at_bottom(380, 220, "left"), 1)
    return {
        "single_left_room_ahead": ("One animal facing left with space in front; settles after hysteresis.",
                                   repeat([good], 14)),
        "single_right_no_lead_space": ("One animal facing right, pressed against the right edge.",
                                       repeat([det(animal(560, 240, "right"), 1)], 14)),
        "single_limb_clipped_bottom": ("Back paw on the bottom edge - limb_clip should win priority.",
                                       repeat([clipped], 14)),
        "single_frontal": ("Animal facing the camera - lead_space gates out (frontal yaw).",
                           repeat([det(animal(320, 220, "frontal"), 1)], 14)),
        "single_eyes_hidden": ("Both eyes at zero confidence - eye_thirds gates out, centrality decides the slot.",
                               repeat([det(animal(380, 220, "left", overrides={KP.LEFT_EYE: (320, 190, 0.0), KP.RIGHT_EYE: (322, 188, 0.0)}), 1)], 14)),
        "acquire_and_lose": ("Empty frames, an animal for 10 frames, then empty again (NONE <-> SINGLE notices).",
                             repeat([], 3) + repeat([good], 10) + repeat([], 3)),
        "hysteresis_flip": ("Good -> paw clipped for 8 frames -> good: rule states flip only after N agreeing frames.",
                            repeat([good], 10) + repeat([clipped], 8) + repeat([good], 10)),
        "flicker_suppressed": ("A 1-frame clipped glitch inside a good sequence must not change the output.",
                               repeat([good], 10) + repeat([clipped], 1) + repeat([good], 6)),
        "dominant_big_and_small": ("A large animal plus a small background one -> DOMINANT.",
                                   repeat([det(animal(300, 230, "left", 1.2), 1), det(animal(560, 120, "right", 0.35), 2)], 14)),
        "facing_pair": ("Two similar animals facing each other -> FACING_PAIR; odd_count gates out.",
                        repeat([det(animal(200, 240, "right", 0.6), 1), det(animal(450, 240, "left", 0.6), 2)], 14)),
        "two_facing_away": ("Two similar animals facing away from each other (even count, not a pair).",
                            repeat([det(animal(180, 240, "left", 0.6), 1), det(animal(470, 240, "right", 0.6), 2)], 14)),
        "three_aligned": ("Three similar animals all facing left -> GROUP_ALIGNED, odd count.",
                          repeat([det(animal(x, 250, "left", 0.45), i + 1) for i, x in enumerate((150, 330, 510))], 14)),
        "four_with_edge_member": ("Four animals, one half out of frame at the left edge -> odd_count reframing.",
                                  repeat([det(animal(x, 250, "left", 0.45), i + 1) for i, x in enumerate((15, 220, 380, 540))], 14)),
        "primary_switch": ("Track 1 leaves and an unrelated track 2 appears: per-track hysteresis restarts.",
                           repeat([good], 10) + repeat([det(animal(260, 260, "right"), 2)], 10)),
        "filtered_detections": ("A confident animal plus a low-confidence one and a tiny one, both filtered out.",
                                repeat([good, det(animal(120, 120, "right", 0.5), 2, det_conf=0.1),
                                        det(animal(600, 60, "left", 0.02), 3, margin=1.0)], 10)),
    }


# --- Serialisation of one frame -----------------------------------------------------------------

def _detection_json(d: Detection) -> dict:
    return to_jsonable({"bbox_xyxy": d.bbox_xyxy, "keypoints": d.keypoints, "det_conf": d.det_conf, "track_id": d.track_id})


def detection_from_json(d: dict) -> Detection:
    return Detection(bbox_xyxy=np.array(d["bbox_xyxy"], dtype=np.float32),
                     keypoints=np.array(d["keypoints"], dtype=np.float32),
                     det_conf=float(d["det_conf"]), track_id=d["track_id"])


def _rule_json(o: RuleOutput | None) -> dict | None:
    if o is None:
        return None
    return to_jsonable({"rule": o.rule.value, "state": o.state.value, "reason": o.reason, "message": o.message,
                        "severity": o.severity, "correction": o.correction, "referent": o.referent, "debug": o.debug})


def decision_json(d: Decision) -> dict:
    s = d.subject_set
    return to_jsonable({
        "scene": {
            "mode": s.mode.value, "mode_reason": s.mode_reason, "count": s.count, "count_reliable": s.count_reliable,
            "primary_track_id": s.primary.track_id if s.primary else None,
            "union_bbox": to_jsonable(s.union_bbox),
            "agreed_yaw": s.agreed_orientation.yaw_state if s.agreed_orientation else None,
            "members": [{"track_id": m.track_id, "yaw_state": m.orientation.yaw_state if m.orientation else None,
                         "facing_vector": to_jsonable(m.orientation.facing_vector) if m.orientation else None}
                        for m in s.members],
        },
        "mode_notice": d.mode_notice,
        "raw": {k.value: _rule_json(v) for k, v in sorted(d.raw_outputs.items(), key=lambda kv: kv[0].value)},
        "smoothed": {k.value: _rule_json(v) for k, v in sorted(d.all_outputs.items(), key=lambda kv: kv[0].value)},
        "ready": _rule_json(d.ready),
        "chosen": _rule_json(d.chosen),
    })


def run_frames(frames: list[list[Detection]], rules_config: dict, frame_dims=(FRAME_W, FRAME_H)) -> list[dict]:
    """Feeds frames in order through a fresh model-less Pipeline built from rules_config."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "rules.yaml"
        cfg_path.write_text(yaml.safe_dump(rules_config))
        pipeline = Pipeline(rules_config_path=cfg_path, load_model=False)
        return [decision_json(pipeline.decide(frame, frame_dims)) for frame in frames]


# --- Tracker cases ------------------------------------------------------------------------------

def _box(cx, cy, w=120, h=90):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def tracker_scenarios(max_age: int) -> dict[str, tuple[str, list[list[list[float]]]]]:
    return {
        "two_crossing": ("Two boxes moving through each other keep their ids.",
                         [[_box(100 + 30 * t, 240), _box(540 - 30 * t, 250)] for t in range(16)]),
        "gap_within_max_age": ("A box missing for 3 frames comes back with the same id.",
                               [[_box(200 + 5 * t, 240)] for t in range(5)] + [[] for _ in range(3)]
                               + [[_box(240 + 5 * t, 240)] for t in range(5)]),
        "gap_beyond_max_age": (f"A box missing for {max_age + 2} frames (> track_max_age) gets a new id.",
                               [[_box(200, 240)] for _ in range(3)] + [[] for _ in range(max_age + 2)]
                               + [[_box(200, 240)] for _ in range(3)]),
        "new_arrival": ("A second box appears mid-sequence and gets the next id.",
                        [[_box(150, 200)] for _ in range(4)] + [[_box(150, 200), _box(480, 300)] for _ in range(4)]),
    }


def run_tracker(frames: list[list[list[float]]], model_config: dict) -> list[list[int]]:
    tracker = ByteTracker(model_config)
    return [[int(i) for i in tracker.update([np.array(b, dtype=np.float32) for b in boxes])] for boxes in frames]


# --- Entry point --------------------------------------------------------------------------------

def load_configs() -> tuple[dict, dict]:
    rules = yaml.safe_load((REPO_ROOT / "configs/rules.yaml").read_text())
    model = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    return rules, model


def write_android_configs(rules_cfg: dict, model_cfg: dict) -> None:
    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    (ANDROID_ASSETS / "rules.json").write_text(json.dumps(rules_cfg, indent=1) + "\n")
    (ANDROID_ASSETS / "model.json").write_text(json.dumps(model_cfg, indent=1) + "\n")


def main() -> None:
    rules_cfg, model_cfg = load_configs()
    write_android_configs(rules_cfg, model_cfg)
    cases_dir, tracker_dir = GOLDEN_DIR / "cases", GOLDEN_DIR / "tracker"
    cases_dir.mkdir(exist_ok=True)
    tracker_dir.mkdir(exist_ok=True)

    for name, (description, frames) in scenarios().items():
        expected = run_frames(frames, rules_cfg)
        case = {
            "format_version": FORMAT_VERSION, "name": name, "description": description,
            "frame_dims": [FRAME_W, FRAME_H], "rules_config": rules_cfg,
            "frames": [{"detections": [_detection_json(d) for d in frame], "expected": exp}
                       for frame, exp in zip(frames, expected)],
        }
        (cases_dir / f"{name}.json").write_text(json.dumps(case, indent=1) + "\n")
        chosen = [f["chosen"]["message"] if f["chosen"] else None for f in expected]
        print(f"{name}: {len(frames)} frames, last instruction: {chosen[-1]!r}")

    track_cfg = {k: model_cfg[k] for k in ("track_iou_threshold", "track_max_age")}
    for name, (description, frames) in tracker_scenarios(int(track_cfg["track_max_age"])).items():
        case = {"format_version": FORMAT_VERSION, "name": name, "description": description,
                "tracker_config": track_cfg,
                "frames": [{"boxes_xyxy": boxes, "expected_track_ids": ids}
                           for boxes, ids in zip(frames, run_tracker(frames, track_cfg))]}
        (tracker_dir / f"{name}.json").write_text(json.dumps(case, indent=1) + "\n")
        print(f"tracker/{name}: ids {[f['expected_track_ids'] for f in case['frames']][-1]}")


if __name__ == "__main__":
    main()
