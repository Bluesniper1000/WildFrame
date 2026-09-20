# Glossary

Shared vocabulary from `wildframe/contracts.py`, used across `scene.py`, `rules/`, `smoothing.py`, and `explain/`.

## `RuleState` (contracts.py)

- `SATISFIED` — the composition guideline is met.
- `VIOLATED` — the guideline is not met; `RuleOutput.correction` describes how to fix it.
- `INDETERMINATE` — the rule can't be meaningfully evaluated this frame (exclusively `gate()`'s job — `evaluate()` must never return this).

## `RuleOutput` (contracts.py)

The dataclass every rule's `run()` returns: `rule` (`RuleName`), `state` (`RuleState`), `reason` (short machine-readable string), optional `message`, `severity` (0.0–1.0), optional `correction` dict, `referent` (reserved for multi-animal referring expressions, see `PLAN_multi_animal.md` Step 7 — always `None` in the single-animal phase), and a `debug` dict.

## `Detection` vs `Subject`/`Member` vs `SubjectSet`

- **`Detection`** — one raw per-frame output of `pose/backend.py`: bbox, keypoints, detection confidence, track ID. Original frame pixel space.
- **`Subject`** (= `Member`, structurally identical types — a Phase-B "member" is just Phase-A's `Subject`) — the composite thing rules evaluate against: bbox, keypoints, det conf, `Orientation`, track ID, frame dims. In single-animal mode this wraps exactly one `Detection` 1:1.
- **`SubjectSet`** — the scene-level composite `scene.resolve_scene()` builds and rules actually consume: `mode` (`SubjectMode`), `members` (tuple, sorted left-to-right by bbox center x), `primary`, `count`, `count_reliable`, `union_bbox`, `agreed_orientation`, `mode_reason` (a template-ready string, e.g. `"3 of 3 facing left"`). In `SINGLE`/`DOMINANT` mode this degenerates exactly to the pre-multi-animal single-`Subject` case — one member, `primary == that member`, `union_bbox == primary.bbox_xyxy`.

## `SubjectMode` (contracts.py)

Set by `scene.resolve_scene()`; a mode change only commits after `scene.mode_enter_frames` consecutive frames agree (hysteresis stricter than and separate from per-rule hysteresis).

- `NONE` — no valid subject in frame.
- `SINGLE` — exactly one animal.
- `DOMINANT` — multiple animals, but one is large/confident enough (by `scene.dominance_ratio`) to treat as primary and the rest as background.
- `GROUP_ALIGNED` — multiple animals sharing a facing direction (by `scene.agreement_fraction`), evaluated as a group.
- `FACING_PAIR` — two animals facing each other; `odd_count` explicitly gates out of evaluating this mode.
- `MIXED` — multiple animals with no clean dominant/aligned/pair structure.

## `Orientation` (contracts.py)

Output of `orientation.py` for one `Detection`/`Subject`: `facing_vector` (unit vector in image-plane coords, or `None` if undetermined), `yaw_state` (`"frontal"` / `"left"` / `"right"` / `"unknown"`), `confidence` (mean confidence of the keypoints used to build the vector). `orientation.py` already gates `facing_vector` to `None` below `min_axis_conf`, so downstream rules that see a non-`None` vector don't need a separate confidence check.
