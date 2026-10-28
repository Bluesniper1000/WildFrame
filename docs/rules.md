# Composition rules

Every rule subclasses `Rule` (`wildframe/rules/base.py`) and implements two methods:

- `gate(subject_set) -> str | None` — return a short-circuit reason to emit `INDETERMINATE`, else `None`.
- `evaluate(subject_set) -> RuleOutput` — only called if `gate()` returned `None`; must return `SATISFIED` or `VIOLATED`, never `INDETERMINATE`.

Rules consume the composite `SubjectSet` built by `scene.py`, not raw detections (see [`glossary.md`](glossary.md)). Config for each rule is its own sub-block in `configs/rules.yaml`; every numeric threshold there is marked `# UNCALIBRATED` until it's been measured or sourced (see [`calibration.md`](calibration.md)).

`priority_order` (`configs/rules.yaml`, currently `limb_clip > lead_space > (eye_thirds OR centrality) > odd_count`) decides which single output `explain/priority.py` shows the photographer when multiple rules fire in the same frame. A bracketed entry is an OR-group: rules in it are alternative ways to satisfy that priority slot, not independently-ranked rules — see `eye_thirds`/`centrality` below and [`calibration.md`](calibration.md).

## `lead_space` (`wildframe/rules/lead_space.py`)

Checks whether there's enough empty space in front of a moving/facing subject ("lead space" / "nose room").

- **Gate:** no subject → `no_subject`; no orientation or no facing vector → `no_orientation`; subject facing the camera → `frontal_yaw` (lead space is only meaningful for a subject facing left/right).
- **Evaluate:** measures the empty fraction of frame width on the side the subject is facing. Violates if that fraction is below `min_lead_fraction`; severity scales toward 0 as the fraction approaches `min_lead_fraction`. Correction is a camera pan toward the front side (turning the camera the way the subject faces moves the subject the other way in the frame), sized toward `ideal_lead_fraction`. `direction` is the camera movement, which is what the "Pan {direction}" message tells the photographer.
- **Config:** `min_lead_fraction` (0.15), `ideal_lead_fraction` (0.30), `hysteresis_frames` (5).

## `eye_thirds` (`wildframe/rules/eye_thirds.py`)

Checks whether the subject's (more confident) eye sits near a rule-of-thirds intersection.

- **Gate:** no subject → `no_subject`; both eye keypoints below `kpt_conf_threshold` → `eye_not_visible`.
- **Evaluate:** picks the higher-confidence eye, finds the nearest of the four thirds intersections, and computes the distance as a fraction of the frame diagonal. Satisfied if within `thirds_tolerance_frac`; otherwise violates, with a pan/tilt correction toward that intersection.
- **Config:** `kpt_conf_threshold` (0.0), `thirds_tolerance_frac` (0.08), `hysteresis_frames` (5).

## `centrality` (`wildframe/rules/centrality.py`)

Checks whether the subject's bbox center sits near the frame center — an alternative to `eye_thirds`, not a stacked requirement. They form an OR-group in `priority_order`: satisfying either is a valid composition, so a violated `centrality` never surfaces (or blocks the `ready` "click now" signal) while `eye_thirds` is satisfied, and vice versa. See `explain/priority.py: _group_output` / `resolved_outputs`.

- **Gate:** no subject → `no_subject`.
- **Evaluate:** distance from the subject bbox's center to the frame center, as a fraction of the frame diagonal. Satisfied if within `center_tolerance_frac`; otherwise violates, with a pan/tilt correction toward the frame center.
- **Config:** `center_tolerance_frac` (0.12), `hysteresis_frames` (5 — matches its OR-group partner, `eye_thirds`).

## `limb_clip` (`wildframe/rules/limb_clip.py`)

Checks whether any distal limb keypoint (paw/foot) is at or past the frame edge, across all members considered for clipping (`scene.members_for_clipping`), not just the primary subject.

- **Gate:** no members → `no_subject`; every member's keypoints all below `min_kpt_conf` → `no_visible_keypoints`.
- **Evaluate:** walks members in left-to-right order, then limbs in a fixed `LIMB_ORDER`, so the same scene always reports the same offender first (determinism requirement). First limb found within `edge_margin_frac` (fraction of frame width/height, applied per axis) of an edge → violated (severity always 1.0, correction is "zoom out"). No clipped limb found → satisfied.
- **Config:** `edge_margin_frac` (0.003125, i.e. 2px on a 640x640 frame — frame-relative so it scales with resolution), `min_kpt_conf` (0.25), `hysteresis_frames` (3 — reacts faster than the composition rules since a clipped limb is unambiguous, not a matter of degree).

## `odd_count` (`wildframe/rules/odd_count.py`)

Checks whether the number of animals in frame is odd (harder to balance visually when even) and, if even, whether excluding or including one edge animal would fix the parity without disturbing anyone else's framing.

- **Gate:** `count == 0` → `no_subject`; `count == 1` → `single_animal`; `count` above `ceiling.value` → `count_exceeds_ceiling` (too crowded for a deliberate fix); `subject_set.mode == FACING_PAIR` → `facing_pair_mode`; `count_reliable == False` → `count_unreliable` (from `scene.py`'s overlap/confidence-margin checks).
- **Evaluate:** odd count → satisfied. Even count → picks the most-clipped edge member (lowest visible fraction, tie-broken left-to-right), determines whether excluding it (if `visible_fraction < exclude_below_visible`) or including it fixes parity, and checks the resulting crop doesn't clip any other member. Falls back to `INDETERMINATE` (`no_reframing_fixes_count`, `shift_would_clip_another_member`, `shift_does_not_fix_parity`) when no clean fix exists — this is a legitimate outcome, not an error.
- **Config:** `ceiling.value` (6), `edge_margin_frac` (0.03125, i.e. 20px on a 640x640 frame — frame-relative so it scales with resolution), `exclude_below_visible` (0.5), `hysteresis_frames` (5).
- **Source caveat:** `ceiling.source` in `configs/rules.yaml` is currently `"design choice - pending Excell p.?"`. Per `CLAUDE.md` and the rule module's own header comment, this must **not** be attributed to Excell's book (or any source) until the actual page/wording is supplied — don't invent a citation here or anywhere else.
