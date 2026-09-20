# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WildFrame is a deterministic, rule-based real-time composition coach for wildlife photography. A frozen YOLO26-pose model (fine-tuned on AP-10K, exported to ONNX/int8) detects animal keypoints per frame; a pipeline of pure-function rules (lead space, eye-thirds, limb-clip, odd-count) evaluates classic composition guidelines and surfaces one instruction to the photographer via an overlay. No learned components beyond the frozen pose model and ByteTrack's Kalman filter are allowed anywhere in the rule/decision path.

## Commands

```bash
# Install (Windows venv already present at ./venv)
pip install -r requirements.txt

# Run the full test suite
pytest

# Run a single test file / test
pytest tests/test_rules.py
pytest tests/test_rules.py::test_lead_space_violation -v

# Run the live pipeline on an image or video
python -m wildframe.pipeline --source path/to/image.jpg --output out.jpg
python -m wildframe.pipeline --source path/to/video.mp4 --display

# Measure overlap-vs-keypoint-quality (feeds the overlap_iou_max threshold)
python scripts/measure_overlap_keypoints.py
```

There is no lint/format/build step configured (`pyproject.toml` only declares the package name/version). No test config beyond default pytest discovery in `tests/`.

## Architecture

Pipeline (`wildframe/pipeline.py`, class `Pipeline`), one frame at a time:

```
frame -> PoseBackend.infer() -> resolve_scene() -> per-rule .run() -> hysteresis (Smoother)
       -> priority.resolve() -> templates.render() -> overlay.draw_overlay()
```

- **`pose/backend.py`** (`PoseBackend`) — thin onnxruntime wrapper. Does its own letterbox preprocessing and box/keypoint decode (the exported graph has no NMS baked in), then hands boxes to `tracking.ByteTracker` for track IDs. `max_det=1` reproduces the original single-animal baseline; `max_det>1` (configured default) enables multi-animal detection with greedy NMS.
- **`contracts.py`** — the shared dataclass/enum vocabulary every other module imports: `Detection` (raw per-frame output), `Orientation`, `Subject`/`Member` (structurally identical — a Phase-B "member" is just a Phase-A `Subject`), `SubjectSet` (composite multi-animal view), `SubjectMode` enum (`NONE/SINGLE/DOMINANT/GROUP_ALIGNED/FACING_PAIR/MIXED`), `RuleOutput`/`RuleState` (`SATISFIED/VIOLATED/INDETERMINATE`). Read this file first when touching any rule or the scene resolver — it's the contract everything else is built to satisfy.
- **`scene.py`** (`resolve_scene`, `composite_subject`, `referent_label`) — pure function `(detections, frame_dims, config, prev_state) -> (SubjectSet, new_state)`. Filters low-confidence/tiny detections, picks the primary subject (bbox area × mean keypoint conf, tied-broken by track_id then leftmost), classifies the scene into a `SubjectMode`, and builds the composite bbox/orientation/eye-source the rules consume. Mode changes only commit after `mode_enter_frames` consecutive frames agree (hysteresis separate from and stricter than per-rule hysteresis).
- **`rules/`** — one module per composition rule, each subclassing `Rule` (`rules/base.py`): implement `gate()` (return a reason string to short-circuit to `INDETERMINATE`, else `None`) and `evaluate()` (must return `SATISFIED` or `VIOLATED`, never `INDETERMINATE` — that's exclusively `gate()`'s job). Rules consume the composite `SubjectSet`, not raw detections. Current rules: `lead_space.py`, `eye_thirds.py`, `limb_clip.py`, `odd_count.py`.
- **`smoothing.py`** (`Smoother`) — per-rule hysteresis: a raw state only becomes the emitted state after N consecutive frames agree (`hysteresis_frames` per rule in config). Keyed on `(rule, track_id)` so each animal's state is tracked independently; when the pipeline's primary track changes, the predecessor's hysteresis state is dropped rather than left to linger.
- **`explain/priority.py`** (`resolve`) — picks the single output to show the photographer: highest-priority `VIOLATED` first, else highest-priority `SATISFIED`, else highest-priority `INDETERMINATE`. Priority order is config-driven (`configs/rules.yaml: priority_order`), currently `limb_clip > lead_space > eye_thirds > odd_count`.
- **`explain/templates.py`** / **`explain/overlay.py`** — turn a `RuleOutput`/`SubjectSet` into a human-readable message and a drawn frame overlay.
- **`orientation.py`** — derives a facing vector/yaw state from keypoints, feeding `Orientation` on each `Subject`/`Member`.
- **`tracking.py`** (`ByteTracker`) — the one allowed non-deterministic-looking but still deterministic component (Kalman filter + Hungarian matching); assigns stable `track_id`s across frames.

### Config, not code

All thresholds live in `configs/rules.yaml` (per-rule blocks + `scene`/`odd_count` blocks) and `configs/model.yaml` (backend/detection/tracking params), never hardcoded. Every unvalidated threshold is marked `# UNCALIBRATED` in the YAML comment — treat that marker as load-bearing: don't remove it or invent a number to replace it without a cited source or a measurement (e.g. `scene.overlap_iou_max` is meant to be set from `scripts/measure_overlap_keypoints.py`'s output). `odd_count.ceiling` additionally carries a `source` field — gates must not be attributed to a source (e.g. Excell's book) until the user supplies the actual page/wording.

### Invariants to preserve

- Determinism: no learned components beyond the frozen pose model and ByteTrack's Kalman filter.
- Rules stay pure functions returning `RuleOutput`.
- A single-animal frame must produce output identical to the pre-multi-animal pipeline (`SINGLE`/`DOMINANT` mode degenerates exactly to the one-`Subject` case).
- Every `RuleOutput`/template field must exist on the output struct — `tests/test_templates.py` enforces this and must stay green.

## Repo layout notes

- `notebooks/` — exploratory work; see `notebooks/README.md` for the naming convention and what's expected to graduate into `wildframe/`.
- `plans/PLAN_multi_animal.md` — the implementation plan that took the pipeline from single-animal (`max_det=1`) to multi-animal; useful background for why `SubjectSet`/`scene.py` are shaped the way they are.
- `study/` — currently empty stub files (`logging.py`, `viewfinder.py`) for a planned user study harness.
- `paper/`, `ap10k_ft/`, `playground/`, `data/`, `pretrained/` — model training artifacts, pruning/quantization experiments, and dataset/weights storage; not part of the runtime pipeline.
