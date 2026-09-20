# Calibration

Every threshold in `configs/rules.yaml` is marked `# UNCALIBRATED` in a comment until it's been
either measured from data or given a cited source. Per `CLAUDE.md`, that marker is load-bearing:
don't remove it or invent a replacement number without a cited source or a measurement. This doc
tracks *how* each one is meant to get a real value — not the results themselves (those belong in
`notebooks/`, alongside the quantization/optimization experiment notes).

## `scene.overlap_iou_max`

Meant to be set from `scripts/measure_overlap_keypoints.py`'s output. That script runs the
fine-tuned pose model against AP-10K validation annotations, bins ground-truth instances by their
max pairwise bbox IoU with any other animal in the same image, and reports per-bin PCK@0.1 (plus a
single-instance baseline) to a CSV. The threshold should be picked as the IoU bin where keypoint
quality (PCK) degrades enough that `odd_count`/group-mode counting becomes unreliable — i.e. where
`scene.py` should stop trusting the detection count and set `count_reliable = False`.

## Other `# UNCALIBRATED` thresholds

Everything else currently in `configs/rules.yaml` is a placeholder pending its own measurement or
citation:

- `orientation.frontal_yaw_max_deg`, `orientation.min_axis_conf`
- `lead_space.min_lead_fraction`, `lead_space.ideal_lead_fraction`, `lead_space.hysteresis_frames`
- `eye_thirds.kpt_conf_threshold`, `eye_thirds.thirds_tolerance_frac`, `eye_thirds.hysteresis_frames`
- `centrality.center_tolerance_frac`, `centrality.hysteresis_frames`
- `limb_clip.edge_margin_frac`, `limb_clip.min_kpt_conf`, `limb_clip.hysteresis_frames`
- `scene.min_det_conf`, `scene.min_bbox_area_frac`, `scene.dominance_ratio`, `scene.agreement_fraction`, `scene.conf_margin`, `scene.mode_enter_frames`
- `odd_count.ceiling.value`, `odd_count.edge_margin_frac`, `odd_count.exclude_below_visible`, `odd_count.hysteresis_frames`
- `ready.severity_max`, `ready.hysteresis_frames`

None of these have a measurement script yet. If you calibrate one, follow the
`measure_overlap_keypoints.py` pattern: pure, unit-testable scoring functions separated from the
model/file I/O driver, output to a notebook or CSV, and only then update the YAML value and drop
its `# UNCALIBRATED` marker (or replace it with a `source` field, as `odd_count.ceiling` already
has a slot for).

## `odd_count.ceiling.source`

Currently `"design choice - pending Excell p.?"`. Do not fill this in with a page number or
wording unless the user supplies it directly — this applies to `configs/rules.yaml`, code
comments, and these docs alike. See [`rules.md`](rules.md#odd_count-wildframerulesodd_countpy).
