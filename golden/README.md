# Golden fixtures

Reference behaviour of WildFrame's deterministic decision path (detections → scene → rules →
hysteresis → ready/priority → instruction) and of the tracker. A port of the rule engine (the
planned mobile app) is correct when it reproduces every file here. On the Python side,
`tests/test_golden.py` keeps the fixtures in sync with the code.

```bash
python -m golden.generate     # rewrite all fixtures from the current code + configs
pytest tests/test_golden.py   # check the code still matches the committed fixtures
```

Regenerate only deliberately, after a behaviour change you meant to make or a `configs/*.yaml` change,
and review the JSON diff. The parity test also fails if `configs/rules.yaml` or the tracking keys of
`configs/model.yaml` no longer match the config snapshot stored in the fixtures.

## `cases/*.json`: decision path

```jsonc
{
  "format_version": 1,
  "name": "hysteresis_flip",
  "description": "...",
  "frame_dims": [640, 480],              // (width, height) of the original frame
  "rules_config": { ... },               // full configs/rules.yaml the case was generated with
  "frames": [                            // feed IN ORDER into ONE fresh engine instance
    {
      "detections": [                    // input: what PoseBackend.infer() would return
        {"bbox_xyxy": [x1, y1, x2, y2],
         "keypoints": [[x, y, conf] x 17],   // AP-10K order, see wildframe/pose/skeleton.py
         "det_conf": 0.9, "track_id": 1}
      ],
      "expected": {
        "scene":   {"mode", "mode_reason", "count", "count_reliable", "primary_track_id",
                    "union_bbox", "agreed_yaw", "members": [{"track_id", "yaw_state", "facing_vector"}]},
        "mode_notice": "Now tracking: single (1 of 1 member)." | null,
        "raw":      {"<rule>": RuleOutput},   // each rule's run() this frame, referent attached
        "smoothed": {"<rule>": RuleOutput},   // after per-(rule, track) hysteresis
        "ready":    RuleOutput | null,        // aggregate "click now" signal, after its own hysteresis
        "chosen":   RuleOutput | null         // the ONE instruction shown, message rendered
      }
    }
  ]
}
```

`RuleOutput` = `{"rule", "state", "reason", "message", "severity", "correction", "referent", "debug"}`.
`raw`, `smoothed`, `ready` and `chosen` are empty/null on frames where the scene mode is `none`.

Detections carry explicit track ids, so these cases test the decision path on its own. The tracker
has separate fixtures.

## `tracker/*.json`: ByteTracker

```jsonc
{"tracker_config": {"track_iou_threshold": 0.3, "track_max_age": 30},
 "frames": [{"boxes_xyxy": [[x1, y1, x2, y2], ...], "expected_track_ids": [1, 2]}]}
```

## Comparison rules for a port

- **Exact:** structure, keys, strings (states, reasons, messages, modes, notices), ints, bools, nulls.
- **Floats:** relative 1e-6 / absolute 1e-5. Python computes partly in float32 (numpy keypoint arrays),
  so a float64 port can differ in the last digits. The scenarios deliberately keep values away from
  rule thresholds, so a last-digit difference shouldn't flip a state. If one does, that's a real
  precision-sensitivity bug worth looking at, not a reason to loosen the tolerance.
- `debug` dicts are included to help localise a mismatch. A port may match them field by field, or
  treat them as advisory once `raw`/`smoothed`/`chosen` match.

## Known behaviour pinned by these fixtures (not necessarily desired)

- **`ready` is asymmetric** (since 2026-10-28). It drops to not-ready on the same frame a blocking
  violation is committed (`Smoother.commit`), and turns back on only after `ready.hysteresis_frames`
  consecutive all-clear frames. In `hysteresis_flip`, "Zoom out" appears at frame 12, the frame
  `limb_clip` commits, and "click now" returns at frame 26. A port must implement both directions.
- **Per-rule hysteresis can surface a stale instruction on recovery.** In `hysteresis_flip`, frames
  20–21 show a `centrality` violation after the paw is back in frame. The clipped paw stretched the
  bbox, so centrality committed VIOLATED, and its own 5-frame hysteresis is still clearing.
- The first 2 frames of any new subject show `limb_clip: insufficient_history` before hysteresis commits.
