# WildFrame — working notes

Running record of what we've established about the repo, the environment, and the benchmark plan.
Started 2026-10-26; extended as discussions continue.

## 1. Repo state

- GitHub `Bluesniper1000/WildFrame` has one branch (`main`) and one commit (`1864902`, "first commit"). The local checkout matches it exactly.
- `.gitignore` excludes `scripts/`, `ap10k_ft/`, `plans/`, `paper/`, `data/`, `playground/` and `*.pt`. None of these are on GitHub, so a fresh clone has these consequences:
  - `configs/model.yaml` points at `ap10k_ft/weights_quant/best_int8.onnx`, which is missing → the pipeline raises `FileNotFoundError`.
  - `tests/test_measure_overlap_keypoints.py` imports `scripts/measure_overlap_keypoints.py`, which is missing. **The test now skips when the script is absent** (`pytest.importorskip`) instead of erroring during collection. It will run again once the script is restored.
  - Links in `docs/README.md` to `plans/PLAN_multi_animal.md` and `paper/` are broken.
- `CLAUDE.md` lags behind the code. It doesn't mention the `centrality` rule, the `ready` ("click now") signal, or the `[eye_thirds, centrality]` OR-group in `priority_order`. `docs/rules.md` is current.
- `study/logging.py` and `study/viewfinder.py` are empty stubs, and there's no recorded study design.
- `out.jpg` is not a JPEG. It's an MPEG-4 video stream that was written with a `.jpg` name, and its frames already contain an overlay.

## 2. Project summary

WildFrame is a deterministic composition coach for wildlife photography. For each frame:

```
frame -> PoseBackend (ONNX, letterbox 640, own decode + greedy NMS, ByteTrack)
      -> resolve_scene (SubjectMode: NONE/SINGLE/DOMINANT/GROUP_ALIGNED/FACING_PAIR/MIXED, with mode hysteresis)
      -> rules: limb_clip, lead_space, eye_thirds | centrality (OR-group), odd_count
      -> per-(rule, track_id) hysteresis -> ready signal / priority.resolve -> template -> overlay
```

Invariants:
- Nothing learned beyond the frozen pose model and ByteTrack's Kalman filter.
- Rules are pure functions.
- Single-animal output is identical to the pre-multi-animal pipeline.
- Every threshold lives in YAML and is marked `# UNCALIBRATED` until it is measured or sourced.

## 3. Environment

- `./venv` (Python 3.12.10) with `requirements.txt` installed: torch 2.14.1, ultralytics 8.4.189, onnxruntime 1.31.0, onnx 1.23.2, opencv-python 5.0.0.93, numpy 2.5.4, pytest 9.1.1.
- Baseline: `pytest` → **86 passed, 1 skipped** (the overlap-script test above).
- The weights were copied into `ap10k_ft/weights_quant/` (`best_int8.onnx`, `best_fp32.onnx`) so the default config loads. That folder is gitignored.

## 4. Model files

| File | Identity |
|---|---|
| `best_fp32.onnx` (12.8 MB) | YOLO26n-pose, Ultralytics 8.4.142 export, opset 18, dynamic shapes, `end2end=False`, output `(1, 56, 8400)` |
| `best_int8.onnx` (4.2 MB) | onnxruntime-quantized INT8 version of the same model, with the same I/O |
| `yolo26s-pose_ap10k_finetune.pt` | PyTorch checkpoint (Kaggle run `ap10k_ft`) |
| `yolo11n-pose_ap10k_finetune.pt` | PyTorch checkpoint |
| `yolov8n-pose_ap10k_finetune.pt` | PyTorch checkpoint |

- The ONNX output layout matches `pose/backend.py`'s decode (`(1, 56, anchors)`, no NMS).
- **The model sets don't line up.** The ONNX files are 26n, but there is no 26n `.pt`, and there is no FP32/INT8 ONNX export for 26s, 11n or v8n. Those exports are needed for the INT8-delta columns.
- The checkpoints carry a `kpt_oks_sigmas` field. The benchmark plan sidesteps the question of whether Ultralytics used it by evaluating every model externally.

## 5. Smoke test

- The INT8 model on one 1280×720 deer frame detected a SINGLE subject with plausible keypoints in about 66 ms/frame on CPU. This is unbenchmarked: no warm-up, and only one frame.
- **Gap:** single-image mode (`--source img.jpg`) can never produce a verdict. Each rule's hysteresis needs 3–5 agreeing frames, so a single frame always reports `insufficient_history`. This needs a fix before stills are used for demos or paper figures.

## 6. Benchmark plan (pose models only)

**Accuracy:**
- OKS AP / AP50 / AP75 on the AP-10K test split.
- Use one external evaluator for all models: xtcocotools with AP-10K per-keypoint sigmas (what MMPose uses).
- Don't use Ultralytics `val`: for `kpt_shape == [17, 3]` it applies COCO human sigmas, so its numbers can't be compared with published AP-10K results.
- Top-down models (RTMPose, HRNet, ViTPose) report AP with both ground-truth boxes and detector boxes. Only the detector-box number is comparable to YOLO.

**Keypoints the rules consume:**
- Per-keypoint PCK@0.05 or NME, normalized by bbox: eyes, nose, neck, tail root, paws.
- Orientation accuracy: `orientation.compute_orientation` applied to predicted keypoints vs. ground-truth keypoints (left/right/frontal).
- Visibility calibration: AUROC of keypoint confidence for predicting ground-truth visibility. This drives `limb_clip` and the INDETERMINATE gates.

**Efficiency:**
- Params and GFLOPs.
- CPU latency: batch 1, with input size, hardware and runtime stated (ONNX Runtime).
- INT8 deltas for each YOLO model: ΔAP, Δeye error, Δorientation accuracy.

**Fairness caveats:**
- Report the different input resolutions (640 full-frame vs. 256×256 crops) rather than equalizing them.
- Use the same test split for every model.
- Put cited HRNet/ViTPose numbers in a separate, clearly marked block.

**Minimum table columns:** AP (detector boxes), eye PCK, orientation accuracy, CPU latency, INT8 ΔAP.

**What the existing tooling falls short on:** `notebooks/performance_testing.ipynb` uses Ultralytics val plus a homemade PCK@0.1, which is exactly the setup the plan replaces.

## 7. Prior work in the sibling ASTMA-wildlife project

`C:\Dev\Projects\ASTMA-wildlife\pose\` already has:
- the AP-10K data (`pose/data/ap10k`, all 3 splits);
- FP32 and INT8 ONNX exports for yolov8n, yolo11n, yolo26s (both the one-to-many and NMS-free heads) and RTMPose-m;
- a benchmark covering OKS AP (pycocotools with AP-10K sigmas), CPU latency and INT8 ΔAP. Results are in `results/pose_tables.md`.

It does **not** cover YOLO26n (WildFrame's deployed model), per-keypoint PCK/NME, orientation accuracy or visibility AUROC. The WildFrame harness fills those gaps and uses the same AP protocol, so numbers from the two projects can be compared.

Key ASTMA finding: INT8-quantizing the keypoint branch (`cv4`) collapses n-scale YOLO-pose. Keeping those 9 convs in FP32 (the "kpfp32" variant) recovers almost all of the accuracy.

## 8. Benchmark harness (`bench/`)

```bash
python -m bench.run --model <onnx> --kind yolo-o2m|yolo-e2e --family <name> --precision fp32|int8 \
    --data ../ASTMA-wildlife/pose/data/ap10k [--flops-model <fp32 onnx>] [--reuse-preds] [--no-latency]
python -m bench.run --preds <coco results json> --family rtmpose-m --precision fp32 --input-res "256x256 crop"   # score external predictions
python -m bench.table results/bench/*.json --latency-1t results/bench/latency_1thread.json > results/bench/tables.md
python -m bench.quantize --model <fp32 onnx> --out <int8 onnx> [--exclude-regex cv4] --data <ap10k root>
```

`--variant <label>` on `bench.run` distinguishes INT8 recipes of the same family (e.g. `kpfp32`, `allconv`, `deployed`).

Modules:
- `bench/metrics.py` holds the pure scoring functions, unit-tested in `tests/test_bench_metrics.py` (16 tests).
- `bench/predict.py` uses WildFrame's own letterbox.
- `bench/efficiency.py` counts params and GFLOPs from the ONNX graph and measures latency.
- `bench/run.py` is the CLI; `bench/table.py` builds the tables.

**Protocol:**
- **AP:** pycocotools `COCOeval(keypoints)` with `kpt_oks_sigmas` set to the AP-10K (MMPose) sigmas, class-agnostic, test split1.
  - xtcocotools doesn't build on Windows/Python 3.12. For plain OKS it computes the same thing, so the paper can describe this as "COCO keypoint AP with AP-10K sigmas".
  - Prediction settings match Ultralytics val: conf 0.001, NMS IoU 0.7, max_det 300.
  - **Validated:** yolov8n FP32 gives AP 0.434 here, exactly matching ASTMA's independent run.
- **Keypoint metrics** use predictions with score ≥ 0.25 (the deployment `conf_threshold`), matched one-to-one to GT by **box** IoU ≥ 0.5. Matching on boxes keeps the keypoint metrics from being biased toward keypoints that happen to be close.
  - PCK@0.05 uses MMPose "bbox" normalization: the x and y errors are divided by the GT box width and height before taking the norm.
  - NME is the error divided by √(GT box area).
  - Only labeled GT keypoints count.
- **Orientation:** `wildframe.orientation.compute_orientation` with the deployed `configs/rules.yaml` orientation block, applied to predicted keypoints and to GT keypoints (labeled → confidence 1).
  - The headline number is left/right accuracy over GT left/right instances. Predicting "frontal" or "unknown" counts as wrong.
  - Three-way accuracy and the full confusion matrix are in the JSON.
- **Visibility AUROC:** predicted keypoint confidence vs. GT labeled/unlabeled, over matched instances (Mann–Whitney, tie-aware).
- **Efficiency:**
  - Params are counted from the ONNX FP32 initializers.
  - GFLOPs = 2 × MACs over Conv/MatMul/Gemm at 640². This matches Ultralytics' 9.2 for yolov8n-pose.
  - Latency is batch 1 on ONNX Runtime CPU EP, 8 threads, 30 warm-up + 300 timed runs. "Model only" = `session.run`. "e2e" = letterbox + run + decode/NMS at the deployment settings.
- NMS uses `cv2.dnn.NMSBoxes`. **torch's DLLs are blocked by Windows Application Control on this machine**, so nothing in the harness imports torch.

## 9. INT8 quantization of YOLO26n (`bench/quantize.py`)

`bench/quantize.py` is a port of ASTMA's `pose/quantize.py` that uses WildFrame's letterbox for calibration. The recipe:
- static QOperator format (Conv → QLinearConv);
- U8 activations and S8 per-channel weights;
- MinMax calibration on 300 AP-10K **train**-split1 images, seed 0;
- `--exclude-regex cv4` keeps the 9 keypoint-branch convs in FP32.

A manifest JSON is written next to each model.

Two Windows workarounds were needed:
- `onnxruntime.tools` imports torch whenever it's installed, and torch's DLLs are blocked. The script sets `sys.modules["torch"] = None` so ORT's `find_spec` check skips it.
- ORT's symbolic shape inference raises a plain `Exception` on the dynamic-shape YOLO26n export, so the fallback to `skip_symbolic_shape=True` now catches any exception.

The resulting models are in `ap10k_ft/weights_quant/`, which is gitignored:
- `best_int8_kpfp32.onnx`: 102 QLinearConv + 9 FP32 Conv, 5.1 MB.
- `best_int8_allconv.onnx`: 111 QLinearConv, 3.5 MB. This is the same recipe with nothing excluded, used as the ablation.

## 10. Results (AP-10K test split1, Intel Core 7 240H)

INT8 rows: `deployed` = the current `best_int8.onnx` (recipe unknown); `kpfp32` = keypoint branch kept FP32; `allconv` = every conv quantized. The yolov8n/yolo11n INT8 models are ASTMA's exports, built with the same recipe.

| Model | Prec. | AP | AP50 | AP75 | Eye PCK@0.05 | Eye NME | Orient. acc (L/R) | Vis. AUROC | Params (M) | GFLOPs | ms 1 thread (model / e2e) | ΔAP | ΔEye NME | ΔOrient. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **yolo26n** | fp32 | **0.542** | 0.849 | 0.581 | **0.893** | **0.0271** | **0.809** | **0.932** | 2.93 | 7.3 | 84.6 / 93.5 | | | |
| yolo26n | int8 (deployed) | 0.477 | 0.827 | 0.479 | 0.849 | 0.0364 | 0.802 | 0.927 | 2.93 | 7.3 | 66.9 / 69.8 | −0.065 | +0.0093 | −0.006 |
| yolo26n | int8 (allconv) | 0.503 | 0.833 | 0.520 | 0.863 | 0.0339 | 0.800 | 0.928 | 2.93 | 7.3 | 66.8 / 72.1 | −0.039 | +0.0067 | −0.009 |
| yolo26n | int8 (kpfp32) | 0.531 | 0.839 | 0.561 | 0.886 | 0.0289 | 0.797 | 0.929 | 2.93 | 7.3 | 79.3 / 84.2 | −0.011 | +0.0018 | −0.012 |
| yolov8n | fp32 | 0.434 | 0.791 | 0.412 | 0.856 | 0.0360 | 0.766 | 0.920 | 3.36 | 9.2 | 100.6 / 106.2 | | | |
| yolov8n | int8 (allconv) | 0.044 | 0.215 | 0.001 | 0.160 | 0.1246 | 0.753 | 0.900 | 3.36 | 9.2 | 62.4 / 70.6 | −0.390 | +0.0886 | −0.013 |
| yolov8n | int8 (kpfp32) | 0.433 | 0.796 | 0.412 | 0.849 | 0.0375 | 0.774 | 0.921 | 3.36 | 9.2 | 68.8 / 75.2 | −0.000 | +0.0015 | +0.008 |
| yolo11n | fp32 | 0.429 | 0.801 | 0.384 | 0.853 | 0.0387 | 0.772 | 0.918 | 2.93 | 7.5 | 88.3 / 92.9 | | | |
| yolo11n | int8 (allconv) | 0.069 | 0.302 | 0.003 | 0.220 | 0.1118 | 0.781 | 0.906 | 2.93 | 7.5 | 65.0 / 68.8 | −0.360 | +0.0731 | +0.009 |
| yolo11n | int8 (kpfp32) | 0.426 | 0.798 | 0.397 | 0.844 | 0.0400 | 0.772 | 0.915 | 2.93 | 7.5 | 70.7 / 77.2 | −0.003 | +0.0013 | +0.000 |

The full table, including 8-thread latency and per-keypoint PCK / NME / AUROC for the rule keypoints, is in `results/bench/tables.md`. Raw data: `results/bench/*.json` and `results/bench/latency_1thread.json`.

### Takeaways

1. **YOLO26n FP32 is the most accurate n-scale model on every rule-relevant metric**: +0.11 AP over yolov8n/yolo11n, eye NME 0.027 vs 0.036–0.039, and orientation 0.81 vs 0.77. It's also cheaper (7.3 GFLOPs) and faster at 1 thread (93.5 vs 106 ms e2e for yolov8n).
2. **The deployed INT8 is a poor quantization.** Our every-conv recipe runs at the same speed (~70 ms e2e) and is 2.6 AP better (0.503 vs 0.477), so the deployed model's calibration was worse as well as quantizing the keypoint branch.
3. **kpfp32 recovers almost all of the INT8 accuracy for every family.** YOLO26n: −0.011 AP, eye NME +0.0018. yolov8n/yolo11n: within 0.003 AP.
4. **YOLO26n is far more robust to every-conv quantization** than v8n/11n. It loses 0.039 AP, while those two collapse to AP 0.04–0.07 (eye PCK 0.16–0.22). That makes "every-conv INT8" a viable option for YOLO26n only.
5. **Orientation accuracy is nearly unaffected by any INT8 variant** (±0.013), even the collapsed ones. The facing direction comes from coarse head and torso geometry, which survives quantization better than precise localization does.
6. **The latency trade-off at 1 thread** (closest to on-device):

   | YOLO26n variant | e2e ms | vs FP32 | ΔAP |
   |---|---|---|---|
   | FP32 | 93.5 | — | — |
   | INT8 kpfp32 | 84.2 | 10% faster | −0.011 |
   | INT8 every-conv | 72.1 | 23% faster | −0.039 |
   | INT8 deployed | 69.8 | 25% faster | −0.065 |

   kpfp32 keeps the 9 keypoint convs in FP32, so it gives back most of the INT8 speed-up.
7. **8-thread latency is noisy on this laptop** (hybrid P/E cores). For example, yolov8n FP32 shows model-only 43.0 ms > e2e 41.4 ms, and YOLO26n FP32 e2e measured 41.8 ms then 35.3 ms in two sessions. At 8 threads INT8 gives little or no speed-up anyway. Before the paper, repeat the latency runs (e.g. 3 sessions, report the median) with the laptop on mains power and idle.
8. Orientation is still the weak link for `lead_space`. With YOLO26n FP32, 172 of 1206 GT left/right animals are classified "frontal" and 76 as the opposite side. Neck, tail-root and paw PCK@0.05 (0.37–0.53) are far below eye PCK (0.89).
9. Recall at the 0.25 deployment threshold is 92.2% of GT instances (YOLO26n FP32).

### YOLO26s (s-scale comparison, measured 2026-10-28)

These use ASTMA's exports for both heads: `o2m` = one-to-many output + NMS (same decode as YOLO26n); `e2e` = NMS-free top-300 output. Each head is in FP32, INT8 kpfp32 and INT8 every-conv. The model has 10.43 M params and 24.1 GFLOPs, about 3.5× YOLO26n.

**Cross-check:** AP matches ASTMA's independent runs exactly for all four rows it reports (0.583, 0.569, 0.575, 0.557).

| Head | Prec. | AP | AP50 | AP75 | Eye PCK@0.05 | Eye NME | Orient. acc (L/R) | Vis. AUROC | ΔAP | ΔEye NME | ΔOrient. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| o2m | fp32 | **0.583** | 0.875 | 0.631 | 0.909 | 0.0232 | 0.802 | 0.934 | | | |
| o2m | int8 (kpfp32) | 0.569 | 0.870 | 0.610 | 0.900 | 0.0258 | 0.809 | 0.933 | −0.014 | +0.0026 | +0.007 |
| o2m | int8 (allconv) | 0.510 | 0.858 | 0.535 | 0.865 | 0.0333 | 0.814 | 0.930 | −0.073 | +0.0100 | +0.012 |
| e2e | fp32 | 0.575 | 0.874 | 0.625 | 0.911 | 0.0231 | 0.811 | 0.934 | | | |
| e2e | int8 (kpfp32) | 0.557 | 0.857 | 0.596 | 0.906 | 0.0254 | 0.814 | 0.933 | −0.018 | +0.0023 | +0.004 |
| e2e | int8 (allconv) | 0.489 | 0.837 | 0.498 | 0.848 | 0.0356 | 0.820 | 0.930 | −0.086 | +0.0125 | +0.009 |

**YOLO26n vs YOLO26s** (o2m head, 1 thread, e2e ms). All four latencies were measured **in the same session** on 2026-10-28, on mains power:

| Model | AP | Eye NME | Orient. acc (L/R) | Paw PCK@0.05 | 1-thread e2e ms |
|---|---|---|---|---|---|
| YOLO26n fp32 | 0.542 | 0.0271 | 0.809 | 0.44–0.53 | 111.4 |
| YOLO26n int8 kpfp32 (**deployed**) | 0.531 | 0.0289 | 0.797 | 0.41–0.51 | 102.9 |
| YOLO26s fp32 | 0.583 | 0.0232 | 0.802 | 0.50–0.59 | 318.7 (2.9× YOLO26n) |
| YOLO26s int8 kpfp32 | 0.569 | 0.0258 | 0.809 | 0.47–0.58 | 225.4 (2.2× YOLO26n) |

Takeaways:

10. **YOLO26s buys +0.04 AP, about 15% lower eye error and +0.06 paw PCK, for about 3× the latency. It does not improve orientation**, the `lead_space` weak spot: L/R accuracy is 0.802 vs 0.809. With FP32 o2m, 175 of 1321 GT left/right animals are called "frontal" and 84 the opposite side, the same ~20% disagreement rate as YOLO26n. Since this metric applies the *same* heuristic to predicted and GT keypoints, more accurate keypoints not helping suggests the disagreements are borderline cases where `orientation.py` is sensitive to small keypoint shifts. Most are left/right ↔ frontal flips near the `frontal_yaw_max_deg` / eye-offset thresholds, not gross keypoint errors. The metric can't say whether the heuristic itself is *correct*; that needs human facing labels.
11. **The INT8 pattern from YOLO26n holds at s-scale.** kpfp32 costs 0.014–0.018 AP; every-conv costs 0.07–0.09, more than YOLO26n's 0.039.
12. **The NMS-free head is slightly less accurate** (−0.008 AP FP32, −0.012 kpfp32) and **not faster on CPU** (1 thread e2e 325.5 vs 314.0 ms FP32), so it has no advantage for WildFrame.
13. Recall at the 0.25 deployment threshold: 93.4% (o2m), 92.5% (e2e), vs 92.2% for YOLO26n.
14. **Latency drifts between days.** On 2026-10-28 everything was about 1.2–1.6× slower than on 2026-10-26 (YOLO26n FP32 1-thread e2e: 111.4 vs 93.5 ms; YOLO26s FP32 1-thread model-only: 310 ms vs ASTMA's 194 ms), even on mains power. The 1-thread column in `results/bench/tables.md` mixes the 2026-10-26 n-scale rows with the 2026-10-28 YOLO26s rows, so **compare across models only within one session**, as in the table above. The 8-thread numbers are noisy again: the e2e FP32 row shows model-only 97.5 ms > e2e 89.1 ms.

## 11. Mobile app plan

**Approach:** keep the model and the design (YOLO26n pose → pure-function rules → hysteresis → priority → one instruction), and rewrite the runtime natively. Python/OpenCV doesn't ship well to phones (Chaquopy is heavy on Android and not an option on iOS). The rule engine (`scene`, `rules/`, `smoothing`, `priority`, `templates`, `orientation`, `tracking`) is about 1,500 lines of pure logic, so porting it is very manageable.

**Defaults, pending your confirmation:**
- **Android first**, in Kotlin.
- The rule engine in a **Kotlin Multiplatform** module, so it can be shared with iOS later.
- **ONNX Runtime Mobile** for the model, reusing the existing ONNX export and the decode logic in `backend.py`.
- Flutter or React Native would still need native code for the camera and the model, so they don't save much here.

**Open decisions:** platform order, speed target (e.g. ≥10 fps on a mid-range Android), and purpose (study prototype vs Play Store app).

**Phases:**

0. **Golden fixtures** (✅ done 2026-10-28, see below). They lock in the Python behaviour so the port can be checked exactly.
1. **Model for phones.** The x86 INT8 results don't carry over. QOperator QLinearConv runs on the CPU only, so the phone's NNAPI, GPU or Core ML accelerators need re-quantization in **QDQ** format, still keeping `cv4` in FP32. Try input sizes 416/480 vs 640 (expose input size in `bench/run.py` to measure the accuracy cost), and measure latency on 2–3 real phones with ONNX Runtime's `onnxruntime_perf_test`. Ultralytics LiteRT/Core ML exports need torch, which is blocked on this PC, so run those on Kaggle.
2. **Port the rule engine** in this order: contracts → skeleton → orientation → scene → rules → smoothing → priority → templates. Also port the tracker; with ≤10 detections, a brute-force or greedy assignment can replace scipy's Hungarian solver. Done = all of `golden/` passes.
3. **Android app.**
   - Pipeline: CameraX `ImageAnalysis` with `STRATEGY_KEEP_ONLY_LATEST` → letterbox → ORT → decode/NMS → tracker → rule engine.
   - Overlay drawn natively over the camera preview.
   - **Hysteresis is counted in frames** (`hysteresis_frames: 5` is 0.17 s at 30 fps but 0.5 s at 10 fps). Either cap the inference rate or move the config to seconds.
   - **Sensor rotation:** rotate camera frames to the display orientation before inference, or the left/right facing logic behind `lead_space` breaks in portrait.
4. **Field test and user study.** The app can serve as the `study/` harness (`viewfinder.py`, `logging.py` are stubs). Log per-frame rule states and the instruction shown, for the study and for threshold calibration.
5. **iOS.** Reuse the shared KMP module, with AVFoundation for the camera and ORT or Core ML for the model.

### Phase 0: golden fixtures (done)

- `Pipeline.decide(detections, frame_dims)` is the deterministic detections → decision path, split out of `process_frame` with no behaviour change. `Pipeline(load_model=False)` builds it without the ONNX weights.
- `golden/generate.py` (`python -m golden.generate`) writes:
  - **15 decision cases** (`golden/cases/`, 1.8 MB). They cover all scene modes (NONE, SINGLE, DOMINANT, FACING_PAIR, MIXED, GROUP_ALIGNED), every rule winning priority, the odd_count gates and fallbacks, hysteresis flips, a 1-frame flicker being suppressed, acquire/lose, a primary track switch, and filtered detections.
  - **4 tracker cases** (`golden/tracker/`): crossing boxes, gaps inside and beyond `track_max_age`, and a new arrival.
- `golden/README.md` documents the format and the comparison rules for the port: exact strings and states; floats to 1e-6 relative / 1e-5 absolute.
- `tests/test_golden.py` re-runs every case against the current code and fails on any drift, including a `rules.yaml` or tracking-config change that wasn't followed by regeneration. I checked that changing one threshold is caught at the exact field. Suite: **137 passed, 1 skipped.**
- **Found while building these, and fixed (2026-10-28):** "click now" used to lag real violations. In `hysteresis_flip`, "Good composition - click now" stayed on screen for 4 frames after the smoothed `limb_clip` had committed VIOLATED, because `ready` added its own 5-frame hysteresis on top of limb_clip's 3.
  - **Fix:** `ready` is now asymmetric. `Pipeline._resolve_ready` calls the new `Smoother.commit()` to drop ready immediately on any committed blocking violation; turning ready back on still needs `ready.hysteresis_frames` all-clear frames. The `rules.yaml` comment was updated (value unchanged, still `UNCALIBRATED`).
  - **Tests:** 3 new tests in `tests/test_ready.py` (all 3 fail without the fix) and 1 in `tests/test_smoothing.py`. The parity test flagged the 4 affected golden cases before regeneration. Suite: **141 passed, 1 skipped.**
  - **Still open:** on recovery, a stale `centrality` instruction shows for 2 frames (`hysteresis_flip` frames 20–21). That's centrality's own hysteresis, not `ready`.

### Phases 2–3: Android app v0.1 (built 2026-10-28, branch `android-app`)

- **Built in GitHub Actions**, since this PC has no JDK or Android SDK. `.github/workflows/android.yml` runs three things: the Python golden parity test, the Kotlin engine tests, and `assembleDebug`. The APK is uploaded as the `wildframe-debug-apk` artifact. First green run: https://github.com/Bluesniper1000/WildFrame/actions/runs/40733197452 (commit `3c10e14`).
- **`android/engine`** is a pure-Kotlin port of the decision path plus the tracker and the output decoder. Its arithmetic mirrors numpy float32 behaviour, and it **passed every golden fixture on the first CI run**, for both decision cases and tracker cases.
- **`android/app`** combines:
  - CameraX preview and analysis (frames rotated upright, 4:3, keep-latest);
  - letterbox + ONNX Runtime 1.20 on the CPU with the INT8 kpfp32 model;
  - tracker → engine;
  - an overlay with the thirds grid, skeletons, subject box, instruction banner and mode notice.
- **Configs:** `golden.generate` now also writes the app's `assets/rules.json` and `model.json`, and `tests/test_golden.py` fails if they drift from the YAML.
- **APK check:** downloaded and inspected. It contains the manifest, 4 dex files, the model (stored uncompressed), both configs, and ORT native libs for arm64-v8a, armeabi-v7a and x86_64. It's debug-signed. **Not yet run on a real phone.**
- **Two CI fixes:**
  - `overlay` inside `FrameLayout.apply{}` resolved to `ViewGroup.getOverlay()`;
  - `density` inside `TextPaint.apply{}` resolved to TextPaint's own field, which would have made the text 18 px.
- **Known limitations (v0.1):**
  - hysteresis is counted in frames, so it's slower at phone frame rates;
  - CPU only (QDQ re-quantization for NNAPI/GPU is Phase 1);
  - portrait and back camera only;
  - no correction arrows;
  - AGPL-3.0 implications of shipping Ultralytics-derived weights beyond testing.

## 12. Open items

- [ ] **Try the APK on a real phone**: check fps / ms per frame, overlay alignment in portrait, and whether instructions match the scene.
- [ ] Android: time-based hysteresis (or a capped analysis rate) so coaching timing doesn't depend on phone speed.

- [x] **Lead-space pan direction was reversed (fixed 2026-10-29).** The rule reported the direction the *subject* should move in the frame, but the message says "Pan {direction}", which a photographer reads as a camera movement, so every lead-space instruction pointed the wrong way. `lead_space` now reports the camera pan direction (toward the side the subject faces) in `wildframe/rules/lead_space.py` and the Kotlin `Rules.kt`. `test_rules.py` asserts the camera direction. Regenerating `golden/` changed only `direction` and `message` in `single_right_no_lead_space` and `four_with_edge_member`. eye_thirds/centrality still describe `direction_x/y` as subject movement, but no message uses those fields.
- [x] "Click now" lagging a real violation: fixed with asymmetric `ready` hysteresis (§11 Phase 0), and `golden/` regenerated.
- [ ] Decide whether a stale per-rule instruction may flash during recovery (`hysteresis_flip` frames 20–21, centrality). Options: leave it; give rules asymmetric hysteresis too (fast to clear a violation); or suppress lower-priority violations for a few frames after a higher-priority one clears. Any change needs `golden/` regenerated.
- [ ] Mobile Phase 1: QDQ re-quantization, input-size sweep, on-device latency.

- [x] **Deployed model decided (2026-10-26):** `configs/model.yaml: weights_path` now points at `ap10k_ft/weights_quant/best_int8_kpfp32.onnx` (−0.011 AP vs FP32, 10% faster at 1 thread).
  - Tests: 102 passed, 1 skipped.
  - Pipeline smoke run on the deer frame, repeated 12 times so hysteresis can settle: frames 0–1 say `insufficient_history`; from frame 2 on, "Zoom out - the left_front is clipped at the frame bottom". That's correct, since the legs are cut off by the bottom edge. About 34 ms/frame at the default thread count.
  - **That file is gitignored** (`ap10k_ft/`). A fresh clone needs it rebuilt with `python -m bench.quantize --model ap10k_ft/weights_quant/best_fp32.onnx --out ap10k_ft/weights_quant/best_int8_kpfp32.onnx --exclude-regex cv4 --data <ap10k root>`. Its manifest (`best_int8_kpfp32.json`) records the calibration hash, so a rebuild can be checked against it.
- [ ] Restore `scripts/measure_overlap_keypoints.py`, and decide whether `scripts/` should stop being gitignored.
- [x] Run the harness over ASTMA's yolo26s (both heads) for a size comparison (done 2026-10-28, see §10 "YOLO26s").
- [ ] Look into the orientation heuristic's sensitivity. YOLO26s doesn't improve L/R agreement over YOLO26n (takeaway 10), which suggests borderline cases near the frontal thresholds (`frontal_yaw_max_deg`, the 0.35 eye-offset ratio in `orientation.py`). Separately, label a sample with human facing direction to check whether the heuristic is *correct* on GT keypoints at all.
- [ ] Add an RTMPose top-down adapter (port ASTMA `preprocess.rtm_crop`/`rtm_decode`) for GT-box and detector-box rows, or score MMPose results via `--preds`.
- [ ] Repeat the latency runs under controlled conditions (see takeaways 7 and 14, all models in one session), and fold the 1-thread measurement into `bench/run.py` (currently a scratch script).
- [ ] Fix single-image mode hysteresis (§5).
- [ ] Update `CLAUDE.md` for centrality, ready and the OR-group.
