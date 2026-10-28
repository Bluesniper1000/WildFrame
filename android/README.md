# WildFrame for Android

A native port of WildFrame's live composition coach: the back camera feeds YOLO26n pose (ONNX
Runtime, `assets/wildframe_pose.onnx` = the INT8 export with the keypoint branch in FP32), the
Kotlin rule engine picks one instruction, and an overlay draws it over the preview.

## Get the APK

There's no local Android toolchain needed. GitHub Actions (`.github/workflows/android.yml`) builds
every push that touches `android/`, `golden/`, `wildframe/` or `configs/`:

1. Open the repo's **Actions** tab → the latest **Android** run → **Artifacts** → `wildframe-debug-apk`.
2. Unzip it and copy `app-debug.apk` to the phone.
3. Open it on the phone and allow "install unknown apps" for the file manager or browser when asked.
4. Launch **WildFrame** and grant camera access.

It's a debug build (debug-signed, not for the Play Store). Android 8.0 (API 26) or newer.

## Layout

- `engine/`: pure Kotlin (no Android) port of the decision path in `wildframe/`: scene resolution,
  orientation, the five rules, hysteresis (including the asymmetric `ready`), priority, templates,
  the ByteTrack-style tracker, and the model-output decoder. It's tested on the JVM against the
  repo's `golden/` fixtures (`GoldenTest`), so it must reproduce the Python pipeline exactly.
- `app/`: CameraX preview + analysis, letterbox + ONNX Runtime inference (`PoseModel`), the
  per-frame analyzer, and the overlay (thirds grid, skeletons, subject box, instruction banner).
- `app/src/main/assets/rules.json` and `model.json` are generated from `configs/*.yaml` by
  `python -m golden.generate`. `tests/test_golden.py` fails if they drift.

## Known limitations (v0.1)

- **Hysteresis is counted in frames.** On a phone at roughly 8–15 fps, `hysteresis_frames: 5` is
  about 0.3–0.6 s, slower than on a desktop. It needs a time-based setting or a capped analysis rate.
- **CPU only.** The QOperator INT8 model runs on ONNX Runtime's CPU execution provider. NNAPI/GPU
  need a QDQ re-quantization (mobile plan, Phase 1).
- Portrait only, back camera only. No correction arrows yet (text instruction only).
- **Licence:** the pose weights come from an Ultralytics (AGPL-3.0) model. Distributing the app
  beyond personal or research testing has AGPL implications to check first.
