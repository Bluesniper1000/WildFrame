"""Benchmark one pose model on an AP-10K split and write a summary JSON.

  python -m bench.run --model ap10k_ft/weights_quant/best_fp32.onnx --kind yolo-o2m \
      --family yolo26n --precision fp32 --data ../ASTMA-wildlife/pose/data/ap10k

Writes <out>/<family>_<precision>.json (all metrics) and <out>/<family>_<precision>.preds.json
(COCO-format predictions, reused with --reuse-preds). External predictions from another tool
(e.g. MMPose top-down results) can be scored with --preds instead of running a model; latency and
complexity are then skipped unless --model is also given.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from bench import metrics
from bench.ap10k import KEYPOINT_NAMES, RULE_KEYPOINTS, load_gt
from bench.efficiency import count_gflops, count_params, cpu_name, latency_ms
from bench.predict import KINDS, make_session, predict_image, predict_split
from wildframe.pose.backend import REPO_ROOT, _letterbox
from wildframe.pose.skeleton import KP

EYES = [KP.LEFT_EYE, KP.RIGHT_EYE]
PAWS = [KP.LEFT_FRONT_PAW, KP.RIGHT_FRONT_PAW, KP.LEFT_BACK_PAW, KP.RIGHT_BACK_PAW]


def _nan_to_none(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _pooled_pck(matches: list[metrics.Match], kpts: list[int], alpha: float) -> float:
    dist, _, labeled = metrics.keypoint_errors(matches)
    sel = labeled[:, kpts]
    return float(((dist[:, kpts] < alpha) & sel).sum() / sel.sum()) if sel.sum() else float("nan")


def _pooled_nme(matches: list[metrics.Match], kpts: list[int]) -> float:
    _, nme, labeled = metrics.keypoint_errors(matches)
    sel = labeled[:, kpts]
    return float(nme[:, kpts][sel].mean()) if sel.sum() else float("nan")


def score(gt: dict, results: list[dict], orientation_cfg: dict, alpha: float, min_score: float) -> dict:
    """Every accuracy metric for one set of predictions."""
    ap = metrics.oks_ap(gt, results)
    matches, n_gt = metrics.match_predictions(gt, results, min_score=min_score)
    pck, counts = metrics.pck_per_keypoint(matches, alpha)
    nme = metrics.nme_per_keypoint(matches)
    vis_kpt, vis_pooled = metrics.visibility_auroc(matches)
    paw_conf = np.concatenate([[m.pred_kpts[k, 2] for m in matches] for k in PAWS]) if matches else np.zeros(0)
    paw_vis = np.concatenate([[m.gt_kpts[k, 2] > 0 for m in matches] for k in PAWS]) if matches else np.zeros(0)
    return {
        "ap": ap,
        "matching": {"min_score": min_score, "iou": 0.5, "n_gt": n_gt, "n_matched": len(matches),
                     "recall": len(matches) / n_gt if n_gt else None},
        "pck_alpha": alpha,
        "pck": {KEYPOINT_NAMES[k]: _nan_to_none(float(pck[k])) for k in range(17)},
        "pck_counts": {KEYPOINT_NAMES[k]: int(counts[k]) for k in range(17)},
        "nme": {KEYPOINT_NAMES[k]: _nan_to_none(float(nme[k])) for k in range(17)},
        "eye_pck": _nan_to_none(_pooled_pck(matches, EYES, alpha)),
        "eye_nme": _nan_to_none(_pooled_nme(matches, EYES)),
        "rule_kpt_pck": _nan_to_none(_pooled_pck(matches, list(RULE_KEYPOINTS), alpha)),
        "orientation": {k: _nan_to_none(v) for k, v in metrics.orientation_eval(matches, orientation_cfg).items()},
        "visibility_auroc": {
            "per_keypoint": {KEYPOINT_NAMES[k]: _nan_to_none(float(vis_kpt[k])) for k in range(17)},
            "pooled": _nan_to_none(vis_pooled),
            "paws": _nan_to_none(metrics.auroc(paw_conf, paw_vis)),
        },
    }


def measure_latency(sess, gt: dict, images_dir: Path, kind: str, model_cfg: dict, warmup: int, n: int) -> dict:
    """model_only: session.run on a letterboxed 640 input. e2e: letterbox + run + decode/NMS at the
    deployment operating point from configs/model.yaml (conf_threshold, iou_threshold_nms, max_det),
    cycling through real test images (decoded beforehand, so JPEG decoding isn't timed)."""
    infos = sorted(gt["images"], key=lambda im: im["id"])[: max(1, min(n, 50))]
    imgs = [cv2.imread(str(images_dir / im["file_name"])) for im in infos]
    batch = _letterbox(imgs[0], 640)[0]
    name = sess.get_inputs()[0].name
    model_only = latency_ms(lambda: sess.run(None, {name: batch}), warmup, n)

    it = iter(range(10**9))
    conf, iou, max_det = float(model_cfg["conf_threshold"]), float(model_cfg["iou_threshold_nms"]), int(model_cfg["max_det"])
    e2e = latency_ms(lambda: predict_image(sess, imgs[next(it) % len(imgs)], kind, conf=conf, iou=iou, max_det=max_det), warmup, n)
    return {"model_only_ms": model_only, "e2e_ms": e2e,
            "e2e_settings": {"conf": conf, "iou": iou, "max_det": max_det}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="ONNX model to run (and time)")
    p.add_argument("--kind", choices=KINDS, default="yolo-o2m")
    p.add_argument("--family", required=True, help="model name shared by its fp32/int8 rows, e.g. yolo26n")
    p.add_argument("--precision", required=True, choices=["fp32", "int8"])
    p.add_argument("--variant", default="", help="distinguishes recipes of one precision, e.g. kpfp32 / allconv / deployed")
    p.add_argument("--input-res", default="640 full-frame", help="stated in the table, e.g. '256x256 crop'")
    p.add_argument("--preds", help="score an existing COCO-format results JSON instead of running --model")
    p.add_argument("--reuse-preds", action="store_true", help="reuse <out>/<name>.preds.json if present")
    p.add_argument("--data", default=str(REPO_ROOT / "data/ap10k"), help="AP-10K root (annotations/, data/)")
    p.add_argument("--split", default="ap10k-test-split1.json")
    p.add_argument("--flops-model", help="fp32 ONNX to count params/GFLOPs on (defaults to --model if fp32)")
    p.add_argument("--out", default=str(REPO_ROOT / "results/bench"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--pck-alpha", type=float, default=0.05)
    p.add_argument("--match-min-score", type=float, default=None,
                   help="min prediction score to take part in keypoint metrics (default: model.yaml conf_threshold)")
    p.add_argument("--limit", type=int, default=0, help="debug only: first N images")
    a = p.parse_args()

    if not a.model and not a.preds:
        p.error("need --model or --preds")
    data = Path(a.data)
    images_dir = data / "data"
    gt = load_gt(data / "annotations" / a.split)
    if a.limit:
        keep = {im["id"] for im in sorted(gt["images"], key=lambda im: im["id"])[: a.limit]}
        gt["images"] = [im for im in gt["images"] if im["id"] in keep]
        gt["annotations"] = [an for an in gt["annotations"] if an["image_id"] in keep]

    model_cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    rules_cfg = yaml.safe_load((REPO_ROOT / "configs/rules.yaml").read_text())
    min_score = a.match_min_score if a.match_min_score is not None else float(model_cfg["conf_threshold"])

    name = f"{a.family}_{a.precision}" + (f"_{a.variant}" if a.variant else "")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{name}.preds.json"

    sess = make_session(a.model, a.threads) if a.model else None
    t0 = time.time()
    if a.preds:
        results = json.loads(Path(a.preds).read_text())
    elif a.reuse_preds and preds_path.exists() and not a.limit:
        results = json.loads(preds_path.read_text())
    else:
        results = predict_split(sess, gt, images_dir, a.kind)
        if not a.limit:
            preds_path.write_text(json.dumps(results))
    predict_s = time.time() - t0

    summary = {
        "name": name, "family": a.family, "precision": a.precision, "variant": a.variant, "input_res": a.input_res,
        "model": Path(a.model).name if a.model else None, "kind": a.kind if a.model else "external",
        "preds_source": a.preds or str(preds_path.name),
        "split": a.split, "n_images": len(gt["images"]), "n_preds": len(results),
        "protocol": {"evaluator": "pycocotools COCOeval keypoints", "sigmas": "ap10k (MMPose)",
                     "pred_conf": 0.001, "pred_nms_iou": 0.7, "pred_max_det": 300, "class_agnostic": True},
        **score(gt, results, rules_cfg["orientation"], a.pck_alpha, min_score),
        "predict_wall_s": round(predict_s, 1),
        "debug_limit": a.limit or None,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    flops_model = a.flops_model or (a.model if a.precision == "fp32" else None)
    if flops_model:
        summary["params_m"] = count_params(flops_model) / 1e6
        summary["gflops"] = count_gflops(flops_model)
        summary["flops_model"] = Path(flops_model).name

    if sess is not None and not a.no_latency:
        summary["latency"] = measure_latency(sess, gt, images_dir, a.kind, model_cfg, a.warmup, a.n)
        summary["hardware"] = {"cpu": cpu_name(), "threads": a.threads, "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
                               "batch": 1, "input": "640x640 letterbox"}

    summary_path = out_dir / (f"{name}.json" if not a.limit else f"{name}.limit{a.limit}.json")
    summary_path.write_text(json.dumps(summary, indent=1))
    ap, o = summary["ap"], summary["orientation"]
    print(f"{name}: AP {ap['AP']:.3f}  eye PCK@{a.pck_alpha} {summary['eye_pck']}  "
          f"LR orient {o['lr_accuracy']}  vis AUROC {summary['visibility_auroc']['pooled']}  -> {summary_path}")


if __name__ == "__main__":
    main()
