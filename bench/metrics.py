"""Pure scoring functions for the pose benchmark. No model or file I/O here, so every metric can be
unit-tested on synthetic annotations (tests/test_bench_metrics.py).

Inputs are a COCO-format ground-truth dict (see ap10k.load_gt) and a list of COCO-format results:
{"image_id", "category_id", "keypoints": [x, y, conf] * 17, "score", optional "bbox": [x, y, w, h]}.
"""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import dataclass

import numpy as np

from bench.ap10k import AP10K_SIGMAS
from wildframe.contracts import Detection
from wildframe.orientation import compute_orientation

COCO_STAT_NAMES = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
YAW_STATES = ["left", "right", "frontal", "unknown"]


# --- OKS AP -------------------------------------------------------------------------------------

def oks_ap(gt: dict, results: list[dict], sigmas: np.ndarray = AP10K_SIGMAS) -> dict[str, float]:
    """COCO keypoint AP/AR with the given per-keypoint sigmas, class-agnostic.

    pycocotools with kpt_oks_sigmas overridden computes the same OKS as xtcocotools (MMPose's
    evaluator) for plain COCO-style keypoint eval; xtcocotools doesn't build on Windows/py3.12."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        return {k: 0.0 for k in COCO_STAT_NAMES}
    cat_id = gt["categories"][0]["id"]
    results = [dict(r, category_id=cat_id) for r in results]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "keypoints")
        ev.params.kpt_oks_sigmas = np.asarray(sigmas, dtype=np.float64)
        ev.params.imgIds = sorted(coco_gt.getImgIds())
        ev.params.catIds = [cat_id]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(COCO_STAT_NAMES, ev.stats)}


# --- Matching -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    image_id: int
    gt_kpts: np.ndarray    # (17, 3): x, y, visibility flag (0 = unlabeled)
    gt_bbox: np.ndarray    # (4,) xywh
    pred_kpts: np.ndarray  # (17, 3): x, y, confidence
    score: float


def _bbox_from_kpts(kpts: np.ndarray) -> np.ndarray:
    x0, y0 = kpts[:, 0].min(), kpts[:, 1].min()
    return np.array([x0, y0, kpts[:, 0].max() - x0, kpts[:, 1].max() - y0], dtype=np.float64)


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def match_predictions(
    gt: dict, results: list[dict], min_score: float = 0.25, iou_thresh: float = 0.5
) -> tuple[list[Match], int]:
    """Greedy one-to-one matching of predictions to ground truth by *box* IoU, highest score first.

    Matching on boxes rather than keypoints keeps the keypoint metrics unbiased: a prediction isn't
    paired with a GT animal because its keypoints happen to be close. Only predictions with
    score >= min_score take part (the deployment operating point, model.yaml conf_threshold), so the
    keypoint metrics describe what the rules would actually receive. GT instances with no labeled
    keypoints are ignored. Returns (matches, number of GT instances considered)."""
    gts_by_image: dict[int, list[dict]] = {}
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("num_keypoints", 0) == 0:
            continue
        gts_by_image.setdefault(ann["image_id"], []).append(ann)
    preds_by_image: dict[int, list[dict]] = {}
    for r in results:
        if r["score"] >= min_score:
            preds_by_image.setdefault(r["image_id"], []).append(r)

    matches: list[Match] = []
    n_gt = 0
    for image_id in sorted(gts_by_image):
        gts = gts_by_image[image_id]
        n_gt += len(gts)
        gt_boxes = [np.asarray(g["bbox"], dtype=np.float64) for g in gts]
        taken = [False] * len(gts)
        for r in sorted(preds_by_image.get(image_id, []), key=lambda r: -r["score"]):
            kp = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 3)
            box = np.asarray(r["bbox"], dtype=np.float64) if "bbox" in r else _bbox_from_kpts(kp)
            best, best_iou = -1, iou_thresh
            for j, g_box in enumerate(gt_boxes):
                if taken[j]:
                    continue
                iou = _iou_xywh(box, g_box)
                if iou >= best_iou:
                    best, best_iou = j, iou
            if best >= 0:
                taken[best] = True
                g = gts[best]
                matches.append(Match(
                    image_id=image_id,
                    gt_kpts=np.asarray(g["keypoints"], dtype=np.float64).reshape(-1, 3),
                    gt_bbox=gt_boxes[best],
                    pred_kpts=kp,
                    score=float(r["score"]),
                ))
    return matches, n_gt


# --- Per-keypoint localisation ------------------------------------------------------------------

def keypoint_errors(matches: list[Match]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bbox-normalised distance, sqrt-area-normalised distance, labeled mask), each (M, 17).

    The first follows MMPose's PCK 'bbox' normalisation: the x and y error are divided by the GT box
    width and height respectively before taking the norm. The second is the NME convention."""
    if not matches:
        empty = np.zeros((0, 17))
        return empty, empty, empty.astype(bool)
    gt = np.stack([m.gt_kpts for m in matches])
    pred = np.stack([m.pred_kpts for m in matches])
    wh = np.stack([np.maximum(m.gt_bbox[2:4], 1.0) for m in matches])  # (M, 2)
    diff = pred[:, :, :2] - gt[:, :, :2]
    pck_dist = np.linalg.norm(diff / wh[:, None, :], axis=2)
    nme_dist = np.linalg.norm(diff, axis=2) / np.sqrt(wh[:, 0] * wh[:, 1])[:, None]
    labeled = gt[:, :, 2] > 0
    return pck_dist, nme_dist, labeled


def pck_per_keypoint(matches: list[Match], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Per-keypoint PCK@alpha over matched, labeled keypoints. Returns (pck (17,), counts (17,));
    pck is NaN where a keypoint has no labeled instances."""
    pck_dist, _, labeled = keypoint_errors(matches)
    counts = labeled.sum(axis=0)
    hits = ((pck_dist < alpha) & labeled).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, hits / np.maximum(counts, 1), np.nan), counts


def nme_per_keypoint(matches: list[Match]) -> np.ndarray:
    """Per-keypoint mean error over matched, labeled keypoints, normalised by sqrt(GT box area)."""
    _, nme_dist, labeled = keypoint_errors(matches)
    out = np.full(17, np.nan)
    for k in range(17):
        if labeled[:, k].any():
            out[k] = float(nme_dist[labeled[:, k], k].mean())
    return out


# --- Orientation --------------------------------------------------------------------------------

def _yaw_state(kpts: np.ndarray, orientation_cfg: dict) -> str:
    det = Detection(bbox_xyxy=np.zeros(4, np.float32), keypoints=kpts.astype(np.float32), det_conf=1.0)
    return compute_orientation(det, orientation_cfg).yaw_state


def gt_as_confident_keypoints(gt_kpts: np.ndarray) -> np.ndarray:
    """GT keypoints in the predicted-keypoint layout: labeled -> confidence 1, unlabeled -> 0."""
    out = gt_kpts.copy()
    out[:, 2] = (gt_kpts[:, 2] > 0).astype(np.float64)
    return out


def orientation_eval(matches: list[Match], orientation_cfg: dict) -> dict:
    """Facing direction derived by wildframe.orientation from predicted keypoints vs. the same
    derivation from ground truth, with the deployed configs/rules.yaml orientation block.

    lr_accuracy: over GT facing left/right (what lead_space's yaw gate needs), fraction predicted
    the same side. three_way_accuracy: over GT left/right/frontal. GT 'unknown' (too few labeled
    head/torso keypoints) is excluded from both and reported as a count."""
    confusion = {g: {p: 0 for p in YAW_STATES} for g in YAW_STATES}
    for m in matches:
        g = _yaw_state(gt_as_confident_keypoints(m.gt_kpts), orientation_cfg)
        p = _yaw_state(m.pred_kpts, orientation_cfg)
        confusion[g][p] += 1

    def acc(gt_states: list[str]) -> float:
        total = sum(sum(confusion[g].values()) for g in gt_states)
        correct = sum(confusion[g][g] for g in gt_states)
        return correct / total if total else float("nan")

    return {
        "lr_accuracy": acc(["left", "right"]),
        "three_way_accuracy": acc(["left", "right", "frontal"]),
        "n_lr": sum(sum(confusion[g].values()) for g in ["left", "right"]),
        "n_gt_unknown": sum(confusion["unknown"].values()),
        "confusion": confusion,
    }


# --- Visibility calibration ---------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic, ties given average rank.
    NaN if only one class is present."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def visibility_auroc(matches: list[Match]) -> tuple[np.ndarray, float]:
    """AUROC of predicted keypoint confidence for predicting GT visibility (labeled vs. unlabeled),
    over matched instances. Returns (per-keypoint (17,), pooled over all keypoints)."""
    if not matches:
        return np.full(17, np.nan), float("nan")
    conf = np.stack([m.pred_kpts[:, 2] for m in matches])
    vis = np.stack([m.gt_kpts[:, 2] > 0 for m in matches])
    per_kpt = np.array([auroc(conf[:, k], vis[:, k]) for k in range(17)])
    return per_kpt, auroc(conf.ravel(), vis.ravel())
