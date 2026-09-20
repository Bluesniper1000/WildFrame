"""One-off generator for performance_testing.ipynb. Run once, then delete."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src, cid):
    c = nbf.v4.new_markdown_cell(src)
    c.id = cid
    cells.append(c)


def code(src, cid):
    c = nbf.v4.new_code_cell(src)
    c.id = cid
    cells.append(c)


md(
    "# Model performance\n\n"
    "Evaluates the fine-tuned AP-10K pose model (`ap10k_ft/weights/best.pt`) directly against "
    "`data/ap10k` (val split 1):\n\n"
    "1. Bounding-box detection accuracy\n"
    "2. Keypoint accuracy\n"
    "3. Per-keypoint accuracy\n"
    "4. Animal detection + pose association\n"
    "5. Inference latency\n"
    "6. FPS",
    "intro",
)

code(
    '''import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = Path("../ap10k_ft/weights/best.pt")
ANN_FILE = Path("../data/ap10k/annotations/ap10k-val-split1.json")
IMG_DIR = Path("../data/ap10k/data")

KPT_NAMES = [
    "left_eye", "right_eye", "nose", "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
]
IOU_THRESH = 0.5      # box match threshold
PCK_THRESH = 0.1       # keypoint correct if dist < PCK_THRESH * sqrt(gt_box_area)''',
    "imports",
)

md("## Load model + ground truth", "md-load")

code(
    '''model = YOLO(str(WEIGHTS))

coco = json.loads(ANN_FILE.read_text())
images_by_id = {im["id"]: im for im in coco["images"]}
gt_by_image = {}
for ann in coco["annotations"]:
    if ann.get("num_keypoints", 0) == 0:
        continue
    gt_by_image.setdefault(ann["image_id"], []).append(ann)

image_ids = sorted(gt_by_image)
print(f"{len(image_ids)} val images, {sum(len(v) for v in gt_by_image.values())} GT instances")''',
    "load",
)

md("## Run inference over the val split", "md-infer")

code(
    '''def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


img_paths = [IMG_DIR / images_by_id[i]["file_name"] for i in image_ids]
predictions = {}  # image_id -> dict(boxes[N,4], scores[N], kpts[N,17,2])

all_results = []
BATCH = 16
for start in range(0, len(img_paths), BATCH):
    chunk = img_paths[start:start + BATCH]
    all_results.extend(model.predict(chunk, conf=0.25, verbose=False, batch=BATCH))
    if (start // BATCH) % 10 == 0:
        print(f"  {start + len(chunk)}/{len(img_paths)}")

for image_id, r in zip(image_ids, all_results):
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), dtype=np.float32)
    scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,), dtype=np.float32)
    kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None and len(r.boxes) else np.zeros((0, 17, 2), dtype=np.float32)
    predictions[image_id] = {"boxes": boxes, "scores": scores, "kpts": kpts}

print(f"inference done on {len(predictions)} images")''',
    "infer",
)

md(
    "## Ground-truth <-> prediction matching\n\n"
    "Each predicted box is greedily matched (highest confidence first) to the best unmatched "
    "ground-truth box with IoU >= 0.5, mirroring standard detection evaluation. Keypoint and "
    "association metrics below are computed only from this matching, so a wrong box match "
    "directly shows up as a wrong pose.",
    "md-match",
)

code(
    '''def box_iou(a, b):
    """a: [N,4], b: [M,4] xyxy -> [N,M] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def match_greedy(gt_boxes, pred_boxes, pred_scores, iou_thresh=IOU_THRESH):
    """Returns list of (gt_idx, pred_idx, iou) matches, plus unmatched gt/pred index lists."""
    order = np.argsort(-pred_scores)
    ious = box_iou(pred_boxes, gt_boxes)  # [P, G]
    matched_gt, matched_pred = set(), set()
    matches = []
    for p in order:
        if len(gt_boxes) == 0:
            break
        g = int(np.argmax(ious[p])) if len(ious[p]) else -1
        if g == -1 or g in matched_gt:
            candidates = [gg for gg in range(len(gt_boxes)) if gg not in matched_gt]
            if not candidates:
                continue
            g = max(candidates, key=lambda gg: ious[p, gg])
        if ious[p, g] >= iou_thresh:
            matches.append((g, int(p), float(ious[p, g])))
            matched_gt.add(g)
            matched_pred.add(int(p))
    unmatched_gt = [g for g in range(len(gt_boxes)) if g not in matched_gt]
    unmatched_pred = [p for p in range(len(pred_boxes)) if p not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


all_matches = []  # (image_id, gt_ann, pred_boxes_row, pred_kpts_row, iou)
n_gt = n_pred = n_tp = 0

for image_id in image_ids:
    anns = gt_by_image[image_id]
    gt_boxes = np.stack([xywh_to_xyxy(a["bbox"]) for a in anns])
    pred = predictions[image_id]

    matches, unmatched_gt, unmatched_pred = match_greedy(gt_boxes, pred["boxes"], pred["scores"])

    n_gt += len(anns)
    n_pred += len(pred["boxes"])
    n_tp += len(matches)

    for g, p, iou in matches:
        all_matches.append((image_id, anns[g], pred["boxes"][p], pred["kpts"][p], iou))

n_fp = n_pred - n_tp
n_fn = n_gt - n_tp
print(f"GT instances={n_gt}  predictions={n_pred}  matched(TP)={n_tp}  FP={n_fp}  FN={n_fn}")''',
    "match",
)

md("## 1. Bounding-box detection accuracy", "md-box")

code(
    '''box_precision = n_tp / n_pred if n_pred else 0.0
box_recall = n_tp / n_gt if n_gt else 0.0
box_f1 = 2 * box_precision * box_recall / (box_precision + box_recall) if (box_precision + box_recall) else 0.0
mean_iou = float(np.mean([m[4] for m in all_matches])) if all_matches else 0.0

print(f"Box detection @ IoU>={IOU_THRESH}")
print(f"  precision = {box_precision:.4f}")
print(f"  recall    = {box_recall:.4f}")
print(f"  F1        = {box_f1:.4f}")
print(f"  mean IoU of matched boxes = {mean_iou:.4f}")''',
    "box-accuracy",
)

md("## 2. Keypoint accuracy", "md-kpt")

code(
    '''def gt_box_scale(bbox):
    _, _, w, h = bbox
    return np.sqrt(max(w * h, 1e-6))


n_correct = 0
n_visible = 0
per_kpt_correct = np.zeros(17, dtype=np.int64)
per_kpt_total = np.zeros(17, dtype=np.int64)

for image_id, ann, pred_box, pred_kpts, iou in all_matches:
    kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
    scale = gt_box_scale(ann["bbox"])
    thresh = PCK_THRESH * scale

    visible = kpts[:, 2] > 0
    dist = np.linalg.norm(kpts[:, :2] - pred_kpts, axis=1)
    correct = visible & (dist < thresh)

    n_visible += int(visible.sum())
    n_correct += int(correct.sum())
    per_kpt_total += visible.astype(np.int64)
    per_kpt_correct += correct.astype(np.int64)

keypoint_accuracy = n_correct / n_visible if n_visible else 0.0
print(f"PCK@{PCK_THRESH} (dist < {PCK_THRESH} * sqrt(gt box area)), matched instances only")
print(f"  visible GT keypoints = {n_visible}")
print(f"  correct              = {n_correct}")
print(f"  keypoint accuracy    = {keypoint_accuracy:.4f}")''',
    "kpt-accuracy",
)

md("## 3. Per-keypoint accuracy", "md-per-kpt")

code(
    '''per_kpt_acc = np.divide(
    per_kpt_correct, per_kpt_total,
    out=np.zeros_like(per_kpt_correct, dtype=np.float64),
    where=per_kpt_total > 0,
)

order = np.argsort(per_kpt_acc)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([KPT_NAMES[i] for i in order], per_kpt_acc[order])
ax.set_xlabel(f"PCK@{PCK_THRESH}")
ax.set_xlim(0, 1)
ax.set_title("Per-keypoint accuracy")
for i, idx in enumerate(order):
    ax.text(per_kpt_acc[idx] + 0.01, i, f"{per_kpt_acc[idx]:.2f} (n={per_kpt_total[idx]})", va="center", fontsize=8)
fig.tight_layout()
plt.show()''',
    "per-kpt-accuracy",
)

md(
    "## 4. Animal detection + pose association\n\n"
    "Whether each detected animal's predicted keypoints stay attached to the *correct* animal, "
    "not a neighboring one. Reuses the IoU matching above: a wrong association would show up as "
    "a matched box with a good IoU but a poor keypoint score. Also checks multi-animal images "
    "specifically, where mismatched association is most likely.",
    "md-assoc",
)

code(
    '''multi_animal_images = [i for i in image_ids if len(gt_by_image[i]) > 1]

def image_pck(image_id):
    matches_here = [m for m in all_matches if m[0] == image_id]
    correct = total = 0
    for _, ann, _, pred_kpts, _ in matches_here:
        kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        scale = gt_box_scale(ann["bbox"])
        visible = kpts[:, 2] > 0
        dist = np.linalg.norm(kpts[:, :2] - pred_kpts, axis=1)
        correct += int((visible & (dist < PCK_THRESH * scale)).sum())
        total += int(visible.sum())
    return correct / total if total else None

single_pck = [image_pck(i) for i in image_ids if len(gt_by_image[i]) == 1]
multi_pck = [image_pck(i) for i in multi_animal_images]
single_pck = [v for v in single_pck if v is not None]
multi_pck = [v for v in multi_pck if v is not None]

fully_matched_multi = sum(
    1 for i in multi_animal_images
    if len([m for m in all_matches if m[0] == i]) == len(gt_by_image[i])
)

print(f"images with exactly 1 animal: {len(image_ids) - len(multi_animal_images)}")
print(f"images with >1 animal:        {len(multi_animal_images)}")
print(f"  of which all GT animals matched to a prediction: {fully_matched_multi} "
      f"({fully_matched_multi / len(multi_animal_images):.1%})" if multi_animal_images else "")
print(f"mean per-image PCK, single-animal images: {np.mean(single_pck):.4f}" if single_pck else "n/a")
print(f"mean per-image PCK, multi-animal images:  {np.mean(multi_pck):.4f}" if multi_pck else "n/a")''',
    "association",
)

md("## 5. Inference latency", "md-latency")

code(
    '''N_LATENCY = 100
latency_paths = img_paths[:N_LATENCY]

model.predict(latency_paths[0], verbose=False)  # warmup

times_ms = []
for p in latency_paths:
    t0 = time.perf_counter()
    model.predict(p, conf=0.25, verbose=False)
    times_ms.append((time.perf_counter() - t0) * 1000)

times_ms = np.array(times_ms)
print(f"single-image inference over {N_LATENCY} images (CPU, batch=1)")
print(f"  mean   = {times_ms.mean():.1f} ms")
print(f"  median = {np.median(times_ms):.1f} ms")
print(f"  p95    = {np.percentile(times_ms, 95):.1f} ms")
print(f"  min/max = {times_ms.min():.1f} / {times_ms.max():.1f} ms")''',
    "latency",
)

md("## 6. FPS", "md-fps")

code(
    '''fps_single = 1000.0 / times_ms.mean()

t0 = time.perf_counter()
model.predict(latency_paths, conf=0.25, verbose=False, batch=16)
batch_elapsed = time.perf_counter() - t0
fps_batch = len(latency_paths) / batch_elapsed

print(f"FPS (single-image, batch=1):  {fps_single:.2f}")
print(f"FPS (batched, batch=16):      {fps_batch:.2f}")''',
    "fps",
)

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "venv (3.14.6.final.0)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.6",
    },
}

nbf.write(nb, "performance_testing.ipynb")
print("wrote performance_testing.ipynb with", len(cells), "cells")
