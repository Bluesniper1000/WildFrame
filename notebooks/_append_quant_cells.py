"""One-off script to append a quantization comparison section (FP32/FP16/INT8) to
performance_testing.ipynb. Run once from notebooks/, then delete."""
import nbformat as nbf

nb = nbf.read("performance_testing.ipynb", as_version=4)
cells = list(nb["cells"])


def md(src, cid):
    c = nbf.v4.new_markdown_cell(src)
    c.id = cid
    cells.append(c)


def code(src, cid):
    c = nbf.v4.new_code_cell(src)
    c.id = cid
    cells.append(c)


md(
    "## 7. Quantization comparison (FP32 vs FP16 vs INT8)\n\n"
    "Exports the same finetuned checkpoint to ONNX at three precisions and reruns "
    "the six tests above (bounding-box accuracy, keypoint accuracy, per-keypoint accuracy, "
    "animal detection + pose association, latency, FPS) for each, on the identical val split. "
    "INT8 uses static calibration (ONNX Runtime) over a sample of the AP-10K train split.",
    "md-quant-intro",
)

code(
    '''import shutil

QUANT_DIR = Path("../ap10k_ft/weights_quant")
QUANT_DIR.mkdir(exist_ok=True)
CALIB_DIR = Path("../data/ap10k_calib")
N_CALIB = 300''',
    "quant-setup",
)

md(
    "### Build INT8 calibration set\n\n"
    "ONNX Runtime static INT8 quantization needs a sample of real images (with YOLO-pose "
    "labels, purely for the dataloader) to estimate activation ranges. Built once from a random "
    "sample of the train split (not the val split used for evaluation, to avoid calibrating on "
    "test data); reused on rerun if it already exists.",
    "md-calib-build",
)

code(
    '''def build_calibration_set(n=N_CALIB, seed=0):
    if (CALIB_DIR / "data.yaml").exists():
        print(f"calibration set already exists at {CALIB_DIR}")
        return CALIB_DIR / "data.yaml"

    import random

    train_ann = Path("../data/ap10k/annotations/ap10k-train-split1.json")
    coco_train = json.loads(train_ann.read_text())
    images_by_id_train = {im["id"]: im for im in coco_train["images"]}
    anns_by_image_train = {}
    for a in coco_train["annotations"]:
        if a.get("num_keypoints", 0) == 0:
            continue
        anns_by_image_train.setdefault(a["image_id"], []).append(a)

    ids = list(anns_by_image_train.keys())
    random.Random(seed).shuffle(ids)
    ids = ids[:n]

    (CALIB_DIR / "images").mkdir(parents=True, exist_ok=True)
    (CALIB_DIR / "labels").mkdir(parents=True, exist_ok=True)

    for iid in ids:
        im = images_by_id_train[iid]
        w, h = im["width"], im["height"]
        src = IMG_DIR / im["file_name"]
        dst_img = CALIB_DIR / "images" / im["file_name"]
        if not dst_img.exists():
            os.link(src, dst_img)

        lines = []
        for a in anns_by_image_train[iid]:
            x, y, bw, bh = a["bbox"]
            cx, cy, nw, nh = (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h
            row = [0, cx, cy, nw, nh]
            kpts = a["keypoints"]
            for i in range(17):
                kx, ky, kv = kpts[3 * i], kpts[3 * i + 1], kpts[3 * i + 2]
                row += [kx / w, ky / h, kv]
            lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row))
        (CALIB_DIR / "labels" / f"{Path(im['file_name']).stem}.txt").write_text("\\n".join(lines))

    yaml_path = CALIB_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {CALIB_DIR.resolve()}\\ntrain: images\\nval: images\\nkpt_shape: [17, 3]\\nnames:\\n  0: animal\\n"
    )
    print(f"built calibration set: {len(ids)} images -> {CALIB_DIR}")
    return yaml_path


import os

calib_yaml = build_calibration_set()''',
    "calib-build",
)

md("### Export FP32 / FP16 / INT8 ONNX models", "md-quant-export")

code(
    '''def export_variant(quantize, suffix, data=None):
    dst = QUANT_DIR / f"best_{suffix}.onnx"
    if dst.exists():
        print(f"{suffix}: already exported -> {dst}")
        return dst
    kwargs = {"format": "onnx", "dynamic": True}
    if quantize is not None:
        kwargs["quantize"] = quantize
    if data is not None:
        kwargs["data"] = str(data)
    exported = Path(YOLO(str(WEIGHTS)).export(**kwargs))
    shutil.move(str(exported), dst)
    print(f"{suffix}: exported -> {dst}")
    return dst


fp32_path = export_variant(None, "fp32")
fp16_path = export_variant(16, "fp16")
int8_path = export_variant(8, "int8", data=calib_yaml)''',
    "quant-export",
)

md("### Reusable evaluation routine (same 6 tests as above, parameterized by model)", "md-quant-eval-fn")

code(
    '''def evaluate_model(weights_path, label, n_latency=N_LATENCY):
    model_q = YOLO(str(weights_path))

    # -- inference over the val split --
    preds_q = {}
    results_q = []
    for start in range(0, len(img_paths), BATCH):
        chunk = img_paths[start:start + BATCH]
        results_q.extend(model_q.predict(chunk, conf=0.25, verbose=False, batch=BATCH))
    for image_id, r in zip(image_ids, results_q):
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), dtype=np.float32)
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,), dtype=np.float32)
        kpts = (
            r.keypoints.xy.cpu().numpy()
            if r.keypoints is not None and len(r.boxes)
            else np.zeros((0, 17, 2), dtype=np.float32)
        )
        preds_q[image_id] = {"boxes": boxes, "scores": scores, "kpts": kpts}

    # -- GT <-> prediction matching --
    matches_q = []
    n_gt_q = n_pred_q = n_tp_q = 0
    for image_id in image_ids:
        anns = gt_by_image[image_id]
        gt_boxes = np.stack([xywh_to_xyxy(a["bbox"]) for a in anns])
        pred = preds_q[image_id]
        m, _, _ = match_greedy(gt_boxes, pred["boxes"], pred["scores"])
        n_gt_q += len(anns)
        n_pred_q += len(pred["boxes"])
        n_tp_q += len(m)
        for g, p, iou in m:
            matches_q.append((image_id, anns[g], pred["boxes"][p], pred["kpts"][p], iou))

    # -- 1. box detection accuracy --
    precision = n_tp_q / n_pred_q if n_pred_q else 0.0
    recall = n_tp_q / n_gt_q if n_gt_q else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean([m[4] for m in matches_q])) if matches_q else 0.0

    # -- 2 & 3. keypoint accuracy, per-keypoint accuracy --
    n_correct = n_visible = 0
    per_kpt_correct = np.zeros(17, dtype=np.int64)
    per_kpt_total = np.zeros(17, dtype=np.int64)
    for image_id, ann, pred_box, pred_kpts, iou in matches_q:
        kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        thresh = PCK_THRESH * gt_box_scale(ann["bbox"])
        visible = kpts[:, 2] > 0
        dist = np.linalg.norm(kpts[:, :2] - pred_kpts, axis=1)
        correct = visible & (dist < thresh)
        n_visible += int(visible.sum())
        n_correct += int(correct.sum())
        per_kpt_total += visible.astype(np.int64)
        per_kpt_correct += correct.astype(np.int64)
    keypoint_accuracy = n_correct / n_visible if n_visible else 0.0
    per_kpt_acc = np.divide(
        per_kpt_correct, per_kpt_total,
        out=np.zeros_like(per_kpt_correct, dtype=np.float64),
        where=per_kpt_total > 0,
    )

    # -- 4. animal detection + pose association --
    def image_pck_q(image_id):
        here = [m for m in matches_q if m[0] == image_id]
        correct = total = 0
        for _, ann, _, pred_kpts, _ in here:
            kpts = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
            visible = kpts[:, 2] > 0
            dist = np.linalg.norm(kpts[:, :2] - pred_kpts, axis=1)
            correct += int((visible & (dist < PCK_THRESH * gt_box_scale(ann["bbox"]))).sum())
            total += int(visible.sum())
        return correct / total if total else None

    fully_matched_multi = sum(
        1 for i in multi_animal_images
        if len([m for m in matches_q if m[0] == i]) == len(gt_by_image[i])
    )
    single_pck_q = [v for v in (image_pck_q(i) for i in image_ids if len(gt_by_image[i]) == 1) if v is not None]
    multi_pck_q = [v for v in (image_pck_q(i) for i in multi_animal_images) if v is not None]

    # -- 5. latency --
    latency_paths_q = img_paths[:n_latency]
    model_q.predict(latency_paths_q[0], verbose=False)  # warmup
    times_ms_q = []
    for p in latency_paths_q:
        t0 = time.perf_counter()
        model_q.predict(p, conf=0.25, verbose=False)
        times_ms_q.append((time.perf_counter() - t0) * 1000)
    times_ms_q = np.array(times_ms_q)

    # -- 6. FPS --
    fps_single_q = 1000.0 / times_ms_q.mean()
    t0 = time.perf_counter()
    model_q.predict(latency_paths_q, conf=0.25, verbose=False, batch=16)
    fps_batch_q = len(latency_paths_q) / (time.perf_counter() - t0)

    result = {
        "label": label,
        "box_precision": precision,
        "box_recall": recall,
        "box_f1": f1,
        "mean_iou": mean_iou,
        "keypoint_accuracy": keypoint_accuracy,
        "per_kpt_acc": per_kpt_acc,
        "fully_matched_multi_pct": fully_matched_multi / len(multi_animal_images) if multi_animal_images else 0.0,
        "single_pck": float(np.mean(single_pck_q)) if single_pck_q else 0.0,
        "multi_pck": float(np.mean(multi_pck_q)) if multi_pck_q else 0.0,
        "latency_mean_ms": float(times_ms_q.mean()),
        "latency_median_ms": float(np.median(times_ms_q)),
        "latency_p95_ms": float(np.percentile(times_ms_q, 95)),
        "fps_single": fps_single_q,
        "fps_batch": fps_batch_q,
        "onnx_size_mb": Path(weights_path).stat().st_size / 1e6,
    }
    print(
        f"[{label}] box P/R/F1={precision:.3f}/{recall:.3f}/{f1:.3f}  "
        f"IoU={mean_iou:.3f}  PCK@{PCK_THRESH}={keypoint_accuracy:.3f}  "
        f"latency={times_ms_q.mean():.1f}ms  fps={fps_single_q:.2f}  size={result[\'onnx_size_mb\']:.1f}MB"
    )
    return result''',
    "quant-eval-fn",
)

md("### Run all three precisions", "md-quant-run")

code(
    '''quant_results = [
    evaluate_model(fp32_path, "FP32"),
    evaluate_model(fp16_path, "FP16"),
    evaluate_model(int8_path, "INT8"),
]''',
    "quant-run",
)

md("### Comparison table", "md-quant-table")

code(
    '''header = f"{\'precision\':<8}{\'boxF1\':>8}{\'meanIoU\':>9}{\'PCK\':>8}{\'latency(ms)\':>13}{\'FPS\':>8}{\'size(MB)\':>10}"
print(header)
print("-" * len(header))
for r in quant_results:
    print(
        f"{r[\'label\']:<8}{r[\'box_f1\']:>8.3f}{r[\'mean_iou\']:>9.3f}{r[\'keypoint_accuracy\']:>8.3f}"
        f"{r[\'latency_mean_ms\']:>13.1f}{r[\'fps_single\']:>8.2f}{r[\'onnx_size_mb\']:>10.1f}"
    )''',
    "quant-table",
)

md("### Comparison plots", "md-quant-plots")

code(
    '''labels_q = [r["label"] for r in quant_results]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(labels_q, [r["keypoint_accuracy"] for r in quant_results], color=["#4c72b0", "#55a868", "#c44e52"])
axes[0].set_title(f"Keypoint accuracy (PCK@{PCK_THRESH})")
axes[0].set_ylim(0, 1)

axes[1].bar(labels_q, [r["latency_mean_ms"] for r in quant_results], color=["#4c72b0", "#55a868", "#c44e52"])
axes[1].set_title("Mean latency (ms, batch=1)")

axes[2].bar(labels_q, [r["fps_single"] for r in quant_results], color=["#4c72b0", "#55a868", "#c44e52"])
axes[2].set_title("FPS (single-image)")

fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(17)
width = 0.25
for i, r in enumerate(quant_results):
    ax.bar(x + (i - 1) * width, r["per_kpt_acc"], width, label=r["label"])
ax.set_xticks(x)
ax.set_xticklabels(KPT_NAMES, rotation=60, ha="right")
ax.set_ylabel(f"PCK@{PCK_THRESH}")
ax.set_ylim(0, 1)
ax.set_title("Per-keypoint accuracy by precision")
ax.legend()
fig.tight_layout()
plt.show()''',
    "quant-plots",
)

nb["cells"] = cells
nbf.write(nb, "performance_testing.ipynb")
print("appended quantization section,", len(cells), "cells total")
