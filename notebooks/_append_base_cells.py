"""One-off script to append a base-model (pre-finetuning) comparison section to
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
    "## 8. Base model (pre-finetuning) comparison\n\n"
    "How much did finetuning on AP-10K actually help? Evaluates the untouched pretrained "
    "checkpoint (`yolo26n-pose.pt`, COCO person-pose weights — same architecture, same "
    "`kpt_shape=[17,3]`, but never trained on animals) with the identical six tests, "
    "alongside the finetuned PyTorch model and the three quantized ONNX variants above.",
    "md-base-intro",
)

code(
    '''BASE_WEIGHTS = Path("../pretrained/yolo26n-pose.pt")

base_result = evaluate_model(BASE_WEIGHTS, "Base (pretrained)")
finetuned_result = evaluate_model(WEIGHTS, "Finetuned (PT fp32)")

all_results = [base_result, finetuned_result] + quant_results''',
    "base-run",
)

md("### Full comparison table", "md-base-table")

code(
    '''header = f"{\'model\':<20}{\'boxF1\':>8}{\'meanIoU\':>9}{\'PCK\':>8}{\'latency(ms)\':>13}{\'FPS\':>8}{\'size(MB)\':>10}"
print(header)
print("-" * len(header))
for r in all_results:
    print(
        f"{r[\'label\']:<20}{r[\'box_f1\']:>8.3f}{r[\'mean_iou\']:>9.3f}{r[\'keypoint_accuracy\']:>8.3f}"
        f"{r[\'latency_mean_ms\']:>13.1f}{r[\'fps_single\']:>8.2f}{r[\'onnx_size_mb\']:>10.1f}"
    )''',
    "base-table",
)

md("### Full comparison plots", "md-base-plots")

code(
    '''labels_all = [r["label"] for r in all_results]
colors_all = ["#8172b2", "#937860", "#4c72b0", "#55a868", "#c44e52"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].bar(labels_all, [r["keypoint_accuracy"] for r in all_results], color=colors_all)
axes[0].set_title(f"Keypoint accuracy (PCK@{PCK_THRESH})")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(labels_all, [r["box_f1"] for r in all_results], color=colors_all)
axes[1].set_title("Box detection F1")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(labels_all, [r["fps_single"] for r in all_results], color=colors_all)
axes[2].set_title("FPS (single-image)")
axes[2].tick_params(axis="x", rotation=30)

fig.tight_layout()
plt.show()''',
    "base-plots",
)

nb["cells"] = cells
nbf.write(nb, "performance_testing.ipynb")
print("appended base-model comparison section,", len(cells), "cells total")
