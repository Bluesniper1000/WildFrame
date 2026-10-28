"""Run a YOLO-pose ONNX model over an AP-10K split and emit COCO-format keypoint results.

Uses WildFrame's own letterbox (wildframe.pose.backend) so the deployed preprocessing is what gets
scored. Eval settings follow Ultralytics val (conf 0.001, NMS IoU 0.7, max_det 300), which is also
the protocol the sibling ASTMA-wildlife benchmark used, so AP numbers can be cross-checked.

Kinds:
  yolo-o2m  raw output (1, 56, anchors): cx, cy, w, h, score, 17 x (x, y, conf); NMS done here
  yolo-e2e  YOLO26 NMS-free output (1, 300, 57): x1, y1, x2, y2, score, cls, 17 x (x, y, conf)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from wildframe.pose.backend import _cxcywh_to_xyxy, _letterbox

KINDS = ("yolo-o2m", "yolo-e2e")


def make_session(model_path: str | Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Standard greedy NMS, indices in descending score order. OpenCV rather than torchvision:
    torch's DLLs are blocked by Windows Application Control on the dev machine."""
    xywh = np.concatenate([boxes_xyxy[:, :2], boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]], axis=1)
    keep = cv2.dnn.NMSBoxes(xywh.astype(np.float64).tolist(), scores.astype(np.float64).tolist(), 0.0, iou)
    return np.asarray(keep, dtype=np.int64).reshape(-1)


def predict_image(
    sess: ort.InferenceSession, img_bgr: np.ndarray, kind: str,
    conf: float = 0.001, iou: float = 0.7, max_det: int = 300, input_size: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (boxes xyxy (N, 4), scores (N,), keypoints (N, 17, 3)) in original image pixels."""
    batch, scale, pad_x, pad_y = _letterbox(img_bgr, input_size)
    out = sess.run(None, {sess.get_inputs()[0].name: batch})[0][0]
    if kind == "yolo-o2m":
        out = out.T
        out = out[out[:, 4] > conf]
        boxes = _cxcywh_to_xyxy(out[:, 0:4])
        keep = _nms(boxes, out[:, 4], iou)[:max_det] if len(out) else np.zeros(0, dtype=np.int64)
        boxes, scores, kpts = boxes[keep], out[keep, 4], out[keep, 5:].reshape(-1, 17, 3)
    elif kind == "yolo-e2e":
        out = out[out[:, 4] > conf][:max_det]
        boxes, scores, kpts = out[:, 0:4], out[:, 4], out[:, 6:].reshape(-1, 17, 3)
    else:
        raise ValueError(f"unknown kind: {kind}")

    boxes = boxes.astype(np.float64).copy()
    kpts = kpts.astype(np.float64).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
    kpts[..., 1] = (kpts[..., 1] - pad_y) / scale
    return boxes, scores.astype(np.float64), kpts


def predict_split(
    sess: ort.InferenceSession, gt: dict, images_dir: str | Path, kind: str, limit: int = 0,
) -> list[dict]:
    """COCO-format results for every image in `gt` (first `limit` by id if limit > 0).
    Each result also carries "bbox" (xywh) so matching can use the predicted box."""
    images = sorted(gt["images"], key=lambda im: im["id"])[: limit or None]
    results: list[dict] = []
    for info in images:
        img = cv2.imread(str(Path(images_dir) / info["file_name"]))
        if img is None:
            raise FileNotFoundError(Path(images_dir) / info["file_name"])
        boxes, scores, kpts = predict_image(sess, img, kind)
        for b, s, k in zip(boxes, scores, kpts):
            results.append({
                "image_id": info["id"],
                "category_id": 1,
                "keypoints": k.reshape(-1).round(2).tolist(),
                "score": round(float(s), 5),
                "bbox": [round(float(v), 2) for v in (b[0], b[1], b[2] - b[0], b[3] - b[1])],
            })
    return results
