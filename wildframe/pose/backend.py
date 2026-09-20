# YOLO26 wrapper -> keypoints + confidences

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from wildframe.contracts import Detection
from wildframe.tracking import ByteTracker

REPO_ROOT = Path(__file__).resolve().parents[2]


def _letterbox(frame_bgr: np.ndarray, size: int, pad_value: int = 114):
    """Resize preserving aspect ratio onto a (size, size) canvas, padding with pad_value.
    Returns (canvas_rgb_float01_chw_batched, scale, pad_x, pad_y)."""
    h, w = frame_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    import cv2

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = canvas[:, :, ::-1]
    arr = rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr, scale, pad_x, pad_y


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


class PoseBackend:
    """Thin wrapper around onnxruntime.InferenceSession for the AP-10K YOLO26-pose ONNX export.
    Does its own preprocessing (letterbox) and postprocessing (box decode, keypoint extraction) -
    the exported graph has no NMS baked in."""

    def __init__(self, config_path: str | Path = "configs/model.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        self.config = yaml.safe_load(config_path.read_text())

        weights_path = Path(self.config["weights_path"])
        if not weights_path.is_absolute():
            weights_path = REPO_ROOT / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"missing model weights: {weights_path}")

        self.session = ort.InferenceSession(
            str(weights_path), providers=self.config.get("providers", ["CPUExecutionProvider"])
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(self.config["input_size"])
        self.conf_threshold = float(self.config["conf_threshold"])
        self.iou_threshold_nms = float(self.config["iou_threshold_nms"])
        self.max_det = int(self.config["max_det"])
        self.num_keypoints = int(self.config["num_keypoints"])
        self.tracker = ByteTracker(self.config)

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """frame_bgr: (H, W, 3) uint8, OpenCV BGR order, ORIGINAL resolution.
        Returns detections with bbox/keypoints mapped back to (H, W) pixel space."""
        batch, scale, pad_x, pad_y = _letterbox(frame_bgr, self.input_size)
        raw = self.session.run(None, {self.input_name: batch})[0]  # (1, 56, anchors)

        out = raw[0].T  # (anchors, 56)
        cls_conf = out[:, 4]
        keep = cls_conf >= self.conf_threshold
        if not np.any(keep):
            return []

        out = out[keep]
        cls_conf = cls_conf[keep]
        boxes_xyxy = _cxcywh_to_xyxy(out[:, 0:4])
        kpts = out[:, 5:5 + 3 * self.num_keypoints].reshape(-1, self.num_keypoints, 3)

        if self.max_det == 1:
            idx = [int(np.argmax(cls_conf))]
        else:
            idx = _greedy_nms(boxes_xyxy, cls_conf, self.iou_threshold_nms, self.max_det)

        mapped_boxes = []
        mapped_kpts = []
        for i in idx:
            box = boxes_xyxy[i].copy()
            kp = kpts[i].copy()

            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            kp[:, 0] = (kp[:, 0] - pad_x) / scale
            kp[:, 1] = (kp[:, 1] - pad_y) / scale

            mapped_boxes.append(box.astype(np.float32))
            mapped_kpts.append(kp.astype(np.float32))

        track_ids = self.tracker.update(mapped_boxes)

        return [
            Detection(bbox_xyxy=mapped_boxes[n], keypoints=mapped_kpts[n], det_conf=float(cls_conf[i]), track_id=track_ids[n])
            for n, i in enumerate(idx)
        ]

    def reset_tracking(self) -> None:
        """Clears all track state - call this when starting a new, unrelated frame sequence
        (e.g. between independent images/videos sharing one PoseBackend instance)."""
        self.tracker.reset()


def _greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float, max_det: int) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[i], boxes[rest])
        order = rest[ious <= iou_thresh]
    return keep


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[0], b[:, 0])
    iy1 = np.maximum(a[1], b[:, 1])
    ix2 = np.minimum(a[2], b[:, 2])
    iy2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)
