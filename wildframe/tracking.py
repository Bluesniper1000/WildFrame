"""Minimal deterministic multi-object tracker in the SORT/ByteTrack family: a constant-velocity
Kalman filter per track plus IoU-based Hungarian assignment between frames. Runs directly on the
box+score output already produced by pose/backend.py's onnxruntime decode - no dependency on
Ultralytics' `model.track()`, which requires an Ultralytics model object we don't have (backend.py
runs the INT8 ONNX model via raw onnxruntime by design)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# Kalman state: [cx, cy, s, r, vcx, vcy, vs] where s = box area, r = aspect ratio (w/h, held constant).
_F = np.eye(7)
for _i in range(3):
    _F[_i, _i + 4] = 1.0

_H = np.zeros((4, 7))
_H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = max(w * h, 1e-6)
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float64)


def _state_to_bbox(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    s = max(s, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


class _KalmanBoxTracker:
    """One track's constant-velocity Kalman filter over [cx, cy, s, r, vcx, vcy, vs]."""

    def __init__(self, track_id: int, bbox: np.ndarray):
        self.track_id = track_id
        self.mean = np.zeros(7, dtype=np.float64)
        self.mean[:4] = _bbox_to_z(bbox)
        self.covariance = np.eye(7) * 10.0
        self.covariance[4:, 4:] *= 100.0  # high initial uncertainty on velocity

        self.process_noise = np.eye(7) * 1.0
        self.process_noise[4:, 4:] *= 0.01
        self.measurement_noise = np.eye(4) * 1.0

        self.time_since_update = 0
        self.hits = 1

    def predict(self) -> np.ndarray:
        self.mean = _F @ self.mean
        self.covariance = _F @ self.covariance @ _F.T + self.process_noise
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray) -> None:
        z = _bbox_to_z(bbox)
        y = z - _H @ self.mean
        s = _H @ self.covariance @ _H.T + self.measurement_noise
        k = self.covariance @ _H.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ y
        self.covariance = (np.eye(7) - k @ _H) @ self.covariance
        self.time_since_update = 0
        self.hits += 1

    @property
    def bbox(self) -> np.ndarray:
        return _state_to_bbox(self.mean)


class ByteTracker:
    """Frame-to-frame association: predict all tracks, Hungarian-match against this frame's
    detections by IoU, update matched tracks, spawn new tracks for unmatched detections, age
    out tracks that go too long without a match."""

    def __init__(self, config: dict):
        self.iou_threshold = float(config.get("track_iou_threshold", 0.3))
        self.max_age = int(config.get("track_max_age", 30))
        self._tracks: list[_KalmanBoxTracker] = []
        self._next_id = 1

    def update(self, boxes_xyxy: list[np.ndarray]) -> list[int]:
        """boxes_xyxy: this frame's detection boxes, in a fixed order. Returns a track_id per
        detection, same order, assigning new IDs for unmatched detections."""
        for track in self._tracks:
            track.predict()

        track_ids = [None] * len(boxes_xyxy)

        if self._tracks and boxes_xyxy:
            cost = np.zeros((len(self._tracks), len(boxes_xyxy)))
            for i, track in enumerate(self._tracks):
                for j, box in enumerate(boxes_xyxy):
                    cost[i, j] = 1.0 - _box_iou(track.bbox, box)

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks, matched_dets = set(), set()
            for r, c in zip(row_idx, col_idx):
                if 1.0 - cost[r, c] >= self.iou_threshold:
                    self._tracks[r].update(boxes_xyxy[c])
                    track_ids[c] = self._tracks[r].track_id
                    matched_tracks.add(r)
                    matched_dets.add(c)
        else:
            matched_dets = set()

        for j, box in enumerate(boxes_xyxy):
            if j not in matched_dets:
                new_track = _KalmanBoxTracker(self._next_id, box)
                self._next_id += 1
                self._tracks.append(new_track)
                track_ids[j] = new_track.track_id

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return track_ids

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
