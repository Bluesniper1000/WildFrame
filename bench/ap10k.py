"""AP-10K constants and ground-truth loading shared by the benchmark modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wildframe.pose.skeleton import KEYPOINT_NAMES, KP

# Per-keypoint OKS sigmas from MMPose configs/_base_/datasets/ap10k.py (the values MMPose's AP-10K
# evaluator uses). Ultralytics val would instead apply COCO *human* sigmas to any kpt_shape [17, 3].
AP10K_SIGMAS = np.array(
    [0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062,
     0.107, 0.087, 0.089, 0.107, 0.087, 0.089]
)

# Keypoints the composition rules read: eyes (eye_thirds, frontal check), nose/neck/tail root
# (orientation -> lead_space), paws (limb_clip).
RULE_KEYPOINTS = [
    KP.LEFT_EYE, KP.RIGHT_EYE, KP.NOSE, KP.NECK, KP.ROOT_OF_TAIL,
    KP.LEFT_FRONT_PAW, KP.RIGHT_FRONT_PAW, KP.LEFT_BACK_PAW, KP.RIGHT_BACK_PAW,
]

__all__ = ["AP10K_SIGMAS", "KEYPOINT_NAMES", "RULE_KEYPOINTS", "load_gt"]


def load_gt(ann_path: str | Path) -> dict:
    """Load a COCO-format AP-10K annotation file and make it class-agnostic.

    AP-10K annotations carry species-level category ids; keypoint AP here is scored as a single
    "animal" class, so every annotation is mapped onto the first category id."""
    data = json.loads(Path(ann_path).read_text())
    cat_id = data["categories"][0]["id"]
    data["categories"] = [dict(data["categories"][0])]
    for ann in data["annotations"]:
        ann["category_id"] = cat_id
    return data
