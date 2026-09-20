# AP-10K keypoint indices, parent-joint lookup

from enum import IntEnum


class KP(IntEnum):
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE = 2
    NECK = 3
    ROOT_OF_TAIL = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_FRONT_PAW = 7
    RIGHT_SHOULDER = 8
    RIGHT_ELBOW = 9
    RIGHT_FRONT_PAW = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_BACK_PAW = 13
    RIGHT_HIP = 14
    RIGHT_KNEE = 15
    RIGHT_BACK_PAW = 16


NUM_KEYPOINTS = 17

KEYPOINT_NAMES = [k.name.lower() for k in KP]

# COCO-style skeleton edges, transcribed from the AP-10K annotation "skeleton" field.
SKELETON_EDGES = [
    (KP.LEFT_EYE, KP.RIGHT_EYE), (KP.LEFT_EYE, KP.NOSE), (KP.RIGHT_EYE, KP.NOSE),
    (KP.NOSE, KP.NECK), (KP.NECK, KP.ROOT_OF_TAIL),
    (KP.NECK, KP.LEFT_SHOULDER), (KP.LEFT_SHOULDER, KP.LEFT_ELBOW), (KP.LEFT_ELBOW, KP.LEFT_FRONT_PAW),
    (KP.NECK, KP.RIGHT_SHOULDER), (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW), (KP.RIGHT_ELBOW, KP.RIGHT_FRONT_PAW),
    (KP.ROOT_OF_TAIL, KP.LEFT_HIP), (KP.LEFT_HIP, KP.LEFT_KNEE), (KP.LEFT_KNEE, KP.LEFT_BACK_PAW),
    (KP.ROOT_OF_TAIL, KP.RIGHT_HIP), (KP.RIGHT_HIP, KP.RIGHT_KNEE), (KP.RIGHT_KNEE, KP.RIGHT_BACK_PAW),
]

HEAD_TORSO_KEYPOINTS = frozenset({KP.LEFT_EYE, KP.RIGHT_EYE, KP.NOSE, KP.NECK, KP.ROOT_OF_TAIL})

# name -> (proximal, mid, distal) keypoint indices, proximal-to-distal order
LIMBS = {
    "left_front": (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_FRONT_PAW),
    "right_front": (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_FRONT_PAW),
    "left_back": (KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_BACK_PAW),
    "right_back": (KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_BACK_PAW),
}

# Deterministic iteration order for tie-breaking (e.g. limb_clip rule).
LIMB_ORDER = ["left_front", "right_front", "left_back", "right_back"]

# Single-parent tree rooted at NECK (None).
PARENT = {
    KP.LEFT_EYE: KP.NOSE,
    KP.RIGHT_EYE: KP.NOSE,
    KP.NOSE: KP.NECK,
    KP.NECK: None,
    KP.ROOT_OF_TAIL: KP.NECK,
    KP.LEFT_SHOULDER: KP.NECK,
    KP.LEFT_ELBOW: KP.LEFT_SHOULDER,
    KP.LEFT_FRONT_PAW: KP.LEFT_ELBOW,
    KP.RIGHT_SHOULDER: KP.NECK,
    KP.RIGHT_ELBOW: KP.RIGHT_SHOULDER,
    KP.RIGHT_FRONT_PAW: KP.RIGHT_ELBOW,
    KP.LEFT_HIP: KP.ROOT_OF_TAIL,
    KP.LEFT_KNEE: KP.LEFT_HIP,
    KP.LEFT_BACK_PAW: KP.LEFT_KNEE,
    KP.RIGHT_HIP: KP.ROOT_OF_TAIL,
    KP.RIGHT_KNEE: KP.RIGHT_HIP,
    KP.RIGHT_BACK_PAW: KP.RIGHT_KNEE,
}

_KP_TO_LIMB = {kp: name for name, joints in LIMBS.items() for kp in joints}


def limb_of(kpt_index: int) -> str | None:
    """Return the limb name a keypoint belongs to, or None if it's head/torso."""
    return _KP_TO_LIMB.get(KP(kpt_index))
