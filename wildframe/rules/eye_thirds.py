from __future__ import annotations

import math

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet
from wildframe.pose.skeleton import KP
from wildframe.rules.base import Rule
from wildframe.scene import composite_subject


class EyeThirdsRule(Rule):
    name = RuleName.EYE_THIRDS

    def gate(self, subject_set: SubjectSet) -> str | None:
        subject = composite_subject(subject_set)
        if subject is None:
            return "no_subject"

        threshold = float(self.config.get("kpt_conf_threshold", 0.0))
        left_eye = subject.keypoints[KP.LEFT_EYE]
        right_eye = subject.keypoints[KP.RIGHT_EYE]
        if left_eye[2] <= threshold and right_eye[2] <= threshold:
            return "eye_not_visible"
        return None

    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        subject = composite_subject(subject_set)
        tolerance = float(self.config.get("thirds_tolerance_frac", 0.08))
        w, h = subject.frame_dims

        left_eye = subject.keypoints[KP.LEFT_EYE]
        right_eye = subject.keypoints[KP.RIGHT_EYE]
        eye_used = "left" if left_eye[2] >= right_eye[2] else "right"
        eye_point = left_eye[:2] if eye_used == "left" else right_eye[:2]

        thirds_points = [
            (w / 3, h / 3), (2 * w / 3, h / 3),
            (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3),
        ]
        nearest = min(thirds_points, key=lambda p: math.hypot(eye_point[0] - p[0], eye_point[1] - p[1]))
        dist = math.hypot(eye_point[0] - nearest[0], eye_point[1] - nearest[1])
        diag = math.hypot(w, h)
        dist_frac = dist / diag if diag > 0 else 0.0

        debug = {"eye_used": eye_used, "nearest_intersection": nearest, "dist_frac": dist_frac}

        if dist_frac <= tolerance:
            return RuleOutput(rule=self.name, state=RuleState.SATISFIED, reason="ok", debug=debug)

        severity = max(0.0, min(1.0, dist_frac / tolerance - 1.0))
        correction = {
            "axis": "pan_or_tilt",
            "direction_x": "right" if nearest[0] > eye_point[0] else "left",
            "direction_y": "down" if nearest[1] > eye_point[1] else "up",
            "magnitude_px": (nearest[0] - eye_point[0], nearest[1] - eye_point[1]),
        }
        return RuleOutput(
            rule=self.name,
            state=RuleState.VIOLATED,
            reason="eye_off_thirds",
            severity=severity,
            correction=correction,
            debug=debug,
        )
