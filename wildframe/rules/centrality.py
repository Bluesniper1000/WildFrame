"""Centrality composition rule: is the subject's overall position anchored near the frame
center? This is an alternative to eye_thirds, not an additional requirement stacked on top of
it - a photographer satisfying either guideline has a valid composition. See
explain/priority.py's OR-group handling: (eye_thirds, centrality) sit at the same priority slot,
and either one being SATISFIED is enough for that slot to read as "no problem"."""

from __future__ import annotations

import math

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet
from wildframe.rules.base import Rule
from wildframe.scene import composite_subject


class CentralityRule(Rule):
    name = RuleName.CENTRALITY

    def gate(self, subject_set: SubjectSet) -> str | None:
        subject = composite_subject(subject_set)
        if subject is None:
            return "no_subject"
        return None

    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        subject = composite_subject(subject_set)
        tolerance = float(self.config.get("center_tolerance_frac", 0.12))
        w, h = subject.frame_dims

        x1, y1, x2, y2 = subject.bbox_xyxy
        subject_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        frame_center = (w / 2.0, h / 2.0)

        dist = math.hypot(subject_center[0] - frame_center[0], subject_center[1] - frame_center[1])
        diag = math.hypot(w, h)
        dist_frac = dist / diag if diag > 0 else 0.0

        debug = {"subject_center": subject_center, "dist_frac": dist_frac}

        if dist_frac <= tolerance:
            return RuleOutput(rule=self.name, state=RuleState.SATISFIED, reason="ok", debug=debug)

        severity = max(0.0, min(1.0, dist_frac / tolerance - 1.0))
        correction = {
            "axis": "pan_or_tilt",
            "direction_x": "right" if frame_center[0] > subject_center[0] else "left",
            "direction_y": "down" if frame_center[1] > subject_center[1] else "up",
            "magnitude_px": (frame_center[0] - subject_center[0], frame_center[1] - subject_center[1]),
        }
        return RuleOutput(
            rule=self.name,
            state=RuleState.VIOLATED,
            reason="off_center",
            severity=severity,
            correction=correction,
            debug=debug,
        )
