from __future__ import annotations

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet
from wildframe.rules.base import Rule
from wildframe.scene import composite_subject


class LeadSpaceRule(Rule):
    name = RuleName.LEAD_SPACE

    def gate(self, subject_set: SubjectSet) -> str | None:
        subject = composite_subject(subject_set)
        if subject is None:
            return "no_subject"

        # orientation.py already gates facing_vector to None whenever the underlying keypoint
        # confidence is below min_axis_conf, so a non-None facing_vector here already implies
        # adequate confidence - no separate confidence check needed.
        orientation = subject.orientation
        if orientation is None or orientation.facing_vector is None:
            return "no_orientation"
        if orientation.yaw_state == "frontal":
            return "frontal_yaw"
        return None

    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        subject = composite_subject(subject_set)
        min_lead_fraction = float(self.config.get("min_lead_fraction", 0.15))
        ideal_lead_fraction = float(self.config.get("ideal_lead_fraction", 0.30))

        frame_w = subject.frame_dims[0]
        yaw_state = subject.orientation.yaw_state  # "left" or "right" (frontal already gated out)
        x1, _, x2, _ = subject.bbox_xyxy

        if yaw_state == "left":
            front_side = "left"
            lead_space_px = x1
        else:
            front_side = "right"
            lead_space_px = frame_w - x2

        lead_fraction = lead_space_px / frame_w if frame_w > 0 else 0.0

        debug = {"lead_fraction": lead_fraction, "front_side": front_side}

        if lead_fraction < min_lead_fraction:
            severity = max(0.0, min(1.0, (min_lead_fraction - lead_fraction) / min_lead_fraction))
            # Camera pan direction: turning the camera toward the side the subject faces shifts
            # the subject the other way in the frame, opening up space ahead of it.
            direction = front_side
            correction = {
                "axis": "pan",
                "direction": direction,
                "magnitude_frac": max(0.0, ideal_lead_fraction - lead_fraction),
            }
            return RuleOutput(
                rule=self.name,
                state=RuleState.VIOLATED,
                reason="insufficient_lead_space",
                severity=severity,
                correction=correction,
                debug=debug,
            )

        return RuleOutput(rule=self.name, state=RuleState.SATISFIED, reason="ok", debug=debug)
