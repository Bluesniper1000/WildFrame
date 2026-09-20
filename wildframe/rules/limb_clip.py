from __future__ import annotations

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet
from wildframe.pose.skeleton import LIMB_ORDER, LIMBS
from wildframe.rules.base import Rule
from wildframe.scene import members_for_clipping


class LimbClipRule(Rule):
    name = RuleName.LIMB_CLIP

    def gate(self, subject_set: SubjectSet) -> str | None:
        members = members_for_clipping(subject_set)
        if not members:
            return "no_subject"

        min_kpt_conf = float(self.config.get("min_kpt_conf", 0.25))
        if all(bool((m.keypoints[:, 2] < min_kpt_conf).all()) for m in members):
            return "no_visible_keypoints"
        return None

    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        members = members_for_clipping(subject_set)
        min_kpt_conf = float(self.config.get("min_kpt_conf", 0.25))
        edge_margin_frac = float(self.config.get("edge_margin_frac", 0.003125))
        w, h = subject_set.frame_dims
        margin_x = edge_margin_frac * w
        margin_y = edge_margin_frac * h

        checked = []
        # Deterministic order: member position (left-to-right, from scene.py's sort) major,
        # limb order minor - so the same physical scene always reports the same offender first.
        for member_index, member in enumerate(members):
            for limb in LIMB_ORDER:
                _, _, distal = LIMBS[limb]
                x, y, conf = member.keypoints[distal]
                if conf < min_kpt_conf:
                    continue
                checked.append((member_index, limb))

                clipped_edge = None
                if x <= margin_x:
                    clipped_edge = "left"
                elif x >= w - margin_x:
                    clipped_edge = "right"
                elif y <= margin_y:
                    clipped_edge = "top"
                elif y >= h - margin_y:
                    clipped_edge = "bottom"

                if clipped_edge is not None:
                    debug = {"checked": checked, "clipped_limb": limb, "clipped_member_index": member_index}
                    correction = {
                        "axis": "zoom",
                        "direction": "out",
                        "limb": limb,
                        "edge": clipped_edge,
                        "member_index": member_index,
                        "track_id": member.track_id,
                    }
                    return RuleOutput(
                        rule=self.name,
                        state=RuleState.VIOLATED,
                        reason="limb_clipped",
                        severity=1.0,
                        correction=correction,
                        debug=debug,
                    )

        return RuleOutput(
            rule=self.name,
            state=RuleState.SATISFIED,
            reason="ok",
            debug={"checked": checked, "clipped_limb": None},
        )
