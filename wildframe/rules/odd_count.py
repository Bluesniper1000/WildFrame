"""Odd-count composition rule: an even number of animals in frame is harder to balance than
an odd one, so this rule looks for a single edge animal whose exclusion (or inclusion) would
fix the parity, without disturbing any other animal's framing.

Gate provenance: every numeric threshold below comes from config, marked UNCALIBRATED, or (for
`ceiling`) carrying an explicit `source` field pending the real citation - see configs/rules.yaml.
Nothing here is attributed to a source in code; that's tracked in config only."""

from __future__ import annotations

from wildframe.contracts import Member, RuleName, RuleOutput, RuleState, SubjectMode, SubjectSet
from wildframe.rules.base import Rule


def _bbox_area(bbox) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _intersect_area(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _visible_fraction(bbox, frame_dims) -> float:
    w, h = frame_dims
    total = _bbox_area(bbox)
    if total <= 0:
        return 0.0
    return _intersect_area(bbox, (0, 0, w, h)) / total


def _nearest_edge(bbox, frame_dims) -> tuple[str, float]:
    """Returns (edge, distance) where distance is normalized by the respective frame dimension
    (fraction of width for left/right, of height for top/bottom) so it's comparable to a
    frame-relative margin regardless of aspect ratio."""
    w, h = frame_dims
    distances = {
        "left": bbox[0] / w,
        "right": (w - bbox[2]) / w,
        "top": bbox[1] / h,
        "bottom": (h - bbox[3]) / h,
    }
    edge = min(distances, key=lambda k: distances[k])
    return edge, distances[edge]


def _is_edge_member(member: Member, frame_dims, edge_margin_frac: float) -> bool:
    x1, y1, x2, y2 = member.bbox_xyxy
    w, h = frame_dims
    clipped = x1 < 0 or y1 < 0 or x2 > w or y2 > h
    _, dist = _nearest_edge(member.bbox_xyxy, frame_dims)
    return clipped or dist <= edge_margin_frac


def _fully_inside(bbox, rect) -> bool:
    return bbox[0] >= rect[0] and bbox[1] >= rect[1] and bbox[2] <= rect[2] and bbox[3] <= rect[3]


def _exclusion_crop(target_bbox, edge: str, frame_dims):
    w, h = frame_dims
    if edge == "left":
        return (target_bbox[2], 0, w, h)
    if edge == "right":
        return (0, 0, target_bbox[0], h)
    if edge == "top":
        return (0, target_bbox[3], w, h)
    return (0, 0, w, target_bbox[1])


class OddCountRule(Rule):
    name = RuleName.ODD_COUNT

    def gate(self, subject_set: SubjectSet) -> str | None:
        if subject_set.count == 0:
            return "no_subject"
        if subject_set.count == 1:
            return "single_animal"

        ceiling_cfg = self.config.get("ceiling", {})
        ceiling_value = ceiling_cfg["value"] if isinstance(ceiling_cfg, dict) else ceiling_cfg
        if subject_set.count > int(ceiling_value):
            return "count_exceeds_ceiling"

        if subject_set.mode == SubjectMode.FACING_PAIR:
            return "facing_pair_mode"

        if not subject_set.count_reliable:
            return "count_unreliable"

        return None

    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        count = subject_set.count
        if count % 2 == 1:
            return RuleOutput(rule=self.name, state=RuleState.SATISFIED, reason="odd_count", debug={"count": count})

        edge_margin_frac = float(self.config.get("edge_margin_frac", 0.03125))
        exclude_below_visible = float(self.config.get("exclude_below_visible", 0.5))
        frame_dims = subject_set.frame_dims

        edge_members = [(i, m) for i, m in enumerate(subject_set.members) if _is_edge_member(m, frame_dims, edge_margin_frac)]
        if not edge_members:
            return RuleOutput(
                rule=self.name, state=RuleState.INDETERMINATE, reason="no_reframing_fixes_count", debug={"count": count}
            )

        # Deterministic pick: most-clipped (lowest visible fraction) first, tie-broken left-to-right.
        target_index, target = min(
            edge_members, key=lambda pair: (_visible_fraction(pair[1].bbox_xyxy, frame_dims), pair[0])
        )
        visible_fraction = _visible_fraction(target.bbox_xyxy, frame_dims)
        edge, _ = _nearest_edge(target.bbox_xyxy, frame_dims)

        debug = {
            "count": count,
            "target_member_index": target_index,
            "target_track_id": target.track_id,
            "visible_fraction": visible_fraction,
            "edge": edge,
        }

        if visible_fraction < exclude_below_visible:
            action = "exclude"
            new_crop = _exclusion_crop(target.bbox_xyxy, edge, frame_dims)
            others = [m for i, m in enumerate(subject_set.members) if i != target_index]
            if any(not _fully_inside(m.bbox_xyxy, new_crop) for m in others):
                return RuleOutput(
                    rule=self.name,
                    state=RuleState.INDETERMINATE,
                    reason="shift_would_clip_another_member",
                    debug=debug,
                )
            new_count = len(others)
        else:
            # Including a member that's already counted doesn't change the count - correct,
            # if unhelpful, outcome given only this frame's information (no data on animals
            # entirely outside the current field of view).
            action = "include"
            new_count = count

        if new_count % 2 == 0:
            return RuleOutput(
                rule=self.name, state=RuleState.INDETERMINATE, reason="shift_does_not_fix_parity", debug=debug
            )

        correction = {
            "axis": "zoom" if action == "exclude" else "pan",
            "direction": "in" if action == "exclude" else "out",
            "action": action,
            "edge": edge,
            "member_index": target_index,
            "track_id": target.track_id,
        }
        return RuleOutput(
            rule=self.name,
            state=RuleState.VIOLATED,
            reason=f"even_count_fixable_by_{action}",
            severity=1.0,
            correction=correction,
            debug=debug,
        )
