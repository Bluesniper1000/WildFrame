from __future__ import annotations

import cv2
import numpy as np

from wildframe.contracts import RuleName, RuleOutput, RuleState, Subject, SubjectMode, SubjectSet
from wildframe.pose.skeleton import SKELETON_EDGES

STATE_COLOR_BGR = {
    RuleState.SATISFIED: (0, 200, 0),
    RuleState.VIOLATED: (0, 0, 220),
    RuleState.INDETERMINATE: (0, 200, 220),
}
READY_COLOR_BGR = (255, 220, 0)  # distinct from any per-rule state color - "click now"
DRAW_KPT_CONF_THRESHOLD = 0.25
MODE_NOTICE_COLOR = (255, 255, 255)
UNION_BBOX_COLOR = (255, 180, 0)
REFERENT_COLOR = (255, 0, 255)
BANNER_ROW_HEIGHT = 40
GROUP_MODES = (SubjectMode.GROUP_ALIGNED, SubjectMode.FACING_PAIR, SubjectMode.MIXED)


def draw_overlay(
    frame_bgr: np.ndarray,
    subject: Subject | None,
    rule_output: RuleOutput | None,
    mode_notice: str | None = None,
    subject_set: SubjectSet | None = None,
) -> np.ndarray:
    """Returns a NEW annotated frame - bbox + skeleton + instruction banner(s) + correction
    arrow. In group modes (subject_set given), also outlines the union bbox and highlights the
    referent member named by rule_output.correction['member_index'], when present."""
    frame = frame_bgr.copy()
    next_row = 0

    if mode_notice is not None:
        _draw_banner(frame, mode_notice, MODE_NOTICE_COLOR, row=next_row)
        next_row += 1

    if subject is None:
        _draw_banner(frame, "No animal detected", (0, 200, 220), row=next_row)
        return frame

    if rule_output is not None and rule_output.rule == RuleName.READY:
        color = READY_COLOR_BGR
    else:
        color = STATE_COLOR_BGR.get(rule_output.state if rule_output else RuleState.INDETERMINATE, (200, 200, 200))

    if subject_set is not None and subject_set.mode in GROUP_MODES and subject_set.union_bbox is not None:
        ux1, uy1, ux2, uy2 = subject_set.union_bbox.astype(int)
        cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), UNION_BBOX_COLOR, 2)

    x1, y1, x2, y2 = subject.bbox_xyxy.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if subject_set is not None and rule_output is not None:
        member_index = (rule_output.correction or {}).get("member_index")
        if member_index is not None and 0 <= member_index < len(subject_set.members):
            rx1, ry1, rx2, ry2 = subject_set.members[member_index].bbox_xyxy.astype(int)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), REFERENT_COLOR, 3)

    kpts = subject.keypoints
    for a, b in SKELETON_EDGES:
        if kpts[a][2] >= DRAW_KPT_CONF_THRESHOLD and kpts[b][2] >= DRAW_KPT_CONF_THRESHOLD:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, (255, 255, 255), 1)

    for x, y, conf in kpts:
        if conf >= DRAW_KPT_CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    if rule_output is not None:
        message = rule_output.message or f"{rule_output.rule.value}: {rule_output.state.value}"
        _draw_banner(frame, message, color, row=next_row)
        _draw_correction_arrow(frame, rule_output)

    return frame


def _draw_banner(frame: np.ndarray, text: str, color: tuple[int, int, int], row: int = 0) -> None:
    h, w = frame.shape[:2]
    y0 = row * BANNER_ROW_HEIGHT
    y1 = y0 + BANNER_ROW_HEIGHT
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    cv2.putText(frame, text, (10, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _draw_correction_arrow(frame: np.ndarray, rule_output: RuleOutput) -> None:
    correction = rule_output.correction
    if not correction:
        return
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    direction = correction.get("direction")
    if direction == "left":
        end = (cx - 80, cy)
    elif direction == "right":
        end = (cx + 80, cy)
    elif direction == "out":
        cv2.arrowedLine(frame, (cx - 40, cy), (cx - 80, cy), (255, 255, 255), 2, tipLength=0.4)
        cv2.arrowedLine(frame, (cx + 40, cy), (cx + 80, cy), (255, 255, 255), 2, tipLength=0.4)
        return
    else:
        return
    cv2.arrowedLine(frame, (cx, cy), end, (255, 255, 255), 2, tipLength=0.3)
