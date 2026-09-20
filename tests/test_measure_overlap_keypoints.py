import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_overlap_keypoints import (  # noqa: E402
    box_iou,
    build_report,
    compute_pck,
    iou_bin_label,
    match_greedy,
    max_pairwise_iou,
    summarize,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy():
    np.testing.assert_allclose(xywh_to_xyxy([10, 20, 30, 40]), [10, 20, 40, 60])


def test_box_iou_identical_boxes_is_one():
    a = np.array([0, 0, 10, 10], dtype=np.float64)
    b = np.array([[0, 0, 10, 10]], dtype=np.float64)
    assert box_iou(a, b)[0] == 1.0


def test_box_iou_disjoint_boxes_is_zero():
    a = np.array([0, 0, 10, 10], dtype=np.float64)
    b = np.array([[100, 100, 110, 110]], dtype=np.float64)
    assert box_iou(a, b)[0] == 0.0


def test_max_pairwise_iou_single_box_is_zero():
    boxes = [np.array([0, 0, 10, 10], dtype=np.float64)]
    assert max_pairwise_iou(boxes) == [0.0]


def test_max_pairwise_iou_two_overlapping_boxes():
    a = np.array([0, 0, 10, 10], dtype=np.float64)
    b = np.array([5, 0, 15, 10], dtype=np.float64)  # 50% overlap in x, full in y
    overlaps = max_pairwise_iou([a, b])
    expected_iou = 50.0 / 150.0  # intersection 5x10=50, union 100+100-50=150
    assert overlaps[0] == expected_iou
    assert overlaps[1] == expected_iou


def test_iou_bin_label_single_instance():
    assert iou_bin_label(0.0) == "single_instance"


def test_iou_bin_label_bins_correctly():
    assert iou_bin_label(0.05) == "0.0-0.1"
    assert iou_bin_label(0.15) == "0.1-0.2"
    assert iou_bin_label(0.99) == "0.9-1.0"
    assert iou_bin_label(1.0) == "0.9-1.0"


def test_match_greedy_picks_highest_score_first():
    gt_boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float64)
    pred_boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float64)
    pred_scores = np.array([0.9, 0.5])
    matches = match_greedy(gt_boxes, pred_boxes, pred_scores)
    assert len(matches) == 2
    matched_pairs = {(g, p) for g, p, _ in matches}
    assert (0, 0) in matched_pairs
    assert (1, 1) in matched_pairs


def test_match_greedy_below_threshold_unmatched():
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float64)
    pred_boxes = np.array([[50, 50, 60, 60]], dtype=np.float64)
    pred_scores = np.array([0.9])
    matches = match_greedy(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5)
    assert matches == []


def test_compute_pck_perfect_prediction():
    gt = np.array([[10, 10, 1], [20, 20, 1]], dtype=np.float64)
    pred = np.array([[10, 10], [20, 20]], dtype=np.float64)
    bbox = np.array([0, 0, 100, 100], dtype=np.float64)
    assert compute_pck(gt, pred, bbox) == 1.0


def test_compute_pck_ignores_invisible_keypoints():
    gt = np.array([[10, 10, 0], [20, 20, 1]], dtype=np.float64)
    pred = np.array([[999, 999], [20, 20]], dtype=np.float64)
    bbox = np.array([0, 0, 100, 100], dtype=np.float64)
    assert compute_pck(gt, pred, bbox) == 1.0


def test_compute_pck_none_when_no_keypoints_visible():
    gt = np.array([[10, 10, 0], [20, 20, 0]], dtype=np.float64)
    pred = np.array([[10, 10], [20, 20]], dtype=np.float64)
    bbox = np.array([0, 0, 100, 100], dtype=np.float64)
    assert compute_pck(gt, pred, bbox) is None


def _make_ann(ann_id, image_id, bbox, keypoints):
    return {"id": ann_id, "image_id": image_id, "bbox": bbox, "num_keypoints": 1, "keypoints": keypoints}


def test_build_report_single_instance_image():
    gt_by_image = {1: [_make_ann(1, 1, [0, 0, 20, 20], [10, 10, 1])]}
    predictions_by_image = {
        1: {"boxes": np.array([[0, 0, 20, 20]]), "scores": np.array([0.9]), "kpts": np.array([[[10, 10]]])}
    }
    rows = build_report(gt_by_image, predictions_by_image)
    assert len(rows) == 1
    assert rows[0]["iou_bin"] == "single_instance"
    assert rows[0]["matched"] is True
    assert rows[0]["pck"] == 1.0


def test_build_report_overlapping_pair_and_summarize():
    gt_by_image = {
        1: [
            _make_ann(1, 1, [0, 0, 20, 20], [10, 10, 1]),
            _make_ann(2, 1, [10, 0, 20, 20], [20, 10, 1]),  # 50% x-overlap with ann 1
        ]
    }
    predictions_by_image = {
        1: {
            "boxes": np.array([[0, 0, 20, 20], [10, 0, 30, 20]]),
            "scores": np.array([0.9, 0.85]),
            "kpts": np.array([[[10, 10]], [[20, 10]]]),
        }
    }
    rows = build_report(gt_by_image, predictions_by_image)
    assert len(rows) == 2
    assert all(r["iou_bin"] != "single_instance" for r in rows)
    assert all(r["matched"] for r in rows)
    assert all(r["pck"] == 1.0 for r in rows)

    summary = summarize(rows)
    assert len(summary) == 1
    assert summary[0]["n_instances"] == 2
    assert summary[0]["match_rate"] == 1.0
    assert summary[0]["mean_pck"] == 1.0


def test_build_report_unmatched_instance_has_none_pck():
    gt_by_image = {1: [_make_ann(1, 1, [0, 0, 20, 20], [10, 10, 1])]}
    predictions_by_image = {1: {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "kpts": np.zeros((0, 1, 2))}}
    rows = build_report(gt_by_image, predictions_by_image)
    assert rows[0]["matched"] is False
    assert rows[0]["pck"] is None
