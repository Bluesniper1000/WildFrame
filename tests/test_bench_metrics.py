import numpy as np
import pytest

from bench import metrics
from bench.ap10k import AP10K_SIGMAS
from wildframe.pose.skeleton import KP

ORIENT_CFG = {"frontal_yaw_max_deg": 25, "min_axis_conf": 0.3}


def _side_profile_kpts(x0: float, y0: float, facing_right: bool = True) -> np.ndarray:
    """(17, 3) GT keypoints for an animal in profile inside a 200x100 box at (x0, y0), all labeled (v=2)."""
    s = 1 if facing_right else -1
    cx = x0 + 100
    k = np.zeros((17, 3))
    k[:, 2] = 2
    k[KP.NOSE, :2] = (cx + s * 90, y0 + 30)
    k[KP.LEFT_EYE, :2] = (cx + s * 80, y0 + 22)
    k[KP.RIGHT_EYE, :2] = (cx + s * 78, y0 + 24)
    k[KP.NECK, :2] = (cx + s * 50, y0 + 35)
    k[KP.ROOT_OF_TAIL, :2] = (cx - s * 80, y0 + 40)
    for i, kp in enumerate([KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP]):
        k[kp, :2] = (cx + (s * 40 if i < 2 else -s * 60), y0 + 45)
    for kp in [KP.LEFT_ELBOW, KP.RIGHT_ELBOW, KP.LEFT_KNEE, KP.RIGHT_KNEE]:
        k[kp, :2] = (cx, y0 + 70)
    for kp in [KP.LEFT_FRONT_PAW, KP.RIGHT_FRONT_PAW, KP.LEFT_BACK_PAW, KP.RIGHT_BACK_PAW]:
        k[kp, :2] = (cx, y0 + 98)
    return k


def _gt(instances: list[tuple[int, np.ndarray]]) -> dict:
    anns = []
    for i, (image_id, k) in enumerate(instances):
        vis = k[:, 2] > 0
        x0, y0 = k[vis, 0].min(), k[vis, 1].min()
        w, h = k[vis, 0].max() - x0, k[vis, 1].max() - y0
        anns.append({"id": i + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                     "keypoints": k.reshape(-1).tolist(), "num_keypoints": int(vis.sum()),
                     "bbox": [float(x0), float(y0), float(w), float(h)], "area": float(w * h)})
    images = [{"id": iid, "file_name": f"{iid}.jpg", "width": 640, "height": 480} for iid in sorted({i for i, _ in instances})]
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "animal"}]}


def _pred_from_gt(image_id: int, k: np.ndarray, score: float = 0.9, conf: float = 0.9, jitter: np.ndarray | None = None) -> dict:
    p = k.copy()
    p[:, 2] = conf
    if jitter is not None:
        p[:, :2] += jitter
    x0, y0 = k[:, 0].min(), k[:, 1].min()
    return {"image_id": image_id, "category_id": 1, "keypoints": p.reshape(-1).tolist(), "score": score,
            "bbox": [float(x0), float(y0), float(k[:, 0].max() - x0), float(k[:, 1].max() - y0)]}


# --- OKS AP ---

def test_oks_ap_perfect_predictions_score_one():
    pytest.importorskip("pycocotools")
    insts = [(1, _side_profile_kpts(50, 50)), (2, _side_profile_kpts(300, 200, facing_right=False))]
    stats = metrics.oks_ap(_gt(insts), [_pred_from_gt(i, k) for i, k in insts])
    assert stats["AP"] == pytest.approx(1.0)
    assert stats["AP75"] == pytest.approx(1.0)


def test_oks_ap_no_predictions_is_zero():
    assert metrics.oks_ap(_gt([(1, _side_profile_kpts(50, 50))]), [])["AP"] == 0.0


def test_ap10k_sigmas_are_17_and_differ_from_coco_human():
    coco_human = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    assert AP10K_SIGMAS.shape == (17,)
    assert not np.allclose(AP10K_SIGMAS, coco_human)


# --- Matching ---

def test_match_respects_min_score_and_one_to_one():
    k = _side_profile_kpts(50, 50)
    gt = _gt([(1, k)])
    preds = [_pred_from_gt(1, k, score=0.9), _pred_from_gt(1, k, score=0.8), _pred_from_gt(1, k, score=0.1)]
    matches, n_gt = metrics.match_predictions(gt, preds, min_score=0.25)
    assert n_gt == 1
    assert len(matches) == 1
    assert matches[0].score == 0.9


def test_match_rejects_low_iou_box():
    k = _side_profile_kpts(50, 50)
    far = _pred_from_gt(1, _side_profile_kpts(400, 300))
    matches, n_gt = metrics.match_predictions(_gt([(1, k)]), [far])
    assert (len(matches), n_gt) == (0, 1)


def test_match_skips_gt_without_labeled_keypoints():
    k = _side_profile_kpts(50, 50)
    empty = _side_profile_kpts(300, 50)
    empty[:, 2] = 0
    gt = _gt([(1, k)])
    gt["annotations"].append({**gt["annotations"][0], "id": 99, "num_keypoints": 0,
                              "keypoints": empty.reshape(-1).tolist(), "bbox": [300, 50, 200, 100]})
    _, n_gt = metrics.match_predictions(gt, [])
    assert n_gt == 1


# --- PCK / NME ---

def test_pck_uses_per_axis_bbox_normalisation():
    k = _side_profile_kpts(50, 50)  # box ~180 wide x ~76 tall
    w, h = metrics.match_predictions(_gt([(1, k)]), [_pred_from_gt(1, k)])[0][0].gt_bbox[2:]
    jitter = np.zeros((17, 2))
    jitter[KP.NOSE] = (0.04 * w, 0.0)    # 0.04 of width -> inside PCK@0.05
    jitter[KP.NECK] = (0.0, 0.06 * h)    # 0.06 of height -> outside
    matches, _ = metrics.match_predictions(_gt([(1, k)]), [_pred_from_gt(1, k, jitter=jitter)])
    pck, counts = metrics.pck_per_keypoint(matches, alpha=0.05)
    assert pck[KP.NOSE] == 1.0
    assert pck[KP.NECK] == 0.0
    assert pck[KP.LEFT_EYE] == 1.0
    assert counts[KP.NOSE] == 1


def test_pck_ignores_unlabeled_gt_keypoints():
    k = _side_profile_kpts(50, 50)
    k[KP.LEFT_EYE, 2] = 0
    jitter = np.zeros((17, 2))
    jitter[KP.LEFT_EYE] = (500, 500)
    matches, _ = metrics.match_predictions(_gt([(1, k)]), [_pred_from_gt(1, k, jitter=jitter)])
    pck, counts = metrics.pck_per_keypoint(matches)
    assert counts[KP.LEFT_EYE] == 0
    assert np.isnan(pck[KP.LEFT_EYE])


def test_nme_is_distance_over_sqrt_area():
    k = _side_profile_kpts(50, 50)
    gt = _gt([(1, k)])
    w, h = gt["annotations"][0]["bbox"][2:]
    jitter = np.zeros((17, 2))
    jitter[KP.NOSE] = (3.0, 4.0)
    matches, _ = metrics.match_predictions(gt, [_pred_from_gt(1, k, jitter=jitter)])
    nme = metrics.nme_per_keypoint(matches)
    assert nme[KP.NOSE] == pytest.approx(5.0 / np.sqrt(w * h))
    assert nme[KP.NECK] == 0.0


# --- Orientation ---

def test_orientation_perfect_keypoints_agree_with_gt():
    insts = [(1, _side_profile_kpts(50, 50, True)), (2, _side_profile_kpts(50, 50, False))]
    matches, _ = metrics.match_predictions(_gt(insts), [_pred_from_gt(i, k) for i, k in insts])
    res = metrics.orientation_eval(matches, ORIENT_CFG)
    assert res["confusion"]["right"]["right"] == 1
    assert res["confusion"]["left"]["left"] == 1
    assert res["lr_accuracy"] == 1.0
    assert res["n_lr"] == 2


def test_orientation_flipped_prediction_counts_as_wrong():
    k = _side_profile_kpts(50, 50, True)
    flipped = _side_profile_kpts(50, 50, False)
    pred = _pred_from_gt(1, flipped)
    pred["bbox"] = _pred_from_gt(1, k)["bbox"]
    matches, _ = metrics.match_predictions(_gt([(1, k)]), [pred])
    res = metrics.orientation_eval(matches, ORIENT_CFG)
    assert res["confusion"]["right"]["left"] == 1
    assert res["lr_accuracy"] == 0.0


def test_orientation_low_confidence_prediction_is_unknown_and_wrong():
    k = _side_profile_kpts(50, 50, True)
    matches, _ = metrics.match_predictions(_gt([(1, k)]), [_pred_from_gt(1, k, conf=0.05)])
    res = metrics.orientation_eval(matches, ORIENT_CFG)
    assert res["confusion"]["right"]["unknown"] == 1
    assert res["lr_accuracy"] == 0.0


# --- AUROC ---

def test_auroc_perfect_random_and_inverted():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert metrics.auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert metrics.auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert metrics.auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, 40).astype(float)
    labels = rng.random(40) < 0.4
    pos, neg = scores[labels], scores[~labels]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert metrics.auroc(scores, labels) == pytest.approx(expected)


def test_auroc_single_class_is_nan():
    assert np.isnan(metrics.auroc(np.array([0.1, 0.2]), np.array([True, True])))


def test_visibility_auroc_rewards_confidence_tracking_visibility():
    k = _side_profile_kpts(50, 50)
    k[KP.LEFT_BACK_PAW, 2] = 0
    k[KP.RIGHT_BACK_PAW, 2] = 0
    pred = _pred_from_gt(1, k)
    p = np.asarray(pred["keypoints"]).reshape(17, 3)
    p[:, 2] = np.where(k[:, 2] > 0, 0.9, 0.1)
    pred["keypoints"] = p.reshape(-1).tolist()
    matches, _ = metrics.match_predictions(_gt([(1, k)]), [pred])
    _, pooled = metrics.visibility_auroc(matches)
    assert pooled == 1.0
