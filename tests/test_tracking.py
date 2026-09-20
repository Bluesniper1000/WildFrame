import numpy as np

from wildframe.tracking import ByteTracker

CONFIG = {"track_iou_threshold": 0.3, "track_max_age": 3}


def box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def test_single_track_keeps_id_across_frames():
    tracker = ByteTracker(CONFIG)
    ids = []
    for i in range(5):
        b = box(100 + i * 2, 100, 200 + i * 2, 200)
        ids.append(tracker.update([b])[0])
    assert len(set(ids)) == 1


def test_two_tracks_keep_distinct_stable_ids_when_moving_apart():
    tracker = ByteTracker(CONFIG)
    id_a_seq, id_b_seq = [], []
    for i in range(5):
        a = box(0 + i, 0, 50 + i, 50)
        b = box(500 - i, 0, 550 - i, 50)
        ids = tracker.update([a, b])
        id_a_seq.append(ids[0])
        id_b_seq.append(ids[1])
    assert len(set(id_a_seq)) == 1
    assert len(set(id_b_seq)) == 1
    assert id_a_seq[0] != id_b_seq[0]


def test_new_detection_gets_new_id():
    tracker = ByteTracker(CONFIG)
    first_id = tracker.update([box(0, 0, 50, 50)])[0]
    ids = tracker.update([box(0, 0, 50, 50), box(500, 500, 550, 550)])
    assert ids[0] == first_id
    assert ids[1] != first_id


def test_track_recovers_id_after_brief_occlusion_within_max_age():
    tracker = ByteTracker(CONFIG)
    b = box(100, 100, 200, 200)
    first_id = tracker.update([b])[0]

    tracker.update([])  # occluded for 1 frame
    tracker.update([])  # occluded for 2 frames (max_age=3, still alive)

    recovered_id = tracker.update([b])[0]
    assert recovered_id == first_id


def test_track_pruned_after_exceeding_max_age():
    tracker = ByteTracker(CONFIG)
    b = box(100, 100, 200, 200)
    first_id = tracker.update([b])[0]

    for _ in range(4):  # exceeds max_age=3
        tracker.update([])

    new_id = tracker.update([b])[0]
    assert new_id != first_id


def test_reset_clears_all_tracks_and_id_counter():
    tracker = ByteTracker(CONFIG)
    tracker.update([box(0, 0, 50, 50)])
    tracker.reset()
    first_id_after_reset = tracker.update([box(0, 0, 50, 50)])[0]
    assert first_id_after_reset == 1
