import numpy as np

from wildframe.contracts import Orientation, Subject, SubjectMode
from wildframe.scene import SceneState, _select_primary, referent_label, resolve_scene
from tests.synthetic import (
    FRAME_W,
    FRAME_H,
    make_subject,
    make_subject_set,
    facing_left_pose,
    facing_right_pose,
    frontal_pose,
    make_detection,
)

CONFIG = {
    "orientation": {"min_axis_conf": 0.3, "frontal_yaw_max_deg": 25},
    "scene": {
        "min_det_conf": 0.25,
        "min_bbox_area_frac": 0,
        "dominance_ratio": 2.5,
        "agreement_fraction": 0.75,
        "overlap_iou_max": 0.3,
        "conf_margin": 0.05,
        "mode_enter_frames": 3,
    },
}

# mode_enter_frames=1: classification-only tests check the raw per-frame mode, not hysteresis.
NO_HYSTERESIS_CONFIG = {**CONFIG, "scene": {**CONFIG["scene"], "mode_enter_frames": 1}}


def test_no_detections_is_none_mode():
    subject_set, _ = resolve_scene([], (FRAME_W, FRAME_H), CONFIG)
    assert subject_set.mode == SubjectMode.NONE
    assert subject_set.count == 0


def test_single_detection_is_single_mode():
    det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    subject_set, _ = resolve_scene([det], (FRAME_W, FRAME_H), NO_HYSTERESIS_CONFIG)
    assert subject_set.mode == SubjectMode.SINGLE
    assert subject_set.count == 1
    assert subject_set.primary is not None


def test_dominant_when_primary_much_larger():
    big = make_detection(facing_left_pose(150, 200), bbox=(50, 50, 350, 450))
    small = make_detection(facing_right_pose(500, 400), bbox=(480, 380, 520, 420))
    subject_set, _ = resolve_scene([big, small], (FRAME_W, FRAME_H), NO_HYSTERESIS_CONFIG)
    assert subject_set.mode == SubjectMode.DOMINANT


def test_facing_pair_when_mutually_facing():
    left_member = make_detection(facing_right_pose(150, 240), bbox=(100, 200, 200, 280))
    right_member = make_detection(facing_left_pose(450, 240), bbox=(400, 200, 500, 280))
    subject_set, _ = resolve_scene([left_member, right_member], (FRAME_W, FRAME_H), NO_HYSTERESIS_CONFIG)
    assert subject_set.mode == SubjectMode.FACING_PAIR


def test_group_aligned_when_all_face_same_direction():
    a = make_detection(facing_left_pose(100, 100), bbox=(60, 60, 140, 140))
    b = make_detection(facing_left_pose(300, 100), bbox=(260, 60, 340, 140))
    c = make_detection(facing_left_pose(500, 100), bbox=(460, 60, 540, 140))
    subject_set, _ = resolve_scene([a, b, c], (FRAME_W, FRAME_H), NO_HYSTERESIS_CONFIG)
    assert subject_set.mode == SubjectMode.GROUP_ALIGNED
    assert subject_set.agreed_orientation is not None
    assert subject_set.agreed_orientation.yaw_state == "left"


def test_mixed_when_orientations_disagree():
    a = make_detection(facing_left_pose(100, 100), bbox=(60, 60, 140, 140))
    b = make_detection(facing_right_pose(300, 100), bbox=(260, 60, 340, 140))
    c = make_detection(frontal_pose(500, 100), bbox=(460, 60, 540, 140))
    subject_set, _ = resolve_scene([a, b, c], (FRAME_W, FRAME_H), NO_HYSTERESIS_CONFIG)
    assert subject_set.mode == SubjectMode.MIXED


def test_count_reliable_false_on_high_overlap():
    a = make_detection(facing_left_pose(200, 200), bbox=(100, 100, 260, 300))
    b = make_detection(facing_right_pose(220, 200), bbox=(150, 100, 310, 300))
    subject_set, _ = resolve_scene([a, b], (FRAME_W, FRAME_H), CONFIG)
    assert subject_set.count_reliable is False


def test_count_reliable_false_near_conf_floor():
    a = make_detection(facing_left_pose(150, 200), bbox=(100, 150, 200, 250), det_conf=0.26)
    b = make_detection(facing_right_pose(450, 200), bbox=(400, 150, 500, 250), det_conf=0.9)
    subject_set, _ = resolve_scene([a, b], (FRAME_W, FRAME_H), CONFIG)
    assert subject_set.count_reliable is False


def test_count_reliable_true_when_well_separated_and_confident():
    a = make_detection(facing_left_pose(150, 200), bbox=(100, 150, 200, 250), det_conf=0.9)
    b = make_detection(facing_right_pose(450, 200), bbox=(400, 150, 500, 250), det_conf=0.9)
    subject_set, _ = resolve_scene([a, b], (FRAME_W, FRAME_H), CONFIG)
    assert subject_set.count_reliable is True


def test_select_primary_tie_break_by_track_id_then_leftmost():
    orient = Orientation(facing_vector=None, yaw_state="unknown", confidence=0.0)
    kpts = np.full((17, 3), 0.9, dtype=np.float32)
    m_right = Subject(bbox_xyxy=np.array([100, 0, 200, 100]), keypoints=kpts, det_conf=0.9, orientation=orient, track_id=5)
    m_left = Subject(bbox_xyxy=np.array([0, 0, 100, 100]), keypoints=kpts, det_conf=0.9, orientation=orient, track_id=2)
    primary = _select_primary((m_right, m_left))
    assert primary.track_id == 2


def test_select_primary_tie_break_leftmost_when_no_track_id():
    orient = Orientation(facing_vector=None, yaw_state="unknown", confidence=0.0)
    kpts = np.full((17, 3), 0.9, dtype=np.float32)
    m_right = Subject(bbox_xyxy=np.array([100, 0, 200, 100]), keypoints=kpts, det_conf=0.9, orientation=orient)
    m_left = Subject(bbox_xyxy=np.array([0, 0, 100, 100]), keypoints=kpts, det_conf=0.9, orientation=orient)
    primary = _select_primary((m_right, m_left))
    assert primary is m_left


def test_mode_acquisition_from_none_is_immediate():
    # Acquiring a subject for the first time is NOT hysteresis-gated - a single frame with one
    # animal must report SINGLE right away, matching Phase A's single-image behavior.
    state = SceneState()
    det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    subject_set, state = resolve_scene([det], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.SINGLE


def test_mode_to_mode_transition_never_flips_before_threshold():
    state = SceneState()
    single_det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    subject_set, state = resolve_scene([single_det], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.SINGLE

    big = make_detection(facing_left_pose(150, 200), bbox=(50, 50, 350, 450))
    small = make_detection(facing_right_pose(500, 400), bbox=(480, 380, 520, 420))

    modes = []
    for _ in range(2):  # threshold=3, only 2 frames seen
        subject_set, state = resolve_scene([big, small], (FRAME_W, FRAME_H), CONFIG, state)
        modes.append(subject_set.mode)
    assert all(m == SubjectMode.SINGLE for m in modes)


def test_mode_to_mode_transition_commits_on_third_frame():
    state = SceneState()
    single_det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    subject_set, state = resolve_scene([single_det], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.SINGLE

    big = make_detection(facing_left_pose(150, 200), bbox=(50, 50, 350, 450))
    small = make_detection(facing_right_pose(500, 400), bbox=(480, 380, 520, 420))

    last_mode = None
    for _ in range(3):
        subject_set, state = resolve_scene([big, small], (FRAME_W, FRAME_H), CONFIG, state)
        last_mode = subject_set.mode
    assert last_mode == SubjectMode.DOMINANT


def test_losing_subject_holds_committed_mode_until_confirmed():
    state = SceneState()
    det = make_detection(facing_left_pose(FRAME_W / 2, FRAME_H / 2))
    subject_set, state = resolve_scene([det], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.SINGLE

    # Two frames of NONE - not enough to flip (threshold=3).
    for _ in range(2):
        subject_set, state = resolve_scene([], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.SINGLE

    # Third consecutive NONE frame flips it.
    subject_set, state = resolve_scene([], (FRAME_W, FRAME_H), CONFIG, state)
    assert subject_set.mode == SubjectMode.NONE


def test_referent_label_single_member():
    members = (make_subject(bbox=(0, 0, 100, 100)),)
    subject_set = make_subject_set(members, mode=SubjectMode.SINGLE)
    assert referent_label(subject_set) == "the animal"


def test_referent_label_no_members():
    subject_set = make_subject_set((), mode=SubjectMode.NONE)
    assert referent_label(subject_set) == "the frame"


def test_referent_label_group_positions():
    members = (
        make_subject(bbox=(0, 0, 50, 50)),
        make_subject(bbox=(200, 0, 250, 50)),
        make_subject(bbox=(400, 0, 450, 50)),
    )
    subject_set = make_subject_set(members, mode=SubjectMode.MIXED)
    assert referent_label(subject_set, member_index=0) == "the left animal"
    assert referent_label(subject_set, member_index=1) == "one of the middle animals"
    assert referent_label(subject_set, member_index=2) == "the right animal"
    assert referent_label(subject_set, member_index=None) == "the group"
