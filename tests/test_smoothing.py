from wildframe.contracts import RuleName, RuleState
from wildframe.smoothing import Smoother

CONFIG = {
    "lead_space": {"hysteresis_frames": 5},
    "eye_thirds": {"hysteresis_frames": 5},
    "limb_clip": {"hysteresis_frames": 5},
}


def test_initial_state_boundary_commits_on_nth_frame():
    s = Smoother(CONFIG)
    outputs = [s.update(RuleName.LEAD_SPACE, RuleState.SATISFIED) for _ in range(5)]
    assert outputs[:4] == [RuleState.INDETERMINATE] * 4
    assert outputs[4] == RuleState.SATISFIED


def test_flicker_never_commits():
    s = Smoother(CONFIG)
    sequence = [RuleState.SATISFIED, RuleState.VIOLATED] * 2  # 4 frames, threshold=5
    outputs = [s.update(RuleName.LEAD_SPACE, raw) for raw in sequence]
    assert all(o == RuleState.INDETERMINATE for o in outputs)


def test_sustained_flip_commits_on_exact_nth_frame():
    s = Smoother(CONFIG)
    for _ in range(20):
        out = s.update(RuleName.LEAD_SPACE, RuleState.SATISFIED)
    assert out == RuleState.SATISFIED

    outputs = [s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED) for _ in range(5)]
    assert outputs[:4] == [RuleState.SATISFIED] * 4
    assert outputs[4] == RuleState.VIOLATED


def test_rules_tracked_independently():
    s = Smoother(CONFIG)
    for _ in range(5):
        s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED)
    out_eye = None
    for _ in range(5):
        out_eye = s.update(RuleName.EYE_THIRDS, RuleState.SATISFIED)

    assert s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED) == RuleState.VIOLATED
    assert out_eye == RuleState.SATISFIED


def test_reset_isolates_single_rule():
    s = Smoother(CONFIG)
    for _ in range(5):
        s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED)
    for _ in range(5):
        s.update(RuleName.EYE_THIRDS, RuleState.VIOLATED)

    s.reset(RuleName.LEAD_SPACE)

    assert s.update(RuleName.LEAD_SPACE, RuleState.INDETERMINATE) == RuleState.INDETERMINATE
    # EYE_THIRDS untouched by the reset of LEAD_SPACE.
    assert s.update(RuleName.EYE_THIRDS, RuleState.VIOLATED) == RuleState.VIOLATED


def test_commit_bypasses_hysteresis_and_clears_pending():
    s = Smoother(CONFIG)
    for _ in range(5):
        s.update(RuleName.LEAD_SPACE, RuleState.SATISFIED)
    s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED)  # pending 1/5
    s.commit(RuleName.LEAD_SPACE, RuleState.INDETERMINATE)
    assert s.update(RuleName.LEAD_SPACE, RuleState.INDETERMINATE) == RuleState.INDETERMINATE
    outputs = [s.update(RuleName.LEAD_SPACE, RuleState.VIOLATED) for _ in range(5)]
    assert outputs[:4] == [RuleState.INDETERMINATE] * 4  # earlier pending count was cleared
    assert outputs[4] == RuleState.VIOLATED
