"""Golden-fixture parity: the committed golden/ fixtures must match what the Python pipeline does
today. A port of the rule engine (mobile) is checked against the same files, so any behaviour
change here has to be a deliberate `python -m golden.generate` plus a reviewed diff."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from golden.generate import ANDROID_ASSETS, GOLDEN_DIR, detection_from_json, load_configs, run_frames, run_tracker

CASES = sorted((GOLDEN_DIR / "cases").glob("*.json"))
TRACKER_CASES = sorted((GOLDEN_DIR / "tracker").glob("*.json"))


def assert_matches(actual, expected, path="$"):
    """Exact for strings/ints/bools/None and structure; floats to 1e-6 relative / 1e-5 absolute."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), f"{path}: {actual!r} vs {expected!r}"
        assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5), f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for k in expected:
            assert_matches(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_matches(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_fixtures_exist():
    assert CASES and TRACKER_CASES


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_decision_case_matches_pipeline(path: Path):
    case = json.loads(path.read_text())
    frames = [[detection_from_json(d) for d in f["detections"]] for f in case["frames"]]
    actual = json.loads(json.dumps(run_frames(frames, case["rules_config"], tuple(case["frame_dims"]))))
    assert_matches(actual, [f["expected"] for f in case["frames"]])


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_decision_case_uses_current_config(path: Path):
    """A rules.yaml change must be followed by regenerating the fixtures, or the mobile port
    would be validated against thresholds the app no longer uses."""
    rules_cfg, _ = load_configs()
    assert json.loads(path.read_text())["rules_config"] == rules_cfg, "configs/rules.yaml changed: run python -m golden.generate"


@pytest.mark.parametrize("path", TRACKER_CASES, ids=lambda p: p.stem)
def test_tracker_case_matches(path: Path):
    case = json.loads(path.read_text())
    _, model_cfg = load_configs()
    assert case["tracker_config"] == {k: model_cfg[k] for k in case["tracker_config"]}, "configs/model.yaml tracking changed"
    actual = run_tracker([f["boxes_xyxy"] for f in case["frames"]], case["tracker_config"])
    assert actual == [f["expected_track_ids"] for f in case["frames"]]


def test_android_config_assets_match_yaml():
    """The Android app reads JSON copies of configs/*.yaml from its assets; they must not drift."""
    rules_cfg, model_cfg = load_configs()
    assert json.loads((ANDROID_ASSETS / "rules.json").read_text()) == rules_cfg, "run python -m golden.generate"
    assert json.loads((ANDROID_ASSETS / "model.json").read_text()) == model_cfg, "run python -m golden.generate"
