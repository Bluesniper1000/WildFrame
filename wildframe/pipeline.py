"""Multi-animal composition pipeline: frame source -> detection -> scene resolution -> rules ->
smoothing -> priority -> template -> overlay -> output. Deterministic, no learned components
beyond the frozen pose model (and ByteTrack's Kalman filter, allowed by design)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from wildframe.contracts import Detection, RuleName, RuleOutput, RuleState, SubjectMode, SubjectSet
from wildframe.explain import overlay, priority, templates
from wildframe.explain.priority import PriorityItem
from wildframe.pose.backend import PoseBackend, REPO_ROOT
from wildframe.rules.centrality import CentralityRule
from wildframe.rules.eye_thirds import EyeThirdsRule
from wildframe.rules.lead_space import LeadSpaceRule
from wildframe.rules.limb_clip import LimbClipRule
from wildframe.rules.odd_count import OddCountRule
from wildframe.scene import SceneState, composite_subject, referent_label, resolve_scene
from wildframe.smoothing import Smoother


def _parse_priority_order(raw: list) -> list[PriorityItem]:
    """A priority_order entry is either a rule name, or a list of rule names naming an
    OR-group (see explain/priority.py) - either representation round-trips through YAML."""
    parsed: list[PriorityItem] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            parsed.append(tuple(RuleName(r) for r in item))
        else:
            parsed.append(RuleName(item))
    return parsed


@dataclass
class FrameResult:
    annotated_frame: np.ndarray
    chosen: RuleOutput | None
    all_outputs: dict[RuleName, RuleOutput]
    mode_notice: str | None = None


@dataclass
class Decision:
    """Everything the deterministic detections -> instruction path produces for one frame,
    before anything is drawn. This is the contract a port of the rule engine (e.g. mobile)
    must reproduce - see golden/."""

    subject_set: SubjectSet
    chosen: RuleOutput | None
    raw_outputs: dict[RuleName, RuleOutput]
    all_outputs: dict[RuleName, RuleOutput]
    ready: RuleOutput | None
    mode_notice: str | None


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


class Pipeline:
    def __init__(
        self,
        model_config_path: str | Path = "configs/model.yaml",
        rules_config_path: str | Path = "configs/rules.yaml",
        load_model: bool = True,
    ):
        # load_model=False builds only the rule/decision side, for driving decide() with
        # detections directly (golden fixtures, tests) without the ONNX weights present.
        self.backend = PoseBackend(model_config_path) if load_model else None
        self.rules_config = yaml.safe_load(_resolve(rules_config_path).read_text())

        self.rules = {
            RuleName.LEAD_SPACE: LeadSpaceRule(self.rules_config["lead_space"]),
            RuleName.EYE_THIRDS: EyeThirdsRule(self.rules_config["eye_thirds"]),
            RuleName.LIMB_CLIP: LimbClipRule(self.rules_config["limb_clip"]),
            RuleName.ODD_COUNT: OddCountRule(self.rules_config["odd_count"]),
            RuleName.CENTRALITY: CentralityRule(self.rules_config["centrality"]),
        }
        self.smoother = Smoother(self.rules_config)
        self.priority_order = _parse_priority_order(self.rules_config["priority_order"])
        self._last_committed_output: dict[tuple[RuleName, int | None], RuleOutput] = {}
        self._last_primary_track_id: int | None = None
        self._scene_state = SceneState()
        self._last_emitted_mode = SubjectMode.NONE

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        detections = self.backend.infer(frame_bgr)
        h, w = frame_bgr.shape[:2]
        decision = self.decide(detections, (w, h))
        subject_set = decision.subject_set

        if subject_set.mode == SubjectMode.NONE or subject_set.primary is None:
            annotated = overlay.draw_overlay(frame_bgr, None, None, mode_notice=decision.mode_notice)
        else:
            display_subject = composite_subject(subject_set) or subject_set.primary
            annotated = overlay.draw_overlay(
                frame_bgr, display_subject, decision.chosen, mode_notice=decision.mode_notice, subject_set=subject_set
            )
        return FrameResult(
            annotated_frame=annotated, chosen=decision.chosen, all_outputs=decision.all_outputs,
            mode_notice=decision.mode_notice,
        )

    def decide(self, detections: list[Detection], frame_dims: tuple[int, int]) -> Decision:
        """The deterministic per-frame path: detections -> scene -> rules -> hysteresis ->
        ready/priority -> rendered instruction. Stateful across calls (scene mode hysteresis,
        per-rule smoothing), so feed frames in order."""
        subject_set, self._scene_state = resolve_scene(detections, frame_dims, self.rules_config, self._scene_state)

        mode_notice = None
        if subject_set.mode != self._last_emitted_mode:
            mode_notice = templates.render_mode_notice(subject_set)
            self._last_emitted_mode = subject_set.mode

        if subject_set.mode == SubjectMode.NONE or subject_set.primary is None:
            return Decision(subject_set=subject_set, chosen=None, raw_outputs={}, all_outputs={},
                            ready=None, mode_notice=mode_notice)

        primary_track_id = subject_set.primary.track_id

        # Per-rule hysteresis is keyed on (rule, track_id) so each animal's state is tracked
        # independently. When the primary's identity changes, its predecessor's hysteresis
        # state is no longer relevant to it - drop it rather than let it linger unbounded.
        if primary_track_id != self._last_primary_track_id:
            self.smoother.reset(track_id=self._last_primary_track_id)
            self._last_committed_output = {
                k: v for k, v in self._last_committed_output.items() if k[1] != self._last_primary_track_id
            }
            self._last_primary_track_id = primary_track_id

        raw_outputs = {name: rule.run(subject_set) for name, rule in self.rules.items()}
        raw_outputs = {name: self._attach_referent(subject_set, out) for name, out in raw_outputs.items()}
        smoothed_outputs = {
            name: self._apply_smoothing(name, out, primary_track_id) for name, out in raw_outputs.items()
        }

        ready_output = self._resolve_ready(smoothed_outputs, primary_track_id)
        if ready_output.state == RuleState.SATISFIED:
            chosen = templates.render(ready_output)
        else:
            chosen = priority.resolve(smoothed_outputs, self.priority_order)
            if chosen is not None:
                chosen = templates.render(chosen)

        return Decision(subject_set=subject_set, chosen=chosen, raw_outputs=raw_outputs, all_outputs=smoothed_outputs,
                        ready=ready_output, mode_notice=mode_notice)

    @staticmethod
    def _attach_referent(subject_set, raw_output: RuleOutput) -> RuleOutput:
        from dataclasses import replace

        member_index = (raw_output.correction or {}).get("member_index")
        referent = referent_label(subject_set, member_index)
        return replace(raw_output, referent=referent)

    def _resolve_ready(self, smoothed_outputs: dict[RuleName, RuleOutput], track_id: int | None) -> RuleOutput:
        """Aggregate "no blocking violations" signal, computed from the already-smoothed
        per-rule states so it doesn't compete frame-to-frame with the rules it's summarizing.
        Evaluated over `priority.resolved_outputs` (one representative per priority-order slot),
        not the raw per-rule dict directly - so an OR-group (e.g. eye_thirds/centrality) that
        has any SATISFIED member counts as fully resolved here too, not just for display; a
        violated centrality alone must not block "click now" when eye_thirds is fine.
        A VIOLATED slot at or below ready.severity_max doesn't block readiness either - it's
        treated as fully resolved for display too, so "click now" and a lingering minor
        correction never show at once. An INDETERMINATE slot (gated out, e.g. eye not visible)
        also doesn't block it.

        Asymmetric hysteresis, keyed on (RuleName.READY, track_id): turning ready ON needs
        ready.hysteresis_frames agreeing frames, so "click now" doesn't flicker on; turning it
        OFF is immediate. The inputs are already-smoothed rule states, so a blocking violation
        here has already survived its own rule's hysteresis - adding READY's on top used to keep
        "click now" on screen for several frames after, e.g., limb_clip had committed VIOLATED."""
        ready_cfg = self.rules_config.get("ready", {})
        severity_max = float(ready_cfg.get("severity_max", 0.3))

        slots = priority.resolved_outputs(smoothed_outputs, self.priority_order)
        all_clear = all(out.state != RuleState.VIOLATED or out.severity <= severity_max for out in slots)
        raw_ready = RuleOutput(
            rule=RuleName.READY,
            state=RuleState.SATISFIED if all_clear else RuleState.VIOLATED,
            reason="all_clear" if all_clear else "has_blocking_violation",
        )
        if not all_clear:
            self.smoother.commit(RuleName.READY, RuleState.VIOLATED, track_id)
        return self._apply_smoothing(RuleName.READY, raw_ready, track_id)

    def _apply_smoothing(self, rule_name: RuleName, raw_output: RuleOutput, track_id: int | None) -> RuleOutput:
        cache_key = (rule_name, track_id)
        smoothed_state = self.smoother.update(rule_name, raw_output.state, track_id)
        if smoothed_state == raw_output.state:
            # The emitted state matches this frame's raw evaluation, so its reason/correction/
            # debug fields are accurate for what's being shown - cache and use them as-is.
            self._last_committed_output[cache_key] = raw_output
            return raw_output

        # The emitted state is still the previously-committed one (hysteresis hasn't flipped
        # yet); reuse that commit's own fields rather than pairing this frame's mismatched
        # raw reason/correction with a different displayed state.
        cached = self._last_committed_output.get(cache_key)
        if cached is not None and cached.state == smoothed_state:
            return cached
        return RuleOutput(rule=rule_name, state=smoothed_state, reason="insufficient_history")

    def run_on_video(self, source: str | int, display: bool = False, output_path: str | None = None) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source: {source}")

        writer = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.process_frame(frame)

                if output_path is not None:
                    if writer is None:
                        h, w = result.annotated_frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(result.annotated_frame)

                if display:
                    cv2.imshow("WildFrame", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video file path, webcam index, or a single image path")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output", default=None, help="output video/image path")
    args = parser.parse_args()

    pipeline = Pipeline()

    source_path = Path(args.source)
    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise SystemExit(f"could not read image: {args.source}")
        result = pipeline.process_frame(frame)
        if args.output:
            cv2.imwrite(args.output, result.annotated_frame)
        if args.display:
            cv2.imshow("WildFrame", result.annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        message = result.chosen.message if result.chosen else "no detection"
        print(message)
        return

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline.run_on_video(source, display=args.display, output_path=args.output)


if __name__ == "__main__":
    main()
