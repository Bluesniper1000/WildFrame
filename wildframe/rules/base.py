"""Rule interface for lead_space / eye_thirds / limb_clip. Rules consume the composite
SubjectSet built by scene.py, not raw detections - in SINGLE/DOMINANT mode this degenerates
to exactly what Phase A produced (see scene.composite_subject / scene.members_for_clipping)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wildframe.contracts import RuleName, RuleOutput, RuleState, SubjectSet


class Rule(ABC):
    name: RuleName  # class attribute, set by each subclass

    def __init__(self, config: dict):
        """config: this rule's own sub-block from rules.yaml."""
        self.config = config

    @abstractmethod
    def gate(self, subject_set: SubjectSet) -> str | None:
        """Return a reason string if the rule cannot be meaningfully evaluated this frame,
        else None to proceed to evaluate()."""

    @abstractmethod
    def evaluate(self, subject_set: SubjectSet) -> RuleOutput:
        """Called only if gate() returned None. Must return state SATISFIED or VIOLATED -
        never INDETERMINATE, which is exclusively gate()'s job."""

    def run(self, subject_set: SubjectSet) -> RuleOutput:
        reason = self.gate(subject_set)
        if reason is not None:
            return RuleOutput(rule=self.name, state=RuleState.INDETERMINATE, reason=reason)
        return self.evaluate(subject_set)
