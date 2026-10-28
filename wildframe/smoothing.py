"""Per-rule hysteresis: a raw rule state only becomes the emitted state after N consecutive
frames agree. Keyed on (rule, track_id) - Phase A always used track_id=None (single implicit
subject); Phase B re-keys per animal so each track's hysteresis is independent, and a track
that drops out doesn't linger and confuse a different track that later gets the same primary
role."""

from __future__ import annotations

from dataclasses import dataclass

from wildframe.contracts import RuleName, RuleState

_UNSET = object()


@dataclass
class HysteresisState:
    committed_state: RuleState = RuleState.INDETERMINATE
    pending_state: RuleState | None = None
    pending_count: int = 0


class Smoother:
    def __init__(self, rules_config: dict):
        """rules_config: parsed rules.yaml, read as rules_config[rule.value]['hysteresis_frames']."""
        self._hysteresis_frames = {
            rule: int(rules_config.get(rule.value, {}).get("hysteresis_frames", 1)) for rule in RuleName
        }
        self._states: dict[tuple[RuleName, int | None], HysteresisState] = {}

    def _get_state(self, rule: RuleName, track_id: int | None) -> HysteresisState:
        key = (rule, track_id)
        if key not in self._states:
            self._states[key] = HysteresisState()
        return self._states[key]

    def update(self, rule: RuleName, raw_state: RuleState, track_id: int | None = None) -> RuleState:
        state = self._get_state(rule, track_id)
        threshold = self._hysteresis_frames[rule]

        if raw_state == state.committed_state:
            state.pending_state = None
            state.pending_count = 0
        elif raw_state == state.pending_state:
            state.pending_count += 1
            if state.pending_count >= threshold:
                state.committed_state = raw_state
                state.pending_state = None
                state.pending_count = 0
        else:
            state.pending_state = raw_state
            state.pending_count = 1
            if threshold <= 1:
                state.committed_state = raw_state
                state.pending_state = None
                state.pending_count = 0

        return state.committed_state

    def commit(self, rule: RuleName, state: RuleState, track_id: int | None = None) -> None:
        """Commit `state` immediately, bypassing hysteresis and clearing any pending flip. For
        signals that must react at once in one direction (e.g. READY dropping on a blocking
        violation) while still being smoothed in the other."""
        entry = self._get_state(rule, track_id)
        entry.committed_state = state
        entry.pending_state = None
        entry.pending_count = 0

    def reset(self, rule: RuleName | None = None, track_id: int | None = _UNSET) -> None:
        """reset() clears everything. reset(rule) clears that rule across all tracks (Phase A
        usage). reset(track_id=id) clears every rule for one departed track. reset(rule, track_id)
        clears exactly one (rule, track) entry."""
        if rule is None and track_id is _UNSET:
            self._states = {}
        elif rule is not None and track_id is _UNSET:
            self._states = {k: v for k, v in self._states.items() if k[0] != rule}
        elif rule is None:
            self._states = {k: v for k, v in self._states.items() if k[1] != track_id}
        else:
            self._states.pop((rule, track_id), None)
