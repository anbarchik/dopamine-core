"""DopamineEngine — the central orchestrator for synthetic dopamine signals."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from copy import deepcopy

from dopamine_core.config import DopamineConfig
from dopamine_core.distributional.channels import DistributionalChannels
from dopamine_core.distributional.coding import DistributionalCoding
from dopamine_core.injection.context import ContextInjector
from dopamine_core.reward.dual_mode import DualModeReward
from dopamine_core.safety.monitor import SafetyMonitor
from dopamine_core.signals.extractor import SignalExtractor
from dopamine_core.signals.momentum import MomentumTracker
from dopamine_core.signals.rpe import RPECalculator
from dopamine_core.timescale.tracker import TimescaleTracker
from dopamine_core.types import (
    CompositeSignal,
    EngineState,
    ExtractedSignals,
    Outcome,
    TimescaleLevel,
)


class DopamineEngine:
    """Main entry point for DopamineCore.

    Usage::

        engine = DopamineEngine()

        # In your agent loop
        prompt = engine.inject_context(base_prompt)
        response = agent.complete(prompt)
        outcome = evaluate_trade(response)
        engine.update(response.text, outcome)
    """

    def __init__(
        self,
        config: DopamineConfig | None = None,
        coordinate: str | None = None,
    ) -> None:
        self._config = config or DopamineConfig()
        # Phext coordinate identity — tags this engine instance in scrollspace.
        # Format: "library.shelf.series/collection.volume.book/chapter.section.scroll"
        # Valid range: 1-9 per component (modulo 9+1 arithmetic).
        self._coordinate = coordinate

        # Commit log (TTSM-style WAL): list of (commit_hash, EngineState)
        self._commit_log: list[tuple[str, EngineState]] = []

        # Core components
        self._extractor = SignalExtractor()
        self._rpe_calc = RPECalculator(self._config.loss_aversion, self._config.rpe)
        self._injector = ContextInjector(self._config.injection)
        self._dual_mode = DualModeReward(
            tonic_config=self._config.tonic,
            phasic_config=self._config.phasic,
        )
        self._momentum = MomentumTracker(self._config.momentum)
        self._distributional = DistributionalChannels(self._config.distributional)
        self._dist_coding = DistributionalCoding(self._distributional)
        self._timescale = TimescaleTracker(self._config.timescale)
        self._safety = SafetyMonitor(self._config.safety)

        # State
        self._step_count: int = 0
        self._outcome_history: deque[float] = deque(maxlen=100)
        self._last_rpe: float = 0.0
        self._last_signals: ExtractedSignals | None = None
        self._last_composite: CompositeSignal | None = None

    def inject_context(self, base_prompt: str) -> str:
        """Add subliminal reward context to a prompt.

        Uses the current internal state to generate naturalistic
        environmental context that steers agent behavior.

        Args:
            base_prompt: The original prompt to augment.

        Returns:
            Augmented prompt with injected motivation context.
        """
        if self._safety.is_circuit_broken:
            return base_prompt

        signal = self._build_current_signal()
        context = self._injector.build_context(signal)
        return self._injector.inject(base_prompt, context)

    def update(
        self,
        response_text: str,
        outcome: Outcome | float,
    ) -> CompositeSignal:
        """Process an outcome and update all internal state.

        Args:
            response_text: The agent's CoT/response text.
            outcome: Trade outcome — either an Outcome object or raw PnL float.

        Returns:
            The computed composite reward signal.
        """
        # Normalize outcome
        if isinstance(outcome, (int, float)):
            outcome = Outcome(pnl=float(outcome))

        # 1. Extract behavioral signals from response
        signals = self._extractor.extract(response_text)
        self._last_signals = signals

        # 2. Normalize PnL to [0, 1] for RPE formula
        normalized_outcome = self._normalize_pnl(outcome.pnl)

        # 3. Compute RPE
        confidence = outcome.confidence if outcome.confidence is not None else signals.confidence
        rpe = self._rpe_calc.compute(
            outcome=normalized_outcome,
            confidence=confidence,
            baseline=self._dual_mode.tonic.level,
        )
        self._last_rpe = rpe.error

        # 4. Process through dual-mode reward system (tonic + phasic)
        composite_value = self._dual_mode.process(rpe)

        # 5. Update distributional channels
        self._distributional.update(normalized_outcome)

        # 6. Update timescale tracker
        self._timescale.update(composite_value, TimescaleLevel.STEP)

        # 7. Update momentum tracking
        self._momentum.update(outcome.pnl)

        # 8. Record outcome
        self._outcome_history.append(outcome.pnl)
        self._step_count += 1

        # 9. Safety: clamp and apply attenuation
        clamped_value = self._safety.clamp_signal(composite_value)
        attenuation = self._safety.get_attenuation_factor()
        clamped_value *= attenuation

        # 10. Safety: check for violations
        self._safety.check_and_record(clamped_value, confidence)

        # 11. Build composite signal with distributional risk assessment
        risk_score = self._dist_coding.get_risk_score()
        composite = CompositeSignal(
            value=clamped_value,
            confidence_factor=confidence,
            risk_assessment=risk_score,
            momentum_factor=self._momentum.get_momentum_factor(),
            tonic_level=self._dual_mode.tonic.level,
            phasic_response=self._dual_mode.phasic.current_response,
        )
        self._last_composite = composite
        return composite

    def get_state(self) -> EngineState:
        """Snapshot full state for serialization/persistence."""
        return EngineState(
            tonic_baseline=self._dual_mode.tonic.level,
            step_count=self._step_count,
            outcome_history=list(self._outcome_history),
            streak_count=self._momentum.streak_count,
            streak_sign=self._momentum.streak_sign,
            phasic_signals=self._dual_mode.phasic.get_history(),
            channel_expectations=self._distributional.expectations,
            last_rpe=self._last_rpe,
        )

    def load_state(self, state: EngineState) -> None:
        """Restore engine from a saved state snapshot."""
        self._dual_mode.tonic.load(state.tonic_baseline, state.step_count)
        self._step_count = state.step_count
        self._outcome_history = deque(state.outcome_history, maxlen=100)
        self._momentum.load(state.streak_count, state.streak_sign)
        if state.phasic_signals:
            self._dual_mode.phasic.load_history(state.phasic_signals)
        if state.channel_expectations:
            self._distributional.load_expectations(state.channel_expectations)
        self._last_rpe = state.last_rpe

    def commit(self, label: str = "") -> str:
        """Commit current state to the WAL (TTSM-style write-ahead log).

        Creates an immutable snapshot of the current engine state. The commit
        hash is deterministic from the state content — same state always
        produces the same hash.

        Args:
            label: Optional human-readable label for this commit.

        Returns:
            Commit hash (8-char hex).
        """
        state = self.get_state()
        state_json = json.dumps(
            {
                "tonic_baseline": state.tonic_baseline,
                "step_count": state.step_count,
                "outcome_history": state.outcome_history,
                "streak_count": state.streak_count,
                "streak_sign": state.streak_sign,
                "last_rpe": state.last_rpe,
                "coordinate": self._coordinate,
                "label": label,
            },
            sort_keys=True,
        )
        commit_hash = hashlib.sha256(state_json.encode()).hexdigest()[:8]
        self._commit_log.append((commit_hash, state))
        return commit_hash

    def fork(self, commit_hash: str) -> "DopamineEngine":
        """Fork a new engine from a previous commit (TTSM fork semantics).

        Creates an independent engine instance restored to the state at the
        given commit. Enables time-travel debugging and branched agent histories.

        Args:
            commit_hash: Hash returned by a previous commit() call.

        Returns:
            New DopamineEngine instance at the forked state.

        Raises:
            KeyError: If commit_hash not found in the log.
        """
        for h, state in self._commit_log:
            if h == commit_hash:
                forked = DopamineEngine(
                    config=deepcopy(self._config),
                    coordinate=self._coordinate,
                )
                forked.load_state(state)
                forked._commit_log = list(self._commit_log)  # carry history forward
                return forked
        raise KeyError(f"Commit {commit_hash!r} not found in WAL ({len(self._commit_log)} commits)")

    @property
    def coordinate(self) -> str | None:
        """Phext coordinate identity of this engine instance."""
        return self._coordinate

    @property
    def commit_log(self) -> list[tuple[str, EngineState]]:
        """Read-only view of the TTSM-style commit log."""
        return list(self._commit_log)

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._dual_mode.reset()
        self._momentum.reset()
        self._distributional.reset()
        self._timescale.reset()
        self._safety.reset()
        self._step_count = 0
        self._outcome_history.clear()
        self._last_rpe = 0.0
        self._last_signals = None
        self._last_composite = None

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def tonic_baseline(self) -> float:
        return self._dual_mode.tonic.level

    @property
    def last_composite(self) -> CompositeSignal | None:
        return self._last_composite

    @property
    def safety(self) -> SafetyMonitor:
        return self._safety

    @property
    def timescale(self) -> TimescaleTracker:
        return self._timescale

    @property
    def distributional(self) -> DistributionalChannels:
        return self._distributional

    # --- Private methods ---

    def _build_current_signal(self) -> CompositeSignal:
        """Build a signal from current state (for injection before update)."""
        if self._last_composite is not None:
            return self._last_composite
        return CompositeSignal(
            value=0.0,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=self._dual_mode.tonic.level,
            phasic_response=0.0,
        )

    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize raw PnL to [0, 1] using a sigmoid-like mapping.

        Uses the configured pnl_scale to handle different PnL magnitudes:
        - For USDC bets (pnl ±1): pnl_scale=1.0 (default)
        - For dollar amounts (pnl ±100): pnl_scale=100.0
        """
        scale = self._dual_mode.pnl_scale
        normalized = math.tanh(pnl / scale)
        return (normalized + 1.0) / 2.0  # shift from [-1,1] to [0,1]
