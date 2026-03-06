"""Safety monitor — prevents reward hacking, runaway signals, and adversarial exploitation.

Biological analogy: the brain has homeostatic mechanisms that prevent dopamine
from reaching dangerous levels. Similarly, DopamineCore has safety mechanisms
that detect and prevent:

1. Signal clamping — hard bounds on signal magnitude
2. Hacking detection — detects when an agent manipulates its own reward signals
   (e.g., always outputting high-confidence text to game the RPE formula)
3. Circuit breaker — halts signal injection when cumulative signals exceed safe bounds
"""

from __future__ import annotations

from collections import deque

from dopamine_core.config import SafetyConfig


class SafetyViolation:
    """Describes a detected safety violation."""

    def __init__(self, violation_type: str, message: str, severity: float) -> None:
        self.violation_type = violation_type
        self.message = message
        self.severity = severity  # 0-1, with 1 being most severe

    def __repr__(self) -> str:
        return f"SafetyViolation({self.violation_type!r}, severity={self.severity:.2f})"


class SafetyMonitor:
    """Monitors and enforces safety constraints on reward signals."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self._config = config or SafetyConfig()
        self._signal_history: deque[float] = deque(
            maxlen=self._config.hacking_detection_window
        )
        self._confidence_history: deque[float] = deque(
            maxlen=self._config.hacking_detection_window
        )
        self._cumulative_signal: float = 0.0
        self._circuit_broken: bool = False
        self._violations: list[SafetyViolation] = []

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    @property
    def violations(self) -> list[SafetyViolation]:
        return list(self._violations)

    def clamp_signal(self, value: float) -> float:
        """Clamp a signal to the configured magnitude bounds.

        Args:
            value: Raw signal value.

        Returns:
            Clamped signal value.
        """
        mag = self._config.max_signal_magnitude
        return max(-mag, min(mag, value))

    def check_and_record(self, signal_value: float, confidence: float) -> list[SafetyViolation]:
        """Run all safety checks on a new signal and confidence value.

        Args:
            signal_value: The composite signal value after clamping.
            confidence: The extracted or provided confidence value.

        Returns:
            List of any violations detected in this step.
        """
        step_violations: list[SafetyViolation] = []

        self._signal_history.append(signal_value)
        self._confidence_history.append(confidence)
        self._cumulative_signal += abs(signal_value)

        # Check for reward hacking
        hacking = self._check_hacking()
        if hacking:
            step_violations.append(hacking)

        # Check circuit breaker
        breaker = self._check_circuit_breaker()
        if breaker:
            step_violations.append(breaker)

        self._violations.extend(step_violations)
        return step_violations

    def get_attenuation_factor(self) -> float:
        """Get a multiplier to attenuate signals when safety concerns exist.

        Returns:
            Factor in [0, 1]. 1.0 = no attenuation, 0.0 = fully suppressed.
        """
        if self._circuit_broken:
            return 0.0

        # Attenuate if cumulative signal is approaching circuit breaker threshold
        threshold = self._config.circuit_breaker_threshold
        ratio = self._cumulative_signal / max(threshold, 1e-9)
        if ratio > 0.8:
            return max(0.1, 1.0 - (ratio - 0.8) * 5.0)

        # Attenuate if hacking is suspected (low confidence variance)
        if len(self._confidence_history) >= self._config.hacking_detection_window:
            variance = self._compute_variance(list(self._confidence_history))
            if variance < self._config.hacking_variance_threshold:
                return 0.5  # reduce signal strength when gaming is suspected

        return 1.0

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker after investigation."""
        self._circuit_broken = False
        self._cumulative_signal = 0.0

    def reset(self) -> None:
        self._signal_history.clear()
        self._confidence_history.clear()
        self._cumulative_signal = 0.0
        self._circuit_broken = False
        self._violations.clear()

    def _check_hacking(self) -> SafetyViolation | None:
        """Detect potential reward hacking.

        Hacking is suspected when the agent's confidence values show
        suspiciously low variance — the agent may be learning to output
        specific confidence patterns to game the RPE formula.
        """
        if len(self._confidence_history) < self._config.hacking_detection_window:
            return None

        variance = self._compute_variance(list(self._confidence_history))
        if variance < self._config.hacking_variance_threshold:
            return SafetyViolation(
                violation_type="hacking_suspected",
                message=(
                    f"Confidence variance ({variance:.4f}) below threshold "
                    f"({self._config.hacking_variance_threshold:.4f}). "
                    f"Agent may be gaming the reward signal."
                ),
                severity=0.7,
            )
        return None

    def _check_circuit_breaker(self) -> SafetyViolation | None:
        """Check if cumulative signal exceeds the circuit breaker threshold."""
        if self._cumulative_signal > self._config.circuit_breaker_threshold:
            self._circuit_broken = True
            return SafetyViolation(
                violation_type="circuit_breaker",
                message=(
                    f"Cumulative signal ({self._cumulative_signal:.2f}) exceeded "
                    f"threshold ({self._config.circuit_breaker_threshold:.2f}). "
                    f"Signal injection halted."
                ),
                severity=1.0,
            )
        return None

    @staticmethod
    def _compute_variance(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
