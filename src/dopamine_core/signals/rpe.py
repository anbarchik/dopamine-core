"""Reward Prediction Error computation with confidence weighting and loss aversion."""

from __future__ import annotations

from dopamine_core.config import LossAversionConfig
from dopamine_core.types import RPEResult

_EPSILON = 0.01


class RPECalculator:
    """Computes Reward Prediction Error using the confidence-weighted formula.

    RPE = outcome * (1 - confidence) + (1 - outcome) * (-confidence)

    This means:
    - High confidence + wrong = large negative signal (overconfidence penalty)
    - Low confidence + right = moderate positive signal (lucky win, muted reward)
    - Loss aversion amplifies negative RPE by 1.87x (Kahneman & Tversky)
    """

    def __init__(self, config: LossAversionConfig | None = None) -> None:
        self._config = config or LossAversionConfig()

    def compute(
        self,
        outcome: float,
        confidence: float,
        baseline: float = 0.0,
    ) -> RPEResult:
        """Compute confidence-weighted RPE.

        Args:
            outcome: Normalized outcome in [0, 1] where 1 = full win, 0 = full loss.
            confidence: Agent's extracted confidence in [-1, 1].
                        Mapped to [0, 1] for the formula.
            baseline: Current tonic baseline for prediction comparison.

        Returns:
            RPEResult with raw and loss-aversion-adjusted error.
        """
        # Map confidence from [-1, 1] to [0, 1] for the RPE formula
        conf_normalized = (confidence + 1.0) / 2.0

        # Core RPE formula from the research
        raw_error = outcome * (1.0 - conf_normalized) + (1.0 - outcome) * (-conf_normalized)

        # Apply loss aversion: negative errors are amplified
        if raw_error < 0:
            error = raw_error * self._config.multiplier
        else:
            error = raw_error

        # Surprise = how unexpected relative to baseline
        surprise = abs(raw_error) / max(abs(baseline), _EPSILON)

        return RPEResult(
            prediction=baseline,
            actual=outcome,
            error=error,
            raw_error=raw_error,
            surprise=min(surprise, 10.0),  # cap surprise to avoid extremes
        )
