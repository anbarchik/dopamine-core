"""Reward Prediction Error computation with confidence weighting and loss aversion."""

from __future__ import annotations

from dopamine_core.config import LossAversionConfig, RPEConfig
from dopamine_core.types import RPEResult

_EPSILON = 0.01


class RPECalculator:
    """Computes Reward Prediction Error using the confidence-weighted formula.

    ## Formula (explicit)

    The formula:
        raw_error = outcome * (1 - conf) + (1 - outcome) * (-conf)

    Algebraically simplifies to:
        raw_error = outcome - conf       (confidence is the implicit prediction)

    This treats expressed **confidence as the agent's probability estimate**
    of a positive outcome. RPE is then how much better or worse the actual
    outcome was versus what the agent expected.

    Consequences:
    - High confidence + wrong = large negative signal (overconfidence penalty)
    - Low confidence + right  = large positive signal (pleasant surprise)
    - Low confidence + wrong  = zero signal (expected loss, no update)
    - High confidence + right = zero signal (expected win, no update)

    ## Baseline Blending (RPEConfig.baseline_blend)

    By default (baseline_blend=0.0), the tonic baseline is NOT used in the RPE
    computation — confidence serves as the prediction. This preserves the
    original research behavior.

    When baseline_blend > 0, the learned tonic expectation is blended into the
    prediction. This closes the feedback loop: the system's learned history
    informs how new outcomes are processed.

        effective_conf = (1 - blend) * conf + blend * tonic_normalized

    baseline_blend=0.0 → pure confidence-based (original)
    baseline_blend=0.3 → 30% tonic influence (recommended for persistent agents)
    baseline_blend=1.0 → pure tonic-based (ignores expressed confidence)

    Loss aversion amplifies negative RPE by 1.87x (Kahneman & Tversky).
    """

    def __init__(
        self,
        config: LossAversionConfig | None = None,
        rpe_config: RPEConfig | None = None,
    ) -> None:
        self._config = config or LossAversionConfig()
        self._rpe_config = rpe_config or RPEConfig()

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
                        Mapped to [0, 1] for the formula; serves as the implicit prediction.
            baseline: Current tonic baseline. With baseline_blend=0 (default), used only
                      for surprise normalization. With baseline_blend>0, blended into the
                      effective prediction to close the learning feedback loop.

        Returns:
            RPEResult with raw and loss-aversion-adjusted error.
        """
        # Map confidence from [-1, 1] to [0, 1]
        # This is the implicit prediction: conf_normalized ≈ P(win)
        conf_normalized = (confidence + 1.0) / 2.0

        # Optional baseline blending: let learned tonic history inform the prediction.
        # Blend=0.0 preserves original behavior; blend=0.3 recommended for persistent agents.
        blend = self._rpe_config.baseline_blend
        if blend > 0.0 and abs(baseline) > _EPSILON:
            # Normalize tonic from its natural range [-2, 2] to [0, 1]
            tonic_normalized = max(0.0, min(1.0, (baseline + 2.0) / 4.0))
            effective_conf = (1.0 - blend) * conf_normalized + blend * tonic_normalized
        else:
            effective_conf = conf_normalized

        # RPE = outcome - effective_confidence (algebraic simplification)
        # Kept in original form for clarity and consistency with the research paper
        raw_error = outcome * (1.0 - effective_conf) + (1.0 - outcome) * (-effective_conf)

        # Apply loss aversion: negative errors (losses vs expectation) are amplified
        if raw_error < 0:
            error = raw_error * self._config.multiplier
        else:
            error = raw_error

        # Surprise = magnitude of error relative to tonic baseline magnitude
        surprise = abs(raw_error) / max(abs(baseline), _EPSILON)

        return RPEResult(
            prediction=effective_conf,   # the effective prediction (conf + optional tonic blend)
            actual=outcome,
            error=error,
            raw_error=raw_error,
            surprise=min(surprise, 10.0),  # cap surprise to avoid extremes
        )
