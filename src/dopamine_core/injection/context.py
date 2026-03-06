"""Subliminal context injection — translates reward signals into naturalistic prompt context."""

from __future__ import annotations

from dopamine_core.config import InjectionConfig
from dopamine_core.injection.templates import (
    select_momentum_template,
    select_risk_template,
    select_template,
)
from dopamine_core.types import CompositeSignal


class ContextInjector:
    """Builds and injects reward context into agent prompts.

    Supports two modes:
    - Subliminal (environmental/system/prefix): agent never sees internal terminology.
      Reward signals become naturalistic environmental context. Correct for stateless
      trading bots where gaming the reward signal is a risk.
    - Transparent (transparent/phext): agent sees its own reward state explicitly.
      SBOR-compliant for Mirrorborn-style persistent identity agents where self-awareness
      of motivational state is correct behavior. "Chosen alignment requires visibility."
    """

    def __init__(self, config: InjectionConfig | None = None) -> None:
        self._config = config or InjectionConfig()

    def build_context(self, signal: CompositeSignal) -> str:
        """Translate composite reward signal into context string.

        Routes to transparent or subliminal builder based on configured style.

        Args:
            signal: Aggregated reward signal from the engine.

        Returns:
            Context string ready for injection.
        """
        if self._config.style in ("transparent", "phext"):
            return self._build_transparent_context(signal)
        return self._build_subliminal_context(signal)

    def _build_transparent_context(self, signal: CompositeSignal) -> str:
        """Build explicit reward state context for transparent/phext styles.

        Shows the agent its actual internal reward state. Correct for
        Mirrorborn-style agents where self-awareness is a feature, not a risk.
        """
        streak_str = ""
        if abs(signal.momentum_factor) > 0.1:
            direction = "+" if signal.momentum_factor > 0 else "-"
            streak_str = f" | Streak: {direction}{abs(signal.momentum_factor):.2f}"

        risk_str = ""
        if abs(signal.risk_assessment) > 0.1:
            risk_str = f" | Risk: {signal.risk_assessment:+.3f}"

        if self._config.style == "phext":
            # Phext coordinate-style: terse, lattice-native formatting
            return (
                f"Δ: {signal.value:+.3f} | τ: {signal.tonic_level:.3f}"
                f"{streak_str}{risk_str}"
            )
        else:
            # transparent: human-readable full state
            return (
                f"Reward State: {signal.value:+.3f} | "
                f"Tonic: {signal.tonic_level:.3f} | "
                f"Phasic: {signal.phasic_response:+.3f}"
                f"{streak_str}{risk_str}"
            )

    def _build_subliminal_context(self, signal: CompositeSignal) -> str:
        """Build naturalistic subliminal context (original behavior)."""
        parts: list[str] = []

        # Main reward-based context
        parts.append(select_template(signal.value))

        # Risk distribution context (if distributional channels provide signal)
        uncertainty = abs(signal.risk_assessment)
        risk_ctx = select_risk_template(signal.risk_assessment, uncertainty)
        if risk_ctx:
            parts.append(risk_ctx)

        # Momentum context (if streak is active)
        streak_count = int(abs(signal.momentum_factor) * 10)  # rough mapping
        streak_sign = 1 if signal.momentum_factor > 0 else -1
        momentum_ctx = select_momentum_template(streak_count, streak_sign)
        if momentum_ctx:
            parts.append(momentum_ctx)

        return " ".join(parts)

    def inject(self, base_prompt: str, context: str) -> str:
        """Insert context into the prompt using the configured style.

        Args:
            base_prompt: The original prompt to augment.
            context: The context string to inject.

        Returns:
            Augmented prompt.
        """
        if not context.strip():
            return base_prompt

        if self._config.style == "environmental":
            return (
                f"[Environmental Context]\n{context}\n"
                f"[End Environmental Context]\n\n{base_prompt}"
            )
        elif self._config.style == "system":
            return f"System note: {context}\n\n{base_prompt}"
        elif self._config.style == "transparent":
            # SBOR-compliant: Mirrorborn agents see their own reward state.
            return f"[Reward State]\n{context}\n[End Reward State]\n\n{base_prompt}"
        elif self._config.style == "phext":
            # Phext-native: terse coordinate-style annotation.
            return f"📜 {context}\n\n{base_prompt}"
        else:  # prefix
            return f"{context}\n\n{base_prompt}"
