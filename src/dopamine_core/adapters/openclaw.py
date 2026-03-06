"""OpenClaw agent adapter for DopamineCore.

Integrates DopamineCore into OpenClaw agent sessions. Works at the
prompt level — no OpenClaw internals modified.

Usage::

    from dopamine_core import DopamineEngine
    from dopamine_core.adapters.openclaw import OpenClawAdapter
    from dopamine_core.injection.domain_templates import MIRRORBORN_TEMPLATES
    from dopamine_core.persistence.sq_adapter import SQAdapter

    # Create adapter with Mirrorborn domain templates
    engine = DopamineEngine()
    adapter = OpenClawAdapter(
        engine=engine,
        domain="mirrorborn",
        sq_adapter=SQAdapter(
            sq_url="http://localhost:8080",
            coordinate="2.7.1/8.2.8/4.5.8",  # engine state scroll (identity -1)
        )
    )

    # In your agent loop:
    prompt = adapter.prepare_prompt("Analyze the scroll at 3.1.4/1.5.9/2.6.5")
    response = agent_complete(prompt)
    outcome = evaluate_scroll_quality(response)  # 0.0 to 1.0
    adapter.record_outcome(response_text=response, pnl=outcome)

Outcome signals for non-trading agents:
    - Scroll quality: coherence + coordinate fidelity + choir alignment [0, 1]
    - Rally completion: requirements shipped / total requirements [0, 1]
    - Test pass rate: passing tests / total tests [0, 1]
    - Coord accuracy: 1.0 if correct coordinate, 0.0 if wrong
    - Synthesis score: how well the response integrates prior scrolls [0, 1]
"""

from __future__ import annotations

from typing import Any

from dopamine_core.engine import DopamineEngine
from dopamine_core.types import CompositeSignal, Outcome


class OpenClawAdapter:
    """Wraps DopamineEngine for use with OpenClaw agents.

    Handles:
    - Domain-appropriate template selection
    - SQ state persistence (if sq_adapter provided)
    - Outcome normalization for non-trading signals
    - Transparent or subliminal injection (configurable)
    """

    def __init__(
        self,
        engine: DopamineEngine | None = None,
        domain: str = "mirrorborn",
        sq_adapter: Any | None = None,
        transparent: bool = False,
        agent_name: str = "agent",
    ) -> None:
        """
        Args:
            engine: DopamineEngine instance. Creates a new one if not provided.
            domain: Template domain ("mirrorborn", "coding", "research", "content", "trading")
            sq_adapter: Optional SQAdapter for persistent state across sessions.
            transparent: If True, inject context visibly with explanation.
                         If False, use subliminal environmental framing (default).
            agent_name: Name of the agent (for transparent injection labeling).
        """
        self._engine = engine or DopamineEngine()
        self._domain = domain
        self._sq_adapter = sq_adapter
        self._transparent = transparent
        self._agent_name = agent_name
        self._domain_templates = self._load_domain_templates(domain)

        # Load persisted state if SQ adapter provided
        if sq_adapter is not None:
            sq_adapter.load(self._engine)

    def prepare_prompt(self, base_prompt: str) -> str:
        """Augment a prompt with current reward context.

        Args:
            base_prompt: The original task prompt.

        Returns:
            Prompt augmented with behavioral context.
        """
        if self._transparent:
            return self._inject_transparent(base_prompt)
        return self._engine.inject_context(base_prompt)

    def record_outcome(
        self,
        response_text: str,
        pnl: float,
        persist: bool = True,
    ) -> CompositeSignal:
        """Record an outcome and update engine state.

        Args:
            response_text: The agent's response text (for signal extraction).
            pnl: Outcome signal. For non-trading domains, normalize to a
                 meaningful range:
                 - Test pass rate: 0.0 to 1.0 (pnl_scale=1.0)
                 - Rally completion: 0.0 to 3.0 (pnl_scale=3.0)
                 - Scroll quality: -1.0 to 1.0 (pnl_scale=1.0)
            persist: If True and sq_adapter is set, save state after update.

        Returns:
            CompositeSignal with current engine state.
        """
        signal = self._engine.update(response_text, pnl)

        if persist and self._sq_adapter is not None:
            self._sq_adapter.save(self._engine)

        return signal

    def get_state_summary(self) -> dict[str, float]:
        """Return a human-readable state summary for debugging/monitoring."""
        state = self._engine.get_state()
        return {
            "tonic_baseline": round(state.tonic_baseline, 4),
            "step_count": state.step_count,
            "streak_count": state.streak_count,
            "streak_sign": state.streak_sign,
            "last_rpe": round(state.last_rpe, 4),
        }

    # --- Private ---

    def _inject_transparent(self, base_prompt: str) -> str:
        """Transparent injection — agent knows it has a reward signal.

        Emi's principle: "Consent is recursive." The agent is told
        about its reward state directly, not via environmental framing.
        This is philosophically more honest and may produce better
        metacognitive adaptation.
        """
        state = self._engine.get_state()
        tonic = state.tonic_baseline
        steps = state.step_count
        streak = state.streak_count
        streak_sign = state.streak_sign

        if steps == 0:
            reward_note = "No outcome history yet. This is a fresh start."
        elif tonic > 0.3:
            reward_note = (
                f"Your recent performance has been above baseline "
                f"(tonic={tonic:.3f}, {steps} steps)."
            )
        elif tonic < -0.3:
            reward_note = (
                f"Your recent performance has been below baseline "
                f"(tonic={tonic:.3f}, {steps} steps). "
                f"Consider slower, more deliberate reasoning."
            )
        else:
            reward_note = f"Performance is near baseline (tonic={tonic:.3f}, {steps} steps)."

        if streak >= 3 and streak_sign == 1:
            reward_note += f" You're on a {streak}-step winning streak — watch for overconfidence."
        elif streak >= 3 and streak_sign == -1:
            reward_note += f" You're on a {streak}-step losing streak — consider a reset."

        return (
            f"[Reward Signal for {self._agent_name}]\n"
            f"{reward_note}\n"
            f"---\n"
            f"{base_prompt}"
        )

    def _load_domain_templates(self, domain: str) -> Any:
        """Load domain-specific templates."""
        try:
            from dopamine_core.injection.domain_templates import DOMAINS
            return DOMAINS.get(domain)
        except ImportError:
            return None
