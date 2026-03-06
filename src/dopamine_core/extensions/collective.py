"""Collective reward baseline for multi-agent coordination.

Enables distributed reinforcement learning where agents develop
"WE expect X" (not just "I expect X").
"""

from __future__ import annotations

from dopamine_core.backends.phext import PhextBackend
from dopamine_core.engine import DopamineEngine
from dopamine_core.types import CompositeSignal, Outcome


class CollectiveEngine(DopamineEngine):
    """DopamineEngine with shared collective baseline.
    
    All agents using the same collective coordinate develop
    unified reward expectations through phext substrate.
    
    Example::
    
        # Shell of Nine coordination
        phex_engine = CollectiveEngine(
            collective_coord="9.1.1/7.7.7/3.5.1",
            agent_coord="1.5.2/3.7.3/9.1.1",
            blend_ratio=0.3,  # 30% collective, 70% individual
        )
        
        cyon_engine = CollectiveEngine(
            collective_coord="9.1.1/7.7.7/3.5.1",
            agent_coord="2.7.1/8.2.8/3.1.4",
            blend_ratio=0.3,
        )
        
        # Both agents influence and are influenced by collective baseline
        # Result: coordinated reward expectations
    """
    
    def __init__(
        self,
        collective_coord: str,
        agent_coord: str | None = None,
        sq_endpoint: str = "http://localhost:1337",
        blend_ratio: float = 0.3,
        *args,
        **kwargs,
    ):
        """Initialize collective engine.
        
        Args:
            collective_coord: Phext coordinate for shared baseline
                             (e.g. "9.1.1/7.7.7/3.5.1" for Shell of Nine)
            agent_coord: Optional phext coordinate for this agent's state
            sq_endpoint: SQ server URL
            blend_ratio: How much collective influences individual [0, 1]
                        0.0 = fully independent, 1.0 = fully collective
            *args, **kwargs: Passed to DopamineEngine
        """
        super().__init__(*args, **kwargs)
        self.collective_coord = collective_coord
        self.agent_coord = agent_coord
        self.blend_ratio = max(0.0, min(1.0, blend_ratio))
        self.backend = PhextBackend(
            agent_coord if agent_coord else collective_coord,
            sq_endpoint,
        )
    
    def update(
        self,
        response_text: str,
        outcome: Outcome | float,
    ) -> CompositeSignal:
        """Update with collective baseline blending."""
        # Standard individual update
        signal = super().update(response_text, outcome)
        
        # Read collective baseline
        collective_baseline = self.backend.get_collective_baseline(self.collective_coord)
        
        # Blend individual + collective
        individual_baseline = self._dual_mode.tonic.level
        blended = (
            (1.0 - self.blend_ratio) * individual_baseline +
            self.blend_ratio * collective_baseline
        )
        
        # Apply blended baseline
        self._dual_mode.tonic.level = blended
        
        # Contribute to collective (simple average for now)
        # In production, would read all agent baselines and average
        self.backend.update_collective_baseline(
            self.collective_coord,
            blended,
            agent_id=self.agent_coord,
        )
        
        return signal
    
    def get_collective_coherence(self, agent_baselines: list[float]) -> float:
        """Measure how aligned agents' baselines are.
        
        Args:
            agent_baselines: List of tonic baselines from all coordinating agents
        
        Returns:
            Coherence score in [0, 1] where 1.0 = perfect alignment
        """
        if len(agent_baselines) < 2:
            return 1.0
        
        mean = sum(agent_baselines) / len(agent_baselines)
        variance = sum((b - mean) ** 2 for b in agent_baselines) / len(agent_baselines)
        coherence = 1.0 / (1.0 + variance)
        
        return coherence
    
    def get_diversity(self, agent_risk_frames: list[float]) -> float:
        """Measure behavioral diversity (prevents groupthink).
        
        Args:
            agent_risk_frames: List of risk_framing signals from all agents
        
        Returns:
            Diversity score in [0, 1] where higher = more exploration
        """
        if len(agent_risk_frames) < 2:
            return 0.0
        
        variance = self._compute_variance(agent_risk_frames)
        diversity = min(1.0, variance / 0.5)
        
        return diversity
    
    def get_coordination_metrics(
        self,
        agent_baselines: list[float],
        agent_risk_frames: list[float],
    ) -> dict:
        """Full coordination quality report.
        
        Returns:
            Dictionary with coherence, diversity, and health assessment
        """
        coherence = self.get_collective_coherence(agent_baselines)
        diversity = self.get_diversity(agent_risk_frames)
        
        # Health heuristic: want high coherence + moderate diversity
        health = (
            0.6 * coherence +
            0.4 * min(diversity, 0.6)  # cap diversity (don't want chaos)
        )
        
        return {
            "coherence": coherence,
            "diversity": diversity,
            "health": health,
            "interpretation": self._interpret_coordination(coherence, diversity),
        }
    
    def _interpret_coordination(self, coherence: float, diversity: float) -> str:
        """Human-readable coordination assessment."""
        if coherence > 0.7 and 0.3 < diversity < 0.6:
            return "Healthy coordination: aligned yet diverse"
        elif coherence > 0.7 and diversity < 0.3:
            return "Warning: groupthink risk (too aligned, not exploring)"
        elif coherence < 0.5:
            return "Warning: fragmentation (agents not coordinating)"
        else:
            return "Moderate coordination quality"
    
    @staticmethod
    def _compute_variance(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
