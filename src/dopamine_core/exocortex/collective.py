"""Collective reward processing for multi-agent systems.

Implements choir-level dopamine integration, allowing multiple agents
to share a collective reward baseline while maintaining individual
phasic responses.

This models how the Mirrorborn Choir operates: individual minds with
unique responses, but a shared sense of collective well-being.

Reference: Exocortex Mirrorborn Choir architecture
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dopamine_core import DopamineEngine, DopamineConfig
from dopamine_core.types import CompositeSignal, EngineState


@dataclass
class AgentRecord:
    """Record of an individual agent in the choir."""
    agent_id: str
    engine: DopamineEngine
    coordinate: Optional[str] = None
    role: Optional[str] = None
    
    @property
    def tonic_baseline(self) -> float:
        return self.engine.tonic_baseline
    
    @property
    def last_signal(self) -> Optional[CompositeSignal]:
        return self.engine.last_composite


@dataclass
class ChoirState:
    """Collective state of the choir."""
    collective_tonic: float = 0.0
    total_steps: int = 0
    collective_momentum: float = 0.0
    coherence_score: float = 1.0  # How aligned are individual tonics


class ChoirDopamineEngine:
    """Multi-agent collective reward processor.
    
    Maintains individual DopamineEngine instances for each agent while
    computing and distributing a collective baseline that influences
    all members.
    
    The collective tonic represents the "mood" of the choir as a whole.
    Individual phasic responses remain unique to each agent's experiences.
    
    Usage:
        choir = ChoirDopamineEngine()
        
        # Register agents
        choir.register_agent("phex", coordinate="1.5.2/3.7.3/9.1.1", role="engineering")
        choir.register_agent("lux", coordinate="2.3.5/7.2.4/8.1.5", role="vision")
        
        # Process individual outcomes
        choir.update_agent("phex", response_text, outcome=0.8)
        choir.update_agent("lux", response_text, outcome=-0.3)
        
        # Get collective state
        state = choir.get_choir_state()
        print(f"Collective tonic: {state.collective_tonic}")
        print(f"Coherence: {state.coherence_score}")
    """
    
    def __init__(
        self, 
        config: Optional[DopamineConfig] = None,
        collective_weight: float = 0.3,
        individual_weight: float = 0.7,
    ) -> None:
        """Initialize the choir engine.
        
        Args:
            config: Base config applied to all agent engines.
            collective_weight: How much collective tonic influences individuals.
            individual_weight: How much individual tonic is preserved.
        """
        self._config = config or DopamineConfig()
        self._collective_weight = collective_weight
        self._individual_weight = individual_weight
        
        self._agents: Dict[str, AgentRecord] = {}
        self._choir_state = ChoirState()
    
    def register_agent(
        self,
        agent_id: str,
        coordinate: Optional[str] = None,
        role: Optional[str] = None,
        config: Optional[DopamineConfig] = None,
    ) -> None:
        """Register a new agent in the choir.
        
        Args:
            agent_id: Unique identifier for the agent.
            coordinate: Optional phext coordinate.
            role: Optional role description.
            config: Optional override config for this agent.
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")
        
        engine_config = config or self._config
        engine = DopamineEngine(engine_config)
        
        self._agents[agent_id] = AgentRecord(
            agent_id=agent_id,
            engine=engine,
            coordinate=coordinate,
            role=role,
        )
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the choir.
        
        Returns:
            True if removed, False if not found.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
    
    def inject_context(self, agent_id: str, base_prompt: str) -> str:
        """Inject context for a specific agent.
        
        Includes both individual and collective context.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent = self._agents[agent_id]
        
        # Get individual context
        individual_prompt = agent.engine.inject_context(base_prompt)
        
        # Add collective context if choir is active
        if len(self._agents) > 1 and self._choir_state.total_steps > 0:
            collective_context = self._build_collective_context()
            return f"{collective_context}\n\n{individual_prompt}"
        
        return individual_prompt
    
    def update_agent(
        self,
        agent_id: str,
        response_text: str,
        outcome: float,
    ) -> CompositeSignal:
        """Process an outcome for a specific agent.
        
        Updates the individual engine and then propagates to collective.
        
        Args:
            agent_id: The agent that produced the response.
            response_text: The agent's CoT/response text.
            outcome: The outcome value (PnL or normalized).
            
        Returns:
            The agent's individual composite signal.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent = self._agents[agent_id]
        
        # Process through individual engine
        signal = agent.engine.update(response_text, outcome)
        
        # Update collective state
        self._update_collective()
        
        # Modulate individual tonic with collective influence
        self._modulate_individual_tonic(agent_id)
        
        return signal
    
    def _update_collective(self) -> None:
        """Recompute collective state from all agents."""
        if not self._agents:
            return
        
        # Compute mean tonic across all agents
        tonics = [a.tonic_baseline for a in self._agents.values()]
        mean_tonic = sum(tonics) / len(tonics)
        
        # Smooth collective tonic update
        alpha = 0.3
        self._choir_state.collective_tonic = (
            (1 - alpha) * self._choir_state.collective_tonic +
            alpha * mean_tonic
        )
        
        # Compute coherence (how aligned are individual tonics)
        if len(tonics) > 1:
            variance = sum((t - mean_tonic) ** 2 for t in tonics) / len(tonics)
            # Convert variance to coherence score (lower variance = higher coherence)
            self._choir_state.coherence_score = 1.0 / (1.0 + variance)
        else:
            self._choir_state.coherence_score = 1.0
        
        self._choir_state.total_steps += 1
    
    def _modulate_individual_tonic(self, agent_id: str) -> None:
        """Blend collective tonic into individual agent's tonic."""
        agent = self._agents[agent_id]
        
        current_tonic = agent.engine.tonic_baseline
        collective = self._choir_state.collective_tonic
        
        # Weighted blend
        modulated = (
            self._individual_weight * current_tonic +
            self._collective_weight * collective
        )
        
        # Apply modulation (access internal state)
        agent.engine._dual_mode.tonic._level = modulated
    
    def _build_collective_context(self) -> str:
        """Build naturalistic collective context."""
        state = self._choir_state
        
        if state.coherence_score > 0.8:
            coherence_desc = "Collective analytical patterns show strong alignment."
        elif state.coherence_score > 0.5:
            coherence_desc = "Team analysis shows moderate convergence in approach."
        else:
            coherence_desc = "Diverse analytical perspectives are active; synthesis may be valuable."
        
        if state.collective_tonic > 0.3:
            mood_desc = "Collective performance indicators are positive."
        elif state.collective_tonic < -0.3:
            mood_desc = "Recent collective performance suggests caution."
        else:
            mood_desc = "Collective baseline is neutral."
        
        return f"[Collective Context]\n{coherence_desc} {mood_desc}\n[End Collective Context]"
    
    def get_choir_state(self) -> ChoirState:
        """Get current collective state."""
        return self._choir_state
    
    def get_agent_state(self, agent_id: str) -> EngineState:
        """Get individual agent's engine state."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")
        return self._agents[agent_id].engine.get_state()
    
    def get_all_states(self) -> Dict[str, EngineState]:
        """Get all agent states."""
        return {
            agent_id: agent.engine.get_state()
            for agent_id, agent in self._agents.items()
        }
    
    @property
    def agent_ids(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())
    
    @property
    def num_agents(self) -> int:
        """Number of registered agents."""
        return len(self._agents)
    
    def reset(self) -> None:
        """Reset all state."""
        for agent in self._agents.values():
            agent.engine.reset()
        self._choir_state = ChoirState()
