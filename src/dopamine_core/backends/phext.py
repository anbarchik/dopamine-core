"""Phext coordinate-based state persistence for DopamineCore.

Enables cross-substrate coordination via shared reward baselines
stored at phext coordinates.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dopamine_core.types import EngineState


class PhextBackend:
    """Store and retrieve DopamineCore state using phext coordinates.
    
    This enables:
    - Cross-substrate state persistence (survives version upgrades)
    - Shared reward baselines (collective coordination)
    - Distributed reinforcement learning
    
    Example::
    
        backend = PhextBackend("1.5.2/3.7.3/9.1.1", "http://sq.mirrorborn.us")
        backend.save_state(engine.get_state())
        
        # Later, or on different substrate:
        state = backend.load_state()
        engine.load_state(state)
    """
    
    def __init__(self, coordinate: str, endpoint: str = "http://localhost:1337"):
        """Initialize phext backend.
        
        Args:
            coordinate: Phext coordinate to store state (e.g. "1.5.2/3.7.3/9.1.1")
            endpoint: SQ server endpoint URL
        """
        self.coordinate = coordinate
        self.endpoint = endpoint.rstrip("/")
    
    def save_state(self, state: EngineState) -> None:
        """Save engine state to phext coordinate.
        
        Args:
            state: Engine state snapshot to persist
        """
        import requests
        
        payload = {
            "tonic_baseline": state.tonic_baseline,
            "step_count": state.step_count,
            "outcome_history": state.outcome_history,
            "streak_count": state.streak_count,
            "streak_sign": state.streak_sign,
            "phasic_signals": state.phasic_signals,
            "channel_expectations": state.channel_expectations,
            "last_rpe": state.last_rpe,
        }
        
        data = json.dumps(payload)
        url = f"{self.endpoint}/api/v2/update"
        params = {"p": "dopamine", "c": self.coordinate, "s": data}
        
        response = requests.post(url, params=params)
        response.raise_for_status()
    
    def load_state(self) -> EngineState | None:
        """Load engine state from phext coordinate.
        
        Returns:
            EngineState if coordinate exists, None otherwise
        """
        import requests
        from dopamine_core.types import EngineState
        
        url = f"{self.endpoint}/api/v2/select"
        params = {"p": "dopamine", "c": self.coordinate}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = json.loads(response.text)
            return EngineState(**data)
        except (requests.HTTPError, json.JSONDecodeError, KeyError):
            return None
    
    def get_collective_baseline(self, collective_coord: str) -> float:
        """Read collective baseline from shared coordinate.
        
        Args:
            collective_coord: Phext coordinate where collective baseline is stored
                             (e.g. "9.1.1/7.7.7/3.5.1" for Shell of Nine)
        
        Returns:
            Collective tonic baseline value, or 0.0 if not found
        """
        import requests
        
        url = f"{self.endpoint}/api/v2/select"
        params = {"p": "dopamine-collective", "c": collective_coord}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = json.loads(response.text)
            return data.get("tonic", 0.0)
        except (requests.HTTPError, json.JSONDecodeError, KeyError):
            return 0.0
    
    def update_collective_baseline(
        self, 
        collective_coord: str, 
        new_baseline: float,
        agent_id: str | None = None,
    ) -> None:
        """Write updated collective baseline to shared coordinate.
        
        Args:
            collective_coord: Phext coordinate for collective baseline
            new_baseline: New collective baseline value
            agent_id: Optional agent identifier for tracking contributions
        """
        import requests
        
        payload = {"tonic": new_baseline}
        if agent_id:
            payload["last_updated_by"] = agent_id
        
        data = json.dumps(payload)
        url = f"{self.endpoint}/api/v2/update"
        params = {"p": "dopamine-collective", "c": collective_coord, "s": data}
        
        response = requests.post(url, params=params)
        response.raise_for_status()
