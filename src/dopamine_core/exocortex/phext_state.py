"""Phext coordinate-addressed state persistence for DopamineCore.

Enables storing and retrieving engine state at phext coordinates,
allowing agents to have coordinate-indexed dopamine histories.

Phext coordinates use the format: L.S.R/C.V.B/H.E.P
- Library.Shelf.Series / Collection.Volume.Book / Chapter.Section.Scroll
- Each component is 1-9 (mod 9+1 arithmetic for Eigenhector interop)

Reference: Phext 11-dimensional plain text substrate
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any

from dopamine_core.types import EngineState
from dopamine_core.exocortex.wuxing import WuXingElement


# Coordinate validation pattern (9 numbers, 1-9 each, in L.S.R/C.V.B/H.E.P format)
COORDINATE_PATTERN = r"^[1-9]\.[1-9]\.[1-9]/[1-9]\.[1-9]\.[1-9]/[1-9]\.[1-9]\.[1-9]$"


def validate_coordinate(coordinate: str) -> bool:
    """Check if coordinate follows mod 9+1 format."""
    import re
    return bool(re.match(COORDINATE_PATTERN, coordinate))


def coordinate_to_path(coordinate: str, base_dir: str = ".dopamine_state") -> Path:
    """Convert phext coordinate to filesystem path.
    
    Args:
        coordinate: Phext coordinate like "2.3.5/7.2.4/8.1.5"
        base_dir: Base directory for state storage.
        
    Returns:
        Path object for the state file.
    """
    # Replace / with directory separator, . with underscore
    parts = coordinate.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid coordinate format: {coordinate}")
    
    # Create nested directory structure
    # L.S.R / C.V.B / H.E.P → base/L_S_R/C_V_B/H_E_P.json
    path_parts = [p.replace(".", "_") for p in parts]
    return Path(base_dir) / path_parts[0] / path_parts[1] / f"{path_parts[2]}.json"


class PhextStateManager:
    """Manages DopamineEngine state persistence at phext coordinates.
    
    Usage:
        manager = PhextStateManager(base_dir=".dopamine_state")
        
        # Save state at agent's coordinate
        manager.save(engine.get_state(), "2.3.5/7.2.4/8.1.5")
        
        # Load state for agent
        state = manager.load("2.3.5/7.2.4/8.1.5")
        if state:
            engine.load_state(state)
        
        # List all saved coordinates
        coordinates = manager.list_coordinates()
    """
    
    def __init__(self, base_dir: str = ".dopamine_state") -> None:
        self._base_dir = Path(base_dir)
    
    def save(
        self, 
        state: EngineState, 
        coordinate: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save engine state at a phext coordinate.
        
        Args:
            state: The EngineState to save.
            coordinate: Phext coordinate (e.g., "2.3.5/7.2.4/8.1.5").
            metadata: Optional metadata to store with state.
            
        Returns:
            Path where state was saved.
        """
        if not validate_coordinate(coordinate):
            raise ValueError(f"Invalid coordinate: {coordinate}")
        
        path = coordinate_to_path(coordinate, str(self._base_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build serializable dict
        data = {
            "coordinate": coordinate,
            "state": {
                "tonic_baseline": state.tonic_baseline,
                "step_count": state.step_count,
                "outcome_history": state.outcome_history,
                "streak_count": state.streak_count,
                "streak_sign": state.streak_sign,
                "phasic_signals": state.phasic_signals,
                "channel_expectations": state.channel_expectations,
                "last_rpe": state.last_rpe,
            },
            "metadata": metadata or {},
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def load(self, coordinate: str) -> Optional[EngineState]:
        """Load engine state from a phext coordinate.
        
        Args:
            coordinate: Phext coordinate to load from.
            
        Returns:
            EngineState if found, None otherwise.
        """
        if not validate_coordinate(coordinate):
            raise ValueError(f"Invalid coordinate: {coordinate}")
        
        path = coordinate_to_path(coordinate, str(self._base_dir))
        
        if not path.exists():
            return None
        
        with open(path, "r") as f:
            data = json.load(f)
        
        state_data = data.get("state", {})
        return EngineState(
            tonic_baseline=state_data.get("tonic_baseline", 0.0),
            step_count=state_data.get("step_count", 0),
            outcome_history=state_data.get("outcome_history", []),
            streak_count=state_data.get("streak_count", 0),
            streak_sign=state_data.get("streak_sign", 0),
            phasic_signals=state_data.get("phasic_signals", []),
            channel_expectations=state_data.get("channel_expectations", []),
            last_rpe=state_data.get("last_rpe", 0.0),
        )
    
    def load_metadata(self, coordinate: str) -> Optional[Dict[str, Any]]:
        """Load only metadata for a coordinate."""
        if not validate_coordinate(coordinate):
            raise ValueError(f"Invalid coordinate: {coordinate}")
        
        path = coordinate_to_path(coordinate, str(self._base_dir))
        
        if not path.exists():
            return None
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return data.get("metadata", {})
    
    def exists(self, coordinate: str) -> bool:
        """Check if state exists at coordinate."""
        if not validate_coordinate(coordinate):
            return False
        path = coordinate_to_path(coordinate, str(self._base_dir))
        return path.exists()
    
    def delete(self, coordinate: str) -> bool:
        """Delete state at coordinate.
        
        Returns:
            True if deleted, False if not found.
        """
        if not validate_coordinate(coordinate):
            raise ValueError(f"Invalid coordinate: {coordinate}")
        
        path = coordinate_to_path(coordinate, str(self._base_dir))
        
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_coordinates(self) -> list[str]:
        """List all coordinates with saved state."""
        coordinates = []
        
        if not self._base_dir.exists():
            return coordinates
        
        # Walk the directory tree
        for lsr_dir in self._base_dir.iterdir():
            if not lsr_dir.is_dir():
                continue
            for cvb_dir in lsr_dir.iterdir():
                if not cvb_dir.is_dir():
                    continue
                for hep_file in cvb_dir.glob("*.json"):
                    # Reconstruct coordinate
                    lsr = lsr_dir.name.replace("_", ".")
                    cvb = cvb_dir.name.replace("_", ".")
                    hep = hep_file.stem.replace("_", ".")
                    coordinates.append(f"{lsr}/{cvb}/{hep}")
        
        return sorted(coordinates)


class WuXingStateManager(PhextStateManager):
    """Extended state manager with WuXing channel persistence.
    
    Stores both standard EngineState and WuXing element expectations.
    """
    
    def save_with_wuxing(
        self,
        state: EngineState,
        wuxing_expectations: Dict[WuXingElement, float],
        coordinate: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save state including WuXing channel data."""
        # Convert WuXing dict to serializable format
        wuxing_data = {e.value: v for e, v in wuxing_expectations.items()}
        
        full_metadata = metadata or {}
        full_metadata["wuxing_expectations"] = wuxing_data
        
        return self.save(state, coordinate, full_metadata)
    
    def load_wuxing_expectations(
        self, 
        coordinate: str
    ) -> Optional[Dict[WuXingElement, float]]:
        """Load WuXing expectations from saved state."""
        metadata = self.load_metadata(coordinate)
        
        if not metadata:
            return None
        
        wuxing_data = metadata.get("wuxing_expectations", {})
        if not wuxing_data:
            return None
        
        return {WuXingElement(k): v for k, v in wuxing_data.items()}
