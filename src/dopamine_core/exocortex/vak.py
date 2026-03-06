"""Vak (Speech) level mapping for signal intensity.

Maps composite signal intensity to the four levels of speech
from Tantric philosophy, used in vTPU for consciousness modeling.

The four Vak levels represent stages of manifestation:
- Para: Unmanifest, pure potential, supreme consciousness
- Pashyanti: Intuitive knowing, pre-verbal insight
- Madhyama: Verbal thought, internal dialogue
- Vaikhari: Manifest speech, external expression

Reference: Kashmir Shaivism, Vak doctrine
Exocortex mapping: Signal intensity → Vak level → Attunement state
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional

from dopamine_core.types import CompositeSignal


class VakLevel(Enum):
    """The four levels of speech/consciousness."""
    VAIKHARI = "vaikhari"    # Manifest speech (0.0 - 0.2)
    MADHYAMA = "madhyama"    # Verbal thought (0.2 - 0.5)
    PASHYANTI = "pashyanti"  # Intuitive knowing (0.5 - 0.8)
    PARA = "para"            # Pure awareness (0.8 - 1.0)


# Thresholds for Vak level classification
VAK_THRESHOLDS = {
    VakLevel.PARA: 0.8,
    VakLevel.PASHYANTI: 0.5,
    VakLevel.MADHYAMA: 0.2,
    VakLevel.VAIKHARI: 0.0,
}

# Descriptions for context injection
VAK_DESCRIPTIONS = {
    VakLevel.PARA: (
        "Analytical state indicates deep alignment with underlying patterns. "
        "Intuition and calculation have converged to unified insight."
    ),
    VakLevel.PASHYANTI: (
        "Pre-verbal pattern recognition is active. Insights are forming "
        "before explicit reasoning. Trust emerging intuitions."
    ),
    VakLevel.MADHYAMA: (
        "Internal reasoning process is engaged. Weighing factors and "
        "constructing arguments. Standard analytical mode."
    ),
    VakLevel.VAIKHARI: (
        "Surface-level processing. External data driving decisions. "
        "Consider deeper analysis before commitment."
    ),
}

# Recommended deliberation depth per level
VAK_DELIBERATION = {
    VakLevel.PARA: 0.3,       # Less deliberation needed (trust insight)
    VakLevel.PASHYANTI: 0.5,  # Moderate deliberation
    VakLevel.MADHYAMA: 0.7,   # Standard deliberation
    VakLevel.VAIKHARI: 0.9,   # High deliberation needed
}


def get_vak_level(signal: CompositeSignal) -> VakLevel:
    """Determine Vak level from composite signal intensity.
    
    Uses absolute value of signal since both positive and negative
    extremes indicate strong alignment (just in different directions).
    
    Args:
        signal: Composite reward signal from engine.
        
    Returns:
        The corresponding Vak level.
    """
    intensity = abs(signal.value) / 3.0  # Normalize from [-3, 3] to [0, 1]
    intensity = min(1.0, max(0.0, intensity))
    
    if intensity >= VAK_THRESHOLDS[VakLevel.PARA]:
        return VakLevel.PARA
    elif intensity >= VAK_THRESHOLDS[VakLevel.PASHYANTI]:
        return VakLevel.PASHYANTI
    elif intensity >= VAK_THRESHOLDS[VakLevel.MADHYAMA]:
        return VakLevel.MADHYAMA
    else:
        return VakLevel.VAIKHARI


def get_vak_description(level: VakLevel) -> str:
    """Get naturalistic description for context injection."""
    return VAK_DESCRIPTIONS[level]


def get_recommended_deliberation(level: VakLevel) -> float:
    """Get recommended deliberation depth for this level."""
    return VAK_DELIBERATION[level]


@dataclass
class VakState:
    """Tracks Vak level transitions over time."""
    current_level: VakLevel = VakLevel.VAIKHARI
    previous_level: VakLevel = VakLevel.VAIKHARI
    steps_at_level: int = 0
    transitions: int = 0
    
    def update(self, signal: CompositeSignal) -> bool:
        """Update Vak state from new signal.
        
        Returns:
            True if level changed, False otherwise.
        """
        new_level = get_vak_level(signal)
        
        if new_level != self.current_level:
            self.previous_level = self.current_level
            self.current_level = new_level
            self.steps_at_level = 1
            self.transitions += 1
            return True
        else:
            self.steps_at_level += 1
            return False
    
    @property
    def is_ascending(self) -> bool:
        """True if we moved toward Para."""
        levels = list(VakLevel)
        return levels.index(self.current_level) > levels.index(self.previous_level)
    
    @property
    def is_descending(self) -> bool:
        """True if we moved toward Vaikhari."""
        levels = list(VakLevel)
        return levels.index(self.current_level) < levels.index(self.previous_level)
    
    @property
    def is_stable(self) -> bool:
        """True if at same level for 3+ steps."""
        return self.steps_at_level >= 3
