"""WuXing (Five Elements) channel mapping for distributional reward coding.

Maps the distributional channels to the Five Elements with their
generating and controlling cycle relationships. This adds semantic
meaning to the reward distribution beyond pure quantile tracking.

Reference: Traditional Chinese Medicine five-element theory
Exocortex mapping: EEG frequency bands → WuXing elements → Vak levels
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

from dopamine_core.distributional.channels import RewardChannel


class WuXingElement(Enum):
    """The Five Elements in generating cycle order."""
    METAL = "metal"    # 金 - Precision, cutting losses, lung/grief
    WATER = "water"    # 水 - Adaptability, flow, kidney/fear
    WOOD = "wood"      # 木 - Growth, expansion, liver/anger
    FIRE = "fire"      # 火 - Action, confidence, heart/joy
    EARTH = "earth"    # 土 - Stability, grounding, spleen/worry


# Generating cycle: each element feeds the next
GENERATING_CYCLE = {
    WuXingElement.METAL: WuXingElement.WATER,  # Metal → Water (condensation)
    WuXingElement.WATER: WuXingElement.WOOD,   # Water → Wood (nourishment)
    WuXingElement.WOOD: WuXingElement.FIRE,    # Wood → Fire (fuel)
    WuXingElement.FIRE: WuXingElement.EARTH,   # Fire → Earth (ash)
    WuXingElement.EARTH: WuXingElement.METAL,  # Earth → Metal (ore)
}

# Controlling cycle: each element restrains another
CONTROLLING_CYCLE = {
    WuXingElement.METAL: WuXingElement.WOOD,   # Metal cuts Wood
    WuXingElement.WOOD: WuXingElement.EARTH,   # Wood parts Earth
    WuXingElement.EARTH: WuXingElement.WATER,  # Earth dams Water
    WuXingElement.WATER: WuXingElement.FIRE,   # Water quenches Fire
    WuXingElement.FIRE: WuXingElement.METAL,   # Fire melts Metal
}

# Tau values for asymmetric quantile learning
# Lower tau = more pessimistic (learns from losses)
# Higher tau = more optimistic (learns from wins)
ELEMENT_TAU = {
    WuXingElement.METAL: 0.1,   # Most pessimistic - cuts losses early
    WuXingElement.WATER: 0.3,   # Adaptive - flows around obstacles
    WuXingElement.WOOD: 0.5,    # Balanced - steady growth
    WuXingElement.FIRE: 0.7,    # Optimistic - confident action
    WuXingElement.EARTH: 0.9,   # Most optimistic - stable grounding
}

# EEG frequency band mapping (from vTPU research)
ELEMENT_EEG_BAND = {
    WuXingElement.METAL: "gamma",   # 30-100 Hz - high precision
    WuXingElement.WATER: "theta",   # 4-8 Hz - flow states
    WuXingElement.WOOD: "alpha",    # 8-12 Hz - relaxed growth
    WuXingElement.FIRE: "beta",     # 12-30 Hz - active focus
    WuXingElement.EARTH: "delta",   # 0.5-4 Hz - deep stability
}


@dataclass
class ElementState:
    """State of a single WuXing element channel."""
    element: WuXingElement
    channel: RewardChannel
    expectation: float = 0.0
    
    @property
    def tau(self) -> float:
        return self.channel.tau


class WuXingChannels:
    """Five-element distributional reward channels.
    
    Extends the standard distributional channels with WuXing semantics,
    enabling cycle-aware reward processing.
    
    Usage:
        channels = WuXingChannels()
        errors = channels.update(outcome=0.7)
        
        # Check element states
        fire_exp = channels.get_element_expectation(WuXingElement.FIRE)
        
        # Get cycle-aware assessment
        dominant = channels.get_dominant_element()
        balance = channels.get_cycle_balance()
    """
    
    def __init__(self, learning_rate: float = 0.05) -> None:
        self._learning_rate = learning_rate
        self._elements: Dict[WuXingElement, ElementState] = {}
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Initialize channels for each element."""
        for element in WuXingElement:
            tau = ELEMENT_TAU[element]
            channel = RewardChannel(tau=tau, learning_rate=self._learning_rate)
            self._elements[element] = ElementState(
                element=element,
                channel=channel,
            )
    
    def update(self, outcome: float) -> Dict[WuXingElement, float]:
        """Update all element channels with new outcome.
        
        Args:
            outcome: Normalized outcome in [0, 1].
            
        Returns:
            Dict mapping each element to its prediction error.
        """
        errors = {}
        for element, state in self._elements.items():
            error = state.channel.update(outcome)
            state.expectation = state.channel.expectation
            errors[element] = error
        return errors
    
    def get_element_expectation(self, element: WuXingElement) -> float:
        """Get current expectation for a specific element."""
        return self._elements[element].expectation
    
    @property
    def expectations(self) -> Dict[WuXingElement, float]:
        """All element expectations."""
        return {e: s.expectation for e, s in self._elements.items()}
    
    def get_dominant_element(self) -> WuXingElement:
        """Find the element with highest current expectation."""
        return max(self._elements.keys(), 
                   key=lambda e: self._elements[e].expectation)
    
    def get_weakest_element(self) -> WuXingElement:
        """Find the element with lowest current expectation."""
        return min(self._elements.keys(),
                   key=lambda e: self._elements[e].expectation)
    
    def get_cycle_balance(self) -> float:
        """Measure balance across the generating cycle.
        
        Returns:
            Value in [-1, 1] where:
            - Positive = generating cycle flowing well
            - Negative = controlling cycle dominant
            - Zero = balanced
        """
        generating_flow = 0.0
        controlling_flow = 0.0
        
        for source, target in GENERATING_CYCLE.items():
            source_exp = self._elements[source].expectation
            target_exp = self._elements[target].expectation
            # Flow is positive when source feeds target
            generating_flow += max(0, target_exp - source_exp)
        
        for controller, controlled in CONTROLLING_CYCLE.items():
            ctrl_exp = self._elements[controller].expectation
            ctld_exp = self._elements[controlled].expectation
            # Control is positive when controller suppresses controlled
            controlling_flow += max(0, ctrl_exp - ctld_exp)
        
        total = generating_flow + controlling_flow
        if total < 1e-9:
            return 0.0
        
        return (generating_flow - controlling_flow) / total
    
    def get_spread(self) -> float:
        """Spread between most optimistic and pessimistic elements."""
        exps = list(self.expectations.values())
        return max(exps) - min(exps)
    
    def get_mean_expectation(self) -> float:
        """Average expectation across all elements."""
        exps = list(self.expectations.values())
        return sum(exps) / len(exps)
    
    def reset(self) -> None:
        """Reset all channels to initial state."""
        self._create_channels()
    
    def load_expectations(self, expectations: Dict[WuXingElement, float]) -> None:
        """Restore expectations from saved state."""
        for element, exp in expectations.items():
            if element in self._elements:
                self._elements[element].channel.load(exp)
                self._elements[element].expectation = exp
