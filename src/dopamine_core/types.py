"""Core types for the DopamineCore framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TimescaleLevel(Enum):
    TOKEN = "token"
    STEP = "step"
    EPISODE = "episode"
    SESSION = "session"


@dataclass
class Outcome:
    """Result of a trade/prediction that the agent made."""

    pnl: float
    confidence: float | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedSignals:
    """Behavioral signals extracted from agent CoT reasoning."""

    confidence: float  # [-1, 1] hedging vs certainty
    risk_framing: float  # [-1, 1] risk-seeking vs risk-averse language
    deliberation_depth: float  # [0, 1] reasoning complexity
    temporal_references: float  # [0, 1] how much agent references past outcomes


@dataclass
class RPEResult:
    """Reward Prediction Error computation result."""

    prediction: float
    actual: float
    error: float  # the RPE value (asymmetric with loss aversion)
    raw_error: float  # before loss aversion
    surprise: float  # normalized magnitude


@dataclass
class RewardSignal:
    """A reward signal at a specific timescale."""

    value: float
    source: str
    timescale: TimescaleLevel = TimescaleLevel.STEP
    timestamp: float = 0.0


@dataclass
class CompositeSignal:
    """Aggregated reward signal across all components."""

    value: float  # composite reward signal [-3, 3]
    confidence_factor: float  # extracted confidence
    risk_assessment: float  # from distributional channels
    momentum_factor: float  # streak influence
    tonic_level: float  # current baseline
    phasic_response: float  # current event response


@dataclass
class EngineState:
    """Full serializable state snapshot for persistence."""

    tonic_baseline: float = 0.0
    step_count: int = 0
    outcome_history: list[float] = field(default_factory=list)
    streak_count: int = 0
    streak_sign: int = 0  # 1 for wins, -1 for losses, 0 for none
    phasic_signals: list[tuple[float, int]] = field(default_factory=list)
    channel_expectations: list[float] = field(default_factory=list)
    last_rpe: float = 0.0
