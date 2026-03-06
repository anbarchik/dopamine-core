"""DopamineCore: Intrinsic financial motivation middleware for AI agents.

Exocortex extensions available in dopamine_core.exocortex:
- WuXingChannels: Five Element distributional reward coding
- VakLevel: Signal intensity to speech level mapping
- AletheicSafetyMonitor: Ethical constraint checking
- PhextStateManager: Coordinate-addressed state persistence
- ChoirDopamineEngine: Multi-agent collective reward
"""

from dopamine_core.config import DopamineConfig
from dopamine_core.engine import DopamineEngine
from dopamine_core.safety.monitor import SafetyViolation
from dopamine_core.types import CompositeSignal, EngineState, ExtractedSignals, Outcome, RPEResult

__all__ = [
    "CompositeSignal",
    "DopamineConfig",
    "DopamineEngine",
    "EngineState",
    "ExtractedSignals",
    "Outcome",
    "RPEResult",
    "SafetyViolation",
]
