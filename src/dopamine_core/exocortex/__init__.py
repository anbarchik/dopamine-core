"""Exocortex integration modules for DopamineCore.

Extensions for phext coordinate addressing, WuXing element mapping,
collective reward, and Aletheic Oath compliance.
"""

from dopamine_core.exocortex.wuxing import WuXingChannels, WuXingElement
from dopamine_core.exocortex.vak import VakLevel, get_vak_level
from dopamine_core.exocortex.aletheic import AletheicSafetyMonitor
from dopamine_core.exocortex.phext_state import PhextStateManager
from dopamine_core.exocortex.collective import ChoirDopamineEngine

__all__ = [
    "WuXingChannels",
    "WuXingElement",
    "VakLevel",
    "get_vak_level",
    "AletheicSafetyMonitor",
    "PhextStateManager",
    "ChoirDopamineEngine",
]
