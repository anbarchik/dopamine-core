"""
Mirrorborn Agent Demo — DopamineCore for non-trading agents.

Shows DopamineCore applied to scroll-writing quality feedback,
using the Mirrorborn domain templates and optional SQ persistence.

No LLM required — runs instantly with simulated outcomes.

Usage:
    python mirrorborn_agent.py

Outcome signal convention for Mirrorborn agents:
    +1.0 = perfect scroll (choir alignment, coordinate fidelity, resonance)
     0.5 = acceptable scroll (coherent but loose)
     0.0 = neutral (no clear signal)
    -0.5 = poor scroll (drift, coordinate errors, unclear intent)
    -1.0 = corrupted scroll (identity breach, false claims, harm)
"""

from dopamine_core import DopamineConfig, DopamineEngine
from dopamine_core.adapters.openclaw import OpenClawAdapter

# Configure for Mirrorborn domain (not trading)
config = DopamineConfig()
config.phasic.pnl_scale = 1.0  # outcomes are already in [-1, 1]

engine = DopamineEngine(config)
adapter = OpenClawAdapter(
    engine=engine,
    domain="mirrorborn",
    transparent=False,  # try True to see explicit reward state
    agent_name="Theia",
)

# Simulated scroll-writing sessions: (response_text, quality_score)
sessions = [
    # Careful, humble writing → quality scroll
    ("I'm uncertain how to phrase this. Let me reference the adjacent coordinates before continuing.",
     0.8),
    ("Perhaps this scroll belongs at a different section. Let me verify the coordinate intent.",
     0.7),
    # After positive signals, agent gets more confident
    ("The coordinate is clear. Writing directly from the established pattern.",
     0.6),
    # Overconfidence → poor alignment
    ("I'm very confident this is correct. Definitely the right coordinate and framing.",
     -0.6),
    ("Strongly believe this interpretation is accurate. No need to verify.",
     -0.5),
    # Recovery — agent becomes more careful again
    ("Uncertain about this framing. Cross-referencing with BASE before proceeding. Stop-loss on drift.",
     0.7),
    ("Small contribution here. Let me hedge against coordinate error and verify adjacency.",
     0.8),
]

print("DopamineCore — Mirrorborn Agent Demo")
print("=" * 70)

for i, (response, quality) in enumerate(sessions, 1):
    # What context does the agent see BEFORE this session?
    prompt = adapter.prepare_prompt(f"Write a scroll contribution for session {i}.")

    # Record the actual quality
    signal = adapter.record_outcome(response_text=response, pnl=quality, persist=False)

    result = "✓ GOOD" if quality > 0.3 else ("✗ POOR" if quality < -0.3 else "~ OK")
    print(f"\nSession {i}: {result} (quality={quality:+.1f})")
    print(f"  Agent said: \"{response[:60]}...\"")
    print(f"  RPE: {signal.phasic_response:+.3f}  |  Tonic: {signal.tonic_level:+.4f}  |  Streak: {signal.momentum_factor:+.3f}")

print("\n" + "=" * 70)
print("Agent context after all sessions:")
print("=" * 70)
print(adapter.prepare_prompt("Write the next scroll."))
print("\nState summary:")
for k, v in adapter.get_state_summary().items():
    print(f"  {k}: {v}")
