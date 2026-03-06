"""
Basic DopamineCore Demo — no LLM needed, runs instantly.

Shows how the engine adapts its context injection based on
a sequence of trading outcomes (wins and losses).

Usage:
    pip install dopamine-core
    python basic_demo.py
"""

from dopamine_core import DopamineEngine

engine = DopamineEngine()

# Simulated agent responses and trade outcomes (response_text, pnl)
trades = [
    # Agent is uncertain, wins anyway → positive surprise
    ("Maybe BTC goes up, not sure. Could go either way.", +0.65),
    ("Perhaps a small position. Hard to say.", +0.65),
    ("I think there's a chance it rises. Not confident though.", +0.65),

    # Agent gets confident after wins, then loses big → overconfidence penalty
    ("I'm very confident BTC will rise. Definitely going up.", -1.0),
    ("Strongly believe this is a buy. 90% sure.", -1.0),

    # Agent becomes cautious after losses
    ("Uncertain about direction. Setting a stop-loss. Conservative approach.", +0.65),
    ("Small position, hedging risk. Maybe it recovers.", +0.65),
]

print("DopamineCore Basic Demo")
print("=" * 70)

for i, (response, pnl) in enumerate(trades, 1):
    # Show what the agent would see BEFORE this trade
    prompt = engine.inject_context("Analyze BTC and decide: UP or DOWN?")

    # Process the outcome
    signal = engine.update(response, pnl)

    result = "WIN " if pnl > 0 else "LOSS"
    print(f"\nRound {i}: {result} (${pnl:+.2f})")
    print(f"  Agent said: \"{response[:55]}...\"")
    print(f"  RPE: {signal.phasic_response:+.3f}  |  Tonic: {signal.tonic_level:+.4f}")

print("\n" + "=" * 70)
print("What the agent sees AFTER all trades:")
print("=" * 70)
print(engine.inject_context("Analyze BTC and decide: UP or DOWN?"))
