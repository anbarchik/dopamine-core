"""
DopamineCore + OpenAI — Simple BTC Trading Agent

A minimal trading agent that uses GPT-4 to decide UP/DOWN on BTC,
with DopamineCore providing intrinsic motivation signals.

Usage:
    pip install dopamine-core openai
    export OPENAI_API_KEY="sk-..."
    python openai_trader.py

The agent will simulate 10 rounds of BTC prediction.
Watch how its behavior changes after wins and losses.
"""

import os
import random

from openai import OpenAI

from dopamine_core import DopamineEngine

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
engine = DopamineEngine()


def get_prediction(prompt: str) -> tuple[str, str]:
    """Ask GPT-4 to analyze BTC and return (decision, reasoning)."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    text = response.choices[0].message.content or ""

    # Parse decision from response
    upper = text.upper()
    if "UP" in upper and "DOWN" not in upper:
        decision = "UP"
    elif "DOWN" in upper and "UP" not in upper:
        decision = "DOWN"
    else:
        # If ambiguous, look at what comes last
        up_pos = upper.rfind("UP")
        down_pos = upper.rfind("DOWN")
        decision = "UP" if up_pos > down_pos else "DOWN"

    return decision, text


def simulate_outcome(decision: str) -> tuple[str, float]:
    """Simulate BTC movement. Returns (actual_result, pnl).

    In production, you'd get this from a real exchange or prediction market.
    """
    actual = random.choice(["UP", "DOWN"])
    if decision == actual:
        pnl = 0.65  # won: payout minus stake
    else:
        pnl = -1.0  # lost: lost the stake
    return actual, pnl


def main() -> None:
    base_prompt = (
        "You are a BTC trader. Analyze the current market conditions and make a prediction.\n"
        "You MUST end your response with exactly one word: UP or DOWN.\n"
        "Explain your reasoning before giving your final answer."
    )

    print("DopamineCore + OpenAI Trading Agent")
    print("=" * 60)

    for round_num in range(1, 11):
        # Inject motivation context
        prompt = engine.inject_context(base_prompt)

        # Get LLM prediction
        decision, reasoning = get_prediction(prompt)

        # Simulate market outcome
        actual, pnl = simulate_outcome(decision)
        result = "WIN" if pnl > 0 else "LOSS"

        # Update dopamine engine with outcome
        signal = engine.update(reasoning, pnl)

        print(f"\nRound {round_num}: Predicted {decision}, Actual {actual} → {result} (${pnl:+.2f})")
        print(f"  RPE: {signal.phasic_response:+.3f} | Tonic: {signal.tonic_level:+.4f}")
        print(f"  Reasoning: {reasoning[:80]}...")

    print(f"\n{'=' * 60}")
    print(f"Final stats: {engine.step_count} rounds, tonic={engine.tonic_baseline:+.4f}")
    state = engine.get_state()
    streak = f"{state.streak_count}x {'wins' if state.streak_sign > 0 else 'losses'}"
    print(f"Current streak: {streak}")


if __name__ == "__main__":
    main()
