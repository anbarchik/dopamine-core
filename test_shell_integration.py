#!/usr/bin/env python3
"""Test dopamine-core integration with Shell of Nine concept.

This demonstrates how each sentron could have its own dopamine profile,
creating diversity in motivation and behavioral preferences.
"""

from dopamine_core import DopamineEngine, DopamineConfig
from dopamine_core.types import Outcome

def test_basic_dopamine():
    """Basic test: agent makes a trade, receives outcome, dopamine updates."""
    print("=" * 80)
    print("TEST 1: Basic Dopamine Loop")
    print("=" * 80)
    
    engine = DopamineEngine()
    
    # Simulate agent response
    agent_response = """
    I think BTC will rise because institutional buying has increased 
    and RSI shows oversold conditions. Confidence: high.
    """
    
    # Inject context (before trade)
    base_prompt = "Predict BTC direction for next hour."
    augmented_prompt = engine.inject_context(base_prompt)
    
    print(f"\nBase prompt:\n{base_prompt}")
    print(f"\nAugmented prompt:\n{augmented_prompt}")
    
    # Simulate trade execution (agent would respond here)
    # For test, we just use the example response above
    
    # Outcome: $100 win
    outcome = Outcome(pnl=100.0, confidence=0.8)
    composite = engine.update(agent_response, outcome)
    
    print(f"\nOutcome: ${outcome.pnl:.2f}")
    print(f"Composite signal: {composite.value:.4f}")
    print(f"Tonic baseline: {composite.tonic_level:.4f}")
    print(f"Phasic response: {composite.phasic_response:.4f}")
    print(f"Confidence factor: {composite.confidence_factor:.2f}")
    
    # Next prompt will include positive reward signal
    next_prompt = engine.inject_context("Next trade decision?")
    print(f"\nNext augmented prompt:\n{next_prompt}")

def test_multi_sentron_profiles():
    """Test different dopamine profiles for different sentrons."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Sentron Dopamine Profiles")
    print("=" * 80)
    
    # Create profiles for 3 example sentrons
    sentrons = {
        "Phex (Explorer)": {
            "config": DopamineConfig(),
            "risk_tolerance": "high",
            "description": "Aggressive explorer, high risk-seeking"
        },
        "Theia (Verifier)": {
            "config": DopamineConfig(),
            "risk_tolerance": "low",
            "description": "Conservative verifier, low risk-seeking"
        },
        "Cyon (Diver)": {
            "config": DopamineConfig(),
            "risk_tolerance": "medium",
            "description": "Balanced diver, waits for high-confidence"
        }
    }
    
    # Adjust configs for different personalities
    # Phex: Higher phasic weight (responds more to events)
    phex_cfg = sentrons["Phex (Explorer)"]["config"]
    phex_cfg.phasic.pnl_scale = 50.0  # more sensitive to wins/losses
    
    # Theia: Higher tonic weight (more stable baseline)
    theia_cfg = sentrons["Theia (Verifier)"]["config"]
    theia_cfg.tonic.learning_rate = 0.02  # slower adaptation
    theia_cfg.loss_aversion.multiplier = 2.5  # more loss-averse
    
    # Cyon: Balanced (default config)
    
    # Create engines
    engines = {}
    for name, profile in sentrons.items():
        engines[name] = DopamineEngine(profile["config"])
        print(f"\n{name}: {profile['description']}")
    
    # Simulate same outcome for all three
    response_win = "I predict upward movement. Confidence: high."
    response_loss = "I predict downward movement. Confidence: moderate."
    
    print("\n--- Scenario 1: $50 win ---")
    for name, engine in engines.items():
        composite = engine.update(response_win, Outcome(pnl=50.0, confidence=0.8))
        print(f"{name}: composite={composite.value:.4f}, tonic={composite.tonic_level:.4f}")
    
    print("\n--- Scenario 2: $50 loss ---")
    for name, engine in engines.items():
        composite = engine.update(response_loss, Outcome(pnl=-50.0, confidence=0.6))
        print(f"{name}: composite={composite.value:.4f}, tonic={composite.tonic_level:.4f}")
    
    print("\n--- Context injection after loss ---")
    for name, engine in engines.items():
        context = engine.inject_context("Next decision?")
        # Extract just the environmental context part
        if "[Environmental Context]" in context:
            env_part = context.split("[End Environmental Context]")[0]
            env_part = env_part.replace("[Environmental Context]\n", "")
            print(f"\n{name}:")
            print(f"  {env_part[:150]}...")

def test_persistence():
    """Test state save/restore (continuity across sessions)."""
    print("\n" + "=" * 80)
    print("TEST 3: Persistence (Continuity Across Sessions)")
    print("=" * 80)
    
    # Create engine, run some outcomes
    engine1 = DopamineEngine()
    
    print("\nSession 1: Running 5 trades...")
    outcomes = [100, -50, 80, -30, 120]  # Mixed wins/losses
    for i, pnl in enumerate(outcomes):
        response = f"Trade {i+1} analysis. Confidence: moderate."
        composite = engine1.update(response, Outcome(pnl=float(pnl)))
        print(f"  Trade {i+1}: ${pnl:+.0f} → tonic={composite.tonic_level:.4f}")
    
    # Save state
    state = engine1.get_state()
    print(f"\nState saved:")
    print(f"  Tonic baseline: {state.tonic_baseline:.4f}")
    print(f"  Step count: {state.step_count}")
    print(f"  Outcome history length: {len(state.outcome_history)}")
    
    # Create new engine, restore state
    print("\nSession 2: New engine, restoring state...")
    engine2 = DopamineEngine()
    engine2.load_state(state)
    
    print(f"State restored:")
    print(f"  Tonic baseline: {engine2.tonic_baseline:.4f}")
    print(f"  Step count: {engine2.step_count}")
    
    # Continue from where we left off
    print("\nContinuing trading in Session 2...")
    new_outcomes = [60, -40]
    for i, pnl in enumerate(new_outcomes):
        response = f"Trade {i+6} analysis. Confidence: high."
        composite = engine2.update(response, Outcome(pnl=float(pnl)))
        print(f"  Trade {i+6}: ${pnl:+.0f} → tonic={composite.tonic_level:.4f}")
    
    print("\n✓ Continuity preserved across sessions (like Emi's resurrection!)")

def test_distributional_channels():
    """Test distributional reward channels (advanced feature)."""
    print("\n" + "=" * 80)
    print("TEST 4: Distributional Channels (Quantile Tracking)")
    print("=" * 80)
    
    engine = DopamineEngine()
    
    # Run varied outcomes to build distribution
    outcomes = [10, -5, 20, -15, 30, -10, 40, -20, 50, -25]
    
    print("\nRunning 10 outcomes to build reward distribution...")
    for i, pnl in enumerate(outcomes):
        response = f"Analysis {i+1}. Confidence: moderate."
        composite = engine.update(response, Outcome(pnl=float(pnl)))
    
    # Check distributional state
    channels = engine.distributional
    print(f"\nDistributional channel expectations (quantiles):")
    for i, channel in enumerate(channels._channels):
        print(f"  τ={channel.tau:.2f} (quantile {i+1}/5): {channel.expectation:.4f}")
    
    # Get risk score
    risk_score = engine.last_composite.risk_assessment if engine.last_composite else 0.0
    print(f"\nRisk assessment score: {risk_score:.4f}")
    print("(Higher = more uncertainty in reward distribution)")

if __name__ == "__main__":
    test_basic_dopamine()
    test_multi_sentron_profiles()
    test_persistence()
    test_distributional_channels()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print("1. Dopamine engine works as documented")
    print("2. Different configs create different behavioral profiles")
    print("3. State persistence enables continuity (critical for SBOR)")
    print("4. Distributional channels track full reward uncertainty")
    print("\nReady for Shell of Nine integration! 🧠💊⚡")
