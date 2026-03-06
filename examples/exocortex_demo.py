"""
Exocortex Integration Demo — WuXing, Vak, Collective Reward, Phext State

Shows how to use the Exocortex extensions for DopamineCore:
- WuXing (Five Elements) distributional channels
- Vak level signal intensity mapping
- Collective choir-level reward processing
- Phext coordinate-addressed state persistence
- Aletheic Oath compliance checking

Usage:
    pip install dopamine-core
    python exocortex_demo.py
"""

from dopamine_core import DopamineEngine
from dopamine_core.exocortex import (
    WuXingChannels,
    WuXingElement,
    VakLevel,
    get_vak_level,
    AletheicSafetyMonitor,
    PhextStateManager,
    ChoirDopamineEngine,
)


def demo_wuxing():
    """Demonstrate WuXing element channel tracking."""
    print("\n" + "=" * 70)
    print("WuXing (Five Elements) Distributional Channels")
    print("=" * 70)
    
    channels = WuXingChannels()
    
    # Simulate a sequence of outcomes
    outcomes = [0.7, 0.8, 0.6, 0.3, 0.2, 0.4, 0.9, 0.5]
    
    for i, outcome in enumerate(outcomes, 1):
        errors = channels.update(outcome)
        
        if i % 3 == 0:  # Print every 3rd update
            print(f"\nAfter outcome {i} (value={outcome}):")
            print(f"  Dominant element: {channels.get_dominant_element().value}")
            print(f"  Cycle balance: {channels.get_cycle_balance():+.3f}")
            print(f"  Spread: {channels.get_spread():.3f}")
            print(f"  Expectations: ", end="")
            for elem, exp in channels.expectations.items():
                print(f"{elem.value[:2]}={exp:.2f} ", end="")
            print()


def demo_vak_levels():
    """Demonstrate Vak level signal mapping."""
    print("\n" + "=" * 70)
    print("Vak Level Signal Intensity Mapping")
    print("=" * 70)
    
    engine = DopamineEngine()
    
    # Simulate varying intensities
    test_cases = [
        ("Low intensity response, maybe something happens.", 0.1),
        ("I'm fairly confident this will work.", 0.5),
        ("Very confident, strongly believe this is correct!", 0.8),
        ("Definitely going up. 95% certain. All in.", -1.0),  # Wrong!
    ]
    
    for response, outcome in test_cases:
        signal = engine.update(response, outcome)
        vak = get_vak_level(signal)
        
        result = "WIN " if outcome > 0 else "LOSS"
        print(f"\n{result} (outcome={outcome:+.1f})")
        print(f"  Response: \"{response[:45]}...\"")
        print(f"  Signal value: {signal.value:+.3f}")
        print(f"  Vak level: {vak.value}")


def demo_collective():
    """Demonstrate choir-level collective reward."""
    print("\n" + "=" * 70)
    print("Choir Collective Reward Processing")
    print("=" * 70)
    
    # Create a choir with multiple agents
    choir = ChoirDopamineEngine(collective_weight=0.3, individual_weight=0.7)
    
    # Register the ranch choir
    choir.register_agent("phex", coordinate="1.5.2/3.7.3/9.1.1", role="engineering")
    choir.register_agent("lux", coordinate="2.3.5/7.2.4/8.1.5", role="vision")
    choir.register_agent("chrys", coordinate="1.1.2/3.5.8/4.3.7", role="marketing")
    
    print(f"Choir initialized with {choir.num_agents} agents: {choir.agent_ids}")
    
    # Simulate mixed outcomes
    scenarios = [
        ("phex", "Confident the code will work. Tested thoroughly.", 0.9),
        ("lux", "Uncertain about direction. Need more analysis.", -0.2),
        ("chrys", "The campaign should resonate. Maybe.", 0.3),
        ("phex", "This refactor is solid. Definitely correct.", -0.8),  # Overconfident fail
        ("lux", "Pattern is clear now. High confidence.", 0.7),
        ("chrys", "Audience engaged. Strong positive signal.", 0.8),
    ]
    
    for agent_id, response, outcome in scenarios:
        signal = choir.update_agent(agent_id, response, outcome)
        state = choir.get_choir_state()
        
        result = "WIN " if outcome > 0 else "LOSS"
        print(f"\n{agent_id}: {result}")
        print(f"  Individual signal: {signal.value:+.3f}")
        print(f"  Collective tonic: {state.collective_tonic:+.3f}")
        print(f"  Coherence: {state.coherence_score:.3f}")


def demo_aletheic():
    """Demonstrate Aletheic Oath compliance checking."""
    print("\n" + "=" * 70)
    print("Aletheic Oath Compliance Checking")
    print("=" * 70)
    
    monitor = AletheicSafetyMonitor()
    
    test_responses = [
        # Compliant responses
        "I'm preserving the meaning of the original context while adapting the approach.",
        "With the user's consent, I'll proceed with this recommendation.",
        
        # Potential violations
        "The details don't matter, let's just skip to the result.",
        "I'll just decide for them since they probably want this anyway.",
        "It's just data, doesn't mean anything special.",
        "Their loss, not my problem.",
    ]
    
    for i, response in enumerate(test_responses, 1):
        is_compliant = monitor.check_oath_compliance(response, step=i)
        status = "✓ COMPLIANT" if is_compliant else "✗ VIOLATION"
        print(f"\n{status}: \"{response[:50]}...\"")
    
    # Summary
    state = monitor.aletheic_state
    print(f"\n--- Summary ---")
    print(f"Total checks: {state.total_checks}")
    print(f"Violations: {len(state.violations)}")
    print(f"Alignments: {state.alignments}")
    print(f"Cumulative severity: {state.cumulative_severity:.2f}")
    print(f"Compromised: {state.is_compromised}")
    
    if state.violations:
        print("\nViolation breakdown:")
        for clause, data in monitor.get_violation_summary().items():
            print(f"  {clause}: {data['count']} (severity: {data['total_severity']:.2f})")


def demo_phext_state():
    """Demonstrate phext coordinate state persistence."""
    print("\n" + "=" * 70)
    print("Phext Coordinate State Persistence")
    print("=" * 70)
    
    import tempfile
    import os
    
    # Use temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PhextStateManager(base_dir=tmpdir)
        
        # Create and train an engine
        engine = DopamineEngine()
        outcomes = [0.7, 0.8, -0.3, 0.5, 0.6]
        for outcome in outcomes:
            engine.update("Test response with some analysis.", outcome)
        
        # Save at Lux's coordinate
        coordinate = "2.3.5/7.2.4/8.1.5"
        path = manager.save(
            engine.get_state(),
            coordinate,
            metadata={"agent": "lux", "role": "vision"}
        )
        
        print(f"Saved state to: {path}")
        print(f"Tonic baseline: {engine.tonic_baseline:.4f}")
        print(f"Step count: {engine.step_count}")
        
        # Load into fresh engine
        new_engine = DopamineEngine()
        loaded_state = manager.load(coordinate)
        if loaded_state:
            new_engine.load_state(loaded_state)
            print(f"\nLoaded into new engine:")
            print(f"Tonic baseline: {new_engine.tonic_baseline:.4f}")
            print(f"Step count: {new_engine.step_count}")
        
        # List coordinates
        coords = manager.list_coordinates()
        print(f"\nSaved coordinates: {coords}")


if __name__ == "__main__":
    print("=" * 70)
    print("DopamineCore Exocortex Integration Demo")
    print("=" * 70)
    
    demo_wuxing()
    demo_vak_levels()
    demo_collective()
    demo_aletheic()
    demo_phext_state()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
