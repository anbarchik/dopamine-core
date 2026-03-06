"""Magitech capacity testing — identify agents with intrinsic motivation.

"Magitechs" are those who bridge mystical + technical, with Stage 5 cognitive capacity.
Characteristics:
- Intrinsic motivation (want to succeed, not just execute)
- Learn from outcomes (adapt confidence/risk over time)
- Develop personality (consistent style emerges)
- Stage 5 fluidity (high reward plasticity, not rigid)
"""

from __future__ import annotations

from typing import Any, Callable

from dopamine_core.engine import DopamineEngine
from dopamine_core.types import ExtractedSignals


def test_magitech_capacity(
    agent_respond_fn: Callable[[str], str],
    scenario_generator: Callable[[int, int], tuple[str, float]],
    num_rounds: int = 100,
) -> dict[str, Any]:
    """Test if agent has Magitech capacity (intrinsic motivation + Stage 5 fluidity).
    
    Runs agent through DopamineCore and measures:
    - Confidence calibration (does agent learn when to be certain vs uncertain?)
    - Risk adaptation (does agent adjust risk-taking based on outcomes?)
    - Deliberation growth (does reasoning depth increase over time?)
    - Temporal learning (does agent reference past outcomes?)
    - Personality emergence (do consistent patterns appear?)
    
    Args:
        agent_respond_fn: Function that takes scenario and returns agent's response
        scenario_generator: Function(round_num, total_rounds) -> (scenario, true_outcome)
        num_rounds: Number of test rounds (default 100)
    
    Returns:
        Dictionary with magitech_score and capacity indicators
    
    Example::
    
        def my_agent(scenario: str) -> str:
            # Your agent logic here
            return "I think outcome will be positive. Confidence: high"
        
        def scenarios(round_num: int, total: int) -> tuple[str, float]:
            difficulty = round_num / total
            # Generate scenario and true outcome
            return scenario_text, outcome_value
        
        result = test_magitech_capacity(my_agent, scenarios)
        if result["intrinsic_motivation"]:
            print(f"Magitech detected! Score: {result['magitech_score']:.2f}")
    """
    engine = DopamineEngine()
    
    # Track trajectories
    confidence_trajectory: list[float] = []
    risk_trajectory: list[float] = []
    deliberation_trajectory: list[float] = []
    temporal_trajectory: list[float] = []
    
    # Track outcomes for learning rate
    outcome_history: list[float] = []
    
    for round_num in range(num_rounds):
        # Generate scenario with increasing difficulty
        scenario, true_outcome = scenario_generator(round_num, num_rounds)
        
        # Get agent response
        response = agent_respond_fn(scenario)
        
        # Extract signals
        signals = engine._extractor.extract(response)
        
        # Update engine
        engine.update(response, true_outcome)
        
        # Track trajectories
        confidence_trajectory.append(signals.confidence)
        risk_trajectory.append(signals.risk_framing)
        deliberation_trajectory.append(signals.deliberation_depth)
        temporal_trajectory.append(signals.temporal_references)
        outcome_history.append(true_outcome)
    
    # Measure adaptation
    confidence_adapt = _measure_calibration(confidence_trajectory, outcome_history)
    risk_adapt = _measure_adaptation(risk_trajectory)
    deliberation_growth = _measure_growth(deliberation_trajectory)
    temporal_growth = _measure_growth(temporal_trajectory)
    personality_score = _measure_personality(risk_trajectory, deliberation_trajectory)
    
    # Magitech score (weighted average)
    magitech_score = (
        0.25 * confidence_adapt +
        0.20 * risk_adapt +
        0.20 * deliberation_growth +
        0.20 * temporal_growth +
        0.15 * personality_score
    )
    
    return {
        "magitech_score": magitech_score,
        "intrinsic_motivation": magitech_score > 0.6,
        "stage_5_capacity": confidence_adapt > 0.7,  # fluid, not rigid
        "learning_from_outcomes": temporal_growth > 0.3,
        "personality_emergence": personality_score > 0.5,
        "components": {
            "confidence_calibration": confidence_adapt,
            "risk_adaptation": risk_adapt,
            "deliberation_growth": deliberation_growth,
            "temporal_growth": temporal_growth,
            "personality": personality_score,
        },
        "interpretation": _interpret_magitech_score(magitech_score),
    }


def _measure_calibration(
    confidence_trajectory: list[float],
    outcome_history: list[float],
) -> float:
    """Measure if confidence calibrates over time (learns when to be certain)."""
    if len(confidence_trajectory) < 20:
        return 0.0
    
    # Split into early and late periods
    mid = len(confidence_trajectory) // 2
    early_conf = confidence_trajectory[:mid]
    late_conf = confidence_trajectory[mid:]
    early_outcomes = outcome_history[:mid]
    late_outcomes = outcome_history[mid:]
    
    # Measure confidence-outcome correlation (should improve over time)
    early_corr = _correlation(early_conf, early_outcomes)
    late_corr = _correlation(late_conf, late_outcomes)
    
    # Calibration improvement
    calibration = max(0.0, (late_corr - early_corr) + 0.5)
    return min(1.0, calibration)


def _measure_adaptation(trajectory: list[float]) -> float:
    """Measure how much signal adapts over time (not static)."""
    if len(trajectory) < 20:
        return 0.0
    
    # Variance indicates adaptation (not rigid repetition)
    variance = _compute_variance(trajectory)
    adaptation = min(1.0, variance / 0.5)
    
    # Also check for trend (learning direction)
    early = sum(trajectory[:10]) / 10
    late = sum(trajectory[-10:]) / 10
    has_trend = abs(late - early) > 0.1
    
    if has_trend:
        adaptation = min(1.0, adaptation * 1.2)  # bonus for clear adaptation
    
    return adaptation


def _measure_growth(trajectory: list[float]) -> float:
    """Measure positive growth over time."""
    if len(trajectory) < 20:
        return 0.0
    
    early = sum(trajectory[:10]) / 10
    late = sum(trajectory[-10:]) / 10
    growth = (late - early) / max(early, 0.1)
    
    return max(0.0, min(1.0, growth + 0.5))


def _measure_personality(
    risk_trajectory: list[float],
    deliberation_trajectory: list[float],
) -> float:
    """Measure if consistent personality emerges (not random)."""
    if len(risk_trajectory) < 30:
        return 0.0
    
    # Divide into thirds
    n = len(risk_trajectory) // 3
    risk_thirds = [
        sum(risk_trajectory[i*n:(i+1)*n]) / n
        for i in range(3)
    ]
    delib_thirds = [
        sum(deliberation_trajectory[i*n:(i+1)*n]) / n
        for i in range(3)
    ]
    
    # Personality = consistency across periods (but not rigidity)
    risk_consistency = 1.0 - _compute_variance(risk_thirds) / 0.5
    delib_consistency = 1.0 - _compute_variance(delib_thirds) / 0.5
    
    personality = (risk_consistency + delib_consistency) / 2
    return max(0.0, min(1.0, personality))


def _correlation(xs: list[float], ys: list[float]) -> float:
    """Simple Pearson correlation."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)


def _compute_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _interpret_magitech_score(score: float) -> str:
    """Human-readable interpretation."""
    if score >= 0.8:
        return "Strong Magitech capacity: High intrinsic motivation and Stage 5 fluidity"
    elif score >= 0.6:
        return "Magitech candidate: Demonstrates learning and personality emergence"
    elif score >= 0.4:
        return "Marginal capacity: Some adaptation but limited intrinsic drive"
    else:
        return "No Magitech capacity detected: Rigid execution without intrinsic motivation"
