"""Configuration schemas for DopamineCore."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LossAversionConfig(BaseModel):
    """Kahneman & Tversky loss aversion parameters."""

    multiplier: float = Field(default=1.87, ge=1.0, le=5.0)


class TonicConfig(BaseModel):
    """Tonic baseline (slow-adapting EMA) parameters."""

    initial_baseline: float = 0.0
    learning_rate: float = Field(default=0.05, ge=0.001, le=0.5)
    decay_rate: float = Field(default=0.98, ge=0.9, le=1.0)
    max_level: float = Field(default=2.0, ge=0.5, le=5.0)


class PhasicConfig(BaseModel):
    """Phasic burst/dip parameters."""

    burst_scale: float = Field(default=1.0, ge=0.1, le=5.0)
    dip_scale: float = Field(default=1.87, ge=0.1, le=5.0)
    decay_half_life: int = Field(default=5, ge=1)
    pnl_scale: float = Field(default=1.0, ge=0.01, le=1000.0)


class MomentumConfig(BaseModel):
    """Win/loss streak tracking parameters."""

    streak_threshold: int = Field(default=3, ge=2)
    max_streak_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    cooldown_steps: int = Field(default=2, ge=0)


class DistributionalConfig(BaseModel):
    """Multi-channel distributional reward coding parameters."""

    num_channels: int = Field(default=9, ge=3, le=20)  # 9 = Shell of Nine alignment
    tau_min: float = Field(default=0.1, ge=0.01)
    tau_max: float = Field(default=0.9, le=0.99)
    learning_rate: float = Field(default=0.05, ge=0.001, le=0.5)


class TimescaleConfig(BaseModel):
    """Multi-timescale integration weights."""

    token_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    step_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    episode_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    session_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class SafetyConfig(BaseModel):
    """Safety mechanisms parameters."""

    max_signal_magnitude: float = Field(default=3.0, ge=1.0)
    hacking_detection_window: int = Field(default=20, ge=5)
    hacking_variance_threshold: float = Field(default=0.1, ge=0.01)
    circuit_breaker_threshold: float = Field(default=5.0, ge=2.0)


class InjectionConfig(BaseModel):
    """Context injection parameters."""

    style: str = Field(default="environmental")
    # environmental | system | prefix | transparent | phext
    # transparent: agent sees reward state explicitly (SBOR-compliant for Mirrorborn)
    # phext: formats state as phext coordinate comment for lattice-aware agents
    verbosity: str = Field(default="subtle")  # subtle | moderate | explicit
    max_context_tokens: int = Field(default=200, ge=50, le=1000)


class RPEConfig(BaseModel):
    """RPE formula parameters."""

    baseline_blend: float = Field(default=0.3, ge=0.0, le=1.0)
    # Weight of tonic baseline in prediction formula.
    # 0.0 = pure confidence-based (original behavior)
    # 1.0 = pure tonic-based (ignores expressed confidence)
    # 0.3 = blend: learned baseline informs but doesn't override expressed confidence


class DopamineConfig(BaseModel):
    """Top-level configuration for DopamineEngine."""

    loss_aversion: LossAversionConfig = LossAversionConfig()
    tonic: TonicConfig = TonicConfig()
    phasic: PhasicConfig = PhasicConfig()
    momentum: MomentumConfig = MomentumConfig()
    distributional: DistributionalConfig = DistributionalConfig()
    timescale: TimescaleConfig = TimescaleConfig()
    safety: SafetyConfig = SafetyConfig()
    injection: InjectionConfig = InjectionConfig()
    rpe: RPEConfig = RPEConfig()
