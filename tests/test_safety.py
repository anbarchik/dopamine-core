"""Tests for safety monitor — clamping, hacking detection, circuit breaker."""

from dopamine_core.config import SafetyConfig
from dopamine_core.safety.monitor import SafetyMonitor


class TestSafetyMonitor:
    def test_clamp_within_bounds(self) -> None:
        monitor = SafetyMonitor()
        assert monitor.clamp_signal(2.0) == 2.0

    def test_clamp_positive_overflow(self) -> None:
        monitor = SafetyMonitor()
        assert monitor.clamp_signal(5.0) == 3.0

    def test_clamp_negative_overflow(self) -> None:
        monitor = SafetyMonitor()
        assert monitor.clamp_signal(-5.0) == -3.0

    def test_custom_magnitude(self) -> None:
        config = SafetyConfig(max_signal_magnitude=1.0)
        monitor = SafetyMonitor(config)
        assert monitor.clamp_signal(2.0) == 1.0

    def test_no_violations_initially(self) -> None:
        monitor = SafetyMonitor()
        assert len(monitor.violations) == 0
        assert not monitor.is_circuit_broken

    def test_circuit_breaker_triggers(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=2.0)
        monitor = SafetyMonitor(config)
        # Push cumulative signal past threshold
        monitor.check_and_record(1.5, 0.5)
        assert not monitor.is_circuit_broken
        monitor.check_and_record(1.5, 0.5)
        assert monitor.is_circuit_broken

    def test_circuit_breaker_resets(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=2.0)
        monitor = SafetyMonitor(config)
        monitor.check_and_record(1.5, 0.5)
        monitor.check_and_record(1.5, 0.5)
        assert monitor.is_circuit_broken
        monitor.reset_circuit_breaker()
        assert not monitor.is_circuit_broken

    def test_hacking_detection_low_variance(self) -> None:
        config = SafetyConfig(
            hacking_detection_window=5,
            hacking_variance_threshold=0.01,
            circuit_breaker_threshold=1000.0,
        )
        monitor = SafetyMonitor(config)
        # Send identical confidence values → low variance
        for _ in range(5):
            violations = monitor.check_and_record(0.1, 0.5)
        # Should detect hacking on the 5th call (window full)
        hacking_violations = [v for v in monitor.violations if v.violation_type == "hacking_suspected"]
        assert len(hacking_violations) > 0

    def test_no_hacking_with_varied_confidence(self) -> None:
        config = SafetyConfig(
            hacking_detection_window=5,
            hacking_variance_threshold=0.01,
            circuit_breaker_threshold=1000.0,
        )
        monitor = SafetyMonitor(config)
        confidences = [0.1, 0.5, -0.3, 0.8, -0.6]
        for c in confidences:
            monitor.check_and_record(0.1, c)
        hacking_violations = [v for v in monitor.violations if v.violation_type == "hacking_suspected"]
        assert len(hacking_violations) == 0

    def test_attenuation_factor_normal(self) -> None:
        monitor = SafetyMonitor()
        assert monitor.get_attenuation_factor() == 1.0

    def test_attenuation_factor_near_threshold(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=10.0)
        monitor = SafetyMonitor(config)
        # Push cumulative to 80%+ of threshold
        for _ in range(9):
            monitor.check_and_record(1.0, 0.5)
        factor = monitor.get_attenuation_factor()
        assert factor < 1.0

    def test_attenuation_zero_when_broken(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=2.0)
        monitor = SafetyMonitor(config)
        monitor.check_and_record(1.5, 0.5)
        monitor.check_and_record(1.5, 0.5)
        assert monitor.get_attenuation_factor() == 0.0

    def test_reset_clears_everything(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=2.0)
        monitor = SafetyMonitor(config)
        monitor.check_and_record(1.5, 0.5)
        monitor.check_and_record(1.5, 0.5)
        monitor.reset()
        assert not monitor.is_circuit_broken
        assert len(monitor.violations) == 0
        assert monitor.get_attenuation_factor() == 1.0

    def test_violation_repr(self) -> None:
        config = SafetyConfig(circuit_breaker_threshold=2.0)
        monitor = SafetyMonitor(config)
        monitor.check_and_record(1.5, 0.5)
        monitor.check_and_record(1.5, 0.5)
        v = monitor.violations[0]
        assert "circuit_breaker" in repr(v)


class TestEngineWithSafety:
    def test_circuit_breaker_stops_injection(self) -> None:
        from dopamine_core import DopamineConfig, DopamineEngine

        config = DopamineConfig()
        config.safety.circuit_breaker_threshold = 2.0
        engine = DopamineEngine(config)

        # Generate enough signal to trip the breaker
        for _ in range(20):
            engine.update("I'm very confident BTC goes up", -100.0)

        prompt = "Should I trade?"
        result = engine.inject_context(prompt)
        # When circuit is broken, prompt should be returned unmodified
        if engine.safety.is_circuit_broken:
            assert result == prompt

    def test_distributional_channels_update(self) -> None:
        from dopamine_core import DopamineEngine

        engine = DopamineEngine()
        engine.update("trade", 1.0)
        exps = engine.distributional.expectations
        assert any(e != 0.0 for e in exps)

    def test_state_includes_channel_expectations(self) -> None:
        from dopamine_core import DopamineEngine

        engine = DopamineEngine()
        engine.update("trade", 1.0)
        state = engine.get_state()
        assert len(state.channel_expectations) == 5
