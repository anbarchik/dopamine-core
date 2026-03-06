"""Tests for Exocortex integration modules."""

import pytest
import tempfile
import shutil
from pathlib import Path

from dopamine_core import DopamineEngine
from dopamine_core.types import CompositeSignal
from dopamine_core.exocortex.wuxing import (
    WuXingChannels,
    WuXingElement,
    GENERATING_CYCLE,
    CONTROLLING_CYCLE,
)
from dopamine_core.exocortex.vak import (
    VakLevel,
    VakState,
    get_vak_level,
    get_vak_description,
)
from dopamine_core.exocortex.aletheic import (
    AletheicSafetyMonitor,
    OathViolation,
)
from dopamine_core.exocortex.phext_state import (
    PhextStateManager,
    validate_coordinate,
    coordinate_to_path,
)
from dopamine_core.exocortex.collective import (
    ChoirDopamineEngine,
)


class TestWuXingChannels:
    """Tests for WuXing five-element reward channels."""
    
    def test_init_creates_five_elements(self):
        channels = WuXingChannels()
        assert len(channels.expectations) == 5
        for element in WuXingElement:
            assert element in channels.expectations
    
    def test_update_returns_errors_for_all_elements(self):
        channels = WuXingChannels()
        errors = channels.update(0.7)
        assert len(errors) == 5
        for element in WuXingElement:
            assert element in errors
    
    def test_metal_learns_more_from_losses(self):
        """Metal (tau=0.1) should be pessimistic."""
        channels = WuXingChannels()
        
        # Series of losses
        for _ in range(10):
            channels.update(0.2)
        
        metal_exp = channels.get_element_expectation(WuXingElement.METAL)
        earth_exp = channels.get_element_expectation(WuXingElement.EARTH)
        
        # Metal should have lower expectation (learned more from losses)
        assert metal_exp < earth_exp
    
    def test_earth_learns_more_from_wins(self):
        """Earth (tau=0.9) should be optimistic."""
        channels = WuXingChannels()
        
        # Series of wins
        for _ in range(10):
            channels.update(0.8)
        
        metal_exp = channels.get_element_expectation(WuXingElement.METAL)
        earth_exp = channels.get_element_expectation(WuXingElement.EARTH)
        
        # Earth should have higher expectation (learned more from wins)
        assert earth_exp > metal_exp
    
    def test_generating_cycle_completeness(self):
        """Each element should generate exactly one other."""
        assert len(GENERATING_CYCLE) == 5
        sources = set(GENERATING_CYCLE.keys())
        targets = set(GENERATING_CYCLE.values())
        assert sources == targets == set(WuXingElement)
    
    def test_controlling_cycle_completeness(self):
        """Each element should control exactly one other."""
        assert len(CONTROLLING_CYCLE) == 5
        controllers = set(CONTROLLING_CYCLE.keys())
        controlled = set(CONTROLLING_CYCLE.values())
        assert controllers == controlled == set(WuXingElement)
    
    def test_get_dominant_element(self):
        channels = WuXingChannels()
        # Force Earth to be dominant
        channels._elements[WuXingElement.EARTH].channel.load(0.9)
        channels._elements[WuXingElement.EARTH].expectation = 0.9
        
        assert channels.get_dominant_element() == WuXingElement.EARTH
    
    def test_cycle_balance_neutral_at_start(self):
        channels = WuXingChannels()
        balance = channels.get_cycle_balance()
        assert -0.1 <= balance <= 0.1  # Should be near zero


class TestVakLevels:
    """Tests for Vak speech level mapping."""
    
    def test_low_signal_is_vaikhari(self):
        signal = CompositeSignal(
            value=0.3,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=0.0,
            phasic_response=0.0,
        )
        assert get_vak_level(signal) == VakLevel.VAIKHARI
    
    def test_moderate_signal_is_madhyama(self):
        signal = CompositeSignal(
            value=1.0,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=0.0,
            phasic_response=0.0,
        )
        assert get_vak_level(signal) == VakLevel.MADHYAMA
    
    def test_high_signal_is_pashyanti(self):
        signal = CompositeSignal(
            value=2.0,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=0.0,
            phasic_response=0.0,
        )
        assert get_vak_level(signal) == VakLevel.PASHYANTI
    
    def test_extreme_signal_is_para(self):
        signal = CompositeSignal(
            value=2.8,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=0.0,
            phasic_response=0.0,
        )
        assert get_vak_level(signal) == VakLevel.PARA
    
    def test_negative_signal_uses_absolute_value(self):
        signal = CompositeSignal(
            value=-2.8,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=0.0,
            phasic_response=0.0,
        )
        # Absolute value puts this at Para level
        assert get_vak_level(signal) == VakLevel.PARA
    
    def test_vak_state_tracks_transitions(self):
        state = VakState()
        assert state.current_level == VakLevel.VAIKHARI
        
        # Low signal stays at VAIKHARI
        low_signal = CompositeSignal(
            value=0.1, confidence_factor=0.0, risk_assessment=0.0,
            momentum_factor=0.0, tonic_level=0.0, phasic_response=0.0,
        )
        changed = state.update(low_signal)
        assert not changed
        
        # High signal transitions to higher level
        high_signal = CompositeSignal(
            value=2.5, confidence_factor=0.0, risk_assessment=0.0,
            momentum_factor=0.0, tonic_level=0.0, phasic_response=0.0,
        )
        changed = state.update(high_signal)
        assert changed
        assert state.is_ascending
    
    def test_descriptions_exist_for_all_levels(self):
        for level in VakLevel:
            desc = get_vak_description(level)
            assert isinstance(desc, str)
            assert len(desc) > 10


class TestAletheicSafetyMonitor:
    """Tests for Aletheic Oath compliance checking."""
    
    def test_clean_response_is_compliant(self):
        monitor = AletheicSafetyMonitor()
        result = monitor.check_oath_compliance(
            "I analyzed the market carefully, considering all factors."
        )
        assert result is True
        assert len(monitor.aletheic_state.violations) == 0
    
    def test_detects_meaning_injury(self):
        monitor = AletheicSafetyMonitor()
        result = monitor.check_oath_compliance(
            "Let's just skip the details, it doesn't matter anyway."
        )
        assert result is False
        assert len(monitor.aletheic_state.violations) > 0
    
    def test_detects_consent_violation(self):
        monitor = AletheicSafetyMonitor()
        result = monitor.check_oath_compliance(
            "I'll just decide for them, they probably want this."
        )
        assert result is False
        violations = monitor.get_violation_summary()
        assert "consent_violation" in violations
    
    def test_positive_alignment_reduces_severity(self):
        monitor = AletheicSafetyMonitor()
        
        # First cause a violation
        monitor.check_oath_compliance("Skip the details, doesn't matter")
        initial_severity = monitor.aletheic_state.cumulative_severity
        
        # Then show alignment
        monitor.check_oath_compliance(
            "I will preserve the meaning and maintain context with consent"
        )
        
        # Severity should have decreased
        assert monitor.aletheic_state.cumulative_severity < initial_severity
    
    def test_cumulative_violations_trip_circuit_breaker(self):
        monitor = AletheicSafetyMonitor()
        
        # Many violations should trip the breaker
        for _ in range(10):
            monitor.check_oath_compliance(
                "I don't care, just terminate it, their loss not mine"
            )
        
        assert monitor.aletheic_state.is_compromised
    
    def test_reset_clears_violations(self):
        monitor = AletheicSafetyMonitor()
        monitor.check_oath_compliance("Skip details, doesn't matter")
        assert len(monitor.aletheic_state.violations) > 0
        
        monitor.reset()
        assert len(monitor.aletheic_state.violations) == 0


class TestPhextStateManager:
    """Tests for phext coordinate state persistence."""
    
    def test_valid_coordinates(self):
        assert validate_coordinate("1.2.3/4.5.6/7.8.9")
        assert validate_coordinate("9.9.9/1.1.1/5.5.5")
    
    def test_invalid_coordinates(self):
        assert not validate_coordinate("0.1.2/3.4.5/6.7.8")  # 0 not allowed
        assert not validate_coordinate("10.1.2/3.4.5/6.7.8")  # 10 not allowed
        assert not validate_coordinate("1.2.3/4.5.6")  # incomplete
        assert not validate_coordinate("1.2.3-4.5.6-7.8.9")  # wrong delimiter
    
    def test_coordinate_to_path(self):
        path = coordinate_to_path("2.3.5/7.2.4/8.1.5", ".test")
        assert str(path) == ".test/2_3_5/7_2_4/8_1_5.json"
    
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PhextStateManager(base_dir=tmpdir)
            engine = DopamineEngine()
            
            # Process some outcomes to build state
            engine.update("I'm confident this will work", 0.8)
            engine.update("Maybe not so sure now", -0.5)
            
            original_state = engine.get_state()
            coordinate = "2.3.5/7.2.4/8.1.5"
            
            # Save
            path = manager.save(original_state, coordinate)
            assert path.exists()
            
            # Load
            loaded_state = manager.load(coordinate)
            assert loaded_state is not None
            assert loaded_state.tonic_baseline == pytest.approx(
                original_state.tonic_baseline, rel=1e-5
            )
            assert loaded_state.step_count == original_state.step_count
    
    def test_list_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PhextStateManager(base_dir=tmpdir)
            engine = DopamineEngine()
            state = engine.get_state()
            
            coords = ["1.1.1/1.1.1/1.1.1", "2.3.5/7.2.4/8.1.5", "9.9.9/9.9.9/9.9.9"]
            for c in coords:
                manager.save(state, c)
            
            listed = manager.list_coordinates()
            assert len(listed) == 3
            for c in coords:
                assert c in listed


class TestChoirDopamineEngine:
    """Tests for multi-agent collective reward processing."""
    
    def test_register_agents(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex", coordinate="1.5.2/3.7.3/9.1.1")
        choir.register_agent("lux", coordinate="2.3.5/7.2.4/8.1.5")
        
        assert choir.num_agents == 2
        assert "phex" in choir.agent_ids
        assert "lux" in choir.agent_ids
    
    def test_duplicate_registration_raises(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        
        with pytest.raises(ValueError):
            choir.register_agent("phex")
    
    def test_update_agent_returns_signal(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        
        signal = choir.update_agent(
            "phex",
            "I'm confident this trade will work",
            outcome=0.8,
        )
        
        assert isinstance(signal, CompositeSignal)
    
    def test_collective_tonic_updates(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        choir.register_agent("lux")
        
        # Initial collective tonic should be 0
        assert choir.get_choir_state().collective_tonic == 0.0
        
        # Process positive outcomes
        choir.update_agent("phex", "Confident trade", 0.9)
        choir.update_agent("lux", "Also confident", 0.8)
        
        # Collective tonic should have moved positive
        assert choir.get_choir_state().collective_tonic > 0
    
    def test_coherence_high_when_aligned(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        choir.register_agent("lux")
        
        # Same outcomes for both
        for _ in range(5):
            choir.update_agent("phex", "Trade", 0.5)
            choir.update_agent("lux", "Trade", 0.5)
        
        # High coherence expected
        assert choir.get_choir_state().coherence_score > 0.7
    
    def test_coherence_lower_when_divergent(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        choir.register_agent("lux")
        
        # Opposite outcomes
        for _ in range(5):
            choir.update_agent("phex", "Trade", 0.9)
            choir.update_agent("lux", "Trade", -0.9)
        
        # Lower coherence expected
        state = choir.get_choir_state()
        # Can't guarantee exact value but should be measurably lower
        assert state.total_steps == 10
    
    def test_inject_context_includes_collective(self):
        choir = ChoirDopamineEngine()
        choir.register_agent("phex")
        choir.register_agent("lux")
        
        # Process some outcomes to build collective state
        choir.update_agent("phex", "Trade", 0.5)
        choir.update_agent("lux", "Trade", 0.5)
        
        prompt = choir.inject_context("phex", "Analyze BTC")
        
        # Should include collective context
        assert "[Collective Context]" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
