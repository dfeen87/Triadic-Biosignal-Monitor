"""
Tests for gate module.

Tests instability functional, decision gate, and alert generation.
"""

import pytest
import numpy as np
from core.gate import (
    InstabilityConfig,
    compute_instability_functional,
    decision_gate,
    compute_instability_timeseries,
    generate_alert,
    detect_alert_events,
    ablation_eeg_only,
    ablation_ecg_only,
    risk_score,
    format_alert_message
)


class TestInstabilityConfig:
    """Tests for InstabilityConfig dataclass."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = InstabilityConfig(
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            threshold=2.5
        )
        
        assert config.alpha == 0.4
        assert config.beta == 0.3
        assert config.gamma == 0.3
        assert config.threshold == 2.5
    
    def test_auto_normalize_weights(self):
        """Test automatic normalization of weights."""
        config = InstabilityConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.5,  # Sum = 1.5, should be normalized
            threshold=2.5
        )
        
        # Weights should sum to 1.0 after normalization
        assert np.isclose(config.alpha + config.beta + config.gamma, 1.0)
    
    def test_negative_weight_error(self):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError):
            InstabilityConfig(alpha=-0.1, beta=0.5, gamma=0.5, threshold=2.5)
    
    def test_negative_threshold_error(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError):
            InstabilityConfig(alpha=0.4, beta=0.3, gamma=0.3, threshold=-1.0)


class TestInstabilityFunctional:
    """Tests for instability functional computation."""
    
    def test_compute_instability_functional(self):
        """Test basic instability functional computation."""
        delta_S = 1.0
        delta_I = 0.5
        delta_C = 0.8
        alpha, beta, gamma = 0.4, 0.3, 0.3
        
        delta_phi = compute_instability_functional(
            delta_S, delta_I, delta_C, alpha, beta, gamma
        )
        
        # ΔΦ = 0.4*1.0 + 0.3*0.5 + 0.3*0.8 = 0.4 + 0.15 + 0.24 = 0.79
        expected = 0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.8
        assert np.isclose(delta_phi, expected)
    
    def test_zero_deviations(self):
        """Test with zero deviations."""
        delta_phi = compute_instability_functional(0, 0, 0, 0.4, 0.3, 0.3)
        assert delta_phi == 0.0
    
    def test_negative_deviations(self):
        """Test that negative deviations are handled (absolute value)."""
        delta_phi = compute_instability_functional(-1.0, -0.5, -0.8, 0.4, 0.3, 0.3)
        assert delta_phi > 0  # Should take absolute values


class TestDecisionGate:
    """Tests for decision gate."""
    
    def test_decision_gate_below_threshold(self):
        """Test gate with value below threshold."""
        delta_phi = np.array([1.0, 1.5, 2.0])
        threshold = 2.5
        
        gate = decision_gate(delta_phi, threshold)
        
        assert np.array_equal(gate, np.array([0, 0, 0]))
    
    def test_decision_gate_above_threshold(self):
        """Test gate with value above threshold."""
        delta_phi = np.array([2.0, 2.5, 3.0])
        threshold = 2.5
        
        gate = decision_gate(delta_phi, threshold)
        
        assert np.array_equal(gate, np.array([0, 1, 1]))
    
    def test_decision_gate_exact_threshold(self):
        """Test gate at exact threshold."""
        delta_phi = np.array([2.5])
        threshold = 2.5
        
        gate = decision_gate(delta_phi, threshold)
        
        assert gate[0] == 1  # >= threshold triggers


class TestInstabilityTimeseries:
    """Tests for time series processing."""
    
    def test_compute_instability_timeseries(self):
        """Test instability computation over time series."""
        n_windows = 100
        delta_S_series = np.random.rand(n_windows)
        delta_I_series = np.random.rand(n_windows)
        delta_C_series = np.random.rand(n_windows)
        
        config = InstabilityConfig(alpha=0.4, beta=0.3, gamma=0.3, threshold=2.0)
        
        delta_phi_series, gate_series = compute_instability_timeseries(
            delta_S_series, delta_I_series, delta_C_series, config
        )
        
        assert len(delta_phi_series) == n_windows
        assert len(gate_series) == n_windows
        assert gate_series.dtype == int
        assert np.all(np.isin(gate_series, [0, 1]))
    
    def test_length_mismatch_error(self):
        """Test error on length mismatch."""
        config = InstabilityConfig()
        
        with pytest.raises(ValueError):
            compute_instability_timeseries(
                np.array([1, 2, 3]),
                np.array([1, 2]),  # Different length
                np.array([1, 2, 3]),
                config
            )


class TestAlertGeneration:
    """Tests for alert generation."""
    
    def test_generate_alert(self):
        """Test alert generation."""
        alert = generate_alert(
            timestamp=10.5,
            delta_phi=3.2,
            delta_S=1.5,
            delta_I=0.8,
            delta_C=1.0,
            artifact_level=0.1
        )
        
        assert alert.timestamp == 10.5
        assert alert.delta_phi == 3.2
        assert alert.delta_S == 1.5
        assert alert.delta_I == 0.8
        assert alert.delta_C == 1.0
        assert 0 <= alert.confidence <= 1.0
        assert isinstance(alert.flags, dict)
    
    def test_low_confidence_with_artifacts(self):
        """Test that high artifact level reduces confidence."""
        alert = generate_alert(
            timestamp=10.0,
            delta_phi=3.0,
            delta_S=1.0,
            delta_I=1.0,
            delta_C=1.0,
            artifact_level=0.8  # High artifacts
        )
        
        assert alert.confidence < 0.5


class TestAlertEventDetection:
    """Tests for alert event detection."""
    
    def test_detect_alert_events(self):
        """Test alert event detection."""
        timestamps = np.linspace(0, 100, 100)
        delta_phi_series = np.random.rand(100) * 2
        gate_series = np.zeros(100, dtype=int)
        
        # Create alert period
        gate_series[40:50] = 1
        
        delta_S_series = np.random.rand(100)
        delta_I_series = np.random.rand(100)
        delta_C_series = np.random.rand(100)
        
        alerts = detect_alert_events(
            timestamps, delta_phi_series, gate_series,
            delta_S_series, delta_I_series, delta_C_series
        )
        
        assert len(alerts) > 0
        # Check first alert
        alert = alerts[0]
        assert 40 <= alert.timestamp <= 50


class TestAblationModes:
    """Tests for ablation mode functions."""
    
    def test_ablation_eeg_only(self):
        """Test EEG-only ablation."""
        delta_S = 1.0
        delta_I = 0.5
        
        delta_phi = ablation_eeg_only(delta_S, delta_I, alpha=0.6, beta=0.4)
        
        # Should be weighted sum: 0.6*1.0 + 0.4*0.5 = 0.8
        expected = 0.6 * 1.0 + 0.4 * 0.5
        assert np.isclose(delta_phi, expected)
    
    def test_ablation_ecg_only(self):
        """Test ECG-only ablation."""
        delta_S = 1.0
        delta_I = 0.5
        
        delta_phi = ablation_ecg_only(delta_S, delta_I, alpha=0.6, beta=0.4)
        
        expected = 0.6 * 1.0 + 0.4 * 0.5
        assert np.isclose(delta_phi, expected)


class TestRiskScore:
    """Tests for risk score computation."""
    
    def test_risk_score_below_threshold(self):
        """Test risk score below threshold."""
        delta_phi = 1.0
        threshold = 2.5
        
        risk = risk_score(delta_phi, threshold, scale='linear')
        
        assert 0 <= risk <= 1.0
        # Should be low risk
        assert risk < 0.5
    
    def test_risk_score_above_threshold(self):
        """Test risk score above threshold."""
        delta_phi = 4.0
        threshold = 2.5
        
        risk = risk_score(delta_phi, threshold, scale='linear')
        
        assert 0 <= risk <= 1.0
        # Should be high risk
        assert risk > 0.5
    
    def test_risk_score_sigmoid(self):
        """Test sigmoid risk scoring."""
        delta_phi = 2.5
        threshold = 2.5
        
        risk = risk_score(delta_phi, threshold, scale='sigmoid')
        
        assert 0 <= risk <= 1.0
        # At threshold, sigmoid should be ~0.5
        assert 0.4 < risk < 0.6


class TestAlertFormatting:
    """Tests for alert message formatting."""
    
    def test_format_alert_message(self):
        """Test alert message formatting."""
        alert = generate_alert(
            timestamp=10.5,
            delta_phi=3.2,
            delta_S=1.5,
            delta_I=0.8,
            delta_C=1.0
        )
        
        message = format_alert_message(alert)
        
        assert isinstance(message, str)
        assert 'INSTABILITY ALERT' in message
        assert '10.5' in message
        assert '3.2' in message


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
