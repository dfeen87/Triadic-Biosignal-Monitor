"""
Tests for feature extraction module.

Tests computation of ΔS, ΔI, and ΔC features.
"""

import pytest
import numpy as np
from core.features import (
    compute_bandpower,
    compute_eeg_bandpowers,
    compute_spectral_centroid,
    compute_hrv_metrics,
    compute_lf_hf_ratio,
    compute_delta_S_eeg,
    compute_delta_S_ecg,
    permutation_entropy,
    sample_entropy,
    compute_delta_I,
    magnitude_squared_coherence,
    phase_locking_value,
    compute_delta_C
)


class TestBandpower:
    """Tests for bandpower computation."""
    
    def test_compute_bandpower(self):
        """Test bandpower computation."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        # Create signal with specific frequency
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        
        # Compute power in alpha band (8-13 Hz)
        power = compute_bandpower(signal, fs, (8, 13))
        
        assert power > 0
        assert not np.isnan(power)
        assert not np.isinf(power)


class TestEEGBandpowers:
    """Tests for EEG bandpower computation."""
    
    def test_compute_eeg_bandpowers(self):
        """Test EEG bandpower computation for all bands."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)  # Alpha frequency
        
        bandpowers = compute_eeg_bandpowers(signal, fs)
        
        # Check all standard bands present
        assert 'delta' in bandpowers
        assert 'theta' in bandpowers
        assert 'alpha' in bandpowers
        assert 'beta' in bandpowers
        assert 'gamma' in bandpowers
        
        # All values should be positive
        for band, power in bandpowers.items():
            assert power > 0


class TestSpectralCentroid:
    """Tests for spectral centroid."""
    
    def test_spectral_centroid(self):
        """Test spectral centroid computation."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 20 * t)  # 20 Hz signal
        
        centroid = compute_spectral_centroid(signal, fs)
        
        assert centroid > 0
        # Centroid should be close to 20 Hz
        assert 15 < centroid < 25


class TestHRVMetrics:
    """Tests for HRV metrics."""
    
    def test_hrv_metrics(self):
        """Test HRV metrics computation."""
        # Simulate RR intervals (in seconds)
        mean_rr = 0.8  # 75 BPM
        rr_intervals = np.random.normal(mean_rr, 0.05, 100)
        rr_intervals = np.clip(rr_intervals, 0.5, 1.5)  # Keep realistic
        
        metrics = compute_hrv_metrics(rr_intervals)
        
        assert 'RMSSD' in metrics
        assert 'SDNN' in metrics
        assert 'pNN50' in metrics
        
        # All should be non-negative
        assert metrics['RMSSD'] >= 0
        assert metrics['SDNN'] >= 0
        assert 0 <= metrics['pNN50'] <= 100


class TestLFHFRatio:
    """Tests for LF/HF ratio."""
    
    def test_lf_hf_ratio(self):
        """Test LF/HF ratio computation."""
        # Simulate RR intervals
        mean_rr = 0.8
        rr_intervals = np.random.normal(mean_rr, 0.05, 200)
        rr_intervals = np.clip(rr_intervals, 0.5, 1.5)
        
        lf_hf = compute_lf_hf_ratio(rr_intervals, fs_rr=4.0)
        
        assert lf_hf > 0
        assert not np.isnan(lf_hf)
        # Typical range is 0.5 to 5.0
        assert 0.1 < lf_hf < 10.0


class TestDeltaS:
    """Tests for ΔS computation."""
    
    def test_delta_S_eeg(self):
        """Test ΔS computation for EEG."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        baseline = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(n_samples)
        
        # Current signal with frequency shift
        current = np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(n_samples)
        
        delta_S = compute_delta_S_eeg(current, baseline, fs)
        
        assert delta_S >= 0
        assert not np.isnan(delta_S)
        # Should detect frequency shift
        assert delta_S > 0.01
    
    def test_delta_S_ecg(self):
        """Test ΔS computation for ECG/HRV."""
        # Baseline RR intervals
        baseline_rr = np.random.normal(0.8, 0.05, 100)
        baseline_rr = np.clip(baseline_rr, 0.5, 1.5)
        
        # Current RR intervals with change
        current_rr = np.random.normal(0.7, 0.08, 100)  # Higher HR, more variable
        current_rr = np.clip(current_rr, 0.5, 1.5)
        
        delta_S = compute_delta_S_ecg(current_rr, baseline_rr, fs_rr=4.0)
        
        assert delta_S >= 0
        assert not np.isnan(delta_S)


class TestPermutationEntropy:
    """Tests for permutation entropy."""
    
    def test_permutation_entropy(self):
        """Test permutation entropy computation."""
        # Regular signal (low entropy)
        signal_regular = np.sin(np.linspace(0, 10 * 2 * np.pi, 1000))
        
        # Random signal (high entropy)
        signal_random = np.random.randn(1000)
        
        pe_regular = permutation_entropy(signal_regular, order=3, normalize=True)
        pe_random = permutation_entropy(signal_random, order=3, normalize=True)
        
        assert 0 <= pe_regular <= 1.0
        assert 0 <= pe_random <= 1.0
        # Random should have higher entropy
        assert pe_random > pe_regular


class TestSampleEntropy:
    """Tests for sample entropy."""
    
    def test_sample_entropy(self):
        """Test sample entropy computation."""
        # Regular signal
        signal_regular = np.sin(np.linspace(0, 10 * 2 * np.pi, 1000))
        
        # Random signal
        signal_random = np.random.randn(1000)
        
        se_regular = sample_entropy(signal_regular, m=2)
        se_random = sample_entropy(signal_random, m=2)
        
        assert se_regular >= 0
        assert se_random >= 0
        # Random should have higher entropy
        assert se_random > se_regular


class TestDeltaI:
    """Tests for ΔI computation."""
    
    def test_delta_I(self):
        """Test ΔI computation."""
        # Baseline: random signal
        baseline = np.random.randn(1000)
        
        # Current: more regular signal (lower entropy)
        t = np.linspace(0, 10 * 2 * np.pi, 1000)
        current = np.sin(t) + 0.1 * np.random.randn(1000)
        
        delta_I = compute_delta_I(current, baseline, method='permutation_entropy')
        
        assert delta_I >= 0
        assert not np.isnan(delta_I)
        # Should detect entropy change
        assert delta_I > 0.01


class TestCoherence:
    """Tests for coherence computation."""
    
    def test_magnitude_squared_coherence(self):
        """Test magnitude-squared coherence."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        # Two correlated signals
        signal1 = np.sin(2 * np.pi * 10 * t)
        signal2 = signal1 + 0.1 * np.random.randn(n_samples)
        
        coherence = magnitude_squared_coherence(signal1, signal2, fs)
        
        assert 'mean' in coherence
        assert 0 <= coherence['mean'] <= 1.0
        # Should have high coherence
        assert coherence['mean'] > 0.5


class TestPhaseLockingValueFeatures:
    """Tests for PLV in features module."""
    
    def test_plv(self):
        """Test phase locking value."""
        # Two locked phases
        phase1 = np.linspace(0, 10 * 2 * np.pi, 1000)
        phase2 = phase1 + 0.1
        
        plv = phase_locking_value(phase1, phase2)
        
        assert 0 <= plv <= 1.0
        # Should be high
        assert plv > 0.9


class TestDeltaC:
    """Tests for ΔC computation."""
    
    def test_delta_C(self):
        """Test ΔC computation."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        
        # Baseline: highly correlated signals
        baseline_sig1 = np.sin(2 * np.pi * 10 * t)
        baseline_sig2 = baseline_sig1 + 0.1 * np.random.randn(n_samples)
        
        # Current: less correlated signals
        current_sig1 = np.sin(2 * np.pi * 10 * t)
        current_sig2 = np.sin(2 * np.pi * 15 * t) + 0.5 * np.random.randn(n_samples)
        
        delta_C = compute_delta_C(
            current_sig1, current_sig2,
            baseline_sig1, baseline_sig2,
            fs, method='coherence'
        )
        
        assert delta_C >= 0
        assert not np.isnan(delta_C)
        # Should detect decoupling
        assert delta_C > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
