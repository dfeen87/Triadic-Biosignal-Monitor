"""
Tests for phase extraction module.

Tests Hilbert transform, phase extraction, unwrapping, derivatives, and triadic embedding.
"""

import pytest
import numpy as np
from core.phase import (
    analytic_signal,
    extract_phase,
    unwrap_phase,
    phase_derivative,
    phase_derivative_savgol,
    phase_derivative_diff,
    triadic_embedding,
    instantaneous_frequency,
    phase_locking_value,
    check_phase_quality
)


class TestAnalyticSignal:
    """Tests for analytic_signal function."""
    
    def test_basic_analytic_signal(self):
        """Test analytic signal computation."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        
        analytic = analytic_signal(signal, fs)
        
        assert len(analytic) == len(signal)
        assert analytic.dtype == complex
        # Real part should be similar to original
        assert np.allclose(np.real(analytic), signal, atol=1e-10)
    
    def test_short_signal_error(self):
        """Test that very short signals raise error."""
        signal = np.array([1, 2, 3])
        fs = 256.0
        
        with pytest.raises(ValueError):
            analytic_signal(signal, fs)


class TestPhaseExtraction:
    """Tests for phase extraction."""
    
    def test_extract_phase(self):
        """Test phase extraction from sinusoid."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        freq = 10.0
        signal = np.sin(2 * np.pi * freq * t)
        
        phase = extract_phase(signal, fs, unwrap=True)
        
        assert len(phase) == len(signal)
        assert not np.any(np.isnan(phase))
        # Phase should increase monotonically after unwrapping
        phase_diff = np.diff(phase)
        assert np.all(phase_diff >= -0.1)  # Allow small numerical errors
    
    def test_phase_with_bandpass(self):
        """Test phase extraction with bandpass filtering."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        signal += 0.5 * np.sin(2 * np.pi * 50 * t)  # High freq component
        
        phase = extract_phase(signal, fs, bandpass=(8, 13))
        
        assert len(phase) == len(signal)
        assert not np.any(np.isnan(phase))


class TestPhaseUnwrapping:
    """Tests for phase unwrapping."""
    
    def test_unwrap_phase(self):
        """Test phase unwrapping removes discontinuities."""
        # Create phase with 2π jumps
        phase = np.array([0, np.pi/2, np.pi, -np.pi/2, 0])
        
        unwrapped = unwrap_phase(phase)
        
        # Should be monotonic after unwrapping
        assert len(unwrapped) == len(phase)
        # Check no large jumps
        diff = np.abs(np.diff(unwrapped))
        assert np.all(diff < 2 * np.pi)


class TestPhaseDerivative:
    """Tests for phase derivative computation."""
    
    def test_phase_derivative_savgol(self):
        """Test Savitzky-Golay derivative."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        # Linear phase (constant frequency)
        freq = 10.0
        phase = 2 * np.pi * freq * np.arange(n_samples) / fs
        
        chi = phase_derivative_savgol(phase, fs)
        
        assert len(chi) == len(phase)
        assert not np.any(np.isnan(chi))
        # Should be approximately 2π * freq
        mean_chi = np.mean(chi)
        expected = 2 * np.pi * freq
        assert np.abs(mean_chi - expected) < 1.0  # Allow some error
    
    def test_phase_derivative_diff(self):
        """Test finite difference derivative."""
        fs = 256.0
        phase = np.linspace(0, 10 * 2 * np.pi, 1000)
        
        chi = phase_derivative_diff(phase, fs, method='central')
        
        assert len(chi) == len(phase)
        assert not np.any(np.isnan(chi))
    
    def test_phase_derivative_methods(self):
        """Test that different methods give similar results."""
        fs = 256.0
        phase = np.linspace(0, 10 * 2 * np.pi, 1000)
        
        chi_savgol = phase_derivative(phase, fs, method='savgol')
        chi_diff = phase_derivative(phase, fs, method='diff')
        
        # Should be roughly similar (not exact due to different methods)
        correlation = np.corrcoef(chi_savgol, chi_diff)[0, 1]
        assert correlation > 0.9


class TestTriadicEmbedding:
    """Tests for triadic embedding."""
    
    def test_triadic_embedding(self):
        """Test complete triadic embedding."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        
        embedding = triadic_embedding(signal, fs)
        
        # Check all components present
        assert 't' in embedding
        assert 'phi' in embedding
        assert 'chi' in embedding
        assert 'amplitude' in embedding
        
        # Check lengths
        assert len(embedding['t']) == n_samples
        assert len(embedding['phi']) == n_samples
        assert len(embedding['chi']) == n_samples
        assert len(embedding['amplitude']) == n_samples
        
        # Check no NaN values
        assert not np.any(np.isnan(embedding['phi']))
        assert not np.any(np.isnan(embedding['chi']))


class TestInstantaneousFrequency:
    """Tests for instantaneous frequency."""
    
    def test_constant_frequency(self):
        """Test instantaneous frequency for constant frequency signal."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        freq = 10.0
        t = np.arange(n_samples) / fs
        phase = 2 * np.pi * freq * t
        
        inst_freq = instantaneous_frequency(phase, fs)
        
        assert len(inst_freq) == len(phase)
        # Mean frequency should be close to 10 Hz
        mean_freq = np.mean(inst_freq)
        assert np.abs(mean_freq - freq) < 0.5


class TestPhaseLockingValue:
    """Tests for phase locking value."""
    
    def test_perfectly_locked_phases(self):
        """Test PLV for perfectly locked phases."""
        phase1 = np.linspace(0, 10 * 2 * np.pi, 1000)
        phase2 = phase1 + np.pi / 4  # Constant phase offset
        
        plv = phase_locking_value(phase1, phase2)
        
        # Should be 1.0 for perfect locking
        assert 0.99 < plv <= 1.0
    
    def test_random_phases(self):
        """Test PLV for random phases."""
        phase1 = np.random.uniform(0, 2 * np.pi, 1000)
        phase2 = np.random.uniform(0, 2 * np.pi, 1000)
        
        plv = phase_locking_value(phase1, phase2)
        
        # Should be close to 0 for random phases
        assert plv < 0.2
    
    def test_windowed_plv(self):
        """Test windowed PLV computation."""
        phase1 = np.linspace(0, 10 * 2 * np.pi, 1000)
        phase2 = phase1 + 0.1
        
        plv = phase_locking_value(phase1, phase2, window_size=100)
        
        # Should still be high
        assert plv > 0.9


class TestPhaseQuality:
    """Tests for phase quality checks."""
    
    def test_good_phase_quality(self):
        """Test quality check for good phase."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        # Generate good phase
        freq = 10.0
        phase = 2 * np.pi * freq * np.arange(n_samples) / fs
        chi = phase_derivative(phase, fs)
        
        quality_score, flags = check_phase_quality(phase, chi, fs)
        
        assert 0 <= quality_score <= 1.0
        assert flags['no_nan']
        assert flags['no_inf']
        assert flags['continuous_phase']
        assert flags['reasonable_derivative']
    
    def test_detect_nan_phase(self):
        """Test detection of NaN in phase."""
        phase = np.array([0, 1, 2, np.nan, 4])
        chi = np.array([1, 1, 1, 1, 1])
        fs = 256.0
        
        quality_score, flags = check_phase_quality(phase, chi, fs)
        
        assert not flags['no_nan']
        assert quality_score < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
