"""
Tests for preprocessing module.

Tests bandpass filtering, artifact detection, quality checks, and signal synchronization.
"""

import pytest
import numpy as np
from core.preprocessing import (
    bandpass_filter,
    notch_filter,
    detect_artifacts_threshold,
    quality_check,
    synchronize_signals,
    remove_baseline_drift,
    preprocess_eeg,
    preprocess_ecg
)


class TestBandpassFilter:
    """Tests for bandpass_filter function."""
    
    def test_basic_filtering(self):
        """Test basic bandpass filtering."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        # Create signal with multiple frequencies
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 1 * t)   # 1 Hz (should be removed)
        signal += np.sin(2 * np.pi * 10 * t)  # 10 Hz (should pass)
        signal += np.sin(2 * np.pi * 60 * t)  # 60 Hz (should be removed)
        
        # Bandpass 5-50 Hz
        filtered = bandpass_filter(signal, fs, lowcut=5.0, highcut=50.0)
        
        assert len(filtered) == len(signal)
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        signal = np.random.randn(1000)
        fs = 256.0
        
        # Invalid cutoff frequencies
        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, lowcut=-1, highcut=50)
        
        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, lowcut=50, highcut=10)
        
        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, lowcut=10, highcut=200)  # > Nyquist


class TestNotchFilter:
    """Tests for notch_filter function."""
    
    def test_notch_filtering(self):
        """Test notch filter removes specific frequency."""
        fs = 256.0
        duration = 5.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz
        signal += np.sin(2 * np.pi * 60 * t)  # 60 Hz (line noise)
        
        filtered = notch_filter(signal, fs, freq=60.0)
        
        assert len(filtered) == len(signal)
        # 60 Hz component should be greatly reduced
        # (We won't check exact attenuation, just that it runs)


class TestArtifactDetection:
    """Tests for artifact detection."""
    
    def test_clean_signal(self):
        """Test that clean signal has few artifacts."""
        fs = 256.0
        signal = np.random.randn(int(5 * fs)) * 0.5  # Low amplitude noise
        
        artifact_mask = detect_artifacts_threshold(signal, fs, threshold=5.0)
        
        # Clean signal should have < 10% artifacts
        artifact_fraction = np.mean(artifact_mask)
        assert artifact_fraction < 0.1
    
    def test_artifact_detection(self):
        """Test that large spikes are detected."""
        fs = 256.0
        signal = np.random.randn(int(5 * fs)) * 0.5
        
        # Add large spike
        signal[int(2.5 * fs)] = 50.0
        
        artifact_mask = detect_artifacts_threshold(signal, fs, threshold=5.0)
        
        # Spike should be detected
        assert artifact_mask[int(2.5 * fs)]


class TestQualityCheck:
    """Tests for quality_check function."""
    
    def test_good_quality_signal(self):
        """Test that good signal passes quality check."""
        fs = 256.0
        signal = np.random.randn(int(10 * fs))
        
        quality_score, flags = quality_check(signal, fs)
        
        assert 0 <= quality_score <= 1.0
        assert flags['sufficient_length']
        assert flags['no_flatline']
        assert flags['adequate_variance']
    
    def test_flatline_detection(self):
        """Test that flatline is detected."""
        fs = 256.0
        signal = np.ones(int(10 * fs))  # Constant signal
        
        quality_score, flags = quality_check(signal, fs)
        
        assert not flags['no_flatline']
        assert quality_score < 1.0
    
    def test_short_signal(self):
        """Test that short signal is flagged."""
        fs = 256.0
        signal = np.random.randn(int(2 * fs))  # Only 2 seconds
        
        quality_score, flags = quality_check(signal, fs, min_length=5.0)
        
        assert not flags['sufficient_length']


class TestSignalSynchronization:
    """Tests for synchronize_signals function."""
    
    def test_synchronization(self):
        """Test signal synchronization."""
        fs = 256.0
        duration = 10.0
        
        # Create two signals with different timestamps
        n1 = int(duration * fs)
        n2 = int(duration * fs * 1.1)  # Slightly different length
        
        sig1 = np.random.randn(n1)
        sig2 = np.random.randn(n2)
        
        ts1 = np.arange(n1) / fs
        ts2 = np.arange(n2) / (fs * 1.1)
        
        sig1_sync, sig2_sync, ts_common = synchronize_signals(
            sig1, sig2, ts1, ts2
        )
        
        # Synchronized signals should have same length
        assert len(sig1_sync) == len(sig2_sync)
        assert len(sig1_sync) == len(ts_common)
        assert not np.any(np.isnan(sig1_sync))
        assert not np.any(np.isnan(sig2_sync))


class TestBaselineDriftRemoval:
    """Tests for remove_baseline_drift function."""
    
    def test_drift_removal(self):
        """Test baseline drift removal."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        t = np.arange(n_samples) / fs
        # Signal with drift
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * t  # Linear drift
        
        corrected = remove_baseline_drift(signal, fs, method='detrend')
        
        assert len(corrected) == len(signal)
        # Mean should be closer to zero after drift removal
        assert abs(np.mean(corrected)) < abs(np.mean(signal))


class TestPreprocessingPipelines:
    """Tests for complete preprocessing pipelines."""
    
    def test_preprocess_eeg(self):
        """Test EEG preprocessing pipeline."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        # Generate synthetic EEG
        t = np.arange(n_samples) / fs
        signal = np.sin(2 * np.pi * 10 * t)  # Alpha rhythm
        signal += 0.1 * np.random.randn(n_samples)
        
        processed, artifact_mask = preprocess_eeg(signal, fs)
        
        assert len(processed) == len(signal)
        assert len(artifact_mask) == len(signal)
        assert not np.any(np.isnan(processed))
        assert artifact_mask.dtype == bool
    
    def test_preprocess_ecg(self):
        """Test ECG preprocessing pipeline."""
        fs = 256.0
        duration = 10.0
        n_samples = int(duration * fs)
        
        # Generate synthetic ECG
        signal = np.random.randn(n_samples) * 0.1
        
        # Add QRS complexes
        for i in range(0, n_samples, int(0.8 * fs)):  # ~75 BPM
            if i + 50 < n_samples:
                signal[i:i+50] += np.random.randn(50)
        
        processed, artifact_mask = preprocess_ecg(signal, fs)
        
        assert len(processed) == len(signal)
        assert len(artifact_mask) == len(signal)
        assert not np.any(np.isnan(processed))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
