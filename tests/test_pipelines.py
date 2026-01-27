"""
Integration tests for pipelines.

Tests EEG-only, ECG-only, coupled, and streaming pipelines end-to-end.
"""

import pytest
import numpy as np
from pipelines.eeg_only import EEGOnlyPipeline, run_eeg_only_pipeline
from pipelines.ecg_only import ECGOnlyPipeline, run_ecg_only_pipeline
from pipelines.coupled import CoupledPipeline, run_coupled_pipeline
from pipelines.streaming import StreamingPipeline
from core.gate import InstabilityConfig
from datasets.synthetic import generate_synthetic_eeg, generate_synthetic_ecg


class TestEEGOnlyPipeline:
    """Tests for EEG-only pipeline."""
    
    @pytest.fixture
    def eeg_signals(self):
        """Generate synthetic EEG signals for testing."""
        fs = 256.0
        baseline = generate_synthetic_eeg(60.0, fs)
        signal = generate_synthetic_eeg(120.0, fs)
        return baseline, signal, fs
    
    def test_pipeline_initialization(self):
        """Test EEG-only pipeline initialization."""
        fs = 256.0
        config = InstabilityConfig(alpha=0.6, beta=0.4, gamma=0.0, threshold=2.5)
        
        pipeline = EEGOnlyPipeline(fs=fs, baseline_duration=60.0, config=config)
        
        assert pipeline.fs == fs
        assert pipeline.baseline_duration == 60.0
        assert pipeline.config.gamma == 0.0  # EEG-only should have no coupling
    
    def test_set_baseline(self, eeg_signals):
        """Test baseline setting."""
        baseline, signal, fs = eeg_signals
        
        pipeline = EEGOnlyPipeline(fs=fs, baseline_duration=60.0)
        baseline_metrics = pipeline.set_baseline(baseline)
        
        assert 'quality_score' in baseline_metrics
        assert pipeline.baseline_eeg is not None
    
    def test_process_window(self, eeg_signals):
        """Test single window processing."""
        baseline, signal, fs = eeg_signals
        
        pipeline = EEGOnlyPipeline(fs=fs, baseline_duration=60.0)
        pipeline.set_baseline(baseline)
        
        # Process 10-second window
        window = signal[:int(10 * fs)]
        result = pipeline.process_window(window)
        
        assert 'delta_phi' in result
        assert 'gate' in result
        assert 'delta_S' in result
        assert 'delta_I' in result
        assert result['delta_C'] == 0.0  # EEG-only has no coupling
        assert result['gate'] in [0, 1]
    
    def test_process_continuous(self, eeg_signals):
        """Test continuous signal processing."""
        baseline, signal, fs = eeg_signals
        
        pipeline = EEGOnlyPipeline(fs=fs, baseline_duration=60.0)
        pipeline.set_baseline(baseline)
        
        results = pipeline.process_continuous(signal, window_size=10.0, overlap=0.5)
        
        assert 'timestamps' in results
        assert 'delta_phi' in results
        assert 'gate' in results
        assert 'alerts' in results
        
        # Check lengths match
        n_windows = len(results['timestamps'])
        assert len(results['delta_phi']) == n_windows
        assert len(results['gate']) == n_windows
    
    def test_run_eeg_only_pipeline(self, eeg_signals):
        """Test convenience function."""
        baseline, signal, fs = eeg_signals
        
        config = {'alpha': 0.6, 'beta': 0.4, 'threshold': 2.5}
        
        results = run_eeg_only_pipeline(signal, baseline, fs, config)
        
        assert 'timestamps' in results
        assert 'config' in results
        assert 'baseline_metrics' in results


class TestECGOnlyPipeline:
    """Tests for ECG-only pipeline."""
    
    @pytest.fixture
    def ecg_signals(self):
        """Generate synthetic ECG signals for testing."""
        fs = 256.0
        baseline = generate_synthetic_ecg(60.0, fs, heart_rate=70.0)
        signal = generate_synthetic_ecg(120.0, fs, heart_rate=75.0)
        return baseline, signal, fs
    
    def test_pipeline_initialization(self):
        """Test ECG-only pipeline initialization."""
        fs = 256.0
        config = InstabilityConfig(alpha=0.6, beta=0.4, gamma=0.0, threshold=2.5)
        
        pipeline = ECGOnlyPipeline(fs=fs, baseline_duration=60.0, config=config)
        
        assert pipeline.fs == fs
        assert pipeline.config.gamma == 0.0
    
    def test_rr_extraction(self, ecg_signals):
        """Test RR interval extraction."""
        baseline, signal, fs = ecg_signals
        
        pipeline = ECGOnlyPipeline(fs=fs, baseline_duration=60.0)
        rr_intervals = pipeline.extract_rr_intervals(signal)
        
        assert len(rr_intervals) > 0
        # RR intervals should be in reasonable range (0.5-1.5 seconds)
        assert np.all((rr_intervals > 0.4) & (rr_intervals < 2.0))
    
    def test_process_window(self, ecg_signals):
        """Test single window processing."""
        baseline, signal, fs = ecg_signals
        
        pipeline = ECGOnlyPipeline(fs=fs, baseline_duration=60.0)
        pipeline.set_baseline(baseline)
        
        window = signal[:int(10 * fs)]
        result = pipeline.process_window(window)
        
        assert 'delta_phi' in result
        assert 'gate' in result
        assert result['delta_C'] == 0.0  # ECG-only has no coupling


class TestCoupledPipeline:
    """Tests for full coupled pipeline."""
    
    @pytest.fixture
    def coupled_signals(self):
        """Generate synthetic coupled signals."""
        fs = 256.0
        eeg_baseline = generate_synthetic_eeg(60.0, fs)
        ecg_baseline = generate_synthetic_ecg(60.0, fs)
        eeg_signal = generate_synthetic_eeg(120.0, fs)
        ecg_signal = generate_synthetic_ecg(120.0, fs)
        return eeg_baseline, ecg_baseline, eeg_signal, ecg_signal, fs
    
    def test_pipeline_initialization(self):
        """Test coupled pipeline initialization."""
        fs = 256.0
        config = InstabilityConfig(alpha=0.4, beta=0.3, gamma=0.3, threshold=2.5)
        
        pipeline = CoupledPipeline(fs=fs, baseline_duration=60.0, config=config)
        
        assert pipeline.fs == fs
        assert pipeline.config.gamma == 0.3  # Coupled should have coupling term
    
    def test_set_baseline(self, coupled_signals):
        """Test baseline setting for coupled mode."""
        eeg_baseline, ecg_baseline, _, _, fs = coupled_signals
        
        pipeline = CoupledPipeline(fs=fs, baseline_duration=60.0)
        baseline_metrics = pipeline.set_baseline(eeg_baseline, ecg_baseline)
        
        assert 'eeg_quality_score' in baseline_metrics
        assert 'ecg_quality_score' in baseline_metrics
        assert pipeline.baseline_eeg is not None
        assert pipeline.baseline_ecg is not None
    
    def test_process_window(self, coupled_signals):
        """Test coupled window processing."""
        eeg_baseline, ecg_baseline, eeg_signal, ecg_signal, fs = coupled_signals
        
        pipeline = CoupledPipeline(fs=fs, baseline_duration=60.0)
        pipeline.set_baseline(eeg_baseline, ecg_baseline)
        
        eeg_window = eeg_signal[:int(10 * fs)]
        ecg_window = ecg_signal[:int(10 * fs)]
        
        result = pipeline.process_window(eeg_window, ecg_window)
        
        assert 'delta_phi' in result
        assert 'gate' in result
        assert 'delta_S' in result
        assert 'delta_I' in result
        assert 'delta_C' in result  # Should have coupling term
        assert result['delta_C'] >= 0  # Coupling deviation should be non-negative
    
    def test_process_continuous(self, coupled_signals):
        """Test continuous coupled processing."""
        eeg_baseline, ecg_baseline, eeg_signal, ecg_signal, fs = coupled_signals
        
        pipeline = CoupledPipeline(fs=fs, baseline_duration=60.0)
        pipeline.set_baseline(eeg_baseline, ecg_baseline)
        
        results = pipeline.process_continuous(eeg_signal, ecg_signal, window_size=10.0)
        
        assert 'timestamps' in results
        assert 'delta_phi' in results
        assert 'delta_S' in results
        assert 'delta_I' in results
        assert 'delta_C' in results
        assert 'alerts' in results
    
    def test_run_coupled_pipeline(self, coupled_signals):
        """Test coupled pipeline convenience function."""
        eeg_baseline, ecg_baseline, eeg_signal, ecg_signal, fs = coupled_signals
        
        config = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3, 'threshold': 2.5}
        
        results = run_coupled_pipeline(
            eeg_signal, ecg_signal,
            eeg_baseline, ecg_baseline,
            fs, config
        )
        
        assert 'timestamps' in results
        assert 'delta_C' in results
        assert len(results['delta_C']) > 0


class TestStreamingPipeline:
    """Tests for streaming pipeline."""
    
    def test_streaming_initialization(self):
        """Test streaming pipeline initialization."""
        fs = 256.0
        
        pipeline = StreamingPipeline(
            fs=fs,
            buffer_duration=10.0,
            baseline_duration=60.0,
            mode='coupled'
        )
        
        assert pipeline.fs == fs
        assert pipeline.buffer_duration == 10.0
        assert pipeline.mode == 'coupled'
    
    def test_buffer_management(self):
        """Test buffer push and ready check."""
        fs = 256.0
        pipeline = StreamingPipeline(fs=fs, buffer_duration=2.0, mode='eeg_only')
        
        # Initially not ready
        assert not pipeline.is_ready()
        
        # Push samples
        samples = np.random.randn(int(2.5 * fs))
        pipeline.push_samples(samples)
        
        # Should be ready now
        assert pipeline.is_ready()
    
    def test_streaming_eeg_only(self):
        """Test streaming in EEG-only mode."""
        fs = 256.0
        pipeline = StreamingPipeline(fs=fs, buffer_duration=5.0, mode='eeg_only')
        
        # Set baseline
        baseline = generate_synthetic_eeg(60.0, fs)
        pipeline.set_baseline(baseline)
        
        # Push data
        signal = generate_synthetic_eeg(10.0, fs)
        pipeline.push_samples(signal)
        
        # Process
        result = pipeline.process_buffer()
        
        if result is not None:  # May be None if buffer not full
            assert 'delta_phi' in result
            assert 'gate' in result
    
    def test_graceful_degradation(self):
        """Test graceful degradation with poor quality."""
        fs = 256.0
        pipeline = StreamingPipeline(fs=fs, buffer_duration=5.0, mode='coupled')
        
        # Set baseline
        eeg_baseline = generate_synthetic_eeg(60.0, fs)
        ecg_baseline = generate_synthetic_ecg(60.0, fs)
        pipeline.set_baseline(eeg_baseline, ecg_baseline)
        
        # Push very noisy data (poor quality)
        eeg_signal = np.random.randn(int(5 * fs)) * 100  # Very noisy
        ecg_signal = np.random.randn(int(5 * fs)) * 100
        
        pipeline.push_samples(eeg_signal, ecg_signal)
        
        # Should handle gracefully
        result = pipeline.process_buffer()
        
        # Should return result with no_decision or fall back to single modality
        assert result is None or 'no_decision' in result
    
    def test_latency_tracking(self):
        """Test latency statistics tracking."""
        fs = 256.0
        pipeline = StreamingPipeline(fs=fs, buffer_duration=5.0, mode='eeg_only')
        
        baseline = generate_synthetic_eeg(60.0, fs)
        pipeline.set_baseline(baseline)
        
        # Process multiple buffers
        for _ in range(3):
            signal = generate_synthetic_eeg(5.0, fs)
            pipeline.push_samples(signal)
            pipeline.process_buffer()
        
        stats = pipeline.get_latency_stats()
        
        if stats:  # May be empty if no processing occurred
            assert 'mean_ms' in stats
            assert stats['mean_ms'] >= 0


class TestPipelineIntegration:
    """Integration tests comparing all pipelines."""
    
    def test_all_pipelines_produce_results(self):
        """Test that all pipelines produce valid results."""
        fs = 256.0
        duration = 60.0
        
        # Generate data
        eeg_baseline = generate_synthetic_eeg(60.0, fs)
        ecg_baseline = generate_synthetic_ecg(60.0, fs)
        eeg_signal = generate_synthetic_eeg(duration, fs)
        ecg_signal = generate_synthetic_ecg(duration, fs)
        
        config = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3, 'threshold': 2.5}
        
        # Run all pipelines
        results_eeg = run_eeg_only_pipeline(eeg_signal, eeg_baseline, fs, config)
        results_ecg = run_ecg_only_pipeline(ecg_signal, ecg_baseline, fs, config)
        results_coupled = run_coupled_pipeline(
            eeg_signal, ecg_signal, eeg_baseline, ecg_baseline, fs, config
        )
        
        # All should produce valid results
        for results in [results_eeg, results_ecg, results_coupled]:
            assert 'timestamps' in results
            assert 'delta_phi' in results
            assert len(results['timestamps']) > 0
    
    def test_coupled_uses_all_terms(self):
        """Test that coupled pipeline uses all three deviation terms."""
        fs = 256.0
        
        eeg_baseline = generate_synthetic_eeg(60.0, fs)
        ecg_baseline = generate_synthetic_ecg(60.0, fs)
        eeg_signal = generate_synthetic_eeg(60.0, fs)
        ecg_signal = generate_synthetic_ecg(60.0, fs)
        
        results = run_coupled_pipeline(
            eeg_signal, ecg_signal,
            eeg_baseline, ecg_baseline,
            fs, config={'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3, 'threshold': 2.5}
        )
        
        # Check that all three terms are computed
        assert 'delta_S' in results
        assert 'delta_I' in results
        assert 'delta_C' in results
        
        # All should be non-negative
        assert np.all(results['delta_S'] >= 0)
        assert np.all(results['delta_I'] >= 0)
        assert np.all(results['delta_C'] >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
