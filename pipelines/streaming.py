"""
Streaming Pipeline for Triadic Biosignal Monitor (Phase 2)

This pipeline implements near-real-time processing with:
- Sliding buffer management
- Latency-aware processing
- Graceful degradation under signal loss
- Resource-aware operation

This is a Phase 2 feature for eventual deployment scenarios.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple, Deque
from collections import deque
import warnings
import time

from core.preprocessing import preprocess_eeg, preprocess_ecg, quality_check
from core.phase import extract_phase
from core.features import (
    compute_delta_S_eeg,
    compute_delta_S_ecg,
    compute_delta_I,
    compute_delta_C
)
from core.gate import (
    InstabilityConfig,
    compute_instability_functional,
    decision_gate,
    generate_alert,
    AlertEvent
)


class StreamingPipeline:
    """
    Near-real-time streaming pipeline for coupled EEG-ECG monitoring.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    buffer_duration : float
        Buffer size in seconds (default: 10.0)
    baseline_duration : float
        Baseline window duration in seconds
    config : InstabilityConfig, optional
        Instability detection configuration
    mode : str, optional
        Processing mode: 'eeg_only', 'ecg_only', or 'coupled' (default: 'coupled')
    """
    
    def __init__(
        self,
        fs: float,
        buffer_duration: float = 10.0,
        baseline_duration: float = 60.0,
        config: Optional[InstabilityConfig] = None,
        mode: str = 'coupled'
    ):
        self.fs = fs
        self.buffer_duration = buffer_duration
        self.baseline_duration = baseline_duration
        self.mode = mode
        
        if config is None:
            if mode == 'eeg_only':
                config = InstabilityConfig(alpha=0.6, beta=0.4, gamma=0.0, threshold=2.5)
            elif mode == 'ecg_only':
                config = InstabilityConfig(alpha=0.6, beta=0.4, gamma=0.0, threshold=2.5)
            else:  # coupled
                config = InstabilityConfig(alpha=0.4, beta=0.3, gamma=0.3, threshold=2.5)
        
        self.config = config
        
        # Buffers
        self.buffer_samples = int(buffer_duration * fs)
        self.eeg_buffer: Deque[float] = deque(maxlen=self.buffer_samples)
        self.ecg_buffer: Deque[float] = deque(maxlen=self.buffer_samples)
        
        # Baseline
        self.baseline_eeg = None
        self.baseline_ecg = None
        self.baseline_phase_eeg = None
        self.baseline_phase_ecg = None
        
        # State tracking
        self.sample_count = 0
        self.last_process_time = 0
        self.processing_times = deque(maxlen=100)  # Track latency
        
        # Signal quality tracking
        self.eeg_quality_history = deque(maxlen=10)
        self.ecg_quality_history = deque(maxlen=10)
        
    def set_baseline(
        self,
        eeg_signal: np.ndarray,
        ecg_signal: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Set baseline from stable epoch.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Baseline EEG signal
        ecg_signal : np.ndarray, optional
            Baseline ECG signal (required for coupled mode)
            
        Returns
        -------
        dict
            Baseline metrics
        """
        # Preprocess and store baseline EEG
        self.baseline_eeg, _ = preprocess_eeg(eeg_signal, self.fs)
        self.baseline_phase_eeg = extract_phase(self.baseline_eeg, self.fs)
        
        baseline_metrics = {'mode': self.mode}
        
        # For coupled or ECG-only modes
        if self.mode in ['coupled', 'ecg_only']:
            if ecg_signal is None:
                raise ValueError(f"ECG baseline required for {self.mode} mode")
            
            self.baseline_ecg, _ = preprocess_ecg(ecg_signal, self.fs)
            self.baseline_phase_ecg = extract_phase(self.baseline_ecg, self.fs)
            
            baseline_metrics['ecg_available'] = True
        else:
            baseline_metrics['ecg_available'] = False
        
        return baseline_metrics
    
    def push_samples(
        self,
        eeg_samples: np.ndarray,
        ecg_samples: Optional[np.ndarray] = None
    ) -> None:
        """
        Push new samples into buffers.
        
        Parameters
        ----------
        eeg_samples : np.ndarray
            New EEG samples
        ecg_samples : np.ndarray, optional
            New ECG samples (required for coupled mode)
        """
        # Add to EEG buffer
        for sample in eeg_samples:
            self.eeg_buffer.append(sample)
        
        # Add to ECG buffer if provided
        if ecg_samples is not None:
            for sample in ecg_samples:
                self.ecg_buffer.append(sample)
        
        self.sample_count += len(eeg_samples)
    
    def is_ready(self) -> bool:
        """
        Check if buffer is full enough to process.
        
        Returns
        -------
        bool
            True if buffer has sufficient data
        """
        if self.mode == 'eeg_only':
            return len(self.eeg_buffer) >= self.buffer_samples
        elif self.mode == 'ecg_only':
            return len(self.ecg_buffer) >= self.buffer_samples
        else:  # coupled
            return (len(self.eeg_buffer) >= self.buffer_samples and 
                   len(self.ecg_buffer) >= self.buffer_samples)
    
    def get_signal_quality_status(self) -> Dict[str, bool]:
        """
        Get current signal quality status based on recent history.
        
        Returns
        -------
        dict
            Quality status flags
        """
        status = {}
        
        if len(self.eeg_quality_history) > 0:
            status['eeg_good'] = np.mean(self.eeg_quality_history) > 0.6
        else:
            status['eeg_good'] = True  # Assume good until proven otherwise
        
        if len(self.ecg_quality_history) > 0:
            status['ecg_good'] = np.mean(self.ecg_quality_history) > 0.6
        else:
            status['ecg_good'] = True
        
        return status
    
    def process_buffer(self) -> Optional[Dict]:
        """
        Process current buffer and compute instability.
        
        Returns
        -------
        dict or None
            Processing result, or None if buffer not ready or processing fails
        """
        start_time = time.time()
        
        # Check if ready
        if not self.is_ready():
            return None
        
        # Check baseline
        if self.baseline_eeg is None:
            warnings.warn("Baseline not set. Cannot process buffer.")
            return None
        
        # Convert buffers to arrays
        eeg_array = np.array(self.eeg_buffer)
        
        # Get signal quality status
        quality_status = self.get_signal_quality_status()
        
        try:
            if self.mode == 'eeg_only':
                result = self._process_eeg_only(eeg_array, quality_status)
            elif self.mode == 'ecg_only':
                ecg_array = np.array(self.ecg_buffer)
                result = self._process_ecg_only(ecg_array, quality_status)
            else:  # coupled
                ecg_array = np.array(self.ecg_buffer)
                result = self._process_coupled(eeg_array, ecg_array, quality_status)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            result['processing_time_ms'] = processing_time * 1000
            result['timestamp'] = self.sample_count / self.fs
            
            return result
            
        except Exception as e:
            warnings.warn(f"Processing failed: {e}")
            return None
    
    def _process_eeg_only(
        self,
        eeg_array: np.ndarray,
        quality_status: Dict
    ) -> Dict:
        """Process EEG-only mode."""
        # Preprocess
        processed_eeg, artifact_mask = preprocess_eeg(eeg_array, self.fs)
        
        # Quality check
        quality_score, quality_flags = quality_check(processed_eeg, self.fs)
        self.eeg_quality_history.append(quality_score)
        
        # Conservative fallback
        if quality_score < 0.5:
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'no_decision': True,
                'reason': 'poor_eeg_quality'
            }
        
        # Compute features
        delta_S = compute_delta_S_eeg(processed_eeg, self.baseline_eeg, self.fs)
        delta_I = compute_delta_I(processed_eeg, self.baseline_eeg, method='permutation_entropy')
        delta_C = 0.0
        
        # Compute instability
        delta_phi = self.config.alpha * delta_S + self.config.beta * delta_I
        gate = 1 if delta_phi >= self.config.threshold else 0
        
        return {
            'delta_phi': float(delta_phi),
            'gate': int(gate),
            'delta_S': float(delta_S),
            'delta_I': float(delta_I),
            'delta_C': 0.0,
            'quality_score': quality_score,
            'no_decision': False
        }
    
    def _process_ecg_only(
        self,
        ecg_array: np.ndarray,
        quality_status: Dict
    ) -> Dict:
        """Process ECG-only mode."""
        # Preprocess
        processed_ecg, artifact_mask = preprocess_ecg(ecg_array, self.fs)
        
        # Quality check
        quality_score, quality_flags = quality_check(processed_ecg, self.fs)
        self.ecg_quality_history.append(quality_score)
        
        # Conservative fallback
        if quality_score < 0.5:
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'no_decision': True,
                'reason': 'poor_ecg_quality'
            }
        
        # Extract RR intervals (simplified)
        # In production, use robust R-peak detector
        rr_current = self._extract_rr_simple(processed_ecg)
        rr_baseline = self._extract_rr_simple(self.baseline_ecg)
        
        if len(rr_current) < 5:
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'no_decision': True,
                'reason': 'insufficient_beats'
            }
        
        # Compute features
        delta_S = compute_delta_S_ecg(rr_current, rr_baseline, fs_rr=4.0)
        delta_I = compute_delta_I(rr_current, rr_baseline, method='permutation_entropy')
        delta_C = 0.0
        
        # Compute instability
        delta_phi = self.config.alpha * delta_S + self.config.beta * delta_I
        gate = 1 if delta_phi >= self.config.threshold else 0
        
        return {
            'delta_phi': float(delta_phi),
            'gate': int(gate),
            'delta_S': float(delta_S),
            'delta_I': float(delta_I),
            'delta_C': 0.0,
            'quality_score': quality_score,
            'no_decision': False
        }
    
    def _process_coupled(
        self,
        eeg_array: np.ndarray,
        ecg_array: np.ndarray,
        quality_status: Dict
    ) -> Dict:
        """Process full coupled mode."""
        # Preprocess both
        processed_eeg, eeg_artifact_mask = preprocess_eeg(eeg_array, self.fs)
        processed_ecg, ecg_artifact_mask = preprocess_ecg(ecg_array, self.fs)
        
        # Quality checks
        eeg_quality, _ = quality_check(processed_eeg, self.fs)
        ecg_quality, _ = quality_check(processed_ecg, self.fs)
        
        self.eeg_quality_history.append(eeg_quality)
        self.ecg_quality_history.append(ecg_quality)
        
        # Graceful degradation
        if eeg_quality < 0.5 and ecg_quality < 0.5:
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'no_decision': True,
                'reason': 'poor_quality_both'
            }
        elif eeg_quality < 0.5:
            # Fall back to ECG-only
            warnings.warn("Poor EEG quality, falling back to ECG-only mode")
            return self._process_ecg_only(ecg_array, quality_status)
        elif ecg_quality < 0.5:
            # Fall back to EEG-only
            warnings.warn("Poor ECG quality, falling back to EEG-only mode")
            return self._process_eeg_only(eeg_array, quality_status)
        
        # Full coupled processing
        rr_current = self._extract_rr_simple(processed_ecg)
        rr_baseline = self._extract_rr_simple(self.baseline_ecg)
        
        # Compute all three terms
        delta_S_eeg = compute_delta_S_eeg(processed_eeg, self.baseline_eeg, self.fs)
        delta_S_ecg = compute_delta_S_ecg(rr_current, rr_baseline, fs_rr=4.0) if len(rr_current) >= 5 else 0.0
        delta_S = 0.5 * delta_S_eeg + 0.5 * delta_S_ecg
        
        delta_I_eeg = compute_delta_I(processed_eeg, self.baseline_eeg, method='permutation_entropy')
        delta_I_ecg = compute_delta_I(rr_current, rr_baseline, method='permutation_entropy') if len(rr_current) >= 5 else 0.0
        delta_I = 0.5 * delta_I_eeg + 0.5 * delta_I_ecg
        
        # Coupling term
        phase_eeg = extract_phase(processed_eeg, self.fs)
        phase_ecg = extract_phase(processed_ecg, self.fs)
        
        delta_C = compute_delta_C(
            processed_eeg, processed_ecg,
            self.baseline_eeg, self.baseline_ecg,
            self.fs,
            method='plv',
            phase1=phase_eeg,
            phase2=phase_ecg,
            baseline_phase1=self.baseline_phase_eeg,
            baseline_phase2=self.baseline_phase_ecg
        )
        
        # Full instability functional
        delta_phi = compute_instability_functional(
            delta_S, delta_I, delta_C,
            self.config.alpha,
            self.config.beta,
            self.config.gamma
        )
        
        gate = 1 if delta_phi >= self.config.threshold else 0
        
        return {
            'delta_phi': float(delta_phi),
            'gate': int(gate),
            'delta_S': float(delta_S),
            'delta_I': float(delta_I),
            'delta_C': float(delta_C),
            'eeg_quality_score': eeg_quality,
            'ecg_quality_score': ecg_quality,
            'no_decision': False
        }
    
    def _extract_rr_simple(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Simplified RR interval extraction."""
        threshold = np.median(ecg_signal) + np.std(ecg_signal)
        peaks = np.where(ecg_signal > threshold)[0]
        
        if len(peaks) < 2:
            return np.array([])
        
        min_distance = int(0.2 * self.fs)
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        return np.diff(filtered_peaks) / self.fs
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get processing latency statistics.
        
        Returns
        -------
        dict
            Latency statistics in milliseconds
        """
        if len(self.processing_times) == 0:
            return {}
        
        times_ms = np.array(self.processing_times) * 1000
        
        return {
            'mean_ms': np.mean(times_ms),
            'median_ms': np.median(times_ms),
            'max_ms': np.max(times_ms),
            'std_ms': np.std(times_ms)
        }
    
    def reset_buffers(self) -> None:
        """Clear all buffers."""
        self.eeg_buffer.clear()
        self.ecg_buffer.clear()
        self.sample_count = 0
    
    def __repr__(self) -> str:
        return (
            f"StreamingPipeline(fs={self.fs}, "
            f"buffer_duration={self.buffer_duration}, "
            f"mode='{self.mode}', "
            f"buffer_fill={len(self.eeg_buffer)}/{self.buffer_samples})"
        )
