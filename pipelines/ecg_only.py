"""
ECG-Only Pipeline for Triadic Biosignal Monitor

This pipeline implements ECG-only ablation analysis:
    ΔΦ_ECG = α|ΔS_H| + β|ΔI_H|
    (no coupling term γ|ΔC|)

This ablation mode is used to prove that the coupling term contributes
additional predictive value beyond single-modality analysis.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from scipy import signal

from core.preprocessing import preprocess_ecg, quality_check
from core.phase import triadic_embedding, check_phase_quality
from core.features import compute_delta_S_ecg, compute_delta_I
from core.gate import (
    InstabilityConfig,
    ablation_ecg_only,
    decision_gate,
    detect_alert_events,
    AlertEvent
)


class ECGOnlyPipeline:
    """
    ECG-only ablation pipeline.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    baseline_duration : float
        Duration of baseline window in seconds
    config : InstabilityConfig, optional
        Configuration for instability detection
        For ECG-only, gamma should be 0 and alpha+beta=1
    """
    
    def __init__(
        self,
        fs: float,
        baseline_duration: float = 60.0,
        config: Optional[InstabilityConfig] = None
    ):
        self.fs = fs
        self.baseline_duration = baseline_duration
        
        # Configure for ECG-only (no coupling term)
        if config is None:
            config = InstabilityConfig(
                alpha=0.6,
                beta=0.4,
                gamma=0.0,
                threshold=2.5
            )
        
        self.config = config
        self.baseline_rr = None
        self.baseline_features = None
        
    def extract_rr_intervals(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Extract RR intervals from ECG signal.
        
        Parameters
        ----------
        ecg_signal : np.ndarray
            ECG signal
            
        Returns
        -------
        np.ndarray
            RR intervals in seconds
            
        Notes
        -----
        This is a simplified placeholder. In production, use a proper
        R-peak detection algorithm (e.g., Pan-Tompkins, or NeuroKit2).
        """
        # For now, use a simple peak detection with distance and prominence.
        # In production, replace with robust R-peak detector.
        min_distance = int(0.4 * self.fs)
        height = np.percentile(ecg_signal, 90)

        peaks, _ = signal.find_peaks(
            ecg_signal,
            distance=min_distance,
            prominence=0.3,
            height=height
        )

        if len(peaks) < 2:
            peaks, _ = signal.find_peaks(
                ecg_signal,
                distance=min_distance,
                prominence=0.2,
                height=np.percentile(ecg_signal, 80)
            )

        if len(peaks) < 2:
            warnings.warn("Insufficient peaks detected in ECG signal")
            return np.array([])

        rr_intervals = np.diff(peaks) / self.fs
        rr_intervals = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 2.0)]

        if len(rr_intervals) == 0:
            warnings.warn("No RR intervals in expected physiological range")
            return np.array([])

        return rr_intervals
    
    def set_baseline(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Establish baseline from stable ECG epoch.
        
        Parameters
        ----------
        ecg_signal : np.ndarray
            Baseline ECG signal (should be from stable period)
            
        Returns
        -------
        dict
            Baseline quality metrics
        """
        # Validate baseline length
        expected_samples = int(self.baseline_duration * self.fs)
        if len(ecg_signal) < expected_samples:
            warnings.warn(
                f"Baseline shorter than expected: {len(ecg_signal)} < {expected_samples} samples"
            )
        
        # Preprocess baseline
        processed_ecg, artifact_mask = preprocess_ecg(ecg_signal, self.fs)
        
        # Extract RR intervals
        self.baseline_rr = self.extract_rr_intervals(processed_ecg)
        
        if len(self.baseline_rr) < 10:
            raise ValueError("Insufficient RR intervals in baseline. Check ECG signal quality.")
        
        # Quality check on raw signal
        quality_score, quality_flags = quality_check(processed_ecg, self.fs)
        
        if quality_score < 0.7:
            warnings.warn(
                f"Baseline quality is low ({quality_score:.2f}). "
                f"Flags: {quality_flags}"
            )
        
        # Store baseline features
        self.baseline_features = {
            'quality_score': quality_score,
            'quality_flags': quality_flags,
            'artifact_fraction': np.mean(artifact_mask),
            'n_beats': len(self.baseline_rr),
            'mean_rr': np.mean(self.baseline_rr),
            'std_rr': np.std(self.baseline_rr)
        }
        
        return self.baseline_features
    
    def process_window(
        self,
        ecg_signal: np.ndarray,
        return_components: bool = True
    ) -> Dict:
        """
        Process single ECG window and compute instability.
        
        Parameters
        ----------
        ecg_signal : np.ndarray
            Current ECG signal window
        return_components : bool, optional
            Whether to return feature components (default: True)
            
        Returns
        -------
        dict
            Processing results with keys:
            - 'delta_phi': ΔΦ_ECG value
            - 'gate': Binary gate value (0 or 1)
            - 'delta_S': Spectral deviation (if return_components)
            - 'delta_I': Information deviation (if return_components)
            - 'quality': Signal quality metrics
        """
        if self.baseline_rr is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Preprocess current window
        processed_ecg, artifact_mask = preprocess_ecg(ecg_signal, self.fs)
        
        # Extract RR intervals
        current_rr = self.extract_rr_intervals(processed_ecg)
        
        # Quality check
        quality_score, quality_flags = quality_check(processed_ecg, self.fs)
        
        # Conservative fallback: no decision under poor quality or insufficient beats
        if quality_score < 0.5 or len(current_rr) < 5:
            warnings.warn("Poor signal quality or insufficient beats. Returning no-decision state.")
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'quality': {'score': quality_score, 'flags': quality_flags, 'n_beats': len(current_rr)},
                'no_decision': True
            }
        
        # Compute features
        delta_S = compute_delta_S_ecg(current_rr, self.baseline_rr, fs_rr=4.0)
        delta_I = compute_delta_I(current_rr, self.baseline_rr, method='permutation_entropy')
        
        # Compute ECG-only instability functional (no coupling term)
        delta_phi = ablation_ecg_only(
            delta_S,
            delta_I,
            alpha=self.config.alpha,
            beta=self.config.beta
        )
        
        # Apply decision gate
        gate = 1 if delta_phi >= self.config.threshold else 0
        
        # Prepare results
        results = {
            'delta_phi': float(delta_phi),
            'gate': int(gate),
            'quality': {
                'score': quality_score,
                'flags': quality_flags,
                'artifact_fraction': float(np.mean(artifact_mask)),
                'n_beats': len(current_rr)
            },
            'no_decision': False
        }
        
        if return_components:
            results.update({
                'delta_S': float(delta_S),
                'delta_I': float(delta_I),
                'delta_C': 0.0  # No coupling in ECG-only mode
            })
        
        return results
    
    def process_continuous(
        self,
        ecg_signal: np.ndarray,
        window_size: float = 10.0,
        overlap: float = 0.5
    ) -> Dict:
        """
        Process continuous ECG signal with sliding windows.
        
        Parameters
        ----------
        ecg_signal : np.ndarray
            Continuous ECG signal
        window_size : float, optional
            Window size in seconds (default: 10.0)
        overlap : float, optional
            Overlap fraction (default: 0.5)
            
        Returns
        -------
        dict
            Results with time series:
            - 'timestamps': Center time of each window
            - 'delta_phi': ΔΦ time series
            - 'gate': Binary gate signal
            - 'delta_S': ΔS time series
            - 'delta_I': ΔI time series
            - 'quality_scores': Quality scores
            - 'alerts': List of AlertEvent objects
        """
        if self.baseline_rr is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Calculate window parameters
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        # Initialize result arrays
        n_windows = (len(ecg_signal) - window_samples) // step_samples + 1
        timestamps = np.zeros(n_windows)
        delta_phi_series = np.zeros(n_windows)
        gate_series = np.zeros(n_windows, dtype=int)
        delta_S_series = np.zeros(n_windows)
        delta_I_series = np.zeros(n_windows)
        quality_scores = np.zeros(n_windows)
        artifact_levels = np.zeros(n_windows)
        
        # Process each window
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            window = ecg_signal[start_idx:end_idx]
            
            # Process window
            result = self.process_window(window, return_components=True)
            
            # Store results
            timestamps[i] = (start_idx + end_idx) / 2 / self.fs
            delta_phi_series[i] = result['delta_phi']
            gate_series[i] = result['gate']
            delta_S_series[i] = result['delta_S']
            delta_I_series[i] = result['delta_I']
            quality_scores[i] = result['quality']['score']
            artifact_levels[i] = result['quality']['artifact_fraction']
        
        # Detect alert events
        alerts = detect_alert_events(
            timestamps=timestamps,
            delta_phi_series=delta_phi_series,
            gate_series=gate_series,
            delta_S_series=delta_S_series,
            delta_I_series=delta_I_series,
            delta_C_series=np.zeros_like(delta_S_series),  # No coupling
            artifact_levels=artifact_levels
        )
        
        return {
            'timestamps': timestamps,
            'delta_phi': delta_phi_series,
            'gate': gate_series,
            'delta_S': delta_S_series,
            'delta_I': delta_I_series,
            'delta_C': np.zeros_like(delta_S_series),  # Always 0 for ECG-only
            'quality_scores': quality_scores,
            'artifact_levels': artifact_levels,
            'alerts': alerts
        }
    
    def __repr__(self) -> str:
        return (
            f"ECGOnlyPipeline(fs={self.fs}, "
            f"baseline_duration={self.baseline_duration}, "
            f"config={self.config})"
        )


def run_ecg_only_pipeline(
    ecg_signal: np.ndarray,
    baseline_ecg: np.ndarray,
    fs: float,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to run complete ECG-only pipeline.
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        Continuous ECG signal to analyze
    baseline_ecg : np.ndarray
        Baseline ECG from stable period
    fs : float
        Sampling frequency in Hz
    config : dict, optional
        Configuration dictionary with keys:
        - 'alpha', 'beta', 'gamma', 'threshold'
        - 'baseline_duration', 'window_size', 'overlap'
        
    Returns
    -------
    dict
        Complete analysis results
    """
    # Parse config
    if config is None:
        config = {}
    
    instability_config = InstabilityConfig(
        alpha=config.get('alpha', 0.6),
        beta=config.get('beta', 0.4),
        gamma=0.0,  # Always 0 for ECG-only
        threshold=config.get('threshold', 2.5)
    )
    
    baseline_duration = config.get('baseline_duration', 60.0)
    window_size = config.get('window_size', 10.0)
    overlap = config.get('overlap', 0.5)
    
    # Initialize pipeline
    pipeline = ECGOnlyPipeline(
        fs=fs,
        baseline_duration=baseline_duration,
        config=instability_config
    )
    
    # Set baseline
    baseline_metrics = pipeline.set_baseline(baseline_ecg)
    
    # Process continuous signal
    results = pipeline.process_continuous(
        ecg_signal,
        window_size=window_size,
        overlap=overlap
    )
    
    # Add baseline info to results
    results['baseline_metrics'] = baseline_metrics
    results['config'] = {
        'alpha': instability_config.alpha,
        'beta': instability_config.beta,
        'gamma': instability_config.gamma,
        'threshold': instability_config.threshold,
        'fs': fs,
        'baseline_duration': baseline_duration,
        'window_size': window_size,
        'overlap': overlap
    }
    
    return results
