"""
Coupled EEG-ECG Pipeline for Triadic Biosignal Monitor

This pipeline implements the full coupled analysis:
    ΔΦ(t) = α|ΔS(t)| + β|ΔI(t)| + γ|ΔC(t)|

This is the complete framework with all three deviation terms including
the cross-modal coupling term ΔC that captures heart-brain coherence.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from core.preprocessing import (
    preprocess_eeg,
    preprocess_ecg,
    synchronize_signals,
    quality_check
)
from core.phase import extract_phase, triadic_embedding
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
    detect_alert_events,
    AlertEvent
)


class CoupledPipeline:
    """
    Full coupled EEG-ECG pipeline with all three terms.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz (should be same for both EEG and ECG)
    baseline_duration : float
        Duration of baseline window in seconds
    config : InstabilityConfig, optional
        Configuration for instability detection with all three weights
    """
    
    def __init__(
        self,
        fs: float,
        baseline_duration: float = 60.0,
        config: Optional[InstabilityConfig] = None
    ):
        self.fs = fs
        self.baseline_duration = baseline_duration
        
        # Configure for full coupled mode (all three terms)
        if config is None:
            config = InstabilityConfig(
                alpha=0.4,  # Spectral/morphological
                beta=0.3,   # Information/entropy
                gamma=0.3,  # Coupling/coherence
                threshold=2.5
            )
        
        self.config = config
        self.baseline_eeg = None
        self.baseline_ecg = None
        self.baseline_phase_eeg = None
        self.baseline_phase_ecg = None
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
        Simplified placeholder. Use robust R-peak detector in production.
        """
        # Simple threshold-based peak detection
        threshold = np.median(ecg_signal) + np.std(ecg_signal)
        peaks = np.where(ecg_signal > threshold)[0]
        
        if len(peaks) < 2:
            warnings.warn("Insufficient peaks detected in ECG signal")
            return np.array([])
        
        # Remove consecutive peaks (within 200ms)
        min_distance = int(0.2 * self.fs)
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        rr_intervals = np.diff(filtered_peaks) / self.fs
        return rr_intervals
    
    def set_baseline(
        self,
        eeg_signal: np.ndarray,
        ecg_signal: np.ndarray,
        eeg_timestamps: Optional[np.ndarray] = None,
        ecg_timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Establish baseline from stable EEG and ECG epochs.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Baseline EEG signal
        ecg_signal : np.ndarray
            Baseline ECG signal
        eeg_timestamps : np.ndarray, optional
            Timestamps for EEG (if signals need synchronization)
        ecg_timestamps : np.ndarray, optional
            Timestamps for ECG (if signals need synchronization)
            
        Returns
        -------
        dict
            Baseline quality metrics
        """
        # Synchronize signals if timestamps provided
        if eeg_timestamps is not None and ecg_timestamps is not None:
            eeg_signal, ecg_signal, _ = synchronize_signals(
                eeg_signal, ecg_signal,
                eeg_timestamps, ecg_timestamps
            )
        
        # Ensure signals have same length
        min_len = min(len(eeg_signal), len(ecg_signal))
        eeg_signal = eeg_signal[:min_len]
        ecg_signal = ecg_signal[:min_len]
        
        # Preprocess baseline signals
        self.baseline_eeg, eeg_artifact_mask = preprocess_eeg(eeg_signal, self.fs)
        self.baseline_ecg, ecg_artifact_mask = preprocess_ecg(ecg_signal, self.fs)
        
        # Extract phases for coupling analysis
        self.baseline_phase_eeg = extract_phase(self.baseline_eeg, self.fs)
        self.baseline_phase_ecg = extract_phase(self.baseline_ecg, self.fs)
        
        # Quality checks
        eeg_quality, eeg_flags = quality_check(self.baseline_eeg, self.fs)
        ecg_quality, ecg_flags = quality_check(self.baseline_ecg, self.fs)
        
        if eeg_quality < 0.7 or ecg_quality < 0.7:
            warnings.warn(
                f"Baseline quality is low. EEG: {eeg_quality:.2f}, ECG: {ecg_quality:.2f}"
            )
        
        # Store baseline features
        self.baseline_features = {
            'eeg_quality_score': eeg_quality,
            'eeg_quality_flags': eeg_flags,
            'eeg_artifact_fraction': np.mean(eeg_artifact_mask),
            'ecg_quality_score': ecg_quality,
            'ecg_quality_flags': ecg_flags,
            'ecg_artifact_fraction': np.mean(ecg_artifact_mask),
            'signal_length': min_len / self.fs  # in seconds
        }
        
        return self.baseline_features
    
    def process_window(
        self,
        eeg_signal: np.ndarray,
        ecg_signal: np.ndarray,
        eeg_timestamps: Optional[np.ndarray] = None,
        ecg_timestamps: Optional[np.ndarray] = None,
        return_components: bool = True
    ) -> Dict:
        """
        Process coupled EEG-ECG window and compute full instability.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Current EEG signal window
        ecg_signal : np.ndarray
            Current ECG signal window
        eeg_timestamps : np.ndarray, optional
            Timestamps for EEG
        ecg_timestamps : np.ndarray, optional
            Timestamps for ECG
        return_components : bool, optional
            Whether to return feature components (default: True)
            
        Returns
        -------
        dict
            Processing results with all three deviation terms
        """
        if self.baseline_eeg is None or self.baseline_ecg is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Synchronize if needed
        if eeg_timestamps is not None and ecg_timestamps is not None:
            eeg_signal, ecg_signal, _ = synchronize_signals(
                eeg_signal, ecg_signal,
                eeg_timestamps, ecg_timestamps
            )
        
        # Ensure same length
        min_len = min(len(eeg_signal), len(ecg_signal))
        eeg_signal = eeg_signal[:min_len]
        ecg_signal = ecg_signal[:min_len]
        
        # Preprocess current windows
        processed_eeg, eeg_artifact_mask = preprocess_eeg(eeg_signal, self.fs)
        processed_ecg, ecg_artifact_mask = preprocess_ecg(ecg_signal, self.fs)
        
        # Quality checks
        eeg_quality, eeg_flags = quality_check(processed_eeg, self.fs)
        ecg_quality, ecg_flags = quality_check(processed_ecg, self.fs)
        
        # Conservative fallback under poor quality
        if eeg_quality < 0.5 or ecg_quality < 0.5:
            warnings.warn("Poor signal quality. Returning no-decision state.")
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'quality': {
                    'eeg_score': eeg_quality,
                    'ecg_score': ecg_quality
                },
                'no_decision': True
            }
        
        # Extract RR intervals for ECG features
        current_rr = self.extract_rr_intervals(processed_ecg)
        baseline_rr = self.extract_rr_intervals(self.baseline_ecg)
        
        if len(current_rr) < 5 or len(baseline_rr) < 5:
            warnings.warn("Insufficient RR intervals. Using EEG-only mode.")
            # Fall back to EEG-only
            delta_S = compute_delta_S_eeg(processed_eeg, self.baseline_eeg, self.fs)
            delta_I = compute_delta_I(processed_eeg, self.baseline_eeg, method='permutation_entropy')
            delta_C = 0.0
        else:
            # Compute all three feature types
            
            # ΔS: Combine EEG and ECG spectral features
            delta_S_eeg = compute_delta_S_eeg(processed_eeg, self.baseline_eeg, self.fs)
            delta_S_ecg = compute_delta_S_ecg(current_rr, baseline_rr, fs_rr=4.0)
            delta_S = 0.5 * delta_S_eeg + 0.5 * delta_S_ecg  # Average of both
            
            # ΔI: Combine EEG and ECG entropy features
            delta_I_eeg = compute_delta_I(processed_eeg, self.baseline_eeg, method='permutation_entropy')
            delta_I_ecg = compute_delta_I(current_rr, baseline_rr, method='permutation_entropy')
            delta_I = 0.5 * delta_I_eeg + 0.5 * delta_I_ecg  # Average of both
            
            # ΔC: EEG-ECG coupling deviation
            current_phase_eeg = extract_phase(processed_eeg, self.fs)
            current_phase_ecg = extract_phase(processed_ecg, self.fs)
            
            delta_C = compute_delta_C(
                processed_eeg, processed_ecg,
                self.baseline_eeg, self.baseline_ecg,
                self.fs,
                method='plv',
                phase1=current_phase_eeg,
                phase2=current_phase_ecg,
                baseline_phase1=self.baseline_phase_eeg,
                baseline_phase2=self.baseline_phase_ecg
            )
        
        # Compute full instability functional with all three terms
        delta_phi = compute_instability_functional(
            delta_S, delta_I, delta_C,
            self.config.alpha,
            self.config.beta,
            self.config.gamma
        )
        
        # Apply decision gate
        gate = 1 if delta_phi >= self.config.threshold else 0
        
        # Prepare results
        results = {
            'delta_phi': float(delta_phi),
            'gate': int(gate),
            'quality': {
                'eeg_score': eeg_quality,
                'ecg_score': ecg_quality,
                'eeg_artifact_fraction': float(np.mean(eeg_artifact_mask)),
                'ecg_artifact_fraction': float(np.mean(ecg_artifact_mask))
            },
            'no_decision': False
        }
        
        if return_components:
            results.update({
                'delta_S': float(delta_S),
                'delta_I': float(delta_I),
                'delta_C': float(delta_C)
            })
        
        return results
    
    def process_continuous(
        self,
        eeg_signal: np.ndarray,
        ecg_signal: np.ndarray,
        window_size: float = 10.0,
        overlap: float = 0.5
    ) -> Dict:
        """
        Process continuous coupled EEG-ECG signals with sliding windows.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Continuous EEG signal
        ecg_signal : np.ndarray
            Continuous ECG signal
        window_size : float, optional
            Window size in seconds (default: 10.0)
        overlap : float, optional
            Overlap fraction (default: 0.5)
            
        Returns
        -------
        dict
            Complete results with all three deviation components
        """
        if self.baseline_eeg is None or self.baseline_ecg is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Ensure signals same length
        min_len = min(len(eeg_signal), len(ecg_signal))
        eeg_signal = eeg_signal[:min_len]
        ecg_signal = ecg_signal[:min_len]
        
        # Calculate window parameters
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        # Initialize result arrays
        n_windows = (min_len - window_samples) // step_samples + 1
        timestamps = np.zeros(n_windows)
        delta_phi_series = np.zeros(n_windows)
        gate_series = np.zeros(n_windows, dtype=int)
        delta_S_series = np.zeros(n_windows)
        delta_I_series = np.zeros(n_windows)
        delta_C_series = np.zeros(n_windows)
        eeg_quality_scores = np.zeros(n_windows)
        ecg_quality_scores = np.zeros(n_windows)
        artifact_levels = np.zeros(n_windows)
        
        # Process each window
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            eeg_window = eeg_signal[start_idx:end_idx]
            ecg_window = ecg_signal[start_idx:end_idx]
            
            # Process window
            result = self.process_window(
                eeg_window, ecg_window,
                return_components=True
            )
            
            # Store results
            timestamps[i] = (start_idx + end_idx) / 2 / self.fs
            delta_phi_series[i] = result['delta_phi']
            gate_series[i] = result['gate']
            delta_S_series[i] = result['delta_S']
            delta_I_series[i] = result['delta_I']
            delta_C_series[i] = result['delta_C']
            eeg_quality_scores[i] = result['quality']['eeg_score']
            ecg_quality_scores[i] = result['quality']['ecg_score']
            
            # Average artifact level
            artifact_levels[i] = 0.5 * (
                result['quality']['eeg_artifact_fraction'] +
                result['quality']['ecg_artifact_fraction']
            )
        
        # Detect alert events
        alerts = detect_alert_events(
            timestamps=timestamps,
            delta_phi_series=delta_phi_series,
            gate_series=gate_series,
            delta_S_series=delta_S_series,
            delta_I_series=delta_I_series,
            delta_C_series=delta_C_series,
            artifact_levels=artifact_levels
        )
        
        return {
            'timestamps': timestamps,
            'delta_phi': delta_phi_series,
            'gate': gate_series,
            'delta_S': delta_S_series,
            'delta_I': delta_I_series,
            'delta_C': delta_C_series,
            'eeg_quality_scores': eeg_quality_scores,
            'ecg_quality_scores': ecg_quality_scores,
            'artifact_levels': artifact_levels,
            'alerts': alerts
        }
    
    def __repr__(self) -> str:
        return (
            f"CoupledPipeline(fs={self.fs}, "
            f"baseline_duration={self.baseline_duration}, "
            f"config={self.config})"
        )


def run_coupled_pipeline(
    eeg_signal: np.ndarray,
    ecg_signal: np.ndarray,
    baseline_eeg: np.ndarray,
    baseline_ecg: np.ndarray,
    fs: float,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to run complete coupled EEG-ECG pipeline.
    
    Parameters
    ----------
    eeg_signal : np.ndarray
        Continuous EEG signal to analyze
    ecg_signal : np.ndarray
        Continuous ECG signal to analyze
    baseline_eeg : np.ndarray
        Baseline EEG from stable period
    baseline_ecg : np.ndarray
        Baseline ECG from stable period
    fs : float
        Sampling frequency in Hz
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    dict
        Complete analysis results with all three terms
    """
    # Parse config
    if config is None:
        config = {}
    
    instability_config = InstabilityConfig(
        alpha=config.get('alpha', 0.4),
        beta=config.get('beta', 0.3),
        gamma=config.get('gamma', 0.3),
        threshold=config.get('threshold', 2.5)
    )
    
    baseline_duration = config.get('baseline_duration', 60.0)
    window_size = config.get('window_size', 10.0)
    overlap = config.get('overlap', 0.5)
    
    # Initialize pipeline
    pipeline = CoupledPipeline(
        fs=fs,
        baseline_duration=baseline_duration,
        config=instability_config
    )
    
    # Set baseline
    baseline_metrics = pipeline.set_baseline(baseline_eeg, baseline_ecg)
    
    # Process continuous signals
    results = pipeline.process_continuous(
        eeg_signal, ecg_signal,
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
