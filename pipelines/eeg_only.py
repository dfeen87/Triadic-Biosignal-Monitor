"""
EEG-Only Pipeline for Triadic Biosignal Monitor

This pipeline implements EEG-only ablation analysis:
    ΔΦ_EEG = α|ΔS_B| + β|ΔI_B|
    (no coupling term γ|ΔC|)

This ablation mode is used to prove that the coupling term contributes
additional predictive value beyond single-modality analysis.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from core.config_adapter import adapt_config
from core.preprocessing import preprocess_eeg, quality_check
from core.phase import triadic_embedding, check_phase_quality
from core.features import compute_delta_S_eeg, compute_delta_I
from core.gate import (
    InstabilityConfig,
    ablation_eeg_only,
    decision_gate,
    detect_alert_events,
    AlertEvent
)


class EEGOnlyPipeline:
    """
    EEG-only ablation pipeline.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    baseline_duration : float
        Duration of baseline window in seconds
    config : InstabilityConfig, optional
        Configuration for instability detection
        For EEG-only, gamma should be 0 and alpha+beta=1
    entropy_kwargs : dict, optional
        Keyword arguments forwarded to the entropy function (e.g. order, delay,
        normalize for permutation_entropy).
    delta_I_method : str, optional
        Entropy method to use for ΔI (default: 'permutation_entropy').
    """
    
    def __init__(
        self,
        fs: float,
        baseline_duration: float = 60.0,
        config: Optional[InstabilityConfig] = None,
        entropy_kwargs: Optional[Dict] = None,
        delta_I_method: str = 'permutation_entropy',
    ):
        self.fs = fs
        self.baseline_duration = baseline_duration
        
        # Configure for EEG-only (no coupling term)
        if config is None:
            config = InstabilityConfig(
                alpha=0.6,
                beta=0.4,
                gamma=0.0,
                threshold=2.5
            )
        
        self.config = config
        self._entropy_kwargs = entropy_kwargs or {}
        self._delta_I_method = delta_I_method
        self.baseline_eeg = None
        self.baseline_features = None
        
    def set_baseline(self, eeg_signal: np.ndarray) -> Dict[str, float]:
        """
        Establish baseline from stable EEG epoch.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Baseline EEG signal (should be from stable period)
            
        Returns
        -------
        dict
            Baseline quality metrics
        """
        # Validate baseline length
        expected_samples = int(self.baseline_duration * self.fs)
        if len(eeg_signal) < expected_samples:
            warnings.warn(
                f"Baseline shorter than expected: {len(eeg_signal)} < {expected_samples} samples"
            )
        
        # Preprocess baseline
        self.baseline_eeg, artifact_mask = preprocess_eeg(eeg_signal, self.fs)
        
        # Quality check
        quality_score, quality_flags = quality_check(self.baseline_eeg, self.fs)
        
        if quality_score < 0.7:
            warnings.warn(
                f"Baseline quality is low ({quality_score:.2f}). "
                f"Flags: {quality_flags}"
            )
        
        # Store baseline features for reference
        self.baseline_features = {
            'quality_score': quality_score,
            'quality_flags': quality_flags,
            'artifact_fraction': np.mean(artifact_mask)
        }
        
        return self.baseline_features
    
    def process_window(
        self,
        eeg_signal: np.ndarray,
        return_components: bool = True
    ) -> Dict:
        """
        Process single EEG window and compute instability.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Current EEG signal window
        return_components : bool, optional
            Whether to return feature components (default: True)
            
        Returns
        -------
        dict
            Processing results with keys:
            - 'delta_phi': ΔΦ_EEG value
            - 'gate': Binary gate value (0 or 1)
            - 'delta_S': Spectral deviation (if return_components)
            - 'delta_I': Information deviation (if return_components)
            - 'quality': Signal quality metrics
        """
        if self.baseline_eeg is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Preprocess current window
        processed_eeg, artifact_mask = preprocess_eeg(eeg_signal, self.fs)
        
        # Quality check
        quality_score, quality_flags = quality_check(processed_eeg, self.fs)
        
        # Conservative fallback: no decision under poor quality
        if quality_score < 0.5:
            warnings.warn("Poor signal quality. Returning no-decision state.")
            return {
                'delta_phi': 0.0,
                'gate': 0,
                'quality': {'score': quality_score, 'flags': quality_flags},
                'no_decision': True
            }
        
        # Compute features
        delta_S = compute_delta_S_eeg(processed_eeg, self.baseline_eeg, self.fs)
        delta_I = compute_delta_I(
            processed_eeg, self.baseline_eeg,
            method=self._delta_I_method,
            **self._entropy_kwargs
        )
        
        # Compute EEG-only instability functional (no coupling term)
        delta_phi = ablation_eeg_only(
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
                'artifact_fraction': float(np.mean(artifact_mask))
            },
            'no_decision': False
        }
        
        if return_components:
            results.update({
                'delta_S': float(delta_S),
                'delta_I': float(delta_I),
                'delta_C': 0.0  # No coupling in EEG-only mode
            })
        
        return results
    
    def process_continuous(
        self,
        eeg_signal: np.ndarray,
        window_size: float = 10.0,
        overlap: float = 0.5
    ) -> Dict:
        """
        Process continuous EEG signal with sliding windows.
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            Continuous EEG signal
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
        if self.baseline_eeg is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Validate overlap
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")

        # Calculate window parameters
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))

        # Guard: signal too short for even one window
        if len(eeg_signal) < window_samples:
            warnings.warn(
                f"Signal length ({len(eeg_signal)} samples) is shorter than one window "
                f"({window_samples} samples). Returning empty outputs."
            )
            return {
                'timestamps': np.array([]),
                'delta_phi': np.array([]),
                'gate': np.array([], dtype=int),
                'delta_S': np.array([]),
                'delta_I': np.array([]),
                'delta_C': np.array([]),
                'quality_scores': np.array([]),
                'artifact_levels': np.array([]),
                'alerts': []
            }
        
        # Initialize result arrays
        n_windows = (len(eeg_signal) - window_samples) // step_samples + 1
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
            
            window = eeg_signal[start_idx:end_idx]
            
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
            'delta_C': np.zeros_like(delta_S_series),  # Always 0 for EEG-only
            'quality_scores': quality_scores,
            'artifact_levels': artifact_levels,
            'alerts': alerts
        }
    
    def __repr__(self) -> str:
        return (
            f"EEGOnlyPipeline(fs={self.fs}, "
            f"baseline_duration={self.baseline_duration}, "
            f"config={self.config})"
        )


def run_eeg_only_pipeline(
    eeg_signal: np.ndarray,
    baseline_eeg: np.ndarray,
    fs: float,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to run complete EEG-only pipeline.
    
    Parameters
    ----------
    eeg_signal : np.ndarray
        Continuous EEG signal to analyze
    baseline_eeg : np.ndarray
        Baseline EEG from stable period
    fs : float
        Sampling frequency in Hz
    config : dict, optional
        Configuration dictionary. Accepts both the nested YAML structure
        (``instability_functional``, ``sliding_window``, ``features``, …) and
        the legacy flat layout (``alpha``, ``window_size``, …).
        
    Returns
    -------
    dict
        Complete analysis results
    """
    # Resolve nested YAML structure → flat params
    adapted = adapt_config(config)

    instability_config = InstabilityConfig(
        alpha=adapted['alpha'],
        beta=adapted['beta'],
        gamma=0.0,  # Always 0 for EEG-only
        threshold=adapted['threshold'],
    )

    baseline_duration = adapted['baseline_duration']
    window_size = adapted['window_size']
    overlap = adapted['overlap']
    entropy_kwargs = {
        'order': adapted['entropy_order'],
        'delay': adapted['entropy_delay'],
        'normalize': adapted['entropy_normalize'],
    }

    # Initialize pipeline
    pipeline = EEGOnlyPipeline(
        fs=fs,
        baseline_duration=baseline_duration,
        config=instability_config,
        entropy_kwargs=entropy_kwargs,
        delta_I_method=adapted['delta_I_method'],
    )

    # Set baseline
    baseline_metrics = pipeline.set_baseline(baseline_eeg)

    # Process continuous signal
    results = pipeline.process_continuous(
        eeg_signal,
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
