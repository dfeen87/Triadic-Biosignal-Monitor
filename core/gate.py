"""
Instability Gate Module for Triadic Biosignal Monitor

This module implements the unified instability functional:
    ΔΦ(t) = α|ΔS(t)| + β|ΔI(t)| + γ|ΔC(t)|

And the deterministic decision gate:
    G(t) = 1{ΔΦ(t) ≥ τ}

All parameters (α, β, γ, τ) are fixed a priori for prospective validation.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings


@dataclass
class InstabilityConfig:
    """
    Configuration for instability detection.
    
    Attributes
    ----------
    alpha : float
        Weight for spectral/morphological deviation ΔS
    beta : float
        Weight for information/entropy deviation ΔI
    gamma : float
        Weight for coupling/coherence deviation ΔC
    threshold : float
        Decision threshold τ for gate
    """
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    threshold: float = 2.5
    
    def __post_init__(self):
        """Validate configuration."""
        # Check weights sum to 1
        weight_sum = self.alpha + self.beta + self.gamma
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            warnings.warn(
                f"Weights sum to {weight_sum}, not 1.0. "
                f"Normalizing: α={self.alpha}, β={self.beta}, γ={self.gamma}"
            )
            # Normalize
            total = self.alpha + self.beta + self.gamma
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
        
        # Check all weights are non-negative
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("All weights must be non-negative")
        
        # Check threshold is positive
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")


def compute_instability_functional(
    delta_S: float,
    delta_I: float,
    delta_C: float,
    alpha: float,
    beta: float,
    gamma: float
) -> float:
    """
    Compute instability functional ΔΦ(t).
    
    Parameters
    ----------
    delta_S : float
        Spectral/morphological deviation
    delta_I : float
        Information/entropy deviation
    delta_C : float
        Coupling/coherence deviation
    alpha : float
        Weight for ΔS
    beta : float
        Weight for ΔI
    gamma : float
        Weight for ΔC
        
    Returns
    -------
    float
        Instability functional value ΔΦ(t)
        
    Notes
    -----
    ΔΦ(t) = α|ΔS(t)| + β|ΔI(t)| + γ|ΔC(t)|
    
    Constraint: α + β + γ = 1
    """
    # Take absolute values (deviations are already typically positive, but ensure)
    delta_phi = alpha * np.abs(delta_S) + beta * np.abs(delta_I) + gamma * np.abs(delta_C)
    
    return delta_phi


def decision_gate(
    delta_phi: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Apply deterministic decision gate.
    
    Parameters
    ----------
    delta_phi : np.ndarray
        Instability functional values over time
    threshold : float
        Decision threshold τ
        
    Returns
    -------
    np.ndarray
        Binary gate signal (1 = instability detected, 0 = stable)
        
    Notes
    -----
    G(t) = 1{ΔΦ(t) ≥ τ}
    
    This is a hard threshold - no adaptive modification during deployment.
    """
    gate = (delta_phi >= threshold).astype(int)
    return gate


def compute_instability_timeseries(
    delta_S_series: np.ndarray,
    delta_I_series: np.ndarray,
    delta_C_series: np.ndarray,
    config: InstabilityConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instability functional and gate over time series.
    
    Parameters
    ----------
    delta_S_series : np.ndarray
        Time series of ΔS values
    delta_I_series : np.ndarray
        Time series of ΔI values
    delta_C_series : np.ndarray
        Time series of ΔC values
    config : InstabilityConfig
        Configuration with weights and threshold
        
    Returns
    -------
    delta_phi_series : np.ndarray
        Time series of ΔΦ(t) values
    gate_series : np.ndarray
        Binary gate signal
    """
    # Validate input lengths
    n = len(delta_S_series)
    if len(delta_I_series) != n or len(delta_C_series) != n:
        raise ValueError("All input series must have the same length")
    
    # Compute ΔΦ(t) for each time point
    delta_phi_series = np.zeros(n)
    for i in range(n):
        delta_phi_series[i] = compute_instability_functional(
            delta_S_series[i],
            delta_I_series[i],
            delta_C_series[i],
            config.alpha,
            config.beta,
            config.gamma
        )
    
    # Apply decision gate
    gate_series = decision_gate(delta_phi_series, config.threshold)
    
    return delta_phi_series, gate_series


@dataclass
class AlertEvent:
    """
    Structure for an instability alert event.
    
    Attributes
    ----------
    timestamp : float
        Time of alert in seconds
    delta_phi : float
        ΔΦ(t) value at alert
    delta_S : float
        ΔS component
    delta_I : float
        ΔI component
    delta_C : float
        ΔC component
    confidence : float
        Confidence score (0-1)
    flags : dict
        Quality/confidence flags
    """
    timestamp: float
    delta_phi: float
    delta_S: float
    delta_I: float
    delta_C: float
    confidence: float
    flags: Dict[str, bool]


def generate_alert(
    timestamp: float,
    delta_phi: float,
    delta_S: float,
    delta_I: float,
    delta_C: float,
    artifact_level: float = 0.0,
    missing_data: bool = False,
    sync_drift: bool = False
) -> AlertEvent:
    """
    Generate explainable alert with component breakdown.
    
    Parameters
    ----------
    timestamp : float
        Time of alert in seconds
    delta_phi : float
        Total instability functional value
    delta_S : float
        Spectral/morphological deviation component
    delta_I : float
        Information/entropy deviation component
    delta_C : float
        Coupling/coherence deviation component
    artifact_level : float, optional
        Fraction of artifacts in window (default: 0.0)
    missing_data : bool, optional
        Whether data is missing (default: False)
    sync_drift : bool, optional
        Whether synchronization has drifted (default: False)
        
    Returns
    -------
    AlertEvent
        Complete alert event with all components and flags
        
    Notes
    -----
    Every alert is fully explainable and traceable to ΔΦ components.
    Confidence flags indicate data quality issues.
    """
    # Compute confidence score
    confidence_factors = []
    
    # Artifact level (lower is better)
    confidence_factors.append(1.0 - artifact_level)
    
    # No missing data
    confidence_factors.append(0.0 if missing_data else 1.0)
    
    # No sync drift
    confidence_factors.append(0.0 if sync_drift else 1.0)
    
    confidence = np.mean(confidence_factors)
    
    # Quality flags
    flags = {
        'low_artifacts': artifact_level < 0.2,
        'no_missing_data': not missing_data,
        'good_sync': not sync_drift,
        'high_confidence': confidence > 0.7
    }
    
    return AlertEvent(
        timestamp=timestamp,
        delta_phi=delta_phi,
        delta_S=delta_S,
        delta_I=delta_I,
        delta_C=delta_C,
        confidence=confidence,
        flags=flags
    )


def detect_alert_events(
    timestamps: np.ndarray,
    delta_phi_series: np.ndarray,
    gate_series: np.ndarray,
    delta_S_series: np.ndarray,
    delta_I_series: np.ndarray,
    delta_C_series: np.ndarray,
    artifact_levels: Optional[np.ndarray] = None,
    min_duration: float = 0.0
) -> List[AlertEvent]:
    """
    Detect and generate alert events from gate signal.
    
    Parameters
    ----------
    timestamps : np.ndarray
        Time array in seconds
    delta_phi_series : np.ndarray
        ΔΦ(t) time series
    gate_series : np.ndarray
        Binary gate signal
    delta_S_series : np.ndarray
        ΔS time series
    delta_I_series : np.ndarray
        ΔI time series
    delta_C_series : np.ndarray
        ΔC time series
    artifact_levels : np.ndarray, optional
        Artifact levels over time
    min_duration : float, optional
        Minimum duration (seconds) for valid alert (default: 0.0)
        
    Returns
    -------
    list of AlertEvent
        List of detected alert events
    """
    alerts = []
    
    # Find transitions from 0 to 1 (alert onset)
    gate_diff = np.diff(gate_series, prepend=0)
    onset_indices = np.where(gate_diff == 1)[0]
    offset_indices = np.where(gate_diff == -1)[0]
    
    # Handle case where alert extends to end
    if len(onset_indices) > len(offset_indices):
        offset_indices = np.append(offset_indices, len(gate_series) - 1)
    
    # Process each alert period
    for onset_idx, offset_idx in zip(onset_indices, offset_indices):
        # Check duration
        duration = timestamps[offset_idx] - timestamps[onset_idx]
        if duration < min_duration:
            continue
        
        # Use peak ΔΦ value during alert period
        alert_window = slice(onset_idx, offset_idx + 1)
        peak_idx = onset_idx + np.argmax(delta_phi_series[alert_window])
        
        # Get artifact level
        artifact_level = 0.0
        if artifact_levels is not None:
            artifact_level = artifact_levels[peak_idx]
        
        # Generate alert
        alert = generate_alert(
            timestamp=timestamps[peak_idx],
            delta_phi=delta_phi_series[peak_idx],
            delta_S=delta_S_series[peak_idx],
            delta_I=delta_I_series[peak_idx],
            delta_C=delta_C_series[peak_idx],
            artifact_level=artifact_level
        )
        
        alerts.append(alert)
    
    return alerts


def ablation_eeg_only(
    delta_S: float,
    delta_I: float,
    alpha: float = 0.6,
    beta: float = 0.4
) -> float:
    """
    EEG-only ablation: ΔΦ_EEG = α|ΔS| + β|ΔI| (no coupling term).
    
    Parameters
    ----------
    delta_S : float
        Spectral/morphological deviation
    delta_I : float
        Information/entropy deviation
    alpha : float, optional
        Weight for ΔS (default: 0.6)
    beta : float, optional
        Weight for ΔI (default: 0.4)
        
    Returns
    -------
    float
        ΔΦ_EEG value
    """
    # Normalize weights
    total = alpha + beta
    alpha_norm = alpha / total
    beta_norm = beta / total
    
    return alpha_norm * np.abs(delta_S) + beta_norm * np.abs(delta_I)


def ablation_ecg_only(
    delta_S: float,
    delta_I: float,
    alpha: float = 0.6,
    beta: float = 0.4
) -> float:
    """
    ECG-only ablation: ΔΦ_ECG = α|ΔS| + β|ΔI| (no coupling term).
    
    Parameters
    ----------
    delta_S : float
        Spectral/morphological deviation
    delta_I : float
        Information/entropy deviation
    alpha : float, optional
        Weight for ΔS (default: 0.6)
    beta : float, optional
        Weight for ΔI (default: 0.4)
        
    Returns
    -------
    float
        ΔΦ_ECG value
    """
    # Normalize weights
    total = alpha + beta
    alpha_norm = alpha / total
    beta_norm = beta / total
    
    return alpha_norm * np.abs(delta_S) + beta_norm * np.abs(delta_I)


def risk_score(
    delta_phi: float,
    threshold: float,
    scale: str = 'linear'
) -> float:
    """
    Convert ΔΦ value to risk score.
    
    Parameters
    ----------
    delta_phi : float
        Instability functional value
    threshold : float
        Decision threshold τ
    scale : str, optional
        Scaling: 'linear' or 'sigmoid' (default: 'linear')
        
    Returns
    -------
    float
        Risk score from 0.0 (stable) to 1.0 (high risk)
    """
    if scale == 'linear':
        # Linear scaling: 0 at threshold/2, 1.0 at 2*threshold
        risk = (delta_phi - threshold/2) / (1.5 * threshold)
        risk = np.clip(risk, 0.0, 1.0)
        
    elif scale == 'sigmoid':
        # Sigmoid scaling centered at threshold
        x = (delta_phi - threshold) / threshold
        risk = 1.0 / (1.0 + np.exp(-5 * x))  # Steepness factor of 5
        
    else:
        raise ValueError(f"Unknown scale: {scale}")
    
    return risk


def format_alert_message(alert: AlertEvent) -> str:
    """
    Format alert event as human-readable message.
    
    Parameters
    ----------
    alert : AlertEvent
        Alert event to format
        
    Returns
    -------
    str
        Formatted alert message
    """
    msg = f"""
INSTABILITY ALERT
=================
Timestamp: {alert.timestamp:.2f} s
ΔΦ(t): {alert.delta_phi:.3f}
Confidence: {alert.confidence:.2%}

Component Breakdown:
  ΔS (spectral): {alert.delta_S:.3f}
  ΔI (information): {alert.delta_I:.3f}
  ΔC (coupling): {alert.delta_C:.3f}

Quality Flags:
  Low artifacts: {alert.flags['low_artifacts']}
  No missing data: {alert.flags['no_missing_data']}
  Good sync: {alert.flags['good_sync']}
  High confidence: {alert.flags['high_confidence']}
"""
    return msg.strip()
