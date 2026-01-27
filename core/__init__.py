"""
Core Module for Triadic Biosignal Monitor

This module provides the foundational signal processing functions for
operator-based heart-brain monitoring via triadic spiral-time embeddings.

Modules:
--------
preprocessing : Signal filtering, artifact rejection, synchronization
phase : Phase extraction via Hilbert transform and triadic embedding
features : Computation of ΔS, ΔI, ΔC deviation terms
gate : Instability functional ΔΦ(t) and decision gate
metrics : Performance evaluation (lead-time, false alarms, ROC)

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

__version__ = "0.1.0"

# Import key functions for convenience
from .preprocessing import (
    bandpass_filter,
    notch_filter,
    preprocess_eeg,
    preprocess_ecg,
    synchronize_signals,
    quality_check
)

from .phase import (
    analytic_signal,
    extract_phase,
    unwrap_phase,
    phase_derivative,
    triadic_embedding,
    instantaneous_frequency,
    phase_locking_value
)

__all__ = [
    # Preprocessing
    'bandpass_filter',
    'notch_filter',
    'preprocess_eeg',
    'preprocess_ecg',
    'synchronize_signals',
    'quality_check',
    
    # Phase extraction
    'analytic_signal',
    'extract_phase',
    'unwrap_phase',
    'phase_derivative',
    'triadic_embedding',
    'instantaneous_frequency',
    'phase_locking_value',
]
