"""
Pipelines Module for Triadic Biosignal Monitor

This module provides analysis pipelines for different operational modes:
- EEG-only: Ablation mode using only EEG signal
- ECG-only: Ablation mode using only ECG/HRV signal
- Coupled: Full mode using both EEG and ECG with coupling term
- Streaming: Near-real-time processing mode (Phase 2)

All pipelines support the same interface for consistency and comparability.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

__version__ = "0.1.0"

# Import pipeline classes for convenience
from .eeg_only import EEGOnlyPipeline
from .ecg_only import ECGOnlyPipeline
from .coupled import CoupledPipeline

__all__ = [
    'EEGOnlyPipeline',
    'ECGOnlyPipeline',
    'CoupledPipeline',
]
