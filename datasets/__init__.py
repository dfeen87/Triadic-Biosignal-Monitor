"""
Datasets Module for Triadic Biosignal Monitor

This module provides dataset loading and synthetic signal generation:
- loaders: Load EEG/ECG data from various formats (EDF, FIF, WFDB, etc.)
- synthetic: Generate synthetic signals with controlled regime changes for testing

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

__version__ = "0.1.0"

# Import key functions for convenience
from .loaders import (
    load_edf,
    load_fif,
    load_csv,
    load_numpy,
    auto_load
)

from .synthetic import (
    generate_synthetic_eeg,
    generate_synthetic_ecg,
    add_regime_change,
    generate_test_dataset
)

__all__ = [
    # Loaders
    'load_edf',
    'load_fif',
    'load_csv',
    'load_numpy',
    'auto_load',
    
    # Synthetic generators
    'generate_synthetic_eeg',
    'generate_synthetic_ecg',
    'add_regime_change',
    'generate_test_dataset',
]
