"""
Preprocessing Module for Triadic Biosignal Monitor

This module provides robust signal preprocessing functions including:
- Bandpass filtering
- Artifact detection and rejection
- Signal synchronization
- Quality assessment

All functions are deterministic and explicitly handle edge cases.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, Union
import warnings


def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
    method: str = 'butterworth'
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal (1D array)
    fs : float
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    order : int, optional
        Filter order (default: 4)
    method : str, optional
        Filter type: 'butterworth' or 'chebyshev' (default: 'butterworth')
        
    Returns
    -------
    np.ndarray
        Filtered signal
        
    Raises
    ------
    ValueError
        If cutoff frequencies are invalid or fs is too low
        
    Notes
    -----
    Uses zero-phase filtering (filtfilt) to avoid phase distortion.
    Edge effects are handled by padding.
    """
    # Validation
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive")
    if lowcut >= highcut:
        raise ValueError("lowcut must be less than highcut")
    if highcut >= fs / 2:
        raise ValueError(f"highcut ({highcut} Hz) must be less than Nyquist frequency ({fs/2} Hz)")
    if len(sig) < 3 * order:
        raise ValueError(f"Signal too short for filter order {order}")
        
    # Normalize frequencies to Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design filter
    if method == 'butterworth':
        b, a = signal.butter(order, [low, high], btype='band')
    elif method == 'chebyshev':
        b, a = signal.cheby1(order, 0.1, [low, high], btype='band')
    else:
        raise ValueError(f"Unknown filter method: {method}")
    
    # Apply zero-phase filter
    try:
        filtered = signal.filtfilt(b, a, sig, padlen=3*order)
    except Exception as e:
        warnings.warn(f"filtfilt failed: {e}. Falling back to lfilter.")
        filtered = signal.lfilter(b, a, sig)
        
    return filtered


def notch_filter(
    sig: np.ndarray,
    fs: float,
    freq: float = 60.0,
    quality: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove line noise (50/60 Hz).
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    freq : float, optional
        Frequency to remove in Hz (default: 60.0 for US)
    quality : float, optional
        Quality factor (default: 30.0)
        
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    
    # Design notch filter
    b, a = signal.iirnotch(w0, quality)
    
    # Apply filter
    filtered = signal.filtfilt(b, a, sig)
    
    return filtered


def detect_artifacts_threshold(
    sig: np.ndarray,
    fs: float,
    threshold: float = 5.0,
    window_size: float = 1.0
) -> np.ndarray:
    """
    Detect artifacts using threshold-based method.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    threshold : float, optional
        Z-score threshold for artifact detection (default: 5.0)
    window_size : float, optional
        Window size in seconds for local statistics (default: 1.0)
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = artifact, False = clean)
        
    Notes
    -----
    Detects artifacts based on:
    1. Amplitude exceeding threshold * std
    2. Rapid changes (high derivative)
    """
    n_samples = len(sig)
    window_samples = int(window_size * fs)
    artifact_mask = np.zeros(n_samples, dtype=bool)
    
    # Compute robust statistics (median-based)
    median = np.median(sig)
    mad = np.median(np.abs(sig - median))
    robust_std = 1.4826 * mad  # MAD to std conversion
    
    # Amplitude-based detection
    amplitude_outliers = np.abs(sig - median) > threshold * robust_std
    artifact_mask |= amplitude_outliers
    
    # Derivative-based detection (detect jumps)
    if n_samples > 1:
        diff = np.diff(sig)
        diff_median = np.median(diff)
        diff_mad = np.median(np.abs(diff - diff_median))
        diff_std = 1.4826 * diff_mad
        
        derivative_outliers = np.abs(diff - diff_median) > threshold * diff_std
        # Extend to original length
        derivative_mask = np.zeros(n_samples, dtype=bool)
        derivative_mask[1:] |= derivative_outliers
        derivative_mask[:-1] |= derivative_outliers
        artifact_mask |= derivative_mask
    
    # Dilate artifact regions (expand by window_size/2 on each side)
    half_window = window_samples // 2
    if half_window > 0:
        artifact_mask = np.convolve(artifact_mask.astype(float), 
                                     np.ones(half_window), 
                                     mode='same') > 0
    
    return artifact_mask


def quality_check(
    sig: np.ndarray,
    fs: float,
    min_length: float = 5.0,
    max_artifact_fraction: float = 0.3
) -> Tuple[float, Dict[str, bool]]:
    """
    Assess signal quality.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    min_length : float, optional
        Minimum required signal length in seconds (default: 5.0)
    max_artifact_fraction : float, optional
        Maximum allowed fraction of artifacts (default: 0.3)
        
    Returns
    -------
    quality_score : float
        Quality score from 0.0 (poor) to 1.0 (excellent)
    flags : dict
        Dictionary of quality flags:
        - 'sufficient_length': bool
        - 'low_artifacts': bool
        - 'no_flatline': bool
        - 'adequate_variance': bool
    """
    flags = {}
    
    # Check length
    duration = len(sig) / fs
    flags['sufficient_length'] = duration >= min_length
    
    # Check for artifacts
    artifact_mask = detect_artifacts_threshold(sig, fs)
    artifact_fraction = np.mean(artifact_mask)
    flags['low_artifacts'] = artifact_fraction < max_artifact_fraction
    
    # Check for flatline (constant signal)
    variance = np.var(sig)
    flags['no_flatline'] = variance > 1e-10
    
    # Check for adequate variance (not too flat)
    # Use 1e-6 for better numerical stability than 1e-10
    normalized_std = np.std(sig) / (np.abs(np.mean(sig)) + 1e-6)
    flags['adequate_variance'] = normalized_std > 0.01
    
    # Compute overall quality score
    quality_score = np.mean(list(flags.values()))
    
    return quality_score, flags


def synchronize_signals(
    sig1: np.ndarray,
    sig2: np.ndarray,
    ts1: np.ndarray,
    ts2: np.ndarray,
    method: str = 'interpolate'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synchronize two signals with different timestamps.
    
    Parameters
    ----------
    sig1 : np.ndarray
        First signal
    sig2 : np.ndarray
        Second signal
    ts1 : np.ndarray
        Timestamps for sig1 (in seconds)
    ts2 : np.ndarray
        Timestamps for sig2 (in seconds)
    method : str, optional
        Synchronization method: 'interpolate' or 'nearest' (default: 'interpolate')
        
    Returns
    -------
    sig1_sync : np.ndarray
        Synchronized first signal
    sig2_sync : np.ndarray
        Synchronized second signal
    ts_common : np.ndarray
        Common timestamp array
        
    Notes
    -----
    Uses linear interpolation to resample signals onto common timebase.
    Common timebase is the intersection of the two timestamp ranges.
    """
    # Find common time range
    t_start = max(ts1[0], ts2[0])
    t_end = min(ts1[-1], ts2[-1])
    
    if t_start >= t_end:
        raise ValueError("No temporal overlap between signals")
    
    # Create common timebase (use higher sampling rate)
    dt1 = np.median(np.diff(ts1))
    dt2 = np.median(np.diff(ts2))
    dt_common = min(dt1, dt2)
    
    ts_common = np.arange(t_start, t_end, dt_common)
    
    # Interpolate both signals onto common timebase
    if method == 'interpolate':
        sig1_sync = np.interp(ts_common, ts1, sig1)
        sig2_sync = np.interp(ts_common, ts2, sig2)
    elif method == 'nearest':
        # Find nearest neighbor indices
        idx1 = np.searchsorted(ts1, ts_common)
        idx2 = np.searchsorted(ts2, ts_common)
        # Clip to valid range
        idx1 = np.clip(idx1, 0, len(sig1) - 1)
        idx2 = np.clip(idx2, 0, len(sig2) - 1)
        sig1_sync = sig1[idx1]
        sig2_sync = sig2[idx2]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return sig1_sync, sig2_sync, ts_common


def remove_baseline_drift(
    sig: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    method: str = 'highpass'
) -> np.ndarray:
    """
    Remove baseline drift from signal.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    cutoff : float, optional
        Cutoff frequency in Hz (default: 0.5)
    method : str, optional
        Method: 'highpass', 'detrend', or 'polynomial' (default: 'highpass')
        
    Returns
    -------
    np.ndarray
        Signal with baseline drift removed
    """
    if method == 'highpass':
        # High-pass filter to remove low frequencies
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(4, normal_cutoff, btype='high')
        corrected = signal.filtfilt(b, a, sig)
        
    elif method == 'detrend':
        # Linear detrend
        corrected = signal.detrend(sig, type='linear')
        
    elif method == 'polynomial':
        # Polynomial detrend (order 3)
        x = np.arange(len(sig))
        coeffs = np.polyfit(x, sig, deg=3)
        trend = np.polyval(coeffs, x)
        corrected = sig - trend
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corrected


def preprocess_eeg(
    eeg_signal: np.ndarray,
    fs: float,
    bandpass: Tuple[float, float] = (0.5, 50.0),
    notch_freq: Optional[float] = 60.0,
    remove_drift: bool = True,
    artifact_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for EEG signal.
    
    Parameters
    ----------
    eeg_signal : np.ndarray
        Raw EEG signal
    fs : float
        Sampling frequency in Hz
    bandpass : tuple of float, optional
        (lowcut, highcut) in Hz (default: (0.5, 50.0))
    notch_freq : float or None, optional
        Notch filter frequency in Hz, or None to skip (default: 60.0)
    remove_drift : bool, optional
        Whether to remove baseline drift (default: True)
    artifact_threshold : float, optional
        Z-score threshold for artifact detection (default: 5.0)
        
    Returns
    -------
    processed_signal : np.ndarray
        Preprocessed EEG signal
    artifact_mask : np.ndarray
        Boolean mask indicating artifacts
    """
    sig = eeg_signal.copy()
    
    # 1. Remove baseline drift
    if remove_drift:
        sig = remove_baseline_drift(sig, fs, cutoff=0.5)
    
    # 2. Bandpass filter
    sig = bandpass_filter(sig, fs, bandpass[0], bandpass[1])
    
    # 3. Notch filter (line noise)
    if notch_freq is not None:
        sig = notch_filter(sig, fs, freq=notch_freq)
    
    # 4. Detect artifacts
    artifact_mask = detect_artifacts_threshold(sig, fs, threshold=artifact_threshold)
    
    return sig, artifact_mask


def preprocess_ecg(
    ecg_signal: np.ndarray,
    fs: float,
    bandpass: Tuple[float, float] = (0.5, 40.0),
    notch_freq: Optional[float] = 60.0,
    remove_drift: bool = True,
    artifact_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for ECG signal.
    
    Parameters
    ----------
    ecg_signal : np.ndarray
        Raw ECG signal
    fs : float
        Sampling frequency in Hz
    bandpass : tuple of float, optional
        (lowcut, highcut) in Hz (default: (0.5, 40.0))
    notch_freq : float or None, optional
        Notch filter frequency in Hz, or None to skip (default: 60.0)
    remove_drift : bool, optional
        Whether to remove baseline drift (default: True)
    artifact_threshold : float, optional
        Z-score threshold for artifact detection (default: 5.0)
        
    Returns
    -------
    processed_signal : np.ndarray
        Preprocessed ECG signal
    artifact_mask : np.ndarray
        Boolean mask indicating artifacts
    """
    sig = ecg_signal.copy()
    
    # 1. Remove baseline drift
    if remove_drift:
        sig = remove_baseline_drift(sig, fs, cutoff=0.5)
    
    # 2. Bandpass filter
    sig = bandpass_filter(sig, fs, bandpass[0], bandpass[1])
    
    # 3. Notch filter (line noise)
    if notch_freq is not None:
        sig = notch_filter(sig, fs, freq=notch_freq)
    
    # 4. Detect artifacts
    artifact_mask = detect_artifacts_threshold(sig, fs, threshold=artifact_threshold)
    
    return sig, artifact_mask
