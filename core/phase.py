"""
Phase Extraction Module for Triadic Biosignal Monitor

This module implements phase extraction via Hilbert transform and
triadic embedding ψ(t) = (t, ϕ(t), χ(t)) where:
- ϕ(t) is instantaneous phase
- χ(t) = ∂ϕ/∂t is phase torsion (phase acceleration)

All functions are deterministic and handle edge effects explicitly.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline
from typing import Tuple, Optional, Dict
import warnings


def analytic_signal(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute analytic signal via Hilbert transform.
    
    Parameters
    ----------
    sig : np.ndarray
        Real-valued input signal
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    np.ndarray
        Complex-valued analytic signal
        
    Notes
    -----
    The analytic signal z(t) = sig(t) + i*H[sig(t)] where H is the Hilbert transform.
    This allows extraction of instantaneous amplitude and phase.
    """
    if len(sig) < 4:
        raise ValueError("Signal too short for Hilbert transform")
    
    # Compute Hilbert transform
    analytic = signal.hilbert(sig)
    
    return analytic


def extract_phase(
    sig: np.ndarray,
    fs: float,
    bandpass: Optional[Tuple[float, float]] = None,
    unwrap: bool = True
) -> np.ndarray:
    """
    Extract instantaneous phase ϕ(t) from signal.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    bandpass : tuple of float, optional
        (lowcut, highcut) for pre-filtering, or None (default: None)
    unwrap : bool, optional
        Whether to unwrap phase (default: True)
        
    Returns
    -------
    np.ndarray
        Instantaneous phase in radians
        
    Notes
    -----
    Phase is extracted from the analytic signal: ϕ(t) = arg(H[sig(t)]).
    Unwrapping removes 2π discontinuities for smoother derivatives.
    """
    # Optional pre-filtering
    if bandpass is not None:
        from .preprocessing import bandpass_filter
        sig_filtered = bandpass_filter(sig, fs, bandpass[0], bandpass[1])
    else:
        sig_filtered = sig
    
    # Compute analytic signal
    analytic = analytic_signal(sig_filtered, fs)
    
    # Extract phase
    phase = np.angle(analytic)
    
    # Unwrap phase to remove 2π jumps
    if unwrap:
        phase = unwrap_phase(phase)
    
    return phase


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to remove 2π discontinuities.
    
    Parameters
    ----------
    phase : np.ndarray
        Wrapped phase in radians (typically in [-π, π])
        
    Returns
    -------
    np.ndarray
        Unwrapped phase in radians
        
    Notes
    -----
    Uses numpy's unwrap function which detects jumps > π and adds
    multiples of 2π to maintain continuity.
    """
    return np.unwrap(phase)


def phase_derivative_savgol(
    phase: np.ndarray,
    fs: float,
    window_length: Optional[int] = None,
    polyorder: int = 3
) -> np.ndarray:
    """
    Compute phase derivative χ(t) = ∂ϕ/∂t using Savitzky-Golay filter.
    
    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians
    fs : float
        Sampling frequency in Hz
    window_length : int, optional
        Window length for Savitzky-Golay filter (must be odd).
        If None, automatically determined as ~0.1 seconds.
    polyorder : int, optional
        Polynomial order (default: 3)
        
    Returns
    -------
    np.ndarray
        Phase derivative in rad/s
        
    Notes
    -----
    Savitzky-Golay filter provides noise-robust differentiation by
    fitting local polynomials. This is more stable than simple finite differences.
    """
    # Determine window length if not specified
    if window_length is None:
        window_length = int(0.1 * fs)  # 100 ms window
        if window_length % 2 == 0:  # Must be odd
            window_length += 1
        window_length = max(polyorder + 2, window_length)  # Minimum required
    
    # Validate window length
    if window_length >= len(phase):
        window_length = len(phase) - 1
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < polyorder + 2:
            # Fall back to simple diff
            warnings.warn("Signal too short for Savitzky-Golay, using simple diff")
            return phase_derivative_diff(phase, fs)
    
    # Compute derivative using Savitzky-Golay
    try:
        chi = signal.savgol_filter(phase, window_length, polyorder, deriv=1, delta=1.0/fs)
    except Exception as e:
        warnings.warn(f"Savitzky-Golay failed: {e}. Falling back to finite diff.")
        chi = phase_derivative_diff(phase, fs)
    
    return chi


def phase_derivative_diff(
    phase: np.ndarray,
    fs: float,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute phase derivative using finite differences.
    
    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Difference method: 'central', 'forward', or 'backward' (default: 'central')
        
    Returns
    -------
    np.ndarray
        Phase derivative in rad/s
    """
    dt = 1.0 / fs
    
    if method == 'central':
        # Central difference (more accurate)
        chi = np.gradient(phase, dt)
    elif method == 'forward':
        # Forward difference
        chi = np.diff(phase, prepend=phase[0]) / dt
    elif method == 'backward':
        # Backward difference
        chi = np.diff(phase, append=phase[-1]) / dt
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return chi


def phase_derivative_spline(
    phase: np.ndarray,
    fs: float,
    smoothing: Optional[float] = None
) -> np.ndarray:
    """
    Compute phase derivative using spline interpolation.
    
    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians
    fs : float
        Sampling frequency in Hz
    smoothing : float, optional
        Smoothing parameter for spline (default: automatic)
        
    Returns
    -------
    np.ndarray
        Phase derivative in rad/s
        
    Notes
    -----
    Fits a smoothing spline and computes analytical derivative.
    More robust to noise than finite differences.
    """
    t = np.arange(len(phase)) / fs
    
    # Determine smoothing parameter if not provided
    if smoothing is None:
        smoothing = len(phase)  # Moderate smoothing
    
    # Fit spline
    try:
        spline = UnivariateSpline(t, phase, s=smoothing)
        # Compute derivative
        chi = spline.derivative()(t)
    except Exception as e:
        warnings.warn(f"Spline fitting failed: {e}. Falling back to Savitzky-Golay.")
        chi = phase_derivative_savgol(phase, fs)
    
    return chi


def phase_derivative(
    phase: np.ndarray,
    fs: float,
    method: str = 'savgol',
    **kwargs
) -> np.ndarray:
    """
    Compute phase derivative χ(t) = ∂ϕ/∂t using specified method.
    
    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Method: 'savgol', 'diff', or 'spline' (default: 'savgol')
    **kwargs
        Additional arguments passed to specific derivative function
        
    Returns
    -------
    np.ndarray
        Phase derivative in rad/s
        
    Notes
    -----
    'savgol' (Savitzky-Golay) is recommended for most cases as it provides
    noise-robust differentiation. 'diff' is faster but noisier. 'spline'
    is smoothest but computationally expensive.
    """
    if method == 'savgol':
        return phase_derivative_savgol(phase, fs, **kwargs)
    elif method == 'diff':
        return phase_derivative_diff(phase, fs, **kwargs)
    elif method == 'spline':
        return phase_derivative_spline(phase, fs, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'savgol', 'diff', or 'spline'.")


def triadic_embedding(
    sig: np.ndarray,
    fs: float,
    bandpass: Optional[Tuple[float, float]] = None,
    derivative_method: str = 'savgol'
) -> Dict[str, np.ndarray]:
    """
    Compute triadic embedding ψ(t) = (t, ϕ(t), χ(t)).
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    bandpass : tuple of float, optional
        (lowcut, highcut) for pre-filtering (default: None)
    derivative_method : str, optional
        Method for computing χ(t): 'savgol', 'diff', or 'spline' (default: 'savgol')
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 't': time array in seconds
        - 'phi': instantaneous phase ϕ(t) in radians
        - 'chi': phase torsion χ(t) in rad/s
        - 'amplitude': instantaneous amplitude (optional)
        
    Notes
    -----
    This is the core operator for the triadic spiral-time embedding.
    χ(t) = ∂ϕ/∂t serves as a sensitive marker for regime transitions.
    """
    # Time array
    t = np.arange(len(sig)) / fs
    
    # Extract phase
    phi = extract_phase(sig, fs, bandpass=bandpass, unwrap=True)
    
    # Compute phase derivative
    chi = phase_derivative(phi, fs, method=derivative_method)
    
    # Compute instantaneous amplitude (optional, for completeness)
    if bandpass is not None:
        from .preprocessing import bandpass_filter
        sig_filtered = bandpass_filter(sig, fs, bandpass[0], bandpass[1])
    else:
        sig_filtered = sig
    
    analytic = analytic_signal(sig_filtered, fs)
    amplitude = np.abs(analytic)
    
    return {
        't': t,
        'phi': phi,
        'chi': chi,
        'amplitude': amplitude
    }


def instantaneous_frequency(
    phase: np.ndarray,
    fs: float,
    method: str = 'savgol'
) -> np.ndarray:
    """
    Compute instantaneous frequency from phase.
    
    Parameters
    ----------
    phase : np.ndarray
        Unwrapped phase in radians
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Derivative method (default: 'savgol')
        
    Returns
    -------
    np.ndarray
        Instantaneous frequency in Hz
        
    Notes
    -----
    Instantaneous frequency f(t) = (1/2π) * ∂ϕ/∂t = χ(t) / (2π)
    """
    chi = phase_derivative(phase, fs, method=method)
    freq = chi / (2 * np.pi)
    return freq


def phase_coherence(
    phase1: np.ndarray,
    phase2: np.ndarray,
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Compute phase coherence between two phase signals.
    
    Parameters
    ----------
    phase1 : np.ndarray
        First phase signal (radians)
    phase2 : np.ndarray
        Second phase signal (radians)
    window_size : int, optional
        Window size for local coherence computation.
        If None, computes global coherence.
        
    Returns
    -------
    np.ndarray
        Phase coherence (0 = no coherence, 1 = perfect coherence)
        
    Notes
    -----
    Coherence is computed as |<exp(i(ϕ1 - ϕ2))>| where <> is averaging.
    """
    # Phase difference
    phase_diff = phase1 - phase2
    
    # Complex representation
    complex_diff = np.exp(1j * phase_diff)
    
    if window_size is None:
        # Global coherence
        coherence = np.abs(np.mean(complex_diff))
        coherence = np.full(len(phase1), coherence)
    else:
        # Local coherence (sliding window)
        coherence = np.zeros(len(phase1))
        half_window = window_size // 2
        
        for i in range(len(phase1)):
            start = max(0, i - half_window)
            end = min(len(phase1), i + half_window + 1)
            coherence[i] = np.abs(np.mean(complex_diff[start:end]))
    
    return coherence


def phase_locking_value(
    phase1: np.ndarray,
    phase2: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) between two signals.
    
    Parameters
    ----------
    phase1 : np.ndarray
        First phase signal (radians)
    phase2 : np.ndarray
        Second phase signal (radians)
    window_size : int
        Window size for PLV computation
        
    Returns
    -------
    np.ndarray
        Phase locking value (0 = no locking, 1 = perfect locking)
        
    Notes
    -----
    PLV measures phase synchronization between signals.
    Computed as |<exp(i(ϕ1 - ϕ2))>| over sliding windows.
    """
    return phase_coherence(phase1, phase2, window_size=window_size)


def check_phase_quality(
    phase: np.ndarray,
    chi: np.ndarray,
    fs: float
) -> Tuple[float, Dict[str, bool]]:
    """
    Assess quality of phase extraction.
    
    Parameters
    ----------
    phase : np.ndarray
        Extracted phase (radians)
    chi : np.ndarray
        Phase derivative (rad/s)
    fs : float
        Sampling frequency (Hz)
        
    Returns
    -------
    quality_score : float
        Quality score from 0.0 to 1.0
    flags : dict
        Quality assessment flags
        
    Notes
    -----
    Checks for:
    - Phase continuity (no large jumps after unwrapping)
    - Reasonable derivative values
    - No NaN or infinite values
    """
    flags = {}
    
    # Check for NaN or inf
    flags['no_nan'] = not (np.any(np.isnan(phase)) or np.any(np.isnan(chi)))
    flags['no_inf'] = not (np.any(np.isinf(phase)) or np.any(np.isinf(chi)))
    
    # Check phase continuity (jumps should be small after unwrapping)
    phase_diff = np.abs(np.diff(phase))
    flags['continuous_phase'] = np.percentile(phase_diff, 99) < 2 * np.pi
    
    # Check derivative is reasonable (not too extreme)
    # For typical biosignals, chi should be in a reasonable range
    chi_abs = np.abs(chi)
    flags['reasonable_derivative'] = np.percentile(chi_abs, 99) < 100 * 2 * np.pi  # < 100 Hz
    
    # Overall quality
    quality_score = np.mean(list(flags.values()))
    
    return quality_score, flags
