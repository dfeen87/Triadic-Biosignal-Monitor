"""
Feature Extraction Module for Triadic Biosignal Monitor

This module computes deviation features from baseline:
- ΔS: Spectral/morphological deviation
- ΔI: Information/entropy deviation
- ΔC: Coupling/coherence deviation (EEG-ECG)

All deviations are baseline-normalized and interpretable.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, Optional, Tuple, List
import warnings


# ========================================
# Spectral/Morphological Features (ΔS)
# ========================================

def compute_bandpower(
    sig: np.ndarray,
    fs: float,
    freq_band: Tuple[float, float],
    method: str = 'welch'
) -> float:
    """
    Compute power in a frequency band.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    freq_band : tuple of float
        (low, high) frequency band in Hz
    method : str, optional
        Method: 'welch' or 'fft' (default: 'welch')
        
    Returns
    -------
    float
        Band power
    """
    if method == 'welch':
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    elif method == 'fft':
        freqs = np.fft.rfftfreq(len(sig), 1/fs)
        fft = np.fft.rfft(sig)
        psd = np.abs(fft) ** 2 / len(sig)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Find indices in frequency band
    idx = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    
    # Integrate power
    band_power = np.trapz(psd[idx], freqs[idx])
    
    return band_power


def compute_eeg_bandpowers(
    sig: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Compute EEG power in standard frequency bands.
    
    Parameters
    ----------
    sig : np.ndarray
        EEG signal
    fs : float
        Sampling frequency in Hz
    bands : dict, optional
        Dictionary of band names to (low, high) frequency ranges.
        Default: standard EEG bands (delta, theta, alpha, beta, gamma)
        
    Returns
    -------
    dict
        Dictionary of band names to power values
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    bandpowers = {}
    for band_name, freq_range in bands.items():
        bandpowers[band_name] = compute_bandpower(sig, fs, freq_range)
    
    return bandpowers


def compute_spectral_centroid(sig: np.ndarray, fs: float) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    float
        Spectral centroid in Hz
    """
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    
    # Compute centroid as weighted mean
    centroid = np.sum(freqs * psd) / np.sum(psd)
    
    return centroid


def compute_hrv_metrics(
    rr_intervals: np.ndarray
) -> Dict[str, float]:
    """
    Compute time-domain HRV metrics from RR intervals.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds
        
    Returns
    -------
    dict
        Dictionary with HRV metrics:
        - 'RMSSD': Root mean square of successive differences
        - 'SDNN': Standard deviation of NN intervals
        - 'pNN50': Percentage of successive RR intervals differing by > 50ms
    """
    if len(rr_intervals) < 2:
        return {'RMSSD': 0.0, 'SDNN': 0.0, 'pNN50': 0.0}
    
    # RMSSD
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # SDNN
    sdnn = np.std(rr_intervals, ddof=1)
    
    # pNN50
    nn50 = np.sum(np.abs(diff_rr) > 0.05)  # 50 ms = 0.05 s
    pnn50 = 100 * nn50 / len(diff_rr) if len(diff_rr) > 0 else 0.0
    
    return {
        'RMSSD': rmssd,
        'SDNN': sdnn,
        'pNN50': pnn50
    }


def compute_lf_hf_ratio(
    rr_intervals: np.ndarray,
    fs_rr: float = 4.0
) -> float:
    """
    Compute LF/HF ratio from RR intervals.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds
    fs_rr : float, optional
        Effective sampling rate for RR series (default: 4 Hz)
        
    Returns
    -------
    float
        LF/HF ratio
        
    Notes
    -----
    LF (Low Frequency): 0.04-0.15 Hz
    HF (High Frequency): 0.15-0.4 Hz
    """
    if len(rr_intervals) < 10:
        return 1.0  # Neutral ratio
    
    # Resample RR intervals to uniform time base
    time_rr = np.cumsum(rr_intervals)
    time_uniform = np.arange(0, time_rr[-1], 1.0/fs_rr)
    rr_uniform = np.interp(time_uniform, time_rr, rr_intervals)
    
    # Compute PSD
    freqs, psd = signal.welch(rr_uniform, fs=fs_rr, nperseg=min(len(rr_uniform), 256))
    
    # LF and HF power
    lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], 
                        freqs[(freqs >= 0.04) & (freqs < 0.15)])
    hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], 
                        freqs[(freqs >= 0.15) & (freqs < 0.4)])
    
    # LF/HF ratio
    lf_hf = lf_power / (hf_power + 1e-10)  # Avoid division by zero
    
    return lf_hf


def compute_delta_S_eeg(
    sig: np.ndarray,
    baseline_sig: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> float:
    """
    Compute spectral/morphological deviation ΔS for EEG.
    
    Parameters
    ----------
    sig : np.ndarray
        Current EEG signal window
    baseline_sig : np.ndarray
        Baseline EEG signal
    fs : float
        Sampling frequency in Hz
    bands : dict, optional
        Frequency bands to analyze
        
    Returns
    -------
    float
        Normalized ΔS (z-score)
        
    Notes
    -----
    Computes deviation in:
    1. Band powers (delta, theta, alpha, beta, gamma)
    2. Spectral centroid
    """
    # Compute bandpowers for current and baseline
    current_bp = compute_eeg_bandpowers(sig, fs, bands)
    baseline_bp = compute_eeg_bandpowers(baseline_sig, fs, bands)
    
    # Compute deviations for each band
    deviations = []
    for band in current_bp.keys():
        current = current_bp[band]
        baseline = baseline_bp[band]
        # Normalized deviation (z-score style)
        dev = (current - baseline) / (baseline + 1e-10)
        deviations.append(np.abs(dev))
    
    # Spectral centroid deviation
    current_centroid = compute_spectral_centroid(sig, fs)
    baseline_centroid = compute_spectral_centroid(baseline_sig, fs)
    centroid_dev = np.abs(current_centroid - baseline_centroid) / (baseline_centroid + 1e-10)
    deviations.append(centroid_dev)
    
    # Aggregate deviations
    delta_S = np.mean(deviations)
    
    return delta_S


def compute_delta_S_ecg(
    rr_intervals: np.ndarray,
    baseline_rr: np.ndarray,
    fs_rr: float = 4.0
) -> float:
    """
    Compute spectral/morphological deviation ΔS for ECG/HRV.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        Current RR intervals
    baseline_rr : np.ndarray
        Baseline RR intervals
    fs_rr : float, optional
        Sampling rate for RR series (default: 4 Hz)
        
    Returns
    -------
    float
        Normalized ΔS
        
    Notes
    -----
    Computes deviation in:
    1. HRV metrics (RMSSD, SDNN)
    2. LF/HF ratio
    """
    # Current metrics
    current_hrv = compute_hrv_metrics(rr_intervals)
    current_lf_hf = compute_lf_hf_ratio(rr_intervals, fs_rr)
    
    # Baseline metrics
    baseline_hrv = compute_hrv_metrics(baseline_rr)
    baseline_lf_hf = compute_lf_hf_ratio(baseline_rr, fs_rr)
    
    # Compute deviations
    deviations = []
    
    # HRV time-domain metrics
    for key in ['RMSSD', 'SDNN']:
        current = current_hrv[key]
        baseline = baseline_hrv[key]
        dev = np.abs(current - baseline) / (baseline + 1e-10)
        deviations.append(dev)
    
    # LF/HF ratio
    lf_hf_dev = np.abs(current_lf_hf - baseline_lf_hf) / (baseline_lf_hf + 1e-10)
    deviations.append(lf_hf_dev)
    
    # Aggregate
    delta_S = np.mean(deviations)
    
    return delta_S


# ========================================
# Information/Entropy Features (ΔI)
# ========================================

def permutation_entropy(
    sig: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Compute permutation entropy.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    order : int, optional
        Embedding dimension (default: 3)
    delay : int, optional
        Time delay (default: 1)
    normalize : bool, optional
        Normalize to [0, 1] (default: True)
        
    Returns
    -------
    float
        Permutation entropy
        
    Notes
    -----
    Permutation entropy quantifies complexity by analyzing ordinal patterns.
    Lower values indicate reduced complexity/adaptability.
    """
    n = len(sig)
    
    # Create embedded matrix
    permutations = {}
    for i in range(n - delay * (order - 1)):
        # Extract embedded vector
        idx = np.arange(order) * delay + i
        embedded = sig[idx]
        
        # Get permutation pattern (argsort gives rank order)
        pattern = tuple(np.argsort(embedded))
        
        # Count pattern
        permutations[pattern] = permutations.get(pattern, 0) + 1
    
    # Compute entropy
    total = sum(permutations.values())
    if total == 0:
        return 0.0
    
    pe = 0.0
    for count in permutations.values():
        if count > 0:
            p = count / total
            pe -= p * np.log2(p)
    
    # Normalize
    if normalize:
        max_entropy = np.log2(np.math.factorial(order))
        pe = pe / max_entropy if max_entropy > 0 else 0.0
    
    return pe


def sample_entropy(
    sig: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute sample entropy.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    m : int, optional
        Pattern length (default: 2)
    r : float, optional
        Tolerance (default: 0.2 * std(sig))
        
    Returns
    -------
    float
        Sample entropy
        
    Notes
    -----
    Sample entropy measures regularity and unpredictability.
    Lower values indicate more regular (less complex) signals.
    """
    n = len(sig)
    
    if r is None:
        r = 0.2 * np.std(sig, ddof=1)
    
    def _maxdist(xi, xj):
        """Maximum distance between embedded vectors."""
        return np.max(np.abs(xi - xj))
    
    def _phi(m_val):
        """Count similar patterns."""
        patterns = np.array([sig[i:i + m_val] for i in range(n - m_val)])
        count = 0
        for i in range(len(patterns) - 1):
            for j in range(i + 1, len(patterns)):
                if _maxdist(patterns[i], patterns[j]) <= r:
                    count += 1
        return count
    
    # Count matches for m and m+1
    matches_m = _phi(m)
    matches_m1 = _phi(m + 1)
    
    # Compute sample entropy
    if matches_m > 0 and matches_m1 > 0:
        sampen = -np.log(matches_m1 / matches_m)
    else:
        sampen = np.inf  # No regularity found
    
    return sampen if np.isfinite(sampen) else 10.0  # Cap at reasonable value


def approximate_entropy(
    sig: np.ndarray,
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute approximate entropy.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal
    m : int, optional
        Pattern length (default: 2)
    r : float, optional
        Tolerance (default: 0.2 * std(sig))
        
    Returns
    -------
    float
        Approximate entropy
    """
    n = len(sig)
    
    if r is None:
        r = 0.2 * np.std(sig, ddof=1)
    
    def _phi(m_val):
        patterns = np.array([sig[i:i + m_val] for i in range(n - m_val + 1)])
        phi_vals = []
        
        for i in range(len(patterns)):
            # Count similar patterns
            matches = np.sum([np.max(np.abs(patterns[i] - patterns[j])) <= r 
                            for j in range(len(patterns))])
            phi_vals.append(matches / len(patterns))
        
        return np.mean(np.log(phi_vals))
    
    return _phi(m) - _phi(m + 1)


def compute_delta_I(
    sig: np.ndarray,
    baseline_sig: np.ndarray,
    method: str = 'permutation_entropy',
    **kwargs
) -> float:
    """
    Compute information/entropy deviation ΔI.
    
    Parameters
    ----------
    sig : np.ndarray
        Current signal window
    baseline_sig : np.ndarray
        Baseline signal
    method : str, optional
        Entropy method: 'permutation_entropy', 'sample_entropy', 'approximate_entropy'
        (default: 'permutation_entropy')
    **kwargs
        Additional parameters for entropy functions
        
    Returns
    -------
    float
        Normalized ΔI (baseline-normalized absolute deviation)
        
    Notes
    -----
    A decrease in entropy indicates reduced complexity and adaptability,
    which may precede instability events.
    """
    # Select entropy function
    if method == 'permutation_entropy':
        entropy_fn = permutation_entropy
    elif method == 'sample_entropy':
        entropy_fn = sample_entropy
    elif method == 'approximate_entropy':
        entropy_fn = approximate_entropy
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute entropies
    current_entropy = entropy_fn(sig, **kwargs)
    baseline_entropy = entropy_fn(baseline_sig, **kwargs)
    
    # Normalized deviation
    delta_I = np.abs(current_entropy - baseline_entropy) / (baseline_entropy + 1e-10)
    
    return delta_I


# ========================================
# Coupling/Coherence Features (ΔC)
# ========================================

def magnitude_squared_coherence(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: float,
    freq_bands: Optional[List[Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Compute magnitude-squared coherence between two signals.
    
    Parameters
    ----------
    sig1 : np.ndarray
        First signal
    sig2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency in Hz
    freq_bands : list of tuples, optional
        Frequency bands to compute coherence in
        
    Returns
    -------
    dict
        Coherence values for each frequency band
    """
    # Compute coherence
    freqs, coh = signal.coherence(sig1, sig2, fs=fs, nperseg=min(len(sig1), 256))
    
    if freq_bands is None:
        # Return mean coherence
        return {'mean': np.mean(coh)}
    
    # Compute coherence in each band
    coherences = {}
    for low, high in freq_bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        coherences[f'{low}-{high}Hz'] = np.mean(coh[idx])
    
    return coherences


def phase_locking_value(
    phase1: np.ndarray,
    phase2: np.ndarray,
    window_size: Optional[int] = None
) -> float:
    """
    Compute Phase Locking Value (PLV) between two phase signals.
    
    Parameters
    ----------
    phase1 : np.ndarray
        First phase signal (radians)
    phase2 : np.ndarray
        Second phase signal (radians)
    window_size : int, optional
        Window size for local PLV. If None, compute global PLV.
        
    Returns
    -------
    float
        Phase locking value (0 = no locking, 1 = perfect locking)
    """
    # Phase difference
    phase_diff = phase1 - phase2
    
    # Complex representation
    complex_diff = np.exp(1j * phase_diff)
    
    if window_size is None:
        # Global PLV
        plv = np.abs(np.mean(complex_diff))
    else:
        # Mean over sliding windows
        plvs = []
        for i in range(len(phase1) - window_size + 1):
            window = complex_diff[i:i + window_size]
            plvs.append(np.abs(np.mean(window)))
        plv = np.mean(plvs)
    
    return plv


def compute_delta_C(
    sig1: np.ndarray,
    sig2: np.ndarray,
    baseline_sig1: np.ndarray,
    baseline_sig2: np.ndarray,
    fs: float,
    method: str = 'plv',
    phase1: Optional[np.ndarray] = None,
    phase2: Optional[np.ndarray] = None,
    baseline_phase1: Optional[np.ndarray] = None,
    baseline_phase2: Optional[np.ndarray] = None
) -> float:
    """
    Compute coupling/coherence deviation ΔC between EEG and ECG.
    
    Parameters
    ----------
    sig1 : np.ndarray
        Current first signal (e.g., EEG)
    sig2 : np.ndarray
        Current second signal (e.g., ECG)
    baseline_sig1 : np.ndarray
        Baseline first signal
    baseline_sig2 : np.ndarray
        Baseline second signal
    fs : float
        Sampling frequency in Hz
    method : str, optional
        Method: 'plv' or 'coherence' (default: 'plv')
    phase1, phase2 : np.ndarray, optional
        Pre-computed phases (for PLV method)
    baseline_phase1, baseline_phase2 : np.ndarray, optional
        Pre-computed baseline phases (for PLV method)
        
    Returns
    -------
    float
        Normalized ΔC (baseline-normalized absolute deviation)
        
    Notes
    -----
    Measures loss of heart-brain coherence/coupling, which may
    indicate regulatory breakdown.
    """
    if method == 'plv':
        # Use phase locking value
        if phase1 is None or phase2 is None:
            # Extract phases
            from .phase import extract_phase
            phase1 = extract_phase(sig1, fs)
            phase2 = extract_phase(sig2, fs)
        
        if baseline_phase1 is None or baseline_phase2 is None:
            from .phase import extract_phase
            baseline_phase1 = extract_phase(baseline_sig1, fs)
            baseline_phase2 = extract_phase(baseline_sig2, fs)
        
        current_coupling = phase_locking_value(phase1, phase2)
        baseline_coupling = phase_locking_value(baseline_phase1, baseline_phase2)
        
    elif method == 'coherence':
        # Use magnitude-squared coherence
        current_coh = magnitude_squared_coherence(sig1, sig2, fs)
        baseline_coh = magnitude_squared_coherence(baseline_sig1, baseline_sig2, fs)
        
        current_coupling = current_coh['mean']
        baseline_coupling = baseline_coh['mean']
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalized deviation
    delta_C = np.abs(current_coupling - baseline_coupling) / (baseline_coupling + 1e-10)
    
    return delta_C
