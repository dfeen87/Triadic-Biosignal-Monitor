"""
Synthetic Signal Generator for Triadic Biosignal Monitor

Generates synthetic EEG and ECG signals with controlled regime changes
for testing and validation of the instability detection framework.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


def generate_pink_noise(n_samples: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generate pink (1/f) noise.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    alpha : float, optional
        Spectral slope (default: 1.0 for pink noise)
        
    Returns
    -------
    np.ndarray
        Pink noise signal
    """
    # Generate white noise in frequency domain
    white = np.fft.rfft(np.random.randn(n_samples))
    
    # Apply 1/f^alpha filter
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero at DC
    
    pink_spectrum = white / (freqs ** (alpha / 2))
    
    # Transform back to time domain
    pink = np.fft.irfft(pink_spectrum, n=n_samples)
    
    # Normalize
    pink = pink / np.std(pink)
    
    return pink


def generate_oscillation(
    n_samples: int,
    fs: float,
    freq: float,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate sinusoidal oscillation.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    fs : float
        Sampling frequency in Hz
    freq : float
        Oscillation frequency in Hz
    amplitude : float, optional
        Amplitude (default: 1.0)
    phase : float, optional
        Phase offset in radians (default: 0.0)
        
    Returns
    -------
    np.ndarray
        Oscillatory signal
    """
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_synthetic_eeg(
    duration: float,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float, float]]] = None,
    noise_level: float = 0.2
) -> np.ndarray:
    """
    Generate synthetic EEG signal with multiple frequency bands.
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float, optional
        Sampling frequency in Hz (default: 256.0)
    bands : dict, optional
        Dictionary mapping band names to (freq_range, amplitude) tuples.
        Default: standard EEG bands
    noise_level : float, optional
        Background noise level (default: 0.2)
        
    Returns
    -------
    np.ndarray
        Synthetic EEG signal
    """
    n_samples = int(duration * fs)
    
    # Default EEG bands with typical frequencies and amplitudes
    if bands is None:
        bands = {
            'delta': (2.0, 30.0),    # 0.5-4 Hz, high amplitude
            'theta': (6.0, 20.0),    # 4-8 Hz
            'alpha': (10.0, 40.0),   # 8-13 Hz, dominant in relaxed state
            'beta': (20.0, 15.0),    # 13-30 Hz
            'gamma': (40.0, 5.0)     # 30-50 Hz, low amplitude
        }
    
    # Initialize with pink noise background
    eeg = noise_level * generate_pink_noise(n_samples, alpha=1.0)
    
    # Add oscillatory components
    for band_name, (freq, amplitude) in bands.items():
        # Add some frequency jitter for realism
        freq_jitter = freq * 0.1 * np.random.randn()
        osc = generate_oscillation(n_samples, fs, freq + freq_jitter, amplitude)
        
        # Add amplitude modulation for realism
        t = np.arange(n_samples) / fs
        mod_freq = 0.1  # Slow amplitude modulation
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
        
        eeg += osc * modulation
    
    # Normalize
    eeg = eeg / np.std(eeg)
    
    return eeg


def generate_synthetic_ecg(
    duration: float,
    fs: float = 256.0,
    heart_rate: float = 70.0,
    hrv_std: float = 0.05
) -> np.ndarray:
    """
    Generate synthetic ECG signal with realistic QRS complexes.
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float, optional
        Sampling frequency in Hz (default: 256.0)
    heart_rate : float, optional
        Mean heart rate in BPM (default: 70.0)
    hrv_std : float, optional
        Heart rate variability (std of RR intervals in seconds) (default: 0.05)
        
    Returns
    -------
    np.ndarray
        Synthetic ECG signal
    """
    n_samples = int(duration * fs)
    ecg = np.zeros(n_samples)
    
    # Mean RR interval
    rr_mean = 60.0 / heart_rate  # seconds
    
    # Generate beat times with HRV
    t = 0
    beat_times = []
    while t < duration:
        # Add HRV noise
        rr = rr_mean + hrv_std * np.random.randn()
        rr = max(rr, 0.3)  # Minimum 0.3s (200 BPM)
        t += rr
        if t < duration:
            beat_times.append(t)
    
    # Add QRS complexes at beat times
    for beat_time in beat_times:
        beat_idx = int(beat_time * fs)
        
        # QRS complex shape (simplified)
        qrs_duration = int(0.08 * fs)  # 80ms
        qrs_start = max(0, beat_idx - qrs_duration // 2)
        qrs_end = min(n_samples, beat_idx + qrs_duration // 2)
        
        # Create QRS complex (biphasic)
        qrs_samples = qrs_end - qrs_start
        t_qrs = np.linspace(-1, 1, qrs_samples)
        
        # Q wave (small negative)
        q_wave = -0.2 * np.exp(-((t_qrs + 0.5) ** 2) / 0.01)
        
        # R wave (large positive)
        r_wave = 1.0 * np.exp(-(t_qrs ** 2) / 0.005)
        
        # S wave (small negative)
        s_wave = -0.3 * np.exp(-((t_qrs - 0.5) ** 2) / 0.01)
        
        qrs = q_wave + r_wave + s_wave
        
        # Add to ECG
        ecg[qrs_start:qrs_end] += qrs
        
        # Add T wave (after QRS)
        t_wave_start = beat_idx + int(0.15 * fs)  # 150ms after R peak
        t_wave_end = t_wave_start + int(0.15 * fs)  # 150ms duration
        
        if t_wave_end < n_samples:
            t_wave_samples = t_wave_end - t_wave_start
            t_t = np.linspace(-1, 1, t_wave_samples)
            t_wave = 0.3 * np.exp(-(t_t ** 2) / 0.2)
            ecg[t_wave_start:t_wave_end] += t_wave
    
    # Add baseline wander
    baseline_freq = 0.2  # Hz
    t = np.arange(n_samples) / fs
    baseline = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
    ecg += baseline
    
    # Add noise
    ecg += 0.05 * np.random.randn(n_samples)
    
    # Normalize
    ecg = ecg / np.max(np.abs(ecg))
    
    return ecg


def add_regime_change(
    signal: np.ndarray,
    fs: float,
    change_time: float,
    change_type: str = 'frequency',
    change_magnitude: float = 1.5,
    transition_duration: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    Add a regime change to synthetic signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Original signal
    fs : float
        Sampling frequency in Hz
    change_time : float
        Time of regime change in seconds
    change_type : str, optional
        Type of change: 'frequency', 'amplitude', 'complexity', 'entropy'
        (default: 'frequency')
    change_magnitude : float, optional
        Magnitude of change (default: 1.5)
    transition_duration : float, optional
        Duration of transition in seconds (default: 2.0)
        
    Returns
    -------
    modified_signal : np.ndarray
        Signal with regime change
    change_info : dict
        Information about the change
    """
    change_idx = int(change_time * fs)
    transition_samples = int(transition_duration * fs)
    
    modified = signal.copy()
    
    if change_type == 'frequency':
        # Increase dominant frequency
        after_change = modified[change_idx:]
        
        # Apply high-pass filter to shift spectrum
        from scipy import signal as sp_signal
        sos = sp_signal.butter(4, 15, btype='high', fs=fs, output='sos')
        shifted = sp_signal.sosfilt(sos, after_change)
        
        # Smooth transition
        transition = np.linspace(0, 1, transition_samples)
        modified[change_idx:change_idx + transition_samples] = (
            (1 - transition) * modified[change_idx:change_idx + transition_samples] +
            transition * shifted[:transition_samples]
        )
        modified[change_idx + transition_samples:] = shifted[transition_samples:]
        
    elif change_type == 'amplitude':
        # Increase amplitude
        transition = np.linspace(1, change_magnitude, transition_samples)
        modified[change_idx:change_idx + transition_samples] *= transition
        modified[change_idx + transition_samples:] *= change_magnitude
        
    elif change_type == 'complexity':
        # Add high-frequency noise to increase complexity
        after_change = modified[change_idx:]
        noise = 0.3 * change_magnitude * np.random.randn(len(after_change))
        
        # Smooth transition
        transition = np.linspace(0, 1, transition_samples)
        modified[change_idx:change_idx + transition_samples] += transition * noise[:transition_samples]
        modified[change_idx + transition_samples:] += noise[transition_samples:]
        
    elif change_type == 'entropy':
        # Reduce entropy by making signal more regular
        after_change = modified[change_idx:]
        
        # Add strong periodic component
        t = np.arange(len(after_change)) / fs
        periodic = 0.5 * change_magnitude * np.sin(2 * np.pi * 10 * t)
        
        # Smooth transition
        transition = np.linspace(0, 1, transition_samples)
        modified[change_idx:change_idx + transition_samples] += transition * periodic[:transition_samples]
        modified[change_idx + transition_samples:] += periodic[transition_samples:]
        
    else:
        raise ValueError(f"Unknown change type: {change_type}")
    
    change_info = {
        'change_time': change_time,
        'change_idx': change_idx,
        'change_type': change_type,
        'change_magnitude': change_magnitude,
        'transition_duration': transition_duration
    }
    
    return modified, change_info


def generate_test_dataset(
    duration: float = 300.0,
    fs: float = 256.0,
    n_events: int = 3,
    event_types: Optional[List[str]] = None,
    baseline_duration: float = 60.0
) -> Dict:
    """
    Generate complete test dataset with baseline and multiple events.
    
    Parameters
    ----------
    duration : float, optional
        Total duration in seconds (default: 300 = 5 minutes)
    fs : float, optional
        Sampling frequency in Hz (default: 256.0)
    n_events : int, optional
        Number of instability events (default: 3)
    event_types : list of str, optional
        Types of events to generate (default: mix of types)
    baseline_duration : float, optional
        Duration of clean baseline at start (default: 60.0)
        
    Returns
    -------
    dict
        Dataset with keys:
        - 'eeg': EEG signal
        - 'ecg': ECG signal
        - 'eeg_baseline': Baseline EEG
        - 'ecg_baseline': Baseline ECG
        - 'fs': Sampling frequency
        - 'events': List of event information dicts
        - 'ground_truth': Binary array (1 = event, 0 = stable)
    """
    if event_types is None:
        event_types = ['frequency', 'amplitude', 'complexity']
    
    # Generate baseline signals
    eeg_baseline = generate_synthetic_eeg(baseline_duration, fs)
    ecg_baseline = generate_synthetic_ecg(baseline_duration, fs, heart_rate=70.0)
    
    # Generate full duration signals
    eeg = generate_synthetic_eeg(duration, fs)
    ecg = generate_synthetic_ecg(duration, fs, heart_rate=70.0)
    
    # Add events
    events = []
    ground_truth = np.zeros(int(duration * fs), dtype=int)
    
    # Ensure events don't overlap and happen after baseline
    min_event_time = baseline_duration + 10.0
    max_event_time = duration - 30.0  # Leave buffer at end
    event_spacing = (max_event_time - min_event_time) / (n_events + 1)
    
    for i in range(n_events):
        event_time = min_event_time + (i + 1) * event_spacing
        event_type = event_types[i % len(event_types)]
        
        # Add change to EEG
        eeg, eeg_change_info = add_regime_change(
            eeg, fs, event_time,
            change_type=event_type,
            change_magnitude=1.5 + 0.5 * np.random.rand()
        )
        
        # Add correlated change to ECG (with slight delay)
        ecg_change_time = event_time + 0.5  # 500ms delay
        ecg, ecg_change_info = add_regime_change(
            ecg, fs, ecg_change_time,
            change_type='frequency' if event_type == 'frequency' else 'amplitude',
            change_magnitude=1.3 + 0.3 * np.random.rand()
        )
        
        # Mark ground truth (event window)
        event_start = int(event_time * fs)
        event_duration = int(10.0 * fs)  # 10 seconds
        event_end = min(event_start + event_duration, len(ground_truth))
        ground_truth[event_start:event_end] = 1
        
        events.append({
            'event_number': i + 1,
            'event_time': event_time,
            'event_type': event_type,
            'eeg_change': eeg_change_info,
            'ecg_change': ecg_change_info
        })
    
    return {
        'eeg': eeg,
        'ecg': ecg,
        'eeg_baseline': eeg_baseline,
        'ecg_baseline': ecg_baseline,
        'fs': fs,
        'duration': duration,
        'baseline_duration': baseline_duration,
        'events': events,
        'ground_truth': ground_truth,
        'n_samples': len(eeg)
    }


def generate_artifact_burst(
    duration: float,
    fs: float,
    artifact_type: str = 'motion'
) -> np.ndarray:
    """
    Generate artifact burst for robustness testing.
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float
        Sampling frequency in Hz
    artifact_type : str, optional
        Type: 'motion', 'muscle', 'electrode' (default: 'motion')
        
    Returns
    -------
    np.ndarray
        Artifact signal
    """
    n_samples = int(duration * fs)
    
    if artifact_type == 'motion':
        # Low-frequency high-amplitude drift
        artifact = 5.0 * generate_oscillation(n_samples, fs, 0.5)
        artifact += 2.0 * np.random.randn(n_samples)
        
    elif artifact_type == 'muscle':
        # High-frequency noise
        artifact = 3.0 * np.random.randn(n_samples)
        # Add some structure
        for freq in [50, 100, 150]:
            artifact += 0.5 * generate_oscillation(n_samples, fs, freq)
            
    elif artifact_type == 'electrode':
        # Sudden jumps and baseline shifts
        artifact = np.zeros(n_samples)
        n_jumps = np.random.randint(3, 8)
        jump_locs = np.random.choice(n_samples, n_jumps, replace=False)
        
        for loc in jump_locs:
            artifact[loc:] += np.random.randn() * 2.0
            
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    return artifact


def save_synthetic_dataset(
    dataset: Dict,
    filepath: str
) -> None:
    """
    Save synthetic dataset to NPZ file.
    
    Parameters
    ----------
    dataset : dict
        Dataset dictionary from generate_test_dataset()
    filepath : str
        Output filepath (.npz)
    """
    np.savez(
        filepath,
        eeg=dataset['eeg'],
        ecg=dataset['ecg'],
        eeg_baseline=dataset['eeg_baseline'],
        ecg_baseline=dataset['ecg_baseline'],
        ground_truth=dataset['ground_truth'],
        fs=dataset['fs'],
        duration=dataset['duration'],
        baseline_duration=dataset['baseline_duration'],
        n_samples=dataset['n_samples']
    )
    
    print(f"Saved synthetic dataset to {filepath}")
    print(f"  Duration: {dataset['duration']:.1f}s")
    print(f"  Events: {len(dataset['events'])}")
    print(f"  Sampling rate: {dataset['fs']} Hz")
