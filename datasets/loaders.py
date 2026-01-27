"""
Data Loaders for Triadic Biosignal Monitor

Supports loading EEG and ECG data from various formats:
- EDF (European Data Format) - common for clinical EEG
- FIF (Neuromag/MNE format) - common in research
- CSV/TSV - simple text formats
- NumPy arrays - .npy/.npz files
- WFDB (PhysioNet) - common for ECG/physiological signals

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path


def load_numpy(
    filepath: Union[str, Path],
    key: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Load data from NumPy file (.npy or .npz).
    
    Parameters
    ----------
    filepath : str or Path
        Path to .npy or .npz file
    key : str, optional
        Key for .npz files (e.g., 'data', 'signal')
        
    Returns
    -------
    data : np.ndarray
        Loaded signal data
    metadata : dict
        Metadata dictionary (may be empty for .npy files)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        data = np.load(filepath)
        metadata = {'filename': filepath.name, 'format': 'npy'}
        
    elif filepath.suffix == '.npz':
        npz_file = np.load(filepath)
        
        # If key specified, use it
        if key is not None:
            if key not in npz_file:
                raise ValueError(f"Key '{key}' not found in .npz file. Available: {list(npz_file.keys())}")
            data = npz_file[key]
        else:
            # Try common keys
            for common_key in ['data', 'signal', 'eeg', 'ecg', 'arr_0']:
                if common_key in npz_file:
                    data = npz_file[common_key]
                    key = common_key
                    break
            else:
                # Just use first array
                key = list(npz_file.keys())[0]
                data = npz_file[key]
                warnings.warn(f"No standard key found. Using '{key}'")
        
        # Extract metadata
        metadata = {
            'filename': filepath.name,
            'format': 'npz',
            'key': key,
            'available_keys': list(npz_file.keys())
        }
        
        # Check for fs (sampling frequency)
        if 'fs' in npz_file:
            metadata['fs'] = float(npz_file['fs'])
        if 'sampling_rate' in npz_file:
            metadata['fs'] = float(npz_file['sampling_rate'])
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    
    return data, metadata


def load_csv(
    filepath: Union[str, Path],
    column: Optional[Union[int, str]] = 0,
    delimiter: str = ',',
    skip_header: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Load data from CSV/TSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV/TSV file
    column : int or str, optional
        Column index (int) or name (str) to load (default: 0)
    delimiter : str, optional
        Delimiter character (default: ',')
    skip_header : int, optional
        Number of header rows to skip (default: 0)
        
    Returns
    -------
    data : np.ndarray
        Loaded signal data
    metadata : dict
        Metadata including column info
    """
    filepath = Path(filepath)
    
    # Read file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    lines = lines[skip_header:]
    
    # Parse data
    if isinstance(column, str):
        # Assume first line is header with column names
        header = lines[0].strip().split(delimiter)
        if column not in header:
            raise ValueError(f"Column '{column}' not found. Available: {header}")
        column_idx = header.index(column)
        lines = lines[1:]  # Skip header line
    else:
        column_idx = column
    
    # Extract column
    data = []
    for line in lines:
        values = line.strip().split(delimiter)
        try:
            data.append(float(values[column_idx]))
        except (ValueError, IndexError):
            warnings.warn(f"Skipping invalid line: {line.strip()}")
            continue
    
    data = np.array(data)
    
    metadata = {
        'filename': filepath.name,
        'format': 'csv',
        'column': column,
        'delimiter': delimiter,
        'n_samples': len(data)
    }
    
    return data, metadata


def load_edf(
    filepath: Union[str, Path],
    channel: Optional[Union[int, str]] = 0,
    return_all_channels: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Load data from EDF (European Data Format) file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .edf file
    channel : int or str, optional
        Channel index or name (default: 0)
    return_all_channels : bool, optional
        If True, return all channels as 2D array (default: False)
        
    Returns
    -------
    data : np.ndarray
        Signal data (1D if single channel, 2D if all channels)
    metadata : dict
        Metadata including sampling rate, channel info
        
    Notes
    -----
    Requires pyedflib or mne package.
    """
    filepath = Path(filepath)
    
    try:
        import pyedflib
        
        # Open EDF file
        with pyedflib.EdfReader(str(filepath)) as edf:
            n_channels = edf.signals_in_file
            channel_labels = edf.getSignalLabels()
            sampling_rates = [edf.getSampleFrequency(i) for i in range(n_channels)]
            
            if return_all_channels:
                # Load all channels
                data = np.array([edf.readSignal(i) for i in range(n_channels)])
                channel_used = 'all'
            else:
                # Load single channel
                if isinstance(channel, str):
                    if channel not in channel_labels:
                        raise ValueError(f"Channel '{channel}' not found. Available: {channel_labels}")
                    channel_idx = channel_labels.index(channel)
                else:
                    channel_idx = channel
                
                data = edf.readSignal(channel_idx)
                channel_used = channel_labels[channel_idx]
            
            # Get sampling rate
            fs = sampling_rates[0] if isinstance(channel, int) and channel == 0 else sampling_rates[channel_idx]
            
            metadata = {
                'filename': filepath.name,
                'format': 'edf',
                'fs': fs,
                'n_channels': n_channels,
                'channel_labels': channel_labels,
                'channel_used': channel_used,
                'duration_sec': len(data) / fs if not return_all_channels else data.shape[1] / fs
            }
    
    except ImportError:
        # Try MNE as fallback
        try:
            import mne
            
            raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
            
            if return_all_channels:
                data, times = raw.get_data(return_times=True)
                channel_used = 'all'
            else:
                if isinstance(channel, str):
                    if channel not in raw.ch_names:
                        raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")
                    data, times = raw.get_data(picks=channel, return_times=True)
                    channel_used = channel
                else:
                    data, times = raw.get_data(picks=channel, return_times=True)
                    channel_used = raw.ch_names[channel]
                
                data = data.squeeze()
            
            metadata = {
                'filename': filepath.name,
                'format': 'edf',
                'fs': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'channel_labels': raw.ch_names,
                'channel_used': channel_used,
                'duration_sec': raw.times[-1]
            }
        
        except ImportError:
            raise ImportError(
                "EDF loading requires either 'pyedflib' or 'mne' package. "
                "Install with: pip install pyedflib or pip install mne"
            )
    
    return data, metadata


def load_fif(
    filepath: Union[str, Path],
    channel: Optional[Union[int, str]] = 0,
    return_all_channels: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Load data from FIF (Neuromag/MNE) file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to .fif file
    channel : int or str, optional
        Channel index or name (default: 0)
    return_all_channels : bool, optional
        If True, return all channels (default: False)
        
    Returns
    -------
    data : np.ndarray
        Signal data
    metadata : dict
        Metadata including sampling rate
        
    Notes
    -----
    Requires mne package.
    """
    filepath = Path(filepath)
    
    try:
        import mne
        
        raw = mne.io.read_raw_fif(str(filepath), preload=True, verbose=False)
        
        if return_all_channels:
            data, times = raw.get_data(return_times=True)
            channel_used = 'all'
        else:
            if isinstance(channel, str):
                if channel not in raw.ch_names:
                    raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")
                data, times = raw.get_data(picks=channel, return_times=True)
                channel_used = channel
            else:
                data, times = raw.get_data(picks=channel, return_times=True)
                channel_used = raw.ch_names[channel]
            
            data = data.squeeze()
        
        metadata = {
            'filename': filepath.name,
            'format': 'fif',
            'fs': raw.info['sfreq'],
            'n_channels': len(raw.ch_names),
            'channel_labels': raw.ch_names,
            'channel_used': channel_used,
            'duration_sec': raw.times[-1]
        }
        
    except ImportError:
        raise ImportError("FIF loading requires 'mne' package. Install with: pip install mne")
    
    return data, metadata


def load_wfdb(
    filepath: Union[str, Path],
    channel: Optional[Union[int, str]] = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Load data from WFDB (PhysioNet) format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to record (without extension)
    channel : int or str, optional
        Channel index or name (default: 0)
        
    Returns
    -------
    data : np.ndarray
        Signal data
    metadata : dict
        Metadata including sampling rate
        
    Notes
    -----
    Requires wfdb package.
    """
    filepath = Path(filepath)
    
    try:
        import wfdb
        
        # Read record (filepath without extension)
        record = wfdb.rdrecord(str(filepath.with_suffix('')))
        
        # Extract channel
        if isinstance(channel, str):
            if channel not in record.sig_name:
                raise ValueError(f"Channel '{channel}' not found. Available: {record.sig_name}")
            channel_idx = record.sig_name.index(channel)
        else:
            channel_idx = channel
        
        data = record.p_signal[:, channel_idx]
        
        metadata = {
            'filename': filepath.name,
            'format': 'wfdb',
            'fs': record.fs,
            'n_channels': record.n_sig,
            'channel_labels': record.sig_name,
            'channel_used': record.sig_name[channel_idx],
            'units': record.units[channel_idx],
            'duration_sec': len(data) / record.fs
        }
        
    except ImportError:
        raise ImportError("WFDB loading requires 'wfdb' package. Install with: pip install wfdb")
    
    return data, metadata


def auto_load(
    filepath: Union[str, Path],
    channel: Optional[Union[int, str]] = 0,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Automatically detect format and load data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to data file
    channel : int or str, optional
        Channel to load (default: 0)
    **kwargs
        Additional arguments passed to specific loader
        
    Returns
    -------
    data : np.ndarray
        Signal data
    metadata : dict
        Metadata dictionary
    """
    filepath = Path(filepath)
    
    # Detect format from extension
    suffix = filepath.suffix.lower()
    
    if suffix == '.npy' or suffix == '.npz':
        return load_numpy(filepath, **kwargs)
    elif suffix == '.csv':
        return load_csv(filepath, column=channel, delimiter=',', **kwargs)
    elif suffix == '.tsv':
        return load_csv(filepath, column=channel, delimiter='\t', **kwargs)
    elif suffix == '.edf':
        return load_edf(filepath, channel=channel, **kwargs)
    elif suffix == '.fif':
        return load_fif(filepath, channel=channel, **kwargs)
    elif suffix == '.dat' or suffix == '.hea':
        # WFDB format
        return load_wfdb(filepath, channel=channel, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_dataset_pair(
    eeg_filepath: Union[str, Path],
    ecg_filepath: Union[str, Path],
    eeg_channel: Optional[Union[int, str]] = 0,
    ecg_channel: Optional[Union[int, str]] = 0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load paired EEG and ECG datasets.
    
    Parameters
    ----------
    eeg_filepath : str or Path
        Path to EEG data
    ecg_filepath : str or Path
        Path to ECG data
    eeg_channel : int or str, optional
        EEG channel to load
    ecg_channel : int or str, optional
        ECG channel to load
        
    Returns
    -------
    eeg_data : np.ndarray
        EEG signal
    ecg_data : np.ndarray
        ECG signal
    metadata : dict
        Combined metadata
    """
    eeg_data, eeg_meta = auto_load(eeg_filepath, channel=eeg_channel)
    ecg_data, ecg_meta = auto_load(ecg_filepath, channel=ecg_channel)
    
    # Combine metadata
    metadata = {
        'eeg': eeg_meta,
        'ecg': ecg_meta,
        'fs_match': np.isclose(eeg_meta.get('fs', 0), ecg_meta.get('fs', 0))
    }
    
    # Warn if sampling rates don't match
    if not metadata['fs_match']:
        warnings.warn(
            f"Sampling rates differ: EEG={eeg_meta.get('fs')} Hz, "
            f"ECG={ecg_meta.get('fs')} Hz. Signals may need resampling."
        )
    
    return eeg_data, ecg_data, metadata


def validate_signal(
    signal: np.ndarray,
    fs: float,
    min_duration: float = 5.0,
    check_nans: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate signal quality and format.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to validate
    fs : float
        Sampling frequency in Hz
    min_duration : float, optional
        Minimum duration in seconds (default: 5.0)
    check_nans : bool, optional
        Check for NaN values (default: True)
        
    Returns
    -------
    is_valid : bool
        True if signal passes all checks
    issues : list of str
        List of validation issues found
    """
    issues = []
    
    # Check dimensionality
    if signal.ndim != 1:
        issues.append(f"Signal must be 1D, got {signal.ndim}D")
    
    # Check length
    duration = len(signal) / fs
    if duration < min_duration:
        issues.append(f"Signal too short: {duration:.1f}s < {min_duration}s")
    
    # Check for NaN/inf
    if check_nans:
        if np.any(np.isnan(signal)):
            issues.append("Signal contains NaN values")
        if np.any(np.isinf(signal)):
            issues.append("Signal contains infinite values")
    
    # Check variance
    if np.var(signal) < 1e-10:
        issues.append("Signal has near-zero variance (flatline)")
    
    # Check sampling rate
    if fs <= 0:
        issues.append(f"Invalid sampling rate: {fs} Hz")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues
