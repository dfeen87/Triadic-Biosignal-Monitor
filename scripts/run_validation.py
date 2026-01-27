"""
Batch Validation Runner for Triadic Biosignal Monitor

This script runs validation analyses on datasets and computes performance metrics.
Supports all three modes: EEG-only, ECG-only, and coupled.

Usage:
    python run_validation.py --config configs/default.yaml --data dataset.npz --output results/
    python run_validation.py --mode coupled --synthetic --duration 300 --n-events 3

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import argparse
import yaml
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.eeg_only import run_eeg_only_pipeline
from pipelines.ecg_only import run_ecg_only_pipeline
from pipelines.coupled import run_coupled_pipeline
from datasets.loaders import auto_load, load_dataset_pair, validate_signal
from datasets.synthetic import generate_test_dataset
from core.metrics import (
    compute_performance_metrics,
    compute_lead_time,
    compute_false_alarm_rate,
    ablation_comparison,
    format_performance_report,
    format_ablation_report
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_validation(
    eeg_signal: np.ndarray,
    ecg_signal: Optional[np.ndarray],
    eeg_baseline: np.ndarray,
    ecg_baseline: Optional[np.ndarray],
    fs: float,
    config: Dict,
    mode: str = 'coupled',
    ground_truth: Optional[np.ndarray] = None
) -> Dict:
    """
    Run validation on single dataset.
    
    Parameters
    ----------
    eeg_signal : np.ndarray
        EEG signal to analyze
    ecg_signal : np.ndarray or None
        ECG signal (required for coupled/ecg_only modes)
    eeg_baseline : np.ndarray
        Baseline EEG
    ecg_baseline : np.ndarray or None
        Baseline ECG (required for coupled/ecg_only modes)
    fs : float
        Sampling frequency
    config : dict
        Configuration dictionary
    mode : str
        Processing mode: 'eeg_only', 'ecg_only', 'coupled'
    ground_truth : np.ndarray, optional
        Ground truth labels for performance evaluation
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} validation")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run appropriate pipeline
    if mode == 'eeg_only':
        results = run_eeg_only_pipeline(
            eeg_signal=eeg_signal,
            baseline_eeg=eeg_baseline,
            fs=fs,
            config=config
        )
    elif mode == 'ecg_only':
        if ecg_signal is None or ecg_baseline is None:
            raise ValueError("ECG signal and baseline required for ECG-only mode")
        results = run_ecg_only_pipeline(
            ecg_signal=ecg_signal,
            baseline_ecg=ecg_baseline,
            fs=fs,
            config=config
        )
    elif mode == 'coupled':
        if ecg_signal is None or ecg_baseline is None:
            raise ValueError("ECG signal and baseline required for coupled mode")
        results = run_coupled_pipeline(
            eeg_signal=eeg_signal,
            ecg_signal=ecg_signal,
            baseline_eeg=eeg_baseline,
            baseline_ecg=ecg_baseline,
            fs=fs,
            config=config
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    processing_time = time.time() - start_time
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"  Windows processed: {len(results['timestamps'])}")
    print(f"  Alerts generated: {len(results['alerts'])}")
    print(f"  Mean ΔΦ: {np.mean(results['delta_phi']):.3f}")
    print(f"  Max ΔΦ: {np.max(results['delta_phi']):.3f}")
    
    # Compute performance metrics if ground truth available
    if ground_truth is not None:
        print(f"\n{'-'*60}")
        print("Computing performance metrics...")
        
        # Ensure ground truth matches signal length
        n_samples = len(eeg_signal)
        if len(ground_truth) != n_samples:
            print(f"Warning: Ground truth length mismatch. Trimming to {n_samples} samples.")
            ground_truth = ground_truth[:n_samples]
        
        # Convert gate signal to per-sample predictions
        predictions = np.zeros(n_samples, dtype=int)
        window_size = config.get('window_size', 10.0)
        for i, (timestamp, gate) in enumerate(zip(results['timestamps'], results['gate'])):
            if gate == 1:
                idx_center = int(timestamp * fs)
                idx_start = max(0, idx_center - int(window_size * fs / 2))
                idx_end = min(n_samples, idx_center + int(window_size * fs / 2))
                predictions[idx_start:idx_end] = 1
        
        # Compute metrics
        duration = len(eeg_signal) / fs
        metrics = compute_performance_metrics(predictions, ground_truth, duration)
        
        print(format_performance_report(metrics))
        
        # Compute lead times
        alert_times = results['timestamps'][results['gate'] == 1]
        event_times = np.where(np.diff(ground_truth.astype(int)) == 1)[0] / fs
        
        if len(alert_times) > 0 and len(event_times) > 0:
            lead_times = compute_lead_time(alert_times, event_times, max_lead_time=600.0)
            valid_lead_times = lead_times[~np.isnan(lead_times)]
            
            if len(valid_lead_times) > 0:
                print(f"\nLead Time Statistics:")
                print(f"  Mean: {np.mean(valid_lead_times):.1f} seconds")
                print(f"  Median: {np.median(valid_lead_times):.1f} seconds")
                print(f"  Range: [{np.min(valid_lead_times):.1f}, {np.max(valid_lead_times):.1f}] seconds")
        
        results['performance_metrics'] = {
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
            'precision': metrics.precision,
            'f1_score': metrics.f1_score,
            'false_alarm_rate': metrics.false_alarm_rate,
            'true_positives': metrics.true_positives,
            'false_positives': metrics.false_positives,
            'true_negatives': metrics.true_negatives,
            'false_negatives': metrics.false_negatives
        }
    
    results['processing_time'] = processing_time
    results['mode'] = mode
    
    return results


def run_ablation_study(
    eeg_signal: np.ndarray,
    ecg_signal: np.ndarray,
    eeg_baseline: np.ndarray,
    ecg_baseline: np.ndarray,
    fs: float,
    config: Dict,
    ground_truth: Optional[np.ndarray] = None
) -> Dict:
    """
    Run complete ablation study comparing all three modes.
    
    Parameters
    ----------
    eeg_signal : np.ndarray
        EEG signal
    ecg_signal : np.ndarray
        ECG signal
    eeg_baseline : np.ndarray
        Baseline EEG
    ecg_baseline : np.ndarray
        Baseline ECG
    fs : float
        Sampling frequency
    config : dict
        Configuration
    ground_truth : np.ndarray, optional
        Ground truth labels
        
    Returns
    -------
    dict
        Ablation study results
    """
    print(f"\n{'='*60}")
    print("ABLATION STUDY: Comparing EEG-only, ECG-only, and Coupled modes")
    print(f"{'='*60}")
    
    # Run all three modes
    results_eeg = run_single_validation(
        eeg_signal, None, eeg_baseline, None,
        fs, config, mode='eeg_only', ground_truth=ground_truth
    )
    
    results_ecg = run_single_validation(
        None, ecg_signal, None, ecg_baseline,
        fs, config, mode='ecg_only', ground_truth=ground_truth
    )
    
    results_coupled = run_single_validation(
        eeg_signal, ecg_signal, eeg_baseline, ecg_baseline,
        fs, config, mode='coupled', ground_truth=ground_truth
    )
    
    # Compare performance metrics if available
    if ground_truth is not None and all(
        'performance_metrics' in r for r in [results_eeg, results_ecg, results_coupled]
    ):
        print(f"\n{'='*60}")
        print("ABLATION COMPARISON")
        print(f"{'='*60}")
        
        # Extract predictions for comparison
        n_samples = len(eeg_signal)
        duration = n_samples / fs
        
        preds_eeg = np.zeros(n_samples, dtype=int)
        preds_ecg = np.zeros(n_samples, dtype=int)
        preds_coupled = np.zeros(n_samples, dtype=int)
        
        window_size = config.get('window_size', 10.0)
        
        for results, preds in [
            (results_eeg, preds_eeg),
            (results_ecg, preds_ecg),
            (results_coupled, preds_coupled)
        ]:
            for timestamp, gate in zip(results['timestamps'], results['gate']):
                if gate == 1:
                    idx_center = int(timestamp * fs)
                    idx_start = max(0, idx_center - int(window_size * fs / 2))
                    idx_end = min(n_samples, idx_center + int(window_size * fs / 2))
                    preds[idx_start:idx_end] = 1
        
        # Compute ablation comparison
        ablation_results = ablation_comparison(
            preds_eeg, preds_ecg, preds_coupled,
            ground_truth, duration
        )
        
        print(format_ablation_report(ablation_results))
        
        # Store comparison results
        comparison = {
            'eeg_only_f1': ablation_results.eeg_only.f1_score,
            'ecg_only_f1': ablation_results.ecg_only.f1_score,
            'coupled_f1': ablation_results.coupled.f1_score,
            'coupling_contribution': ablation_results.coupling_contribution
        }
    else:
        comparison = None
    
    return {
        'eeg_only': results_eeg,
        'ecg_only': results_ecg,
        'coupled': results_coupled,
        'comparison': comparison
    }


def save_results(results: Dict, output_dir: Path, prefix: str = 'validation') -> None:
    """Save validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON summary (excluding large arrays)
    summary = {}
    for key, value in results.items():
        if isinstance(value, dict):
            summary[key] = {k: v for k, v in value.items() 
                          if not isinstance(v, np.ndarray) or len(v) < 100}
        elif not isinstance(value, np.ndarray) or len(value) < 100:
            summary[key] = value
    
    json_path = output_dir / f'{prefix}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to: {json_path}")
    
    # Save full results to NPZ
    npz_data = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            npz_data[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    npz_data[f'{key}_{subkey}'] = subvalue
    
    if npz_data:
        npz_path = output_dir / f'{prefix}_{timestamp}.npz'
        np.savez(npz_path, **npz_data)
        print(f"Saved arrays to: {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run validation for Triadic Biosignal Monitor'
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', type=str, help='Path to dataset file')
    data_group.add_argument('--synthetic', action='store_true', 
                           help='Generate synthetic dataset')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--mode', type=str, choices=['eeg_only', 'ecg_only', 'coupled', 'ablation'],
                       default='coupled', help='Processing mode (default: coupled)')
    
    # Synthetic data options
    parser.add_argument('--duration', type=float, default=300.0,
                       help='Duration for synthetic data (seconds, default: 300)')
    parser.add_argument('--n-events', type=int, default=3,
                       help='Number of events in synthetic data (default: 3)')
    parser.add_argument('--fs', type=float, default=256.0,
                       help='Sampling frequency (Hz, default: 256)')
    
    # Output
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results/)')
    parser.add_argument('--prefix', type=str, default='validation',
                       help='Output file prefix (default: validation)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load or generate data
    if args.synthetic:
        print(f"\nGenerating synthetic dataset...")
        print(f"  Duration: {args.duration}s")
        print(f"  Events: {args.n_events}")
        print(f"  Sampling rate: {args.fs} Hz")
        
        dataset = generate_test_dataset(
            duration=args.duration,
            fs=args.fs,
            n_events=args.n_events
        )
        
        eeg_signal = dataset['eeg']
        ecg_signal = dataset['ecg']
        eeg_baseline = dataset['eeg_baseline']
        ecg_baseline = dataset['ecg_baseline']
        fs = dataset['fs']
        ground_truth = dataset['ground_truth']
        
        print(f"  Generated {len(eeg_signal)} samples ({len(eeg_signal)/fs:.1f}s)")
        
    else:
        print(f"\nLoading dataset from: {args.data}")
        
        # Try to load as NPZ first
        try:
            data = np.load(args.data)
            eeg_signal = data['eeg']
            ecg_signal = data.get('ecg', None)
            eeg_baseline = data['eeg_baseline']
            ecg_baseline = data.get('ecg_baseline', None)
            fs = float(data['fs'])
            ground_truth = data.get('ground_truth', None)
            
            print(f"  Loaded NPZ dataset")
            print(f"  Duration: {len(eeg_signal)/fs:.1f}s")
            print(f"  Sampling rate: {fs} Hz")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
    
    # Validate signals
    print("\nValidating signals...")
    is_valid, issues = validate_signal(eeg_signal, fs)
    if not is_valid:
        print(f"EEG validation issues: {issues}")
    else:
        print("  EEG: OK")
    
    if ecg_signal is not None:
        is_valid, issues = validate_signal(ecg_signal, fs)
        if not is_valid:
            print(f"ECG validation issues: {issues}")
        else:
            print("  ECG: OK")
    
    # Run validation
    try:
        if args.mode == 'ablation':
            if ecg_signal is None:
                print("Error: Ablation study requires both EEG and ECG signals")
                return 1
            
            results = run_ablation_study(
                eeg_signal, ecg_signal,
                eeg_baseline, ecg_baseline,
                fs, config, ground_truth
            )
        else:
            results = run_single_validation(
                eeg_signal, ecg_signal,
                eeg_baseline, ecg_baseline,
                fs, config, args.mode, ground_truth
            )
        
        # Save results
        output_dir = Path(args.output)
        save_results(results, output_dir, args.prefix)
        
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
