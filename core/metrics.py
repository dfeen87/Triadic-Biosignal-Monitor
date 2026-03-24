"""
Metrics Module for Triadic Biosignal Monitor

This module provides performance evaluation metrics:
- Lead-time analysis
- False alarm rate computation
- ROC curve generation
- Ablation comparison

All metrics are designed for prospective validation studies.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    
    Attributes
    ----------
    true_positives : int
        Number of correctly detected events
    false_positives : int
        Number of false alarms
    true_negatives : int
        Number of correctly identified stable periods
    false_negatives : int
        Number of missed events
    sensitivity : float
        TP / (TP + FN)
    specificity : float
        TN / (TN + FP)
    precision : float
        TP / (TP + FP)
    f1_score : float
        Harmonic mean of precision and recall
    false_alarm_rate : float
        FP / total_time (alarms per hour)
    """
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    false_alarm_rate: float


def compute_lead_time(
    prediction_times: np.ndarray,
    event_times: np.ndarray,
    max_lead_time: float = 600.0
) -> np.ndarray:
    """
    Compute lead time for each predicted event.
    
    Parameters
    ----------
    prediction_times : np.ndarray
        Times of predictions/alerts (seconds)
    event_times : np.ndarray
        Times of actual events (seconds)
    max_lead_time : float, optional
        Maximum allowable lead time in seconds (default: 600 = 10 min)
        
    Returns
    -------
    np.ndarray
        Lead times for each prediction (negative = late, positive = early)
        NaN if no matching event within max_lead_time
        
    Notes
    -----
    Lead time = event_time - prediction_time
    Positive lead time means early warning.
    """
    lead_times = np.full(len(prediction_times), np.nan)
    
    for i, pred_time in enumerate(prediction_times):
        # Find closest future event
        future_events = event_times[event_times >= pred_time]
        
        if len(future_events) > 0:
            closest_event = future_events[0]
            lead_time = closest_event - pred_time
            
            # Only count if within max_lead_time
            if lead_time <= max_lead_time:
                lead_times[i] = lead_time
    
    return lead_times


def lead_time_distribution(
    lead_times: np.ndarray,
    bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lead time distribution.
    
    Parameters
    ----------
    lead_times : np.ndarray
        Array of lead times (may contain NaN)
    bins : int, optional
        Number of histogram bins (default: 20)
        
    Returns
    -------
    counts : np.ndarray
        Histogram counts
    bin_edges : np.ndarray
        Histogram bin edges
    """
    # Remove NaN values
    valid_lead_times = lead_times[~np.isnan(lead_times)]
    
    if len(valid_lead_times) == 0:
        return np.array([]), np.array([])
    
    counts, bin_edges = np.histogram(valid_lead_times, bins=bins)
    
    return counts, bin_edges


def compute_false_alarm_rate(
    prediction_times: np.ndarray,
    event_times: np.ndarray,
    total_duration: float,
    max_lead_time: float = 600.0
) -> Tuple[float, int, int]:
    """
    Compute false alarm rate.
    
    Parameters
    ----------
    prediction_times : np.ndarray
        Times of predictions/alerts (seconds)
    event_times : np.ndarray
        Times of actual events (seconds)
    total_duration : float
        Total monitoring duration in seconds
    max_lead_time : float, optional
        Maximum allowable lead time (default: 600)
        
    Returns
    -------
    rate_per_hour : float
        False alarm rate per hour
    false_alarms : int
        Total number of false alarms
    true_alarms : int
        Total number of true alarms
    """
    true_alarms = 0
    false_alarms = 0
    
    for pred_time in prediction_times:
        # Check if prediction matches an event
        future_events = event_times[event_times >= pred_time]
        
        if len(future_events) > 0:
            closest_event = future_events[0]
            lead_time = closest_event - pred_time
            
            if lead_time <= max_lead_time:
                true_alarms += 1
            else:
                false_alarms += 1
        else:
            false_alarms += 1
    
    # Convert to rate per hour
    duration_hours = total_duration / 3600.0
    rate_per_hour = false_alarms / duration_hours if duration_hours > 0 else np.inf
    
    return rate_per_hour, false_alarms, true_alarms


def roc_curve(
    delta_phi_values: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Parameters
    ----------
    delta_phi_values : np.ndarray
        ΔΦ(t) values
    ground_truth : np.ndarray
        Ground truth labels (1 = event, 0 = no event)
    thresholds : np.ndarray, optional
        Specific thresholds to evaluate. If None, use linspace.
    num_thresholds : int, optional
        Number of thresholds to evaluate (default: 100)
        
    Returns
    -------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Threshold values
    """
    if thresholds is None:
        min_val = np.min(delta_phi_values)
        max_val = np.max(delta_phi_values)
        thresholds = np.linspace(min_val, max_val, num_thresholds)
    
    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    
    for i, thresh in enumerate(thresholds):
        predictions = (delta_phi_values >= thresh).astype(int)
        
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return fpr, tpr, thresholds


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC) for ROC.
    
    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
        
    Returns
    -------
    float
        AUC score (0.5 = random, 1.0 = perfect)
    """
    # Sort by fpr
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    return auc


def confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, int]:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions (1 = alert, 0 = no alert)
    ground_truth : np.ndarray
        Ground truth labels (1 = event, 0 = no event)
        
    Returns
    -------
    dict
        Dictionary with TP, FP, TN, FN counts
    """
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }


def compute_performance_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    total_duration: float
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions
    ground_truth : np.ndarray
        Ground truth labels
    total_duration : float
        Total duration in seconds
        
    Returns
    -------
    PerformanceMetrics
        Complete performance metrics
    """
    cm = confusion_matrix(predictions, ground_truth)
    tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']
    
    # Sensitivity (recall, true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision (positive predictive value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # False alarm rate (per hour)
    duration_hours = total_duration / 3600.0
    far = fp / duration_hours if duration_hours > 0 else np.inf
    
    return PerformanceMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1_score=f1,
        false_alarm_rate=far
    )


@dataclass
class AblationResults:
    """
    Results from ablation study comparing different modes.
    
    Attributes
    ----------
    eeg_only : PerformanceMetrics
        Performance with EEG-only pipeline
    ecg_only : PerformanceMetrics
        Performance with ECG-only pipeline
    coupled : PerformanceMetrics
        Performance with full coupled pipeline
    coupling_contribution : float
        Added value from coupling term (coupled F1 - max(eeg, ecg) F1)
    """
    eeg_only: PerformanceMetrics
    ecg_only: PerformanceMetrics
    coupled: PerformanceMetrics
    coupling_contribution: float


def ablation_comparison(
    eeg_predictions: np.ndarray,
    ecg_predictions: np.ndarray,
    coupled_predictions: np.ndarray,
    ground_truth: np.ndarray,
    total_duration: float
) -> AblationResults:
    """
    Compare ablation modes to prove coupling contribution.
    
    Parameters
    ----------
    eeg_predictions : np.ndarray
        Predictions from EEG-only pipeline
    ecg_predictions : np.ndarray
        Predictions from ECG-only pipeline
    coupled_predictions : np.ndarray
        Predictions from full coupled pipeline
    ground_truth : np.ndarray
        Ground truth labels
    total_duration : float
        Total duration in seconds
        
    Returns
    -------
    AblationResults
        Comparison results showing coupling contribution
    """
    # Compute metrics for each mode
    eeg_metrics = compute_performance_metrics(eeg_predictions, ground_truth, total_duration)
    ecg_metrics = compute_performance_metrics(ecg_predictions, ground_truth, total_duration)
    coupled_metrics = compute_performance_metrics(coupled_predictions, ground_truth, total_duration)
    
    # Compute coupling contribution
    max_single_f1 = max(eeg_metrics.f1_score, ecg_metrics.f1_score)
    coupling_contribution = coupled_metrics.f1_score - max_single_f1
    
    return AblationResults(
        eeg_only=eeg_metrics,
        ecg_only=ecg_metrics,
        coupled=coupled_metrics,
        coupling_contribution=coupling_contribution
    )


def sensitivity_analysis(
    delta_phi_values: np.ndarray,
    ground_truth: np.ndarray,
    base_threshold: float,
    threshold_range: Tuple[float, float] = (0.5, 2.0),
    num_points: int = 20
) -> Dict[str, np.ndarray]:
    """
    Perform sensitivity analysis on threshold parameter.
    
    Parameters
    ----------
    delta_phi_values : np.ndarray
        ΔΦ(t) values
    ground_truth : np.ndarray
        Ground truth labels
    base_threshold : float
        Base threshold value τ
    threshold_range : tuple of float, optional
        Range to test as multiples of base threshold (default: (0.5, 2.0))
    num_points : int, optional
        Number of points to test (default: 20)
        
    Returns
    -------
    dict
        Dictionary with arrays:
        - 'thresholds': tested threshold values
        - 'sensitivity': sensitivity at each threshold
        - 'specificity': specificity at each threshold
        - 'f1_score': F1 score at each threshold
    """
    threshold_multipliers = np.linspace(threshold_range[0], threshold_range[1], num_points)
    thresholds = base_threshold * threshold_multipliers
    
    sensitivities = np.zeros(num_points)
    specificities = np.zeros(num_points)
    f1_scores = np.zeros(num_points)
    
    for i, thresh in enumerate(thresholds):
        predictions = (delta_phi_values >= thresh).astype(int)
        
        cm = confusion_matrix(predictions, ground_truth)
        tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']
        
        # Sensitivity
        sensitivities[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity
        specificities[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivities[i]
        f1_scores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'thresholds': thresholds,
        'sensitivity': sensitivities,
        'specificity': specificities,
        'f1_score': f1_scores
    }


def bootstrap_confidence_interval(
    metric_values: np.ndarray,
    confidence_level: float = 0.95,
    num_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    metric_values : np.ndarray
        Array of metric values (e.g., lead times, F1 scores)
    confidence_level : float, optional
        Confidence level (default: 0.95 for 95% CI)
    num_bootstrap : int, optional
        Number of bootstrap samples (default: 1000)
        
    Returns
    -------
    mean : float
        Mean of metric
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    # Remove NaN values
    valid_values = metric_values[~np.isnan(metric_values)]
    
    if len(valid_values) == 0:
        return np.nan, np.nan, np.nan
    
    # Bootstrap resampling
    bootstrap_means = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        sample = np.random.choice(valid_values, size=len(valid_values), replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(valid_values)
    
    return mean, lower, upper


def format_performance_report(
    metrics: PerformanceMetrics,
    lead_times: Optional[np.ndarray] = None
) -> str:
    """
    Format performance metrics as human-readable report.
    
    Parameters
    ----------
    metrics : PerformanceMetrics
        Performance metrics to format
    lead_times : np.ndarray, optional
        Lead time values for additional statistics
        
    Returns
    -------
    str
        Formatted report
    """
    report = f"""
PERFORMANCE REPORT
==================

Confusion Matrix:
  True Positives:  {metrics.true_positives}
  False Positives: {metrics.false_positives}
  True Negatives:  {metrics.true_negatives}
  False Negatives: {metrics.false_negatives}

Classification Metrics:
  Sensitivity: {metrics.sensitivity:.3f}
  Specificity: {metrics.specificity:.3f}
  Precision:   {metrics.precision:.3f}
  F1 Score:    {metrics.f1_score:.3f}

Operational Metrics:
  False Alarm Rate: {metrics.false_alarm_rate:.2f} alarms/hour
"""
    
    if lead_times is not None:
        valid_lt = lead_times[~np.isnan(lead_times)]
        if len(valid_lt) > 0:
            mean_lt = np.mean(valid_lt)
            median_lt = np.median(valid_lt)
            std_lt = np.std(valid_lt)
            
            report += f"""
Lead Time Statistics:
  Mean:   {mean_lt:.1f} seconds ({mean_lt/60:.1f} minutes)
  Median: {median_lt:.1f} seconds ({median_lt/60:.1f} minutes)
  Std:    {std_lt:.1f} seconds
  Range:  [{np.min(valid_lt):.1f}, {np.max(valid_lt):.1f}] seconds
"""
    
    return report.strip()


def format_ablation_report(results: AblationResults) -> str:
    """
    Format ablation comparison as human-readable report.
    
    Parameters
    ----------
    results : AblationResults
        Ablation comparison results
        
    Returns
    -------
    str
        Formatted report
    """
    report = f"""
ABLATION COMPARISON REPORT
==========================

EEG-Only Pipeline:
  Sensitivity: {results.eeg_only.sensitivity:.3f}
  Specificity: {results.eeg_only.specificity:.3f}
  F1 Score:    {results.eeg_only.f1_score:.3f}
  FAR:         {results.eeg_only.false_alarm_rate:.2f} alarms/hour

ECG-Only Pipeline:
  Sensitivity: {results.ecg_only.sensitivity:.3f}
  Specificity: {results.ecg_only.specificity:.3f}
  F1 Score:    {results.ecg_only.f1_score:.3f}
  FAR:         {results.ecg_only.false_alarm_rate:.2f} alarms/hour

Full Coupled Pipeline:
  Sensitivity: {results.coupled.sensitivity:.3f}
  Specificity: {results.coupled.specificity:.3f}
  F1 Score:    {results.coupled.f1_score:.3f}
  FAR:         {results.coupled.false_alarm_rate:.2f} alarms/hour

Coupling Contribution:
  ΔF1 = {results.coupling_contribution:+.3f}
  
Interpretation: {'Coupling term adds value' if results.coupling_contribution > 0 else 'Coupling term does not add value'}
"""
    return report.strip()
