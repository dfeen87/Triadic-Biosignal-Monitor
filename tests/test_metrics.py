"""
Tests for metrics module.

Tests performance evaluation metrics including lead-time, false alarms, ROC, and ablations.
"""

import pytest
import numpy as np
from core.metrics import (
    compute_lead_time,
    lead_time_distribution,
    compute_false_alarm_rate,
    roc_curve,
    auc_score,
    confusion_matrix,
    compute_performance_metrics,
    ablation_comparison,
    sensitivity_analysis,
    bootstrap_confidence_interval,
    format_performance_report,
    format_ablation_report
)


class TestLeadTime:
    """Tests for lead time computation."""
    
    def test_compute_lead_time(self):
        """Test lead time computation."""
        prediction_times = np.array([10.0, 30.0, 50.0])
        event_times = np.array([15.0, 35.0, 55.0])
        
        lead_times = compute_lead_time(prediction_times, event_times, max_lead_time=600.0)
        
        assert len(lead_times) == len(prediction_times)
        # Lead times should be positive (early warnings)
        assert lead_times[0] == 5.0
        assert lead_times[1] == 5.0
        assert lead_times[2] == 5.0
    
    def test_no_matching_event(self):
        """Test lead time when no matching event exists."""
        prediction_times = np.array([10.0])
        event_times = np.array([100.0])  # Too far in future
        
        lead_times = compute_lead_time(prediction_times, event_times, max_lead_time=50.0)
        
        # Should be NaN when no event within max_lead_time
        assert np.isnan(lead_times[0])
    
    def test_negative_lead_time(self):
        """Test when prediction comes after event (late)."""
        prediction_times = np.array([20.0])
        event_times = np.array([15.0])
        
        lead_times = compute_lead_time(prediction_times, event_times, max_lead_time=600.0)
        
        # No future event, should be NaN
        assert np.isnan(lead_times[0])


class TestLeadTimeDistribution:
    """Tests for lead time distribution."""
    
    def test_lead_time_distribution(self):
        """Test lead time distribution computation."""
        lead_times = np.array([10.0, 20.0, 15.0, 25.0, 18.0])
        
        counts, bin_edges = lead_time_distribution(lead_times, bins=5)
        
        assert len(counts) == 5
        assert len(bin_edges) == 6  # n+1 edges for n bins
        assert np.sum(counts) == len(lead_times)
    
    def test_with_nans(self):
        """Test distribution with NaN values."""
        lead_times = np.array([10.0, np.nan, 20.0, np.nan, 15.0])
        
        counts, bin_edges = lead_time_distribution(lead_times, bins=5)
        
        # Should only count valid lead times
        assert np.sum(counts) == 3


class TestFalseAlarmRate:
    """Tests for false alarm rate computation."""
    
    def test_false_alarm_rate(self):
        """Test false alarm rate computation."""
        prediction_times = np.array([10.0, 30.0, 50.0, 70.0])
        event_times = np.array([15.0, 55.0])  # Only 2 events
        total_duration = 100.0
        
        rate_per_hour, false_alarms, true_alarms = compute_false_alarm_rate(
            prediction_times, event_times, total_duration, max_lead_time=10.0
        )
        
        assert true_alarms == 2
        assert false_alarms == 2
        # Rate should be 2 FAs / (100/3600) hours
        assert rate_per_hour > 0


class TestROCCurve:
    """Tests for ROC curve computation."""
    
    def test_roc_curve(self):
        """Test ROC curve computation."""
        # Perfect classifier
        delta_phi_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        ground_truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        fpr, tpr, thresholds = roc_curve(delta_phi_values, ground_truth, num_thresholds=10)
        
        assert len(fpr) == len(tpr) == len(thresholds)
        assert np.all((fpr >= 0) & (fpr <= 1))
        assert np.all((tpr >= 0) & (tpr <= 1))
    
    def test_roc_perfect_classifier(self):
        """Test ROC for perfect classifier."""
        delta_phi_values = np.array([1, 2, 3, 4])
        ground_truth = np.array([0, 0, 1, 1])
        
        fpr, tpr, thresholds = roc_curve(delta_phi_values, ground_truth, num_thresholds=5)
        
        # Perfect classifier should achieve TPR=1 with FPR=0 at some threshold
        assert np.any((tpr == 1.0) & (fpr == 0.0))


class TestAUCScore:
    """Tests for AUC score."""
    
    def test_auc_perfect(self):
        """Test AUC for perfect classifier."""
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        
        auc = auc_score(fpr, tpr)
        
        # Perfect classifier should have AUC = 1.0
        assert np.isclose(auc, 1.0, atol=0.01)
    
    def test_auc_random(self):
        """Test AUC for random classifier."""
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 1.0])
        
        auc = auc_score(fpr, tpr)
        
        # Random classifier should have AUC = 0.5
        assert np.isclose(auc, 0.5, atol=0.01)


class TestConfusionMatrix:
    """Tests for confusion matrix."""
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        predictions = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        ground_truth = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        
        cm = confusion_matrix(predictions, ground_truth)
        
        assert 'TP' in cm
        assert 'FP' in cm
        assert 'TN' in cm
        assert 'FN' in cm
        
        # Check values
        assert cm['TP'] == 3  # Correct positives
        assert cm['FP'] == 1  # False positives
        assert cm['TN'] == 3  # Correct negatives
        assert cm['FN'] == 1  # Missed positives
    
    def test_all_positive_predictions(self):
        """Test with all positive predictions."""
        predictions = np.ones(10, dtype=int)
        ground_truth = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        
        cm = confusion_matrix(predictions, ground_truth)
        
        assert cm['TP'] == 5
        assert cm['FP'] == 5
        assert cm['TN'] == 0
        assert cm['FN'] == 0


class TestPerformanceMetrics:
    """Tests for comprehensive performance metrics."""
    
    def test_compute_performance_metrics(self):
        """Test complete performance metrics computation."""
        predictions = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        ground_truth = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        total_duration = 3600.0  # 1 hour
        
        metrics = compute_performance_metrics(predictions, ground_truth, total_duration)
        
        # Check all metrics present
        assert hasattr(metrics, 'sensitivity')
        assert hasattr(metrics, 'specificity')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'false_alarm_rate')
        
        # All metrics should be in valid ranges
        assert 0 <= metrics.sensitivity <= 1
        assert 0 <= metrics.specificity <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.false_alarm_rate >= 0
    
    def test_perfect_classifier_metrics(self):
        """Test metrics for perfect classifier."""
        predictions = np.array([1, 1, 0, 0])
        ground_truth = np.array([1, 1, 0, 0])
        
        metrics = compute_performance_metrics(predictions, ground_truth, 100.0)
        
        # Perfect classifier
        assert metrics.sensitivity == 1.0
        assert metrics.specificity == 1.0
        assert metrics.precision == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.false_alarm_rate == 0.0


class TestAblationComparison:
    """Tests for ablation comparison."""
    
    def test_ablation_comparison(self):
        """Test ablation comparison between modes."""
        ground_truth = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        
        # EEG-only predictions
        eeg_predictions = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        
        # ECG-only predictions
        ecg_predictions = np.array([0, 0, 1, 1, 0, 0, 0, 0])
        
        # Coupled predictions (better)
        coupled_predictions = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        
        results = ablation_comparison(
            eeg_predictions, ecg_predictions, coupled_predictions,
            ground_truth, 100.0
        )
        
        # Check that coupled has positive contribution
        assert results.coupling_contribution >= 0
        
        # Coupled should be best
        assert results.coupled.f1_score >= results.eeg_only.f1_score
        assert results.coupled.f1_score >= results.ecg_only.f1_score


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""
    
    def test_sensitivity_analysis(self):
        """Test threshold sensitivity analysis."""
        delta_phi_values = np.random.rand(100) * 5
        ground_truth = (delta_phi_values > 2.5).astype(int)
        
        results = sensitivity_analysis(
            delta_phi_values, ground_truth,
            base_threshold=2.5,
            threshold_range=(0.5, 2.0),
            num_points=10
        )
        
        assert 'thresholds' in results
        assert 'sensitivity' in results
        assert 'specificity' in results
        assert 'f1_score' in results
        
        assert len(results['thresholds']) == 10
        assert len(results['sensitivity']) == 10


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI computation."""
        metric_values = np.random.normal(10.0, 2.0, 100)
        
        mean, lower, upper = bootstrap_confidence_interval(
            metric_values, confidence_level=0.95, num_bootstrap=100
        )
        
        assert lower < mean < upper
        assert not np.isnan(mean)
        assert not np.isnan(lower)
        assert not np.isnan(upper)
    
    def test_with_nans(self):
        """Test bootstrap with NaN values."""
        metric_values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        
        mean, lower, upper = bootstrap_confidence_interval(metric_values)
        
        # Should handle NaNs gracefully
        assert not np.isnan(mean)


class TestReportFormatting:
    """Tests for report formatting functions."""
    
    def test_format_performance_report(self):
        """Test performance report formatting."""
        predictions = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        ground_truth = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        
        metrics = compute_performance_metrics(predictions, ground_truth, 3600.0)
        report = format_performance_report(metrics)
        
        assert isinstance(report, str)
        assert 'PERFORMANCE REPORT' in report
        assert 'Sensitivity' in report
        assert 'Specificity' in report
    
    def test_format_ablation_report(self):
        """Test ablation report formatting."""
        ground_truth = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        eeg_preds = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        ecg_preds = np.array([0, 0, 1, 1, 0, 0, 0, 0])
        coupled_preds = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        
        results = ablation_comparison(eeg_preds, ecg_preds, coupled_preds, ground_truth, 100.0)
        report = format_ablation_report(results)
        
        assert isinstance(report, str)
        assert 'ABLATION COMPARISON' in report
        assert 'EEG-Only' in report
        assert 'ECG-Only' in report
        assert 'Coupled' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
