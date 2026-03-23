"""
Tests for the config adapter and its integration with the pipelines.

Verifies:
  1. YAML-style ``method: 'PLV'`` (upper-case) does not crash when
     processed through the adapter and passed to compute_delta_C.
  2. The ``order`` parameter from YAML features.delta_I.params is
     respected by compute_delta_I.
  3. The ``window_size`` and ``overlap`` from YAML sliding_window are
     respected by the pipeline convenience function.
"""

import math
import numpy as np
import pytest

from core.config_adapter import adapt_config
from core.features import compute_delta_C, compute_delta_I
from core.phase import extract_phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(duration: float = 5.0, fs: float = 64.0, seed: int = 0) -> np.ndarray:
    """Return a short, reproducible synthetic signal."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.2 * rng.standard_normal(n)
    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# 1.  PLV / plv case-insensitivity
# ---------------------------------------------------------------------------

class TestPLVCaseInsensitivity:
    """YAML delta_C method 'PLV' must not crash."""

    def test_adapt_config_normalises_plv_to_lowercase(self):
        """adapt_config should return 'plv' regardless of YAML casing."""
        yaml_config = {
            'features': {
                'delta_C': {'method': 'PLV'},
            }
        }
        adapted = adapt_config(yaml_config)
        assert adapted['delta_C_method'] == 'plv'

    def test_compute_delta_c_with_uppercase_plv_does_not_crash(self):
        """compute_delta_C(method='PLV') must not raise (function lower()s internally)."""
        fs = 64.0
        sig1 = _make_signal(fs=fs, seed=1)
        sig2 = _make_signal(fs=fs, seed=2)
        bl1 = _make_signal(fs=fs, seed=3)
        bl2 = _make_signal(fs=fs, seed=4)

        # Should not raise even with upper-case 'PLV'
        result = compute_delta_C(sig1, sig2, bl1, bl2, fs, method='PLV')
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_pipeline_yaml_plv_method_does_not_crash(self):
        """adapt_config with method='PLV' produces delta_C_method='plv'."""
        yaml_config = {
            'instability_functional': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3, 'threshold': 2.5},
            'sliding_window': {'window_size': 10.0, 'overlap': 0.5},
            'features': {
                'delta_I': {'method': 'permutation_entropy', 'params': {'order': 3, 'delay': 1}},
                'delta_C': {'method': 'PLV'},
            },
        }
        adapted = adapt_config(yaml_config)
        assert adapted['delta_C_method'] == 'plv'


# ---------------------------------------------------------------------------
# 2.  Entropy order from YAML is respected
# ---------------------------------------------------------------------------

class TestEntropyOrderFromYAML:
    """The order parameter from YAML features.delta_I.params must be used."""

    def test_adapt_config_extracts_entropy_order(self):
        """adapt_config should pick up order from features.delta_I.params."""
        yaml_config = {
            'features': {
                'delta_I': {
                    'method': 'permutation_entropy',
                    'params': {'order': 5, 'delay': 2},
                }
            }
        }
        adapted = adapt_config(yaml_config)
        assert adapted['entropy_order'] == 5
        assert adapted['entropy_delay'] == 2

    def test_entropy_order_affects_result(self):
        """
        compute_delta_I must produce different values for different orders.

        This is sufficient proof that the kwarg is forwarded, not silently
        dropped back to the default.
        """
        rng = np.random.default_rng(42)
        n = 512
        sig = rng.standard_normal(n)
        baseline = rng.standard_normal(n)

        delta_order3 = compute_delta_I(sig, baseline, method='permutation_entropy',
                                        order=3, delay=1)
        delta_order5 = compute_delta_I(sig, baseline, method='permutation_entropy',
                                        order=5, delay=1)

        # Different embedding orders should produce meaningfully different results.
        # A loose absolute tolerance avoids false negatives from floating-point
        # coincidences while still catching a forwarding bug (where both calls
        # would use the same default order and return identical values).
        assert abs(delta_order3 - delta_order5) > 1e-6, (
            "entropy_order=3 and entropy_order=5 returned nearly identical delta_I "
            "values; the order kwarg may not be forwarded correctly."
        )

    def test_adapt_config_default_entropy_order(self):
        """Without YAML params, adapt_config should default to order=3."""
        adapted = adapt_config({})
        assert adapted['entropy_order'] == 3
        assert adapted['entropy_delay'] == 1
        assert adapted['entropy_normalize'] is True

    def test_entropy_order_from_yaml_flows_through_eeg_only_pipeline(self):
        """
        EEGOnlyPipeline constructed via adapt_config stores the YAML order.
        """
        from pipelines.eeg_only import EEGOnlyPipeline
        from core.gate import InstabilityConfig

        yaml_config = {
            'features': {
                'delta_I': {'method': 'permutation_entropy', 'params': {'order': 5, 'delay': 1}},
            }
        }
        adapted = adapt_config(yaml_config)
        entropy_kwargs = {
            'order': adapted['entropy_order'],
            'delay': adapted['entropy_delay'],
            'normalize': adapted['entropy_normalize'],
        }
        pipeline = EEGOnlyPipeline(
            fs=64.0,
            entropy_kwargs=entropy_kwargs,
            delta_I_method=adapted['delta_I_method'],
        )
        assert pipeline._entropy_kwargs['order'] == 5


# ---------------------------------------------------------------------------
# 3.  Window size / overlap from YAML is respected
# ---------------------------------------------------------------------------

class TestWindowParamsFromYAML:
    """sliding_window.window_size and overlap from YAML must control windowing."""

    def test_adapt_config_extracts_window_params(self):
        """adapt_config should read nested sliding_window section."""
        yaml_config = {
            'sliding_window': {'window_size': 4.0, 'overlap': 0.25},
        }
        adapted = adapt_config(yaml_config)
        assert adapted['window_size'] == 4.0
        assert adapted['overlap'] == 0.25

    def test_adapt_config_default_window_params(self):
        """Without YAML settings, adapt_config defaults to 10 s / 50 %."""
        adapted = adapt_config({})
        assert adapted['window_size'] == 10.0
        assert adapted['overlap'] == 0.5

    def test_window_size_controls_number_of_windows(self):
        """
        The number of windows produced should match what window_size and
        overlap imply (independently of the pipeline internals).
        """
        from pipelines.eeg_only import EEGOnlyPipeline
        from datasets.synthetic import generate_synthetic_eeg

        fs = 256.0
        duration = 40.0   # signal length
        window_size = 4.0
        overlap = 0.0     # non-overlapping → easy arithmetic

        baseline = generate_synthetic_eeg(20.0, fs)
        signal = generate_synthetic_eeg(duration, fs)

        pipeline = EEGOnlyPipeline(fs=fs, baseline_duration=10.0)
        pipeline.set_baseline(baseline)

        results = pipeline.process_continuous(
            signal, window_size=window_size, overlap=overlap
        )

        expected_n_windows = int((len(signal) - int(window_size * fs)) //
                                  int(window_size * fs * (1 - overlap))) + 1
        assert len(results['timestamps']) == expected_n_windows

    def test_yaml_window_params_flow_through_run_eeg_only_pipeline(self):
        """
        run_eeg_only_pipeline with a YAML-style config dict produces the
        number of windows implied by the YAML window_size / overlap values.
        """
        from pipelines.eeg_only import run_eeg_only_pipeline
        from datasets.synthetic import generate_synthetic_eeg

        fs = 256.0
        signal = generate_synthetic_eeg(40.0, fs)
        baseline = generate_synthetic_eeg(20.0, fs)

        yaml_config = {
            'instability_functional': {
                'alpha': 0.6, 'beta': 0.4, 'gamma': 0.0, 'threshold': 2.5,
            },
            'sliding_window': {'window_size': 4.0, 'overlap': 0.0},
            'baseline': {'window_seconds': 20.0},
            'features': {
                'delta_I': {'method': 'permutation_entropy',
                             'params': {'order': 3, 'delay': 1}},
            },
        }

        results = run_eeg_only_pipeline(signal, baseline, fs, yaml_config)

        window_size = 4.0
        overlap = 0.0
        n_signal = len(signal)
        w_samples = int(window_size * fs)
        step_samples = int(w_samples * (1 - overlap))
        expected = (n_signal - w_samples) // step_samples + 1

        assert len(results['timestamps']) == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
