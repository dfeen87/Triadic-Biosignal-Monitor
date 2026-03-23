"""
Config Adapter for Triadic Biosignal Monitor

Translates the nested YAML configuration structure into a flat dictionary
that the pipeline convenience functions can consume directly.

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

from typing import Any, Dict


def adapt_config(raw_config: Dict) -> Dict:
    """
    Flatten a nested YAML config dict into the parameters used by pipelines.

    Supports both the full nested YAML structure (e.g. loaded via
    ``yaml.safe_load``) and the older flat dicts that tests pass directly.
    When a key is present at both levels the nested value takes precedence.

    Parameters
    ----------
    raw_config : dict
        Raw configuration dictionary, typically loaded from a YAML file.
        May be ``None`` or empty, in which case all defaults are used.

    Returns
    -------
    dict
        Flat configuration dict with keys:

        - ``alpha`` – spectral/morphological weight
        - ``beta``  – information/entropy weight
        - ``gamma`` – coupling/coherence weight
        - ``threshold`` – decision gate threshold τ
        - ``window_size`` – sliding-window size in seconds
        - ``overlap`` – sliding-window overlap fraction [0, 1)
        - ``baseline_duration`` – baseline window length in seconds
        - ``delta_I_method`` – entropy method name (lower-cased)
        - ``entropy_order`` – permutation-entropy embedding order
        - ``entropy_delay`` – permutation-entropy time delay
        - ``entropy_normalize`` – whether to normalise entropy to [0, 1]
        - ``delta_C_method`` – coupling method name (lower-cased)
    """
    cfg = raw_config or {}

    inst = cfg.get('instability_functional', {})
    window = cfg.get('sliding_window', {})
    baseline = cfg.get('baseline', {})
    delta_i_cfg = cfg.get('features', {}).get('delta_I', {})
    delta_c_cfg = cfg.get('features', {}).get('delta_C', {})
    entropy_params = delta_i_cfg.get('params', {})

    def _get(nested_val: Any, flat_key: str, default: Any) -> Any:
        """Return nested_val if set, else fall back to flat key, else default."""
        if nested_val is not None:
            return nested_val
        return cfg.get(flat_key, default)

    return {
        'alpha': _get(inst.get('alpha'), 'alpha', 0.4),
        'beta': _get(inst.get('beta'), 'beta', 0.3),
        'gamma': _get(inst.get('gamma'), 'gamma', 0.3),
        'threshold': _get(inst.get('threshold'), 'threshold', 2.5),
        'window_size': _get(window.get('window_size'), 'window_size', 10.0),
        'overlap': _get(window.get('overlap'), 'overlap', 0.5),
        'baseline_duration': _get(
            baseline.get('window_seconds'), 'baseline_duration', 60.0
        ),
        'delta_I_method': _get(
            delta_i_cfg.get('method'), 'delta_I_method', 'permutation_entropy'
        ).lower(),
        'entropy_order': _get(entropy_params.get('order'), 'entropy_order', 3),
        'entropy_delay': _get(entropy_params.get('delay'), 'entropy_delay', 1),
        'entropy_normalize': _get(
            entropy_params.get('normalize'), 'entropy_normalize', True
        ),
        'delta_C_method': _get(
            delta_c_cfg.get('method'), 'delta_C_method', 'plv'
        ).lower(),
    }
