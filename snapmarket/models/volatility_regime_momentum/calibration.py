"""Per-volatility-regime momentum calibration.

For each volatility regime we calibrate an independent momentum curve with the exact same
discipline as the classic model (quantile bins, monotone pooling, shrinkage), reusing
`calibrate_fair_probability`. The volatility bin edges are frozen on the training window.
"""
from __future__ import annotations

import numpy as np

from ...features import quantile_bins
from ..momentum_lookup.calibration import calibrate_fair_probability


def calibrate_regime_curves(training_momentum: np.ndarray, training_volatility: np.ndarray,
                            training_outcomes: np.ndarray, volatility_bin_count: int,
                            parameters) -> list:
    """One (bin_edges, probability_per_bin) momentum curve per volatility regime."""
    regime = quantile_bins(training_volatility, training_volatility, volatility_bin_count)
    curves = []
    for regime_index in range(volatility_bin_count):
        in_regime = regime == regime_index
        if in_regime.sum() <= parameters.minimum_samples_per_bin:
            curves.append(None)
            continue
        curves.append(calibrate_fair_probability(
            training_momentum[in_regime], training_outcomes[in_regime], parameters,
        ))
    return curves


def apply_regime_curves(momentum_segment: np.ndarray, volatility_segment: np.ndarray,
                        training_volatility: np.ndarray, volatility_bin_count: int,
                        curves: list) -> np.ndarray:
    """Price each second with the momentum curve of its volatility regime (0.5 fallback)."""
    regime = quantile_bins(volatility_segment, training_volatility, volatility_bin_count)
    probability = np.full(len(momentum_segment), 0.5)
    for regime_index, curve in enumerate(curves):
        if curve is None:
            continue
        in_regime = regime == regime_index
        if not in_regime.any():
            continue
        bin_edges, probability_per_bin = curve
        index = np.clip(np.digitize(momentum_segment[in_regime], bin_edges) - 1,
                        0, len(probability_per_bin) - 1)
        probability[in_regime] = probability_per_bin[index]
    return probability
