"""Calibration of the displayed fair probability for the momentum-lookup model."""
from __future__ import annotations

import numpy as np

from .parameters import MomentumLookupParameters


def _quantile_bin_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, bin_count + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _empirical_probability_per_bin(bin_index: np.ndarray, outcomes: np.ndarray,
                                   parameters: MomentumLookupParameters) -> np.ndarray:
    return np.array([
        outcomes[bin_index == i].mean()
        if (bin_index == i).sum() > parameters.minimum_samples_per_bin else 0.5
        for i in range(parameters.calibration_bin_count)
    ])


def _enforce_monotonicity(probability_per_bin: np.ndarray,
                          parameters: MomentumLookupParameters) -> np.ndarray:
    probability_per_bin = probability_per_bin.copy()
    bin_count = len(probability_per_bin)
    for _ in range(parameters.monotonic_pooling_passes):
        for i in range(bin_count - 1):
            if probability_per_bin[i] > probability_per_bin[i + 1]:
                pooled = (probability_per_bin[i] + probability_per_bin[i + 1]) / 2
                probability_per_bin[i] = probability_per_bin[i + 1] = pooled
    return probability_per_bin


def _shrink_toward_half(probability_per_bin: np.ndarray,
                        parameters: MomentumLookupParameters) -> np.ndarray:
    return 0.5 + parameters.calibration_shrinkage * (probability_per_bin - 0.5)


def calibrate_fair_probability(training_momentum: np.ndarray, training_outcomes: np.ndarray,
                               parameters: MomentumLookupParameters):
    """Return (bin_edges, fair_probability_per_bin): a monotone, shrunk lookup of p_up vs the state."""
    bin_edges = _quantile_bin_edges(training_momentum, parameters.calibration_bin_count)
    bin_index = np.digitize(training_momentum, bin_edges) - 1

    probability_per_bin = _empirical_probability_per_bin(bin_index, training_outcomes, parameters)
    probability_per_bin = _enforce_monotonicity(probability_per_bin, parameters)
    return bin_edges, _shrink_toward_half(probability_per_bin, parameters)
