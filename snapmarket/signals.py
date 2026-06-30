"""Probability signals an informed bettor can act on.

Each builder returns a per-second estimate of P(price up over the horizon), computed with
no look-ahead, ready to feed into `strategies.predictive_bettor` / `regime_aware_bettor`.
The intelligence of an informed attacker lives here; the betting rule (positive expected
value at the quoted odds) is shared.
"""
from __future__ import annotations

import numpy as np

from .features import Features, contract_entries, quantile_bins
from .models.hidden_symmetric_margin import (
    HiddenSymmetricMarginParameters,
    build_internal_probability,
)
from .parameters import SharedParameters


def walk_forward_logistic_probability(
    features: Features,
    shared_parameters: SharedParameters,
    parameters: HiddenSymmetricMarginParameters | None = None,
) -> np.ndarray:
    """Walk-forward logistic estimate of P(up), reusing the model-3 internal estimator."""
    if parameters is None:
        parameters = HiddenSymmetricMarginParameters()
    return build_internal_probability(features, shared_parameters, parameters, rolling=True)


def _contract_up_outcomes(features: Features, entries: np.ndarray,
                          shared_parameters: SharedParameters) -> np.ndarray:
    horizon = shared_parameters.horizon_seconds
    price = features.price
    return (price[entries + horizon] > price[entries]).astype(float)


def _cell_probabilities(momentum_bin: np.ndarray, volatility_bin: np.ndarray,
                        outcomes: np.ndarray, momentum_bin_count: int,
                        volatility_bin_count: int, minimum_samples_per_cell: int) -> dict:
    probabilities = {}
    for m in range(momentum_bin_count):
        for v in range(volatility_bin_count):
            cell = (momentum_bin == m) & (volatility_bin == v)
            probabilities[(m, v)] = (float(outcomes[cell].mean())
                                     if cell.sum() > minimum_samples_per_cell else 0.5)
    return probabilities


def regime_conditional_probability(
    features: Features,
    shared_parameters: SharedParameters,
    training_fraction: float = 0.40,
    momentum_bin_count: int = 12,
    volatility_bin_count: int = 3,
    minimum_samples_per_cell: int = 200,
) -> np.ndarray:
    """P(up) calibrated per (momentum bin, volatility regime) on the training split.

    The house shows a single momentum curve; conditioning on the volatility regime exposes
    where that curve is wrong. Bin edges are frozen on the training contracts (no look-ahead).
    """
    momentum = features.standardized_momentum
    volatility = features.annualized_volatility
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    outcomes = _contract_up_outcomes(features, entries, shared_parameters)

    split_index = int(training_fraction * len(entries))
    training_momentum = momentum[entries[:split_index]]
    training_volatility = volatility[entries[:split_index]]

    training_momentum_bin = quantile_bins(training_momentum, training_momentum, momentum_bin_count)
    training_volatility_bin = quantile_bins(training_volatility, training_volatility, volatility_bin_count)
    cell_probabilities = _cell_probabilities(
        training_momentum_bin, training_volatility_bin, outcomes[:split_index],
        momentum_bin_count, volatility_bin_count, minimum_samples_per_cell,
    )

    momentum_bin = quantile_bins(momentum, training_momentum, momentum_bin_count)
    volatility_bin = quantile_bins(volatility, training_volatility, volatility_bin_count)
    probability_up = np.full(features.number_of_seconds, 0.5)
    for (m, v), probability in cell_probabilities.items():
        probability_up[(momentum_bin == m) & (volatility_bin == v)] = probability
    return probability_up
