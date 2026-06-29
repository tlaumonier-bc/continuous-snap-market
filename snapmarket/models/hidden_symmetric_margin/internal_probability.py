"""Hidden walk-forward estimate of P(price up over the horizon).

This probability is internal/diagnostic only: the model never displays the direction.
It hand-rolls a logistic regression so the estimate stays independent of scikit-learn,
then converts the signal strength into a symmetric margin elsewhere.
"""
from __future__ import annotations

import numpy as np

from ...features import Features, contract_entries
from ...parameters import SharedParameters
from .parameters import HiddenSymmetricMarginParameters

_LINEAR_CLIP = 35.0


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -_LINEAR_CLIP, _LINEAR_CLIP)))


def _feature_matrix(features: Features, index=slice(None)) -> np.ndarray:
    """Feature matrix for the hidden probability model.

    The scaling keeps the hand-written logistic optimiser numerically stable.
    """
    return np.column_stack([
        features.standardized_momentum[index],
        features.absolute_standardized_momentum[index],
        features.momentum_2_seconds[index] * 1e4,
        features.momentum_5_seconds[index] * 1e4,
        features.momentum_15_seconds[index] * 1e4,
        features.momentum_30_seconds[index] * 1e4,
        features.momentum_60_seconds[index] * 1e4,
        features.annualized_volatility[index],
        features.position_in_range_60_seconds[index],
        features.position_in_range_120_seconds[index],
        features.path_efficiency_30_seconds[index],
        features.acceleration[index] * 1e4,
    ]).astype(float)


def _standardize(matrix: np.ndarray):
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (matrix - mean) / scale, mean, scale


def _gradient_descent(standardized_features: np.ndarray, outcomes: np.ndarray,
                      parameters: HiddenSymmetricMarginParameters) -> np.ndarray:
    weights = np.zeros(standardized_features.shape[1] + 1)
    sample_count = len(outcomes)
    for _ in range(parameters.internal_logistic_iterations):
        linear = weights[0] + standardized_features @ weights[1:]
        error = _sigmoid(linear) - outcomes
        gradient = np.empty_like(weights)
        gradient[0] = error.mean()
        gradient[1:] = (standardized_features.T @ error) / sample_count \
            + parameters.internal_logistic_ridge_penalty * weights[1:]
        weights -= parameters.internal_logistic_learning_rate * gradient
    return weights


def _fit_logistic(training_features: np.ndarray, training_outcomes: np.ndarray,
                  parameters: HiddenSymmetricMarginParameters):
    finite = np.isfinite(training_features).all(axis=1) & np.isfinite(training_outcomes)
    feature_matrix = training_features[finite]
    outcomes = training_outcomes[finite].astype(float)
    if len(outcomes) < parameters.minimum_samples_per_bin or outcomes.min() == outcomes.max():
        return None

    standardized_features, mean, scale = _standardize(feature_matrix)
    weights = _gradient_descent(standardized_features, outcomes, parameters)
    return weights, mean, scale


def _predict_logistic(feature_matrix: np.ndarray, fitted,
                      parameters: HiddenSymmetricMarginParameters) -> np.ndarray:
    if fitted is None:
        return np.full(len(feature_matrix), 0.5)
    weights, mean, scale = fitted
    standardized = np.nan_to_num((feature_matrix - mean) / scale,
                                 nan=0.0, posinf=0.0, neginf=0.0)
    probability = _sigmoid(weights[0] + standardized @ weights[1:])
    clip = parameters.internal_probability_clip
    return np.clip(probability, clip, 1.0 - clip)


def _fill_segment(internal_probability: np.ndarray, features: Features, start: int, stop: int,
                  fitted, parameters: HiddenSymmetricMarginParameters,
                  chunk_size: int = 500_000) -> None:
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(stop, chunk_start + chunk_size)
        internal_probability[chunk_start:chunk_stop] = _predict_logistic(
            _feature_matrix(features, slice(chunk_start, chunk_stop)), fitted, parameters,
        )


def _fill_fixed_split(internal_probability, features, entries, outcomes,
                      parameters: HiddenSymmetricMarginParameters) -> None:
    split_index = max(1, int(parameters.training_fraction * len(entries)))
    fitted = _fit_logistic(_feature_matrix(features, entries[:split_index]),
                           outcomes[:split_index], parameters)
    _fill_segment(internal_probability, features, 0, features.number_of_seconds,
                  fitted, parameters)


def _fill_walk_forward(internal_probability, features, entries, outcomes,
                       parameters: HiddenSymmetricMarginParameters) -> None:
    minimum = min(parameters.internal_minimum_training_contracts, len(entries))
    if minimum < parameters.minimum_samples_per_bin:
        return

    retrain_every = max(1, parameters.internal_retrain_contracts)
    window_size = max(minimum, parameters.internal_training_window_contracts)

    for start in range(minimum, len(entries), retrain_every):
        train_start = max(0, start - window_size)
        train_entries = entries[train_start:start]
        fitted = _fit_logistic(_feature_matrix(features, train_entries),
                               outcomes[train_start:start], parameters)

        segment_start = int(entries[start])
        next_start = min(len(entries), start + retrain_every)
        segment_stop = int(entries[next_start]) if next_start < len(entries) \
            else features.number_of_seconds
        _fill_segment(internal_probability, features, segment_start, segment_stop,
                      fitted, parameters)


def build_internal_probability(features: Features, shared_parameters: SharedParameters,
                               parameters: HiddenSymmetricMarginParameters,
                               rolling: bool = True) -> np.ndarray:
    """Walk-forward (or fixed-split) hidden estimate of P(price[t + horizon] > price[t])."""
    horizon = shared_parameters.horizon_seconds
    price = features.price
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    outcomes = (price[entries + horizon] > price[entries]).astype(float)

    internal_probability = np.full(features.number_of_seconds, 0.5)
    if len(entries) == 0:
        return internal_probability

    if rolling:
        _fill_walk_forward(internal_probability, features, entries, outcomes, parameters)
    else:
        _fill_fixed_split(internal_probability, features, entries, outcomes, parameters)
    return internal_probability
