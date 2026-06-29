"""The rolling momentum-lookup model.

Identical calibration to the static momentum-lookup model, applied walk-forward: at each
recompute point, calibrate the fair-price curve on the trailing window of contracts, then
display that curve until the next recompute. Out-of-sample begins only once a full window
of history is available, which `first_evaluation_index` records.
"""
from __future__ import annotations

import numpy as np

from ...features import Features, contract_entries
from ...model import Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from ..momentum_lookup.calibration import calibrate_fair_probability
from .parameters import MomentumLookupRollingParameters

MODEL_NAME = "momentum_lookup_rolling"


def _contract_outcomes(features: Features, entries: np.ndarray,
                       shared_parameters: SharedParameters) -> np.ndarray:
    horizon = shared_parameters.horizon_seconds
    price = features.price
    return (price[entries + horizon] > price[entries]).astype(float)


def _seconds_to_contracts(seconds: int, horizon: int) -> int:
    return max(1, seconds // horizon)


def _apply_lookup(momentum_segment: np.ndarray, bin_edges: np.ndarray,
                  probability_per_bin: np.ndarray) -> np.ndarray:
    index = np.clip(np.digitize(momentum_segment, bin_edges) - 1,
                    0, len(probability_per_bin) - 1)
    return probability_per_bin[index]


def _segment_bounds(entries: np.ndarray, start: int, recompute_contracts: int,
                    number_of_seconds: int) -> tuple[int, int]:
    segment_start = int(entries[start])
    next_start = min(len(entries), start + recompute_contracts)
    segment_stop = int(entries[next_start]) if next_start < len(entries) else number_of_seconds
    return segment_start, segment_stop


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: MomentumLookupRollingParameters) -> Model:
    horizon = shared_parameters.horizon_seconds
    momentum = features.standardized_momentum
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    outcomes = _contract_outcomes(features, entries, shared_parameters)

    window_contracts = _seconds_to_contracts(model_parameters.calibration_window_seconds, horizon)
    recompute_contracts = _seconds_to_contracts(model_parameters.recompute_every_seconds, horizon)

    display_probability = np.full(features.number_of_seconds, 0.5)
    first_evaluation_index = features.number_of_seconds

    for start in range(window_contracts, len(entries), recompute_contracts):
        training_slice = slice(start - window_contracts, start)
        bin_edges, probability_per_bin = calibrate_fair_probability(
            momentum[entries[training_slice]], outcomes[training_slice], model_parameters,
        )
        segment_start, segment_stop = _segment_bounds(entries, start, recompute_contracts,
                                                      features.number_of_seconds)
        display_probability[segment_start:segment_stop] = _apply_lookup(
            momentum[segment_start:segment_stop], bin_edges, probability_per_bin,
        )
        first_evaluation_index = min(first_evaluation_index, segment_start)

    return Model(
        name=MODEL_NAME,
        display_probability=display_probability,
        margin=shared_parameters.house_margin,
        shared_parameters=shared_parameters,
        first_evaluation_index=first_evaluation_index,
    )


register_model(ModelSpecification(
    name=MODEL_NAME,
    description="Momentum-lookup calibration recomputed walk-forward on a trailing window.",
    default_parameters=MomentumLookupRollingParameters,
    build=build,
))
