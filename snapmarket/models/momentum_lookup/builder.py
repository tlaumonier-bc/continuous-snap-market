"""The momentum-lookup model (production v1).

Calibrate p_up on the oracle's own momentum with a fixed train/test split, then display
that calibrated curve with a constant margin.
"""
from __future__ import annotations

import numpy as np

from ...features import Features, contract_entries
from ...model import Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from .calibration import calibrate_fair_probability
from .parameters import MomentumLookupParameters

MODEL_NAME = "momentum_lookup"


def _training_outcomes(features: Features, training_entries: np.ndarray,
                       shared_parameters: SharedParameters) -> np.ndarray:
    horizon = shared_parameters.horizon_seconds
    price = features.price
    return (price[training_entries + horizon] > price[training_entries]).astype(float)


def _displayed_probability(features: Features, bin_edges: np.ndarray,
                           probability_per_bin: np.ndarray) -> np.ndarray:
    index = np.clip(np.digitize(features.standardized_momentum, bin_edges) - 1,
                    0, len(probability_per_bin) - 1)
    return probability_per_bin[index]


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: MomentumLookupParameters) -> Model:
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    split_index = int(model_parameters.training_fraction * len(entries))
    training_entries = entries[:split_index]

    bin_edges, probability_per_bin = calibrate_fair_probability(
        features.standardized_momentum[training_entries],
        _training_outcomes(features, training_entries, shared_parameters),
        model_parameters,
    )
    display_probability = _displayed_probability(features, bin_edges, probability_per_bin)

    return Model(
        name=MODEL_NAME,
        display_probability=display_probability,
        margin=shared_parameters.house_margin,
        shared_parameters=shared_parameters,
        first_evaluation_index=int(entries[split_index]) if split_index < len(entries)
        else features.number_of_seconds,
    )


register_model(ModelSpecification(
    name=MODEL_NAME,
    description="Calibrate p_up on trailing momentum with a fixed split and a constant margin.",
    default_parameters=MomentumLookupParameters,
    build=build,
))
