"""The rolling momentum-and-volatility model.

An improved momentum-lookup: it calibrates one momentum curve per volatility regime,
recomputed walk-forward on a trailing window. This prices the regime dependence that the
single-curve model ignores, closing the gap a regime-aware attacker exploits.
"""
from __future__ import annotations

import numpy as np

from ...features import Features, contract_entries
from ...model import Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from ..rolling import contract_up_outcomes, seconds_to_contracts, walk_forward_segments
from .calibration import apply_regime_curves, calibrate_regime_curves
from .parameters import MomentumVolatilityRollingParameters

MODEL_NAME = "momentum_volatility_rolling"


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: MomentumVolatilityRollingParameters) -> Model:
    horizon = shared_parameters.horizon_seconds
    momentum = features.standardized_momentum
    volatility = features.annualized_volatility
    volatility_bin_count = model_parameters.volatility_bin_count
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    outcomes = contract_up_outcomes(features, entries, shared_parameters)

    window_contracts = seconds_to_contracts(model_parameters.calibration_window_seconds, horizon)
    recompute_contracts = seconds_to_contracts(model_parameters.recompute_every_seconds, horizon)

    display_probability = np.full(features.number_of_seconds, 0.5)
    first_evaluation_index = features.number_of_seconds

    for training_slice, segment_start, segment_stop in walk_forward_segments(
            entries, window_contracts, recompute_contracts, features.number_of_seconds):
        training_entries = entries[training_slice]
        curves = calibrate_regime_curves(
            momentum[training_entries], volatility[training_entries],
            outcomes[training_slice], volatility_bin_count, model_parameters,
        )
        display_probability[segment_start:segment_stop] = apply_regime_curves(
            momentum[segment_start:segment_stop], volatility[segment_start:segment_stop],
            volatility[training_entries], volatility_bin_count, curves,
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
    description="Momentum lookup calibrated per volatility regime, recomputed walk-forward.",
    default_parameters=MomentumVolatilityRollingParameters,
    build=build,
))
