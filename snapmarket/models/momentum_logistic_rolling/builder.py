"""The rolling logistic momentum model.

It displays the walk-forward logistic probability of an up move directly, instead of hiding it
behind a flat 0.50 and a symmetric margin (the hidden-signal model). By pricing what an
informed predictive attacker would model, it removes that attacker's edge: betting a fairly
priced side only pays the vig.
"""
from __future__ import annotations

import numpy as np

from ...features import Features, contract_entries
from ...model import Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from ..hidden_symmetric_margin import HiddenSymmetricMarginParameters
from ..hidden_symmetric_margin.internal_probability import build_internal_probability
from .parameters import MomentumLogisticRollingParameters

MODEL_NAME = "momentum_logistic_rolling"


def _logistic_configuration(parameters: MomentumLogisticRollingParameters) -> HiddenSymmetricMarginParameters:
    """Map this model's parameters onto the shared logistic estimator's configuration."""
    return HiddenSymmetricMarginParameters(
        minimum_samples_per_bin=parameters.minimum_samples_per_fit,
        internal_minimum_training_contracts=parameters.minimum_training_contracts,
        internal_training_window_contracts=parameters.training_window_contracts,
        internal_retrain_contracts=parameters.retrain_contracts,
        internal_logistic_iterations=parameters.logistic_iterations,
        internal_logistic_learning_rate=parameters.logistic_learning_rate,
        internal_logistic_ridge_penalty=parameters.logistic_ridge_penalty,
        internal_probability_clip=parameters.probability_clip,
    )


def _first_priced_second(features: Features, shared_parameters: SharedParameters,
                         parameters: MomentumLogisticRollingParameters) -> int:
    """First second the walk-forward estimator prices, matching build_internal_probability."""
    entries = contract_entries(features.number_of_seconds, shared_parameters)
    minimum = min(parameters.minimum_training_contracts, len(entries))
    if minimum >= len(entries):
        return features.number_of_seconds
    return int(entries[minimum])


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: MomentumLogisticRollingParameters) -> Model:
    probability = build_internal_probability(
        features, shared_parameters, _logistic_configuration(model_parameters), rolling=True,
    )
    display_probability = 0.5 + model_parameters.display_shrinkage * (probability - 0.5)

    return Model(
        name=MODEL_NAME,
        display_probability=display_probability,
        margin=shared_parameters.house_margin,
        shared_parameters=shared_parameters,
        internal_probability=probability,
        first_evaluation_index=_first_priced_second(features, shared_parameters, model_parameters),
    )


register_model(ModelSpecification(
    name=MODEL_NAME,
    description="Displays a walk-forward multi-feature logistic estimate of P(up).",
    default_parameters=MomentumLogisticRollingParameters,
    build=build,
))
