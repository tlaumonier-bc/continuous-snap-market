"""The hidden-signal symmetric-margin model.

Hide the directional P(up) behind a displayed probability of 0.50, and charge a
symmetric margin whenever the hidden signal says the move is predictable.
"""
from __future__ import annotations

import numpy as np

from ...features import Features
from ...model import BookRiskParameters, Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from .information_margin import information_margin_from_probability
from .internal_probability import build_internal_probability
from .parameters import HiddenSymmetricMarginParameters

MODEL_NAME = "hidden_symmetric_margin"


def _book_risk_parameters(parameters: HiddenSymmetricMarginParameters) -> BookRiskParameters | None:
    if (parameters.book_risk_margin_sensitivity == 0.0
            and parameters.terminal_gamma_margin_sensitivity == 0.0):
        return None
    return BookRiskParameters(
        book_risk_margin_sensitivity=parameters.book_risk_margin_sensitivity,
        terminal_gamma_margin_sensitivity=parameters.terminal_gamma_margin_sensitivity,
        stress_sigma_multipliers=parameters.stress_sigma_multipliers,
    )


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: HiddenSymmetricMarginParameters, rolling: bool = True) -> Model:
    internal_probability = build_internal_probability(
        features, shared_parameters, model_parameters, rolling=rolling,
    )
    information_margin = information_margin_from_probability(
        internal_probability, shared_parameters, model_parameters,
    )
    total_static_margin = np.clip(
        shared_parameters.house_margin + information_margin,
        0.0,
        shared_parameters.maximum_total_margin,
    )
    display_probability = np.full(features.number_of_seconds, 0.5)

    return Model(
        name=MODEL_NAME,
        display_probability=display_probability,
        margin=total_static_margin,
        shared_parameters=shared_parameters,
        internal_probability=internal_probability,
        margin_components={"information": information_margin},
        book_risk_parameters=_book_risk_parameters(model_parameters),
    )


register_model(ModelSpecification(
    name=MODEL_NAME,
    description="Hide directional P(up); charge a symmetric margin when predictability is high.",
    default_parameters=HiddenSymmetricMarginParameters,
    build=build,
))
