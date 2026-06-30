"""The guarded volatility-regime momentum model (defence in depth).

It displays the volatility-regime momentum curve (so it prices the direction, defending against
the regime-aware attacker) and, on top, charges an information margin computed from a richer
hidden logistic estimate (defending against the predictive attacker and any residual hidden
signal the displayed curve missed). It also carries the live-book risk margin.
"""
from __future__ import annotations

import numpy as np

from ...features import Features
from ...model import BookRiskParameters, Model
from ...parameters import SharedParameters
from ...registry import ModelSpecification, register_model
from ..hidden_symmetric_margin.information_margin import information_margin_over_display
from ..hidden_symmetric_margin.internal_probability import build_internal_probability
from ..volatility_regime_momentum.builder import build as build_volatility_regime_momentum
from .parameters import GuardedVolatilityRegimeMomentumParameters

MODEL_NAME = "guarded_volatility_regime_momentum"


def _book_risk_parameters(guard) -> BookRiskParameters | None:
    if guard.book_risk_margin_sensitivity == 0.0 and guard.terminal_gamma_margin_sensitivity == 0.0:
        return None
    return BookRiskParameters(
        book_risk_margin_sensitivity=guard.book_risk_margin_sensitivity,
        terminal_gamma_margin_sensitivity=guard.terminal_gamma_margin_sensitivity,
        stress_sigma_multipliers=guard.stress_sigma_multipliers,
    )


def build(features: Features, shared_parameters: SharedParameters,
          model_parameters: GuardedVolatilityRegimeMomentumParameters) -> Model:
    display_model = build_volatility_regime_momentum(
        features, shared_parameters, model_parameters.display)
    display_probability = display_model.display_probability

    hidden_probability = build_internal_probability(
        features, shared_parameters, model_parameters.guard, rolling=True)
    information_margin = information_margin_over_display(
        hidden_probability, display_probability, shared_parameters, model_parameters.guard)

    total_margin = np.clip(
        shared_parameters.house_margin + information_margin,
        0.0,
        shared_parameters.maximum_total_margin,
    )

    return Model(
        name=MODEL_NAME,
        display_probability=display_probability,
        margin=total_margin,
        shared_parameters=shared_parameters,
        internal_probability=hidden_probability,
        margin_components={"information": information_margin},
        book_risk_parameters=_book_risk_parameters(model_parameters.guard),
        first_evaluation_index=display_model.first_evaluation_index,
    )


register_model(ModelSpecification(
    name=MODEL_NAME,
    description="Volatility-regime momentum display plus an information margin from a hidden logistic.",
    default_parameters=GuardedVolatilityRegimeMomentumParameters,
    build=build,
))
