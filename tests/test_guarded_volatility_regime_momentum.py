"""Tests for the guarded volatility-regime momentum model (defence in depth)."""
from __future__ import annotations

import numpy as np

from snapmarket.models import build_model

from .synthetic import (
    fast_guarded_parameters,
    fast_volatility_regime_momentum_parameters,
    shared_parameters,
    synthetic_features,
)


def test_model_is_registered_and_builds():
    features = synthetic_features()
    model = build_model("guarded_volatility_regime_momentum", features, shared_parameters(),
                        fast_guarded_parameters())
    assert model.name == "guarded_volatility_regime_momentum"
    assert len(model.display_probability) == features.number_of_seconds


def test_displayed_probability_stays_within_unit_interval():
    features = synthetic_features()
    model = build_model("guarded_volatility_regime_momentum", features, shared_parameters(),
                        fast_guarded_parameters())
    assert model.display_probability.min() >= 0.0
    assert model.display_probability.max() <= 1.0


def test_display_matches_the_underlying_regime_model():
    features = synthetic_features()
    shared = shared_parameters()
    parameters = fast_guarded_parameters()

    guarded = build_model("guarded_volatility_regime_momentum", features, shared, parameters)
    regime = build_model("volatility_regime_momentum", features, shared,
                         fast_volatility_regime_momentum_parameters())
    assert np.allclose(guarded.display_probability, regime.display_probability)


def test_margin_is_a_per_second_series_at_least_the_vig():
    features = synthetic_features()
    shared = shared_parameters()
    model = build_model("guarded_volatility_regime_momentum", features, shared,
                        fast_guarded_parameters())
    assert not np.isscalar(model.margin)
    assert np.nanmin(model.margin) >= shared.house_margin - 1e-9
