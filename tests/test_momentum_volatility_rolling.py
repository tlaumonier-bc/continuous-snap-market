"""Tests for the rolling momentum-and-volatility model."""
from __future__ import annotations

import numpy as np

from snapmarket.features import contract_entries
from snapmarket.models import build_model

from .synthetic import (
    fast_momentum_volatility_parameters,
    shared_parameters,
    synthetic_features,
)


def test_model_is_registered_and_builds():
    features = synthetic_features()
    model = build_model("momentum_volatility_rolling", features, shared_parameters(),
                        fast_momentum_volatility_parameters())
    assert model.name == "momentum_volatility_rolling"
    assert len(model.display_probability) == features.number_of_seconds


def test_displayed_probability_stays_within_unit_interval():
    features = synthetic_features()
    model = build_model("momentum_volatility_rolling", features, shared_parameters(),
                        fast_momentum_volatility_parameters())
    assert model.display_probability.min() >= 0.0
    assert model.display_probability.max() <= 1.0


def test_evaluation_starts_after_the_first_full_window():
    features = synthetic_features()
    parameters = fast_momentum_volatility_parameters()
    model = build_model("momentum_volatility_rolling", features, shared_parameters(), parameters)

    shared = shared_parameters()
    entries = contract_entries(features.number_of_seconds, shared)
    window_contracts = parameters.calibration_window_seconds // shared.horizon_seconds
    assert model.first_evaluation_index == int(entries[window_contracts])


def test_volatility_conditioning_changes_the_displayed_curve():
    features = synthetic_features()
    shared = shared_parameters()
    parameters = fast_momentum_volatility_parameters()

    one_regime = build_model(
        "momentum_volatility_rolling", features, shared,
        parameters.__class__(**{**parameters.__dict__, "volatility_bin_count": 1}))
    two_regimes = build_model("momentum_volatility_rolling", features, shared, parameters)

    start = two_regimes.first_evaluation_index
    assert not np.allclose(one_regime.display_probability[start:],
                           two_regimes.display_probability[start:])
