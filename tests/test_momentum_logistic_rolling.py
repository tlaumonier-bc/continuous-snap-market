"""Tests for the rolling logistic momentum model."""
from __future__ import annotations

import numpy as np

from snapmarket.features import contract_entries
from snapmarket.models import build_model
from snapmarket.models.momentum_logistic_rolling import MomentumLogisticRollingParameters

from .synthetic import (
    fast_momentum_logistic_parameters,
    shared_parameters,
    synthetic_features,
)


def test_model_is_registered_and_builds():
    features = synthetic_features()
    model = build_model("momentum_logistic_rolling", features, shared_parameters(),
                        fast_momentum_logistic_parameters())
    assert model.name == "momentum_logistic_rolling"
    assert len(model.display_probability) == features.number_of_seconds


def test_displayed_probability_stays_within_unit_interval():
    features = synthetic_features()
    model = build_model("momentum_logistic_rolling", features, shared_parameters(),
                        fast_momentum_logistic_parameters())
    assert model.display_probability.min() >= 0.0
    assert model.display_probability.max() <= 1.0


def test_display_shrinkage_zero_collapses_to_one_half():
    features = synthetic_features()
    parameters = MomentumLogisticRollingParameters(
        minimum_training_contracts=50, training_window_contracts=100,
        retrain_contracts=50, logistic_iterations=10, minimum_samples_per_fit=10,
        display_shrinkage=0.0)
    model = build_model("momentum_logistic_rolling", features, shared_parameters(), parameters)
    assert np.allclose(model.display_probability, 0.5)


def test_first_evaluation_index_matches_the_first_priced_contract():
    features = synthetic_features()
    parameters = fast_momentum_logistic_parameters()
    model = build_model("momentum_logistic_rolling", features, shared_parameters(), parameters)

    entries = contract_entries(features.number_of_seconds, shared_parameters())
    assert model.first_evaluation_index == int(entries[parameters.minimum_training_contracts])
