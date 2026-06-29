"""Tests for the rolling momentum-lookup model and cross-model evaluation alignment."""
from __future__ import annotations

import numpy as np

from snapmarket.experiments import common_evaluation_start
from snapmarket.features import contract_entries
from snapmarket.models import build_model

from .synthetic import fast_rolling_parameters, shared_parameters, synthetic_features


def test_rolling_model_is_registered_and_builds():
    features = synthetic_features()
    model = build_model("momentum_lookup_rolling", features, shared_parameters(),
                        fast_rolling_parameters())
    assert model.name == "momentum_lookup_rolling"
    assert len(model.display_probability) == features.number_of_seconds


def test_displayed_probability_stays_within_unit_interval():
    features = synthetic_features()
    model = build_model("momentum_lookup_rolling", features, shared_parameters(),
                        fast_rolling_parameters())
    assert model.display_probability.min() >= 0.0
    assert model.display_probability.max() <= 1.0


def test_evaluation_starts_after_the_first_full_window():
    features = synthetic_features()
    parameters = fast_rolling_parameters()
    model = build_model("momentum_lookup_rolling", features, shared_parameters(), parameters)

    shared = shared_parameters()
    entries = contract_entries(features.number_of_seconds, shared)
    window_contracts = parameters.calibration_window_seconds // shared.horizon_seconds
    assert model.first_evaluation_index == int(entries[window_contracts])


def test_nothing_is_quoted_with_an_edge_before_the_first_window():
    features = synthetic_features()
    model = build_model("momentum_lookup_rolling", features, shared_parameters(),
                        fast_rolling_parameters())
    assert np.allclose(model.display_probability[:model.first_evaluation_index], 0.5)


def test_common_evaluation_start_aligns_on_the_latest_model():
    features = synthetic_features()
    shared = shared_parameters()
    static_model = build_model("momentum_lookup", features, shared)
    rolling_model = build_model("momentum_lookup_rolling", features, shared,
                                fast_rolling_parameters())

    start = common_evaluation_start([static_model, rolling_model])
    assert start == max(static_model.first_evaluation_index,
                        rolling_model.first_evaluation_index)
