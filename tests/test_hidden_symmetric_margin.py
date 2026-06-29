"""Tests for the hidden-signal symmetric-margin model."""
from __future__ import annotations

import numpy as np

from snapmarket.engine import book_risk_margin, simulate
from snapmarket.model import BookRiskParameters, Model
from snapmarket.models import build_model
from snapmarket.models.hidden_symmetric_margin import information_margin_from_probability
from snapmarket.parameters import SharedParameters
from snapmarket.pricing import quote_odds
from snapmarket.strategies import noise_pool

from .synthetic import (
    fast_hidden_margin_parameters,
    shared_parameters,
    synthetic_features,
)


def test_information_margin_is_symmetric_and_zero_at_one_half():
    shared = SharedParameters()
    parameters = fast_hidden_margin_parameters()
    margin = information_margin_from_probability(np.array([0.40, 0.50, 0.60]), shared, parameters)
    assert margin[0] == margin[2]
    assert margin[1] == 0.0


def test_information_margin_makes_the_stronger_side_non_positive_expected_value():
    shared = SharedParameters()
    parameters = fast_hidden_margin_parameters()
    probability = np.array([0.40, 0.50, 0.60])
    information_margin = information_margin_from_probability(probability, shared, parameters)

    total_margin = shared.house_margin + information_margin[2]
    odds = quote_odds(0.50, total_margin, 0.0, shared)
    assert 1.0 - probability[2] * odds[0] >= parameters.information_margin_buffer - 1e-9
    assert 1.0 - (1.0 - probability[0]) * odds[1] >= parameters.information_margin_buffer - 1e-9


def test_model_hides_internal_direction_behind_a_flat_display():
    features = synthetic_features()
    model = build_model("hidden_symmetric_margin", features, shared_parameters(),
                        fast_hidden_margin_parameters())
    assert model.internal_probability is not None
    assert np.allclose(model.display_probability, 0.5)
    assert np.nanmax(model.margin) >= model.shared_parameters.house_margin


def test_book_risk_margin_activates_only_with_book_risk_parameters():
    features = synthetic_features(1_000)
    shared = shared_parameters()
    model = Model(
        name="risk_model",
        display_probability=np.full(features.number_of_seconds, 0.5),
        margin=shared.house_margin,
        shared_parameters=shared,
        book_risk_parameters=BookRiskParameters(
            book_risk_margin_sensitivity=0.10,
            terminal_gamma_margin_sensitivity=0.10,
        ),
    )

    start_index = 100
    step = 20
    total_up = np.zeros(60)
    total_down = np.zeros(60)
    total_up[10:20] = 1_000
    odds_up = np.full(60, 1.75)
    odds_down = np.full(60, 1.75)

    margin = book_risk_margin(model, features, start_index, step, total_up, total_down,
                              odds_up, odds_down)
    assert margin > 0.0


def test_book_risk_margin_is_zero_without_book_risk_parameters():
    features = synthetic_features(1_000)
    shared = shared_parameters()
    model = Model(
        name="plain_model",
        display_probability=np.full(features.number_of_seconds, 0.5),
        margin=shared.house_margin,
        shared_parameters=shared,
    )
    total_up = np.zeros(60)
    total_down = np.zeros(60)
    total_up[10:20] = 1_000
    margin = book_risk_margin(model, features, 100, 20, total_up, total_down,
                              np.full(60, 1.75), np.full(60, 1.75))
    assert margin == 0.0


def test_simulation_accepts_the_per_second_margin_series():
    features = synthetic_features()
    model = build_model("hidden_symmetric_margin", features, shared_parameters(),
                        fast_hidden_margin_parameters())
    result = simulate(model, features, {"noise": noise_pool(base_stake=10)}, 500, 200, seed=1)
    assert len(result.risk_margin_series) == 200
    assert result.total_volume > 0
