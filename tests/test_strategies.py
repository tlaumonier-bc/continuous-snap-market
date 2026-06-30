"""Tests for the informed bettors and their probability signals."""
from __future__ import annotations

import numpy as np

from snapmarket.signals import (
    regime_conditional_probability,
    walk_forward_logistic_probability,
)
from snapmarket.strategies import (
    lead_lag_bettor,
    noise_pool,
    predictive_bettor,
    regime_aware_bettor,
)

from .synthetic import (
    fast_hidden_margin_parameters,
    shared_parameters,
    synthetic_features,
)


def _generator():
    return np.random.default_rng(0)


def test_bettors_accept_the_quoted_odds_in_their_signature():
    bettor = noise_pool(base_stake=10.0)
    up, down = bettor(5, _generator(), 1.8, 1.9)
    assert up >= 0.0 and down >= 0.0


def test_predictive_bettor_takes_the_positive_expected_value_side():
    probability_up = np.full(10, 0.80)
    bettor = predictive_bettor(probability_up, minimum_edge=0.0)
    up, down = bettor(3, _generator(), odds_up=1.8, odds_down=1.8)
    assert up > 0.0 and down == 0.0


def test_predictive_bettor_passes_when_no_side_clears_the_edge():
    probability_up = np.full(10, 0.50)
    bettor = predictive_bettor(probability_up, minimum_edge=0.05)
    up, down = bettor(3, _generator(), odds_up=1.7, odds_down=1.7)
    assert up == 0.0 and down == 0.0


def test_regime_aware_bettor_uses_the_same_expected_value_rule():
    probability_up = np.full(10, 0.20)
    bettor = regime_aware_bettor(probability_up, minimum_edge=0.0)
    up, down = bettor(3, _generator(), odds_up=1.8, odds_down=1.8)
    assert down > 0.0 and up == 0.0


def test_lead_lag_bettor_follows_the_faster_feed():
    features = synthetic_features(1_000)
    fast_log_price = features.log_price.copy()
    fast_log_price[100] += 0.01            # fast feed above the oracle -> expect an up bet
    bettor = lead_lag_bettor(features, fast_log_price)
    up, down = bettor(100, _generator(), odds_up=1.9, odds_down=1.9)
    assert up > 0.0 and down == 0.0


def test_walk_forward_logistic_probability_is_a_valid_probability():
    features = synthetic_features()
    probability = walk_forward_logistic_probability(
        features, shared_parameters(), fast_hidden_margin_parameters())
    assert probability.shape == (features.number_of_seconds,)
    assert probability.min() >= 0.0 and probability.max() <= 1.0


def test_regime_conditional_probability_is_a_valid_probability():
    features = synthetic_features()
    probability = regime_conditional_probability(
        features, shared_parameters(), momentum_bin_count=4,
        volatility_bin_count=2, minimum_samples_per_cell=5)
    assert probability.shape == (features.number_of_seconds,)
    assert probability.min() >= 0.0 and probability.max() <= 1.0
