"""Tests for the informed bettors and their probability signals."""
from __future__ import annotations

import numpy as np

from snapmarket.signals import (
    regime_conditional_probability,
    walk_forward_logistic_probability,
)
from snapmarket.strategies import (
    demand_responsive_pool,
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


def test_demand_responsive_pool_grows_with_better_odds():
    # Same seed isolates the elasticity: better odds (lower margin) must draw more volume.
    good_odds = demand_responsive_pool(base_stake=100.0)(0, _generator(), 1.90, 1.90)  # margin ~0.05
    bad_odds = demand_responsive_pool(base_stake=100.0)(0, _generator(), 1.50, 1.50)   # margin ~0.25
    assert sum(good_odds) > sum(bad_odds)


def test_demand_responsive_pool_matches_reference_at_reference_margin():
    # At the reference margin the multiplier is 1, so it matches a plain noise pool with the same seed.
    elastic = demand_responsive_pool(base_stake=100.0, reference_margin=0.125)(0, _generator(), 1.75, 1.75)
    plain = noise_pool(base_stake=100.0)(0, _generator(), 1.75, 1.75)
    assert np.allclose(elastic, plain, rtol=1e-3)


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
