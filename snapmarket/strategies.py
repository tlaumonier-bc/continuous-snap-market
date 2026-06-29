"""Bettor strategies.

Every strategy is a closure over the features and its own parameters, exposing the
same call signature so the engine can run any of them against any model:

    bettor(t, random_generator) -> (up_stake, down_stake)

A pooled flow returns stakes on both sides; a selective attacker returns one side or
zero. Stakes are drawn inside the bettor, so each bettor sizes its own flow.
"""
from __future__ import annotations

import numpy as np


def noise_pool(base_stake: float = 50.0, spread: float = 0.10):
    """Uninformed background flow: roughly balanced, small random up/down tilt each second."""
    def bettor(t, random_generator):
        amount = base_stake * random_generator.exponential(1.0)
        up_fraction = float(np.clip(0.5 + random_generator.normal(0, spread), 0, 1))
        return amount * up_fraction, amount * (1 - up_fraction)
    return bettor


def momentum_follower(features, lookback: int = 5, base_stake: float = 50.0, size: float = 1.0):
    """Bet the direction of the trailing move (trend following)."""
    price = features.price
    def bettor(t, random_generator):
        amount = base_stake * size * random_generator.exponential(1.0)
        if price[t] > price[t - lookback]:
            return amount, 0.0
        if price[t] < price[t - lookback]:
            return 0.0, amount
        return 0.0, 0.0
    return bettor


def mean_reversion_fader(features, lookback: int = 5, base_stake: float = 50.0, size: float = 1.0):
    """Bet against the trailing move (mean reversion)."""
    price = features.price
    def bettor(t, random_generator):
        amount = base_stake * size * random_generator.exponential(1.0)
        if price[t] > price[t - lookback]:
            return 0.0, amount
        if price[t] < price[t - lookback]:
            return amount, 0.0
        return 0.0, 0.0
    return bettor


def pocket_follower(features, momentum_high_threshold, momentum_low_threshold,
                    low_volatility_threshold, base_stake: float = 50.0, size: float = 1.0):
    """The Part 10 attacker: follow the move, but only in the low-volatility extreme-momentum
    pocket where the single displayed curve under-prices the tail. Bet nothing elsewhere."""
    volatility = features.annualized_volatility
    momentum = features.standardized_momentum
    def bettor(t, random_generator):
        if volatility[t] < low_volatility_threshold:
            amount = base_stake * size * random_generator.exponential(1.0)
            if momentum[t] >= momentum_high_threshold:
                return amount, 0.0
            if momentum[t] <= momentum_low_threshold:
                return 0.0, amount
        return 0.0, 0.0
    return bettor


def informed_bettor(features, accuracy: float = 0.60, base_stake: float = 50.0, size: float = 1.0):
    """Forward-information benchmark: knows the settlement direction with the given accuracy.
    Upper bound on what private information is worth; not a realistic public strategy."""
    price = features.price
    horizon = 30
    def bettor(t, random_generator):
        amount = base_stake * size * random_generator.exponential(1.0)
        bet_up = (price[t + horizon] > price[t]) == (random_generator.random() < accuracy)
        return (amount, 0.0) if bet_up else (0.0, amount)
    return bettor
