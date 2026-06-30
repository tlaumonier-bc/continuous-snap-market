"""Bettor strategies.

Every strategy is a closure over the features and its own parameters, exposing the
same call signature so the engine can run any of them against any model:

    bettor(t, random_generator, odds_up, odds_down) -> (up_stake, down_stake)

The current quoted odds are passed in, so an informed bettor can size on expected value.
A pooled flow returns stakes on both sides; a selective attacker returns one side or
zero. Stakes are drawn inside the bettor, so each bettor sizes its own flow.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Uninformed and public-signal flow                                          #
# --------------------------------------------------------------------------- #


def noise_pool(base_stake: float = 50.0, spread: float = 0.10):
    """Uninformed background flow: roughly balanced, small random up/down tilt each second."""
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * random_generator.exponential(1.0)
        up_fraction = float(np.clip(0.5 + random_generator.normal(0, spread), 0, 1))
        return amount * up_fraction, amount * (1 - up_fraction)
    return bettor


def momentum_follower(features, lookback: int = 5, base_stake: float = 50.0, size: float = 1.0):
    """Bet the direction of the trailing move (trend following)."""
    price = features.price
    def bettor(t, random_generator, odds_up, odds_down):
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
    def bettor(t, random_generator, odds_up, odds_down):
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
    def bettor(t, random_generator, odds_up, odds_down):
        if volatility[t] < low_volatility_threshold:
            amount = base_stake * size * random_generator.exponential(1.0)
            if momentum[t] >= momentum_high_threshold:
                return amount, 0.0
            if momentum[t] <= momentum_low_threshold:
                return 0.0, amount
        return 0.0, 0.0
    return bettor


def informed_bettor(features, horizon: int, accuracy: float = 0.60,
                    base_stake: float = 50.0, size: float = 1.0):
    """Forward-information benchmark: knows the settlement direction with the given accuracy.
    Upper bound on what private information is worth; not a realistic public strategy."""
    price = features.price
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * size * random_generator.exponential(1.0)
        bet_up = (price[t + horizon] > price[t]) == (random_generator.random() < accuracy)
        return (amount, 0.0) if bet_up else (0.0, amount)
    return bettor


# --------------------------------------------------------------------------- #
#  Informed flow: bet the positive expected-value side                        #
# --------------------------------------------------------------------------- #


def _expected_value_bettor(probability_up, minimum_edge: float,
                           base_stake: float, size: float):
    """Bet the side with the higher expected value, if it clears `minimum_edge`.

    Expected value per $1 staked is `probability * odds - 1` (a push refunds the stake,
    a rare event ignored here). The probability estimate must be built walk-forward so it
    uses no future information.
    """
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * size * random_generator.exponential(1.0)
        probability = probability_up[t]
        expected_value_up = probability * odds_up - 1.0
        expected_value_down = (1.0 - probability) * odds_down - 1.0
        if expected_value_up >= expected_value_down and expected_value_up > minimum_edge:
            return amount, 0.0
        if expected_value_down > minimum_edge:
            return 0.0, amount
        return 0.0, 0.0
    return bettor


def predictive_bettor(probability_up, minimum_edge: float = 0.0,
                      base_stake: float = 50.0, size: float = 1.0):
    """Informed flow from a walk-forward logistic estimate of P(up).

    `probability_up` is a per-second array (see `signals.walk_forward_logistic_probability`).
    The bettor takes the positive-expected-value side at the quoted odds.
    """
    return _expected_value_bettor(probability_up, minimum_edge, base_stake, size)


def regime_aware_bettor(probability_up, minimum_edge: float = 0.0,
                        base_stake: float = 50.0, size: float = 1.0):
    """Informed flow from a volatility-regime-conditional estimate of P(up).

    `probability_up` is a per-second array (see `signals.regime_conditional_probability`).
    Same positive-expected-value rule as `predictive_bettor`; only the signal differs.
    """
    return _expected_value_bettor(probability_up, minimum_edge, base_stake, size)


def lead_lag_bettor(features, fast_log_price, gap_threshold: float = 0.0,
                    base_stake: float = 50.0, size: float = 1.0):
    """Informed flow from a faster reference feed that leads the settlement oracle.

    When the fast feed is above the oracle by more than `gap_threshold`, the oracle is
    likely to rise next, so bet up (and symmetrically for down). The faster feed is the
    private information; the oracle is what the contract settles on.
    """
    oracle_log_price = features.log_price
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * size * random_generator.exponential(1.0)
        gap = fast_log_price[t] - oracle_log_price[t]
        if gap > gap_threshold:
            return amount, 0.0
        if gap < -gap_threshold:
            return 0.0, amount
        return 0.0, 0.0
    return bettor
