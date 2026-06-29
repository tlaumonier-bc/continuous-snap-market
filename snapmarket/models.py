"""Pricing math and model objects.

A fitted model is just two per-second series plus the shared pricing math:
  - display_probability : the probability the displayed odds are built around
  - margin              : the vig (scalar, or a per-second array for models that widen it)

Every model prices through the same `quote_odds`, runs through the same engine, and is
attacked by the same strategies. A new model = a new way to fill those two series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import Features, contract_entries


@dataclass(frozen=True)
class ModelParameters:
    # --- contract ---
    horizon_seconds: int = 30               # contract life; fixed for every bet

    # --- fair price (calibration) ---
    momentum_lookback_seconds: int = 5      # trailing window that defines the momentum state
    volatility_ewma_span: int = 600         # EWMA span for per-second realised volatility
    calibration_bin_count: int = 25         # number of quantile bins in the p_up lookup
    calibration_shrinkage: float = 0.40     # lambda: keep this fraction of each bin's edge over 0.5
    training_fraction: float = 0.40         # share of history used to calibrate the fair price
    minimum_samples_per_bin: int = 50       # bins below this fall back to 0.5
    monotonic_pooling_passes: int = 200     # pool-adjacent-violators sweeps enforcing monotonicity

    # --- pricing ---
    house_margin: float = 0.125             # the vig
    inventory_skew_sensitivity: float = 3.0 # how hard a book imbalance skews the odds
    maximum_odds: float = 5.0               # cap on decimal odds offered on either side

    # --- risk ---
    maximum_net_delta: float = 2.0e4        # hard cap on |net delta| (USDT)

    @property
    def warmup_seconds(self) -> int:
        return self.volatility_ewma_span + self.momentum_lookback_seconds


def quote_odds(display_probability, margin, net_delta_imbalance, parameters: ModelParameters):
    """Decimal (odds_up, odds_down): skewed for inventory and clamped at each side's fair odds.

    Works for any model: pass the model's display probability and margin for the second.
    """
    fair_log_odds = math.log(display_probability / (1.0 - display_probability))
    skewed_up_probability = 1.0 / (1.0 + math.exp(
        -(fair_log_odds + parameters.inventory_skew_sensitivity * net_delta_imbalance)
    ))
    odds_up = min((1.0 - margin) / skewed_up_probability,
                  1.0 / display_probability,
                  parameters.maximum_odds)
    odds_down = min((1.0 - margin) / (1.0 - skewed_up_probability),
                    1.0 / (1.0 - display_probability),
                    parameters.maximum_odds)
    return odds_up, odds_down


@dataclass
class Model:
    name: str
    display_probability: np.ndarray         # per second
    margin: object                          # float, or per-second np.ndarray
    parameters: ModelParameters

    def margin_at(self, t) -> float:
        return float(self.margin) if np.isscalar(self.margin) else float(self.margin[t])

    def quote(self, t: int, net_delta_imbalance: float):
        return quote_odds(self.display_probability[t], self.margin_at(t), net_delta_imbalance, self.parameters)

    def flat_book_odds(self, index):
        """Vectorised odds on a balanced book (imbalance = 0), used for static evaluation."""
        probability = self.display_probability[index]
        margin = self.margin if np.isscalar(self.margin) else self.margin[index]
        odds_up = np.minimum((1.0 - margin) / probability, self.parameters.maximum_odds)
        odds_down = np.minimum((1.0 - margin) / (1.0 - probability), self.parameters.maximum_odds)
        return odds_up, odds_down


def calibrate_fair_probability(training_momentum, training_outcomes, parameters: ModelParameters):
    """Return (bin_edges, fair_probability_per_bin): a monotone, shrunk lookup of p_up vs the state."""
    bin_count = parameters.calibration_bin_count
    bin_edges = np.quantile(training_momentum, np.linspace(0, 1, bin_count + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_index = np.digitize(training_momentum, bin_edges) - 1

    fair_probability_per_bin = np.array([
        training_outcomes[bin_index == i].mean()
        if (bin_index == i).sum() > parameters.minimum_samples_per_bin else 0.5
        for i in range(bin_count)
    ])
    for _ in range(parameters.monotonic_pooling_passes):
        for i in range(bin_count - 1):
            if fair_probability_per_bin[i] > fair_probability_per_bin[i + 1]:
                pooled = (fair_probability_per_bin[i] + fair_probability_per_bin[i + 1]) / 2
                fair_probability_per_bin[i] = fair_probability_per_bin[i + 1] = pooled

    shrunk_probability = 0.5 + parameters.calibration_shrinkage * (fair_probability_per_bin - 0.5)
    return bin_edges, shrunk_probability


def build_momentum_lookup_model(features: Features, parameters: ModelParameters,
                                name: str = "momentum_lookup_v1") -> Model:
    """The current production model (v1): calibrate p_up on the oracle's own momentum,
    fixed train/test split, constant margin."""
    price = features.price
    horizon = parameters.horizon_seconds
    entries = contract_entries(features.number_of_seconds, parameters)
    split_index = int(parameters.training_fraction * len(entries))
    training_entries = entries[:split_index]
    training_outcomes = (price[training_entries + horizon] > price[training_entries]).astype(float)

    bin_edges, probability_per_bin = calibrate_fair_probability(
        features.standardized_momentum[training_entries], training_outcomes, parameters
    )
    index = np.clip(np.digitize(features.standardized_momentum, bin_edges) - 1, 0, len(probability_per_bin) - 1)
    display_probability = probability_per_bin[index]

    return Model(name=name, display_probability=display_probability,
                 margin=parameters.house_margin, parameters=parameters)


def run_golden_tests(parameters: ModelParameters = ModelParameters()) -> None:
    """Invariants the pricing must satisfy, independent of any data."""
    margin = parameters.house_margin

    def approximately_equal(a, b, tolerance=1e-6):
        return abs(a - b) < tolerance

    # 1. balanced book, no momentum => symmetric base odds, overround 1/(1-margin)
    odds = quote_odds(0.50, margin, 0.0, parameters)
    assert approximately_equal(odds[0], 2 * (1 - margin)) and approximately_equal(odds[1], 2 * (1 - margin))
    assert approximately_equal(1 / odds[0] + 1 / odds[1], 1 / (1 - margin))

    # 2. EV frontier: no side is ever quoted above its fair odds (house EV >= 0)
    for probability in [0.40, 0.50, 0.58, 0.62]:
        for imbalance in [-0.9, -0.3, 0.0, 0.3, 0.9]:
            odds = quote_odds(probability, margin, imbalance, parameters)
            assert odds[0] <= 1 / probability + 1e-9 and odds[1] <= 1 / (1 - probability) + 1e-9

    # 3. skew direction: long UP (imbalance > 0) shortens UP, lengthens DOWN
    odds = quote_odds(0.50, margin, 0.3, parameters)
    assert odds[0] < 2 * (1 - margin) < odds[1]

    # 4. momentum direction: uptrend (p_up > 0.5) shortens UP vs the symmetric quote
    assert quote_odds(0.58, margin, 0.0, parameters)[0] < quote_odds(0.50, margin, 0.0, parameters)[0]

    # 5. per-bet house edge at the fair probability equals the vig, both sides
    for probability in [0.45, 0.50, 0.58]:
        odds = quote_odds(probability, margin, 0.0, parameters)
        assert approximately_equal(1 - probability * odds[0], margin, 1e-3)
        assert approximately_equal(1 - (1 - probability) * odds[1], margin, 1e-3)

    print("all golden tests pass")
