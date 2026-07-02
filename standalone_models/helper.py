"""Self-contained helpers for the Snap Market model notebooks.

This single module holds everything the notebooks need except the model itself: data
loading, per-second features, the pricing math, the book simulation engine, the bettor
strategies, and a few numeric utilities. The model is built inside each notebook, step by
step, so the reader sees exactly how it works; the plumbing lives here to keep the
notebooks short.

Nothing here depends on any project package: the notebooks plus this file plus the price
parquet are all that is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600
_PROBABILITY_EPSILON = 1e-6
_DEFAULT_DATA_SEARCH_PATHS = ("data", "../data", ".", "..")


# --------------------------------------------------------------------------- #
#  Shared market parameters                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketParameters:
    """Settings shared by features, pricing, and the engine (not model-specific)."""
    horizon_seconds: int = 30                       # contract life, fixed for every bet
    momentum_lookback_seconds: int = 5              # trailing window defining the momentum state
    volatility_ewma_span: int = 600                 # span of the exponentially weighted volatility
    house_margin: float = 0.125                     # the vig
    inventory_skew_sensitivity: float = 3.0         # how hard a book imbalance skews the odds
    maximum_odds: float = 5.0                        # cap on decimal odds offered on either side
    maximum_total_margin: float = 0.49              # keeps quoted odds at or above roughly 1.0x
    maximum_net_delta: float = 2.0e4                # hard cap on |net delta| (USDT)

    @property
    def warmup_seconds(self) -> int:
        return self.volatility_ewma_span + self.momentum_lookback_seconds


# --------------------------------------------------------------------------- #
#  Data                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriceSeries:
    price: np.ndarray
    log_price: np.ndarray

    @property
    def number_of_seconds(self) -> int:
        return len(self.price)


def _resolve(file_name: str, search_paths) -> str:
    for directory in search_paths:
        candidate = Path(directory) / file_name
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(Path(directory) / file_name) for directory in search_paths)
    raise FileNotFoundError(f"{file_name} not found. Looked in: {searched}")


def load_prices(file_name: str = "btc_pyth_prices.parquet",
                search_paths=_DEFAULT_DATA_SEARCH_PATHS) -> PriceSeries:
    """Load a (timestamp, price) parquet and interpolate onto a contiguous 1-second grid."""
    frame = pd.read_parquet(_resolve(file_name, search_paths))[["timestamp", "price"]].dropna()
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    first, last = int(frame.timestamp.iloc[0]), int(frame.timestamp.iloc[-1])
    grid = np.arange(first, last + 1)
    price = np.interp(grid, frame.timestamp.values, frame.price.values)
    return PriceSeries(price=price, log_price=np.log(price))


def load_fast_feed(file_name: str = "binance_btcusdt_1s_aligned.parquet",
                   column: str = "binance_close", expected_length: int | None = None,
                   search_paths=_DEFAULT_DATA_SEARCH_PATHS) -> PriceSeries:
    """Load the faster reference feed, already aligned 1:1 to the oracle grid."""
    series = pd.read_parquet(_resolve(file_name, search_paths))[column].values
    if expected_length is not None and len(series) != expected_length:
        raise ValueError(f"fast feed length {len(series)} != expected {expected_length}")
    price = np.asarray(series, dtype=float)
    return PriceSeries(price=price, log_price=np.log(price))


# --------------------------------------------------------------------------- #
#  Features                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Features:
    price: np.ndarray
    log_price: np.ndarray
    per_second_volatility: np.ndarray
    annualized_volatility: np.ndarray
    standardized_momentum: np.ndarray

    @property
    def number_of_seconds(self) -> int:
        return len(self.price)


def build_features(prices: PriceSeries, parameters: MarketParameters) -> Features:
    """Per-second momentum and volatility, computed once and shared by everything."""
    log_price = prices.log_price
    number_of_seconds = prices.number_of_seconds
    lookback = parameters.momentum_lookback_seconds

    squared_log_return = np.diff(log_price, prepend=log_price[0]) ** 2
    ewma_variance = pd.Series(squared_log_return).ewm(
        span=parameters.volatility_ewma_span, adjust=False).mean().values
    per_second_volatility = np.sqrt(ewma_variance) + 1e-12
    annualized_volatility = per_second_volatility * np.sqrt(SECONDS_PER_YEAR)

    standardized_momentum = np.zeros(number_of_seconds)
    standardized_momentum[lookback:] = (
        (log_price[lookback:] - log_price[:-lookback])
        / (per_second_volatility[lookback:] * np.sqrt(lookback))
    )
    return Features(price=prices.price, log_price=log_price,
                    per_second_volatility=per_second_volatility,
                    annualized_volatility=annualized_volatility,
                    standardized_momentum=standardized_momentum)


# --------------------------------------------------------------------------- #
#  Numeric utilities                                                           #
# --------------------------------------------------------------------------- #
def contract_entries(number_of_seconds: int, parameters: MarketParameters) -> np.ndarray:
    """Non-overlapping contract entry indices, starting after the warmup."""
    return np.arange(parameters.warmup_seconds,
                     number_of_seconds - parameters.horizon_seconds,
                     parameters.horizon_seconds)


def quantile_bins(values: np.ndarray, reference_values: np.ndarray, number_of_bins: int) -> np.ndarray:
    """Assign each value to a quantile bin whose edges are fixed on reference_values."""
    finite_reference = reference_values[np.isfinite(reference_values)]
    edges = np.quantile(finite_reference, np.linspace(0, 1, number_of_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(values, edges) - 1


def pool_adjacent_violators(probabilities: np.ndarray, passes: int) -> np.ndarray:
    """Enforce a non-decreasing sequence by pooling adjacent out-of-order pairs."""
    pooled = probabilities.copy()
    count = len(pooled)
    for _ in range(passes):
        for i in range(count - 1):
            if pooled[i] > pooled[i + 1]:
                pooled[i] = pooled[i + 1] = (pooled[i] + pooled[i + 1]) / 2
    return pooled


def contract_outcomes(features: Features, entries: np.ndarray, parameters: MarketParameters) -> np.ndarray:
    """1.0 where the price is higher one horizon after each entry, else 0.0."""
    price = features.price
    return (price[entries + parameters.horizon_seconds] > price[entries]).astype(float)


# --------------------------------------------------------------------------- #
#  Pricing                                                                     #
# --------------------------------------------------------------------------- #
def quote_odds(display_probability, margin, net_delta_imbalance, parameters: MarketParameters):
    """Decimal (odds_up, odds_down): skewed for inventory and clamped at each side's fair odds."""
    display_probability = float(np.clip(display_probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON))
    margin = float(np.clip(margin, 0.0, parameters.maximum_total_margin))
    fair_log_odds = math.log(display_probability / (1.0 - display_probability))
    skewed_up_probability = 1.0 / (1.0 + math.exp(
        -(fair_log_odds + parameters.inventory_skew_sensitivity * net_delta_imbalance)))
    odds_up = min((1.0 - margin) / skewed_up_probability, 1.0 / display_probability, parameters.maximum_odds)
    odds_down = min((1.0 - margin) / (1.0 - skewed_up_probability), 1.0 / (1.0 - display_probability),
                    parameters.maximum_odds)
    return odds_up, odds_down


def flat_book_odds(display_probability: np.ndarray, margin, parameters: MarketParameters):
    """Vectorised odds on a balanced book (imbalance = 0), used for static inspection."""
    margin = np.clip(margin, 0.0, parameters.maximum_total_margin)
    odds_up = np.minimum((1.0 - margin) / display_probability, parameters.maximum_odds)
    odds_down = np.minimum((1.0 - margin) / (1.0 - display_probability), parameters.maximum_odds)
    return odds_up, odds_down


# --------------------------------------------------------------------------- #
#  Simulation engine                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class BettorResult:
    pnl: float = 0.0
    stake: float = 0.0

    @property
    def edge(self) -> float:
        return self.pnl / self.stake if self.stake else 0.0


@dataclass
class SimulationResult:
    house_pnl: float
    total_volume: float
    house_edge: float
    max_absolute_net_delta: float
    refused_seconds: int
    per_bettor: dict
    net_delta_series: np.ndarray = field(repr=False)
    pnl_series: np.ndarray = field(repr=False)


def _house_profit(up, down, odds_up, odds_down, price_now, price_then) -> float:
    if price_now > price_then:
        return down - up * (odds_up - 1)
    if price_now < price_then:
        return up - down * (odds_down - 1)
    return 0.0  # a push refunds the stake


def _margin_at(margin, t: int) -> float:
    return float(margin) if np.isscalar(margin) else float(margin[t])


def simulate(display_probability, margin, features: Features, bettors: dict,
             start_index: int, number_of_steps: int, parameters: MarketParameters,
             seed: int = 42) -> SimulationResult:
    """Run the live book for `number_of_steps` seconds from `start_index`.

    A bettor is `bettor(t, random_generator, odds_up, odds_down) -> (up_stake, down_stake)`.
    All bettors are filled at the same odds each second. Returns total house PnL/edge and each
    bettor's isolated PnL/edge.
    """
    horizon = parameters.horizon_seconds
    price = features.price
    random_generator = np.random.default_rng(seed)
    names = list(bettors)

    total_up = np.zeros(number_of_steps)
    total_down = np.zeros(number_of_steps)
    odds_up_recorded = np.zeros(number_of_steps)
    odds_down_recorded = np.zeros(number_of_steps)
    bettor_up = {name: np.zeros(number_of_steps) for name in names}
    bettor_down = {name: np.zeros(number_of_steps) for name in names}

    open_up = open_down = house_pnl = 0.0
    net_delta_series = np.empty(number_of_steps)
    pnl_series = np.empty(number_of_steps)
    refused_seconds = 0
    results = {name: BettorResult() for name in names}

    for step in range(number_of_steps):
        now = start_index + step

        if step - horizon >= 0:
            settle_step = step - horizon
            settle_time = start_index + settle_step
            odds_up = odds_up_recorded[settle_step]
            odds_down = odds_down_recorded[settle_step]
            house_pnl += _house_profit(total_up[settle_step], total_down[settle_step],
                                       odds_up, odds_down, price[now], price[settle_time])
            for name in names:
                up, down = bettor_up[name][settle_step], bettor_down[name][settle_step]
                if up + down > 1e-12:
                    results[name].pnl += -_house_profit(up, down, odds_up, odds_down,
                                                        price[now], price[settle_time])
                    results[name].stake += up + down
            open_up -= total_up[settle_step]
            open_down -= total_down[settle_step]

        net_delta_imbalance = (open_up - open_down) / (open_up + open_down + parameters.maximum_net_delta)
        odds_up, odds_down = quote_odds(display_probability[now], _margin_at(margin, now),
                                        net_delta_imbalance, parameters)

        this_up = {name: 0.0 for name in names}
        this_down = {name: 0.0 for name in names}
        for name in names:
            this_up[name], this_down[name] = bettors[name](now, random_generator, odds_up, odds_down)

        current_net_delta = open_up - open_down
        refused = False
        if current_net_delta >= parameters.maximum_net_delta and sum(this_up.values()) > 0:
            this_up = {name: 0.0 for name in names}; refused = True
        if current_net_delta <= -parameters.maximum_net_delta and sum(this_down.values()) > 0:
            this_down = {name: 0.0 for name in names}; refused = True
        if refused:
            refused_seconds += 1

        odds_up_recorded[step] = odds_up
        odds_down_recorded[step] = odds_down
        for name in names:
            bettor_up[name][step] = this_up[name]
            bettor_down[name][step] = this_down[name]
        total_up[step] = sum(this_up.values())
        total_down[step] = sum(this_down.values())
        open_up += total_up[step]
        open_down += total_down[step]
        net_delta_series[step] = open_up - open_down
        pnl_series[step] = house_pnl

    total_volume = total_up.sum() + total_down.sum()
    return SimulationResult(
        house_pnl=house_pnl, total_volume=total_volume,
        house_edge=house_pnl / total_volume if total_volume else 0.0,
        max_absolute_net_delta=float(np.abs(net_delta_series).max()),
        refused_seconds=refused_seconds, per_bettor=results,
        net_delta_series=net_delta_series, pnl_series=pnl_series)


# --------------------------------------------------------------------------- #
#  Bettor strategies                                                          #
# --------------------------------------------------------------------------- #
def noise_pool(base_stake: float = 50.0, spread: float = 0.10):
    """Uninformed background flow: roughly balanced, small random up/down tilt each second."""
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * random_generator.exponential(1.0)
        up_fraction = float(np.clip(0.5 + random_generator.normal(0, spread), 0, 1))
        return amount * up_fraction, amount * (1 - up_fraction)
    return bettor


def momentum_follower(features: Features, lookback: int = 5, base_stake: float = 50.0, size: float = 1.0):
    """Public trend-following flow: bet the direction of the trailing move."""
    price = features.price
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * size * random_generator.exponential(1.0)
        if price[t] > price[t - lookback]:
            return amount, 0.0
        if price[t] < price[t - lookback]:
            return 0.0, amount
        return 0.0, 0.0
    return bettor


def mean_reversion_fader(features: Features, lookback: int = 5, base_stake: float = 50.0, size: float = 1.0):
    """Public mean-reversion flow: bet against the trailing move."""
    price = features.price
    def bettor(t, random_generator, odds_up, odds_down):
        amount = base_stake * size * random_generator.exponential(1.0)
        if price[t] > price[t - lookback]:
            return 0.0, amount
        if price[t] < price[t - lookback]:
            return amount, 0.0
        return 0.0, 0.0
    return bettor


def regime_aware_bettor(probability_up: np.ndarray, minimum_edge: float = 0.0,
                        base_stake: float = 50.0, size: float = 1.0):
    """Informed attacker: bets the positive expected-value side from a probability estimate."""
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


def lead_lag_bettor(features: Features, fast_log_price: np.ndarray, gap_threshold: float = 0.0,
                    base_stake: float = 50.0, size: float = 1.0):
    """Latency attacker: follow a faster feed that leads the settlement oracle."""
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


# --------------------------------------------------------------------------- #
#  Attacker signal: regime-conditional probability                            #
# --------------------------------------------------------------------------- #
def regime_conditional_probability(features: Features, parameters: MarketParameters,
                                   training_fraction: float = 0.40, momentum_bin_count: int = 12,
                                   volatility_bin_count: int = 3,
                                   minimum_samples_per_cell: int = 200) -> np.ndarray:
    """P(up) calibrated per (momentum bin, volatility regime) on the training split.

    This is the signal an informed 'regime-aware' attacker uses: the raw empirical frequency in
    each cell (no shrinkage), which is exactly what our model shrinks away from. Bin edges are
    frozen on the training contracts (no look-ahead).
    """
    momentum = features.standardized_momentum
    volatility = features.annualized_volatility
    entries = contract_entries(features.number_of_seconds, parameters)
    outcomes = contract_outcomes(features, entries, parameters)

    split_index = int(training_fraction * len(entries))
    training_momentum = momentum[entries[:split_index]]
    training_volatility = volatility[entries[:split_index]]
    training_momentum_bin = quantile_bins(training_momentum, training_momentum, momentum_bin_count)
    training_volatility_bin = quantile_bins(training_volatility, training_volatility, volatility_bin_count)

    cell_probability = {}
    for m in range(momentum_bin_count):
        for v in range(volatility_bin_count):
            cell = (training_momentum_bin == m) & (training_volatility_bin == v)
            cell_probability[(m, v)] = (float(outcomes[:split_index][cell].mean())
                                        if cell.sum() > minimum_samples_per_cell else 0.5)

    momentum_bin = quantile_bins(momentum, training_momentum, momentum_bin_count)
    volatility_bin = quantile_bins(volatility, training_volatility, volatility_bin_count)
    probability_up = np.full(features.number_of_seconds, 0.5)
    for (m, v), probability in cell_probability.items():
        probability_up[(momentum_bin == m) & (volatility_bin == v)] = probability
    return probability_up
