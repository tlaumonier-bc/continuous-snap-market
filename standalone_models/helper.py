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
    # Extra features used only by the logistic estimate (the guarded model). The
    # volatility-regime momentum model uses only the fields above.
    absolute_standardized_momentum: np.ndarray = None
    momentum_2_seconds: np.ndarray = None
    momentum_5_seconds: np.ndarray = None
    momentum_15_seconds: np.ndarray = None
    momentum_30_seconds: np.ndarray = None
    momentum_60_seconds: np.ndarray = None
    position_in_range_60_seconds: np.ndarray = None
    position_in_range_120_seconds: np.ndarray = None
    path_efficiency_30_seconds: np.ndarray = None
    acceleration: np.ndarray = None

    @property
    def number_of_seconds(self) -> int:
        return len(self.price)


def _momentum_over(log_price, window_seconds, number_of_seconds):
    out = np.full(number_of_seconds, np.nan)
    out[window_seconds:] = log_price[window_seconds:] - log_price[:-window_seconds]
    return out


def _position_in_recent_range(price, window_seconds):
    series = pd.Series(price)
    highest = series.rolling(window_seconds, min_periods=window_seconds).max().values
    lowest = series.rolling(window_seconds, min_periods=window_seconds).min().values
    span = highest - lowest
    return np.where(span > 0, (price - lowest) / span, 0.5)


def build_features(prices: PriceSeries, parameters: MarketParameters) -> Features:
    """Per-second momentum and volatility, computed once and shared by everything.

    The first block (volatility and standardized momentum) is all the volatility-regime model
    needs; the rest are extra features the guarded model's logistic estimate uses.
    """
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

    momentum_2 = _momentum_over(log_price, 2, number_of_seconds)
    momentum_5 = _momentum_over(log_price, 5, number_of_seconds)
    momentum_15 = _momentum_over(log_price, 15, number_of_seconds)
    momentum_30 = _momentum_over(log_price, 30, number_of_seconds)
    momentum_60 = _momentum_over(log_price, 60, number_of_seconds)
    absolute_log_return = np.abs(np.diff(log_price, prepend=log_price[0]))
    gross_motion_30 = pd.Series(absolute_log_return).rolling(30, min_periods=30).sum().values
    path_efficiency_30 = np.where(gross_motion_30 > 0, np.abs(momentum_30) / gross_motion_30, np.nan)
    acceleration = momentum_2 - (momentum_5 - momentum_2)

    return Features(price=prices.price, log_price=log_price,
                    per_second_volatility=per_second_volatility,
                    annualized_volatility=annualized_volatility,
                    standardized_momentum=standardized_momentum,
                    absolute_standardized_momentum=np.abs(standardized_momentum),
                    momentum_2_seconds=momentum_2, momentum_5_seconds=momentum_5,
                    momentum_15_seconds=momentum_15, momentum_30_seconds=momentum_30,
                    momentum_60_seconds=momentum_60,
                    position_in_range_60_seconds=_position_in_recent_range(prices.price, 60),
                    position_in_range_120_seconds=_position_in_recent_range(prices.price, 120),
                    path_efficiency_30_seconds=path_efficiency_30,
                    acceleration=acceleration)


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


def expected_value_bettor(probability_up: np.ndarray, minimum_edge: float = 0.0,
                          base_stake: float = 50.0, size: float = 1.0):
    """Informed attacker: bets the positive expected-value side from a probability estimate.

    Feed it the regime-conditional signal for a 'regime-aware' attacker, or the logistic signal
    for a 'predictive' attacker; the betting rule is the same, only the signal differs.
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


def regime_aware_bettor(probability_up: np.ndarray, minimum_edge: float = 0.0,
                        base_stake: float = 50.0, size: float = 1.0):
    """Alias of `expected_value_bettor`, read with the regime-conditional signal in mind."""
    return expected_value_bettor(probability_up, minimum_edge, base_stake, size)


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


# --------------------------------------------------------------------------- #
#  Volatility-regime momentum display (reused by the guarded model)           #
# --------------------------------------------------------------------------- #
def _calibrate_momentum_curve(training_momentum, training_outcomes, momentum_bin_count,
                              minimum_samples_per_bin, monotone_passes, shrinkage):
    edges = np.quantile(training_momentum, np.linspace(0, 1, momentum_bin_count + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_index = np.digitize(training_momentum, edges) - 1
    probability = np.array([
        training_outcomes[bin_index == b].mean() if (bin_index == b).sum() > minimum_samples_per_bin else 0.5
        for b in range(momentum_bin_count)])
    probability = pool_adjacent_violators(probability, monotone_passes)
    return edges, 0.5 + shrinkage * (probability - 0.5)


def build_volatility_regime_display(features: Features, parameters: MarketParameters,
                                    volatility_regime_count: int = 3, momentum_bin_count: int = 25,
                                    calibration_shrinkage: float = 0.40, minimum_samples_per_bin: int = 50,
                                    monotone_passes: int = 200,
                                    calibration_window_seconds: int = 90 * 86_400,
                                    recompute_every_seconds: int = 7 * 86_400):
    """Rolling per-regime momentum display. Returns (display_probability, first_evaluation_second).

    This is the model built step by step in the volatility-regime momentum notebook; the guarded
    model reuses it as its displayed direction.
    """
    momentum = features.standardized_momentum
    volatility = features.annualized_volatility
    entries = contract_entries(features.number_of_seconds, parameters)
    outcomes = contract_outcomes(features, entries, parameters)
    window_contracts = calibration_window_seconds // parameters.horizon_seconds
    recompute_contracts = recompute_every_seconds // parameters.horizon_seconds

    display_probability = np.full(features.number_of_seconds, 0.5)
    first_evaluation_second = features.number_of_seconds

    for start in range(window_contracts, len(entries), recompute_contracts):
        training_slice = slice(start - window_contracts, start)
        training_entries = entries[training_slice]
        training_regime = quantile_bins(volatility[training_entries], volatility[training_entries],
                                        volatility_regime_count)
        curves = []
        for r in range(volatility_regime_count):
            mask = training_regime == r
            curves.append(_calibrate_momentum_curve(
                momentum[training_entries][mask], outcomes[training_slice][mask], momentum_bin_count,
                minimum_samples_per_bin, monotone_passes, calibration_shrinkage)
                if mask.sum() > minimum_samples_per_bin else None)

        segment_start = int(entries[start])
        next_start = min(len(entries), start + recompute_contracts)
        segment_stop = int(entries[next_start]) if next_start < len(entries) else features.number_of_seconds
        segment_regime = quantile_bins(volatility[segment_start:segment_stop],
                                       volatility[training_entries], volatility_regime_count)
        segment_momentum = momentum[segment_start:segment_stop]
        segment_probability = np.full(segment_stop - segment_start, 0.5)
        for r, curve in enumerate(curves):
            if curve is None:
                continue
            in_regime = segment_regime == r
            if in_regime.any():
                edges, probability_per_bin = curve
                index = np.clip(np.digitize(segment_momentum[in_regime], edges) - 1,
                                0, len(probability_per_bin) - 1)
                segment_probability[in_regime] = probability_per_bin[index]
        display_probability[segment_start:segment_stop] = segment_probability
        first_evaluation_second = min(first_evaluation_second, segment_start)

    return display_probability, first_evaluation_second


# --------------------------------------------------------------------------- #
#  Hidden logistic estimate (the guard's information signal)                   #
# --------------------------------------------------------------------------- #
def logistic_feature_matrix(features: Features, index=slice(None)) -> np.ndarray:
    """Feature matrix for the hidden logistic estimate (scaled for a stable optimiser)."""
    return np.column_stack([
        features.standardized_momentum[index],
        features.absolute_standardized_momentum[index],
        features.momentum_2_seconds[index] * 1e4,
        features.momentum_5_seconds[index] * 1e4,
        features.momentum_15_seconds[index] * 1e4,
        features.momentum_30_seconds[index] * 1e4,
        features.momentum_60_seconds[index] * 1e4,
        features.annualized_volatility[index],
        features.position_in_range_60_seconds[index],
        features.position_in_range_120_seconds[index],
        features.path_efficiency_30_seconds[index],
        features.acceleration[index] * 1e4,
    ]).astype(float)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _fit_logistic(training_features, training_outcomes, iterations, learning_rate,
                  ridge_penalty, minimum_samples):
    finite = np.isfinite(training_features).all(axis=1) & np.isfinite(training_outcomes)
    x = training_features[finite]
    y = training_outcomes[finite].astype(float)
    if len(y) < minimum_samples or y.min() == y.max():
        return None
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    weights = np.zeros(x.shape[1] + 1)
    for _ in range(iterations):
        error = _sigmoid(weights[0] + x @ weights[1:]) - y
        gradient = np.empty_like(weights)
        gradient[0] = error.mean()
        gradient[1:] = (x.T @ error) / len(y) + ridge_penalty * weights[1:]
        weights -= learning_rate * gradient
    return weights, mean, scale


def _predict_logistic(feature_matrix, fitted, probability_clip):
    if fitted is None:
        return np.full(len(feature_matrix), 0.5)
    weights, mean, scale = fitted
    x = np.nan_to_num((feature_matrix - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    probability = _sigmoid(weights[0] + x @ weights[1:])
    return np.clip(probability, probability_clip, 1.0 - probability_clip)


def build_walk_forward_logistic_probability(features: Features, parameters: MarketParameters,
                                            minimum_training_contracts: int = 5_000,
                                            training_window_contracts: int = 20_000,
                                            retrain_contracts: int = 5_000, iterations: int = 40,
                                            learning_rate: float = 0.08, ridge_penalty: float = 0.02,
                                            probability_clip: float = 0.02, minimum_samples: int = 50,
                                            chunk_size: int = 500_000) -> np.ndarray:
    """Walk-forward logistic estimate of P(up) on many features (no look-ahead)."""
    entries = contract_entries(features.number_of_seconds, parameters)
    outcomes = contract_outcomes(features, entries, parameters)
    probability_up = np.full(features.number_of_seconds, 0.5)
    minimum = min(minimum_training_contracts, len(entries))
    if minimum < minimum_samples:
        return probability_up

    window = max(minimum, training_window_contracts)
    for start in range(minimum, len(entries), max(1, retrain_contracts)):
        train_start = max(0, start - window)
        train_entries = entries[train_start:start]
        fitted = _fit_logistic(logistic_feature_matrix(features, train_entries),
                               outcomes[train_start:start], iterations, learning_rate,
                               ridge_penalty, minimum_samples)
        segment_start = int(entries[start])
        next_start = min(len(entries), start + max(1, retrain_contracts))
        segment_stop = int(entries[next_start]) if next_start < len(entries) else features.number_of_seconds
        for chunk_start in range(segment_start, segment_stop, chunk_size):
            chunk_stop = min(segment_stop, chunk_start + chunk_size)
            probability_up[chunk_start:chunk_stop] = _predict_logistic(
                logistic_feature_matrix(features, slice(chunk_start, chunk_stop)), fitted, probability_clip)
    return probability_up


def information_margin_over_display(hidden_probability, display_probability,
                                   parameters: MarketParameters,
                                   information_margin_buffer: float = 0.02) -> np.ndarray:
    """Extra symmetric margin so a bettor who knows `hidden_probability` is non-positive.

    On a balanced book the up odds are (1 - margin) / display and the down odds are
    (1 - margin) / (1 - display). A bettor with true probability p breaks even on the up side at
    margin = 1 - display / p, and on the down side at margin = 1 - (1 - display) / (1 - p). We
    charge the larger so neither side is positive, add a buffer, and subtract the vig already taken.
    """
    p = np.clip(np.asarray(hidden_probability, dtype=float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    q = np.clip(np.asarray(display_probability, dtype=float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    up_side = 1.0 - q / p
    down_side = 1.0 - (1.0 - q) / (1.0 - p)
    required = np.maximum(np.maximum(up_side, down_side), 0.0)
    add_on = required + information_margin_buffer - parameters.house_margin
    return np.maximum(add_on, 0.0)
