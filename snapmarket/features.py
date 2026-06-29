"""Per-second features, computed once from a price series and shared by every model
and strategy. Computing them in one place guarantees a model and an attacker that
both look at "momentum" see exactly the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PriceSeries

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 31_557_600


@dataclass(frozen=True)
class Features:
    """All per-second arrays derived from the price feed. Indexed by second."""
    prices: PriceSeries
    per_second_volatility: np.ndarray        # sqrt of EWMA squared log-return (the z denominator)
    annualized_volatility: np.ndarray        # per_second_volatility * sqrt(seconds per year)
    standardized_momentum: np.ndarray        # trailing L-second move / (vol * sqrt(L))
    absolute_standardized_momentum: np.ndarray
    momentum_2_seconds: np.ndarray
    momentum_5_seconds: np.ndarray
    momentum_15_seconds: np.ndarray
    momentum_30_seconds: np.ndarray
    momentum_60_seconds: np.ndarray
    position_in_range_60_seconds: np.ndarray
    position_in_range_120_seconds: np.ndarray
    path_efficiency_30_seconds: np.ndarray
    acceleration: np.ndarray

    @property
    def price(self) -> np.ndarray:
        return self.prices.price

    @property
    def log_price(self) -> np.ndarray:
        return self.prices.log_price

    @property
    def number_of_seconds(self) -> int:
        return self.prices.number_of_seconds


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


def build_features(prices: PriceSeries, parameters) -> Features:
    """Build every per-second feature from a PriceSeries given the model parameters."""
    log_price = prices.log_price
    number_of_seconds = prices.number_of_seconds
    lookback = parameters.momentum_lookback_seconds

    # per-second realised volatility (EWMA of squared log-returns)
    squared_log_return = np.diff(log_price, prepend=log_price[0]) ** 2
    ewma_variance = pd.Series(squared_log_return).ewm(span=parameters.volatility_ewma_span, adjust=False).mean().values
    per_second_volatility = np.sqrt(ewma_variance) + 1e-12
    annualized_volatility = per_second_volatility * np.sqrt(SECONDS_PER_YEAR)

    # volatility-normalised momentum state
    standardized_momentum = np.zeros(number_of_seconds)
    standardized_momentum[lookback:] = (
        (log_price[lookback:] - log_price[:-lookback]) / (per_second_volatility[lookback:] * np.sqrt(lookback))
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

    return Features(
        prices=prices,
        per_second_volatility=per_second_volatility,
        annualized_volatility=annualized_volatility,
        standardized_momentum=standardized_momentum,
        absolute_standardized_momentum=np.abs(standardized_momentum),
        momentum_2_seconds=momentum_2,
        momentum_5_seconds=momentum_5,
        momentum_15_seconds=momentum_15,
        momentum_30_seconds=momentum_30,
        momentum_60_seconds=momentum_60,
        position_in_range_60_seconds=_position_in_recent_range(prices.price, 60),
        position_in_range_120_seconds=_position_in_recent_range(prices.price, 120),
        path_efficiency_30_seconds=path_efficiency_30,
        acceleration=acceleration,
    )


def contract_entries(number_of_seconds: int, parameters) -> np.ndarray:
    """Non-overlapping contract entry indices, starting after the warmup."""
    return np.arange(parameters.warmup_seconds, number_of_seconds - parameters.horizon_seconds,
                     parameters.horizon_seconds)


def quantile_bins(values, reference_values, number_of_bins) -> np.ndarray:
    """Assign each value to a quantile bin whose edges are fixed on reference_values."""
    finite_reference = reference_values[np.isfinite(reference_values)]
    edges = np.quantile(finite_reference, np.linspace(0, 1, number_of_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(values, edges) - 1
