"""Data loading for the Snap Market models.

One job: turn a raw price file into a clean 1-second grid that every notebook and
model shares. Import this instead of re-writing the loader in each notebook.

    from snapmarket.data import load_oracle_prices
    prices = load_oracle_prices()          # looks in ./data, ., and the v1 folder
    price, log_price = prices.price, prices.log_price
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEARCH_PATHS = (Path("data"), Path("."), Path("../1 - Up-Down Model (v1)"))


@dataclass(frozen=True)
class PriceSeries:
    """A price series on a contiguous 1-second grid."""
    price: np.ndarray
    log_price: np.ndarray
    first_timestamp: int
    last_timestamp: int

    @property
    def number_of_seconds(self) -> int:
        return len(self.price)

    def __repr__(self) -> str:
        days = (self.last_timestamp - self.first_timestamp) / 86400
        return (f"PriceSeries({self.number_of_seconds:,} seconds, {days:.0f} days, "
                f"price {self.price.min():,.0f} -> {self.price.max():,.0f})")


def _resolve(file_name: str, search_paths) -> str:
    for directory in search_paths:
        candidate = Path(directory) / file_name
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(Path(d) / file_name) for d in search_paths)
    raise FileNotFoundError(f"{file_name} not found. Looked in: {searched}")


def load_oracle_prices(file_name: str = "btc_pyth_prices.parquet",
                       search_paths=DEFAULT_SEARCH_PATHS) -> PriceSeries:
    """Load a (timestamp, price) parquet and interpolate onto a 1-second grid."""
    frame = pd.read_parquet(_resolve(file_name, search_paths))[["timestamp", "price"]].dropna()
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    first_timestamp, last_timestamp = int(frame.timestamp.iloc[0]), int(frame.timestamp.iloc[-1])

    grid = np.arange(first_timestamp, last_timestamp + 1)
    price = np.interp(grid, frame.timestamp.values, frame.price.values)
    return PriceSeries(price=price, log_price=np.log(price),
                       first_timestamp=first_timestamp, last_timestamp=last_timestamp)


def load_fast_feed(file_name: str = "binance_btcusdt_1s_aligned.parquet",
                   column: str = "binance_close",
                   expected_length: int | None = None,
                   search_paths=DEFAULT_SEARCH_PATHS) -> PriceSeries:
    """Load the faster reference feed, already aligned 1:1 to the oracle grid (for a future v2)."""
    series = pd.read_parquet(_resolve(file_name, search_paths))[column].values
    if expected_length is not None and len(series) != expected_length:
        raise ValueError(f"fast feed length {len(series)} != expected {expected_length}")
    price = np.asarray(series, dtype=float)
    return PriceSeries(price=price, log_price=np.log(price), first_timestamp=0, last_timestamp=len(price) - 1)
