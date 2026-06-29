"""Shared parameters used by every model.

These are the settings that features, pricing, and the engine need regardless of
which model is selected. Model-specific settings live in each model's own parameter
dataclass under `snapmarket/models/<name>/parameters.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedParameters:
    # --- contract ---
    horizon_seconds: int = 30                       # contract life; fixed for every bet

    # --- features ---
    momentum_lookback_seconds: int = 5              # trailing window that defines the momentum state
    volatility_exponentially_weighted_moving_average_span: int = 600

    # --- pricing ---
    house_margin: float = 0.125                     # the vig
    inventory_skew_sensitivity: float = 3.0         # how hard a book imbalance skews the odds
    maximum_odds: float = 5.0                       # cap on decimal odds offered on either side
    maximum_total_margin: float = 0.49              # keep quoted odds at or above roughly 1.0x

    # --- risk ---
    maximum_net_delta: float = 2.0e4                # hard cap on |net delta| (USDT)

    @property
    def warmup_seconds(self) -> int:
        return (
            self.volatility_exponentially_weighted_moving_average_span
            + self.momentum_lookback_seconds
        )
