"""Shared synthetic data for the test suite."""
from __future__ import annotations

import numpy as np

from snapmarket.data import PriceSeries
from snapmarket.features import build_features
from snapmarket.models.hidden_symmetric_margin import HiddenSymmetricMarginParameters
from snapmarket.models.momentum_lookup_rolling import MomentumLookupRollingParameters
from snapmarket.parameters import SharedParameters


def shared_parameters() -> SharedParameters:
    return SharedParameters(volatility_exponentially_weighted_moving_average_span=60)


def fast_hidden_margin_parameters() -> HiddenSymmetricMarginParameters:
    return HiddenSymmetricMarginParameters(
        internal_minimum_training_contracts=50,
        internal_training_window_contracts=100,
        internal_retrain_contracts=50,
        internal_logistic_iterations=10,
        minimum_samples_per_bin=10,
    )


def fast_rolling_parameters() -> MomentumLookupRollingParameters:
    # Small window and cadence so the model rolls within the synthetic horizon.
    return MomentumLookupRollingParameters(
        calibration_window_seconds=900,     # 30 contracts at a 30-second horizon
        recompute_every_seconds=300,        # 10 contracts
        calibration_bin_count=4,
        minimum_samples_per_bin=2,
    )


def synthetic_features(number_of_seconds: int = 5_000):
    t = np.arange(number_of_seconds)
    price = 100_000 + 25 * np.sin(t / 19) + 0.02 * t
    prices = PriceSeries(price=price, log_price=np.log(price),
                         first_timestamp=0, last_timestamp=number_of_seconds - 1)
    return build_features(prices, shared_parameters())
