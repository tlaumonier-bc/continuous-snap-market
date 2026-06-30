"""Shared walk-forward scaffolding for rolling models.

A rolling model recalibrates on a trailing window of contracts and applies the fitted
curve until the next recompute. The schedule and the contract bookkeeping are identical
across rolling models; only the calibration differs, so they live here once.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from ..features import Features
from ..parameters import SharedParameters


def seconds_to_contracts(seconds: int, horizon: int) -> int:
    return max(1, seconds // horizon)


def contract_up_outcomes(features: Features, entries: np.ndarray,
                         shared_parameters: SharedParameters) -> np.ndarray:
    """1.0 where the price is higher one horizon after each contract entry, else 0.0."""
    horizon = shared_parameters.horizon_seconds
    price = features.price
    return (price[entries + horizon] > price[entries]).astype(float)


def walk_forward_segments(entries: np.ndarray, window_contracts: int,
                          recompute_contracts: int,
                          number_of_seconds: int) -> Iterator[tuple[slice, int, int]]:
    """Yield (training_slice, segment_start, segment_stop) for each recompute step.

    `training_slice` indexes the trailing window of contracts; `segment_start:segment_stop`
    is the span of seconds that the curve fitted on that window prices, until the next recompute.
    """
    for start in range(window_contracts, len(entries), recompute_contracts):
        training_slice = slice(start - window_contracts, start)
        segment_start = int(entries[start])
        next_start = min(len(entries), start + recompute_contracts)
        segment_stop = int(entries[next_start]) if next_start < len(entries) else number_of_seconds
        yield training_slice, segment_start, segment_stop
