"""Parameters specific to the rolling momentum-and-volatility model.

The classic momentum model shows a single curve P(up | momentum). This model conditions on
the volatility regime as well, calibrating one momentum curve per volatility bin, recomputed
walk-forward. Defaults match the classic rolling model so the only added ingredient is the
volatility dimension.
"""
from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class MomentumVolatilityRollingParameters:
    # --- rolling schedule ---
    calibration_window_seconds: int = 90 * SECONDS_PER_DAY
    recompute_every_seconds: int = 7 * SECONDS_PER_DAY

    # --- volatility conditioning ---
    volatility_bin_count: int = 3               # number of volatility regimes (terciles)

    # --- momentum calibration within each regime ---
    calibration_bin_count: int = 25
    calibration_shrinkage: float = 0.40
    minimum_samples_per_bin: int = 50
    monotonic_pooling_passes: int = 200
