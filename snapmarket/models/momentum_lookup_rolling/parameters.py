"""Parameters specific to the rolling momentum-lookup model.

Same calibration as the static momentum-lookup model, but the fair-price curve is
recomputed on a trailing window instead of a single fixed train/test split. The window
and recompute cadence are expressed in seconds so they read in calendar terms.
"""
from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class MomentumLookupRollingParameters:
    # --- rolling schedule ---
    calibration_window_seconds: int = 90 * SECONDS_PER_DAY      # trailing window used to calibrate
    recompute_every_seconds: int = 7 * SECONDS_PER_DAY          # recalibrate this often (weekly)

    # --- calibration (same knobs as the static model) ---
    calibration_bin_count: int = 25
    calibration_shrinkage: float = 0.40
    minimum_samples_per_bin: int = 50
    monotonic_pooling_passes: int = 200
