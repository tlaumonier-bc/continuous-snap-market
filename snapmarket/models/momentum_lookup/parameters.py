"""Parameters specific to the momentum-lookup model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumLookupParameters:
    training_fraction: float = 0.40             # share of history used to calibrate the fair price
    calibration_bin_count: int = 25             # number of quantile bins in the p_up lookup
    calibration_shrinkage: float = 0.40         # keep this fraction of each bin's edge over 0.5
    minimum_samples_per_bin: int = 50           # bins below this fall back to 0.5
    monotonic_pooling_passes: int = 200         # pool-adjacent-violators sweeps enforcing monotonicity
