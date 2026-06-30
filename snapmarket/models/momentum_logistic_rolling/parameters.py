"""Parameters specific to the rolling logistic momentum model.

This model displays a walk-forward logistic estimate of P(up) built on many features (the same
estimator the hidden-signal model keeps internal). Making the displayed curve as rich as an
attacker's own model is what removes the attacker's edge.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumLogisticRollingParameters:
    # --- walk-forward schedule (in contracts) ---
    minimum_training_contracts: int = 5_000
    training_window_contracts: int = 20_000
    retrain_contracts: int = 5_000

    # --- logistic fit ---
    logistic_iterations: int = 40
    logistic_learning_rate: float = 0.08
    logistic_ridge_penalty: float = 0.02
    probability_clip: float = 0.02
    minimum_samples_per_fit: int = 50

    # --- display ---
    display_shrinkage: float = 1.0          # 1.0 shows the raw estimate; lower pulls toward 0.5
