"""Parameters specific to the hidden-signal symmetric-margin model."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HiddenSymmetricMarginParameters:
    # --- train/test ---
    training_fraction: float = 0.40
    minimum_samples_per_bin: int = 50

    # --- information margin ---
    information_margin_buffer: float = 0.02     # extra cushion above hidden-signal break-even

    # --- hidden internal logistic model ---
    internal_minimum_training_contracts: int = 5_000
    internal_training_window_contracts: int = 20_000
    internal_retrain_contracts: int = 5_000
    internal_logistic_iterations: int = 40
    internal_logistic_learning_rate: float = 0.08
    internal_logistic_ridge_penalty: float = 0.02
    internal_probability_clip: float = 0.02

    # --- live-book risk margin ---
    book_risk_margin_sensitivity: float = 0.08
    terminal_gamma_margin_sensitivity: float = 0.06
    stress_sigma_multipliers: tuple[float, ...] = field(
        default=(-2.0, -1.0, 0.0, 1.0, 2.0)
    )
