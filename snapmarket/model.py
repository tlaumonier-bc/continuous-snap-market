"""The runtime model object.

A fitted model is two per-second series plus the shared pricing math:
  - display_probability : the probability the displayed odds are built around
  - margin              : the vig (scalar, or a per-second array for models that widen it)

Every model prices through the same `quote_odds`, runs through the same engine, and is
attacked by the same strategies. A new model is a new way to fill those two series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parameters import SharedParameters
from .pricing import quote_odds


@dataclass(frozen=True)
class BookRiskParameters:
    """Opt-in extra margin a model charges for live-book risk.

    A model that wants the engine to add loss-at-risk and terminal-gamma margin attaches
    one of these; a model that leaves it as None pays no extra margin.
    """
    book_risk_margin_sensitivity: float
    terminal_gamma_margin_sensitivity: float
    stress_sigma_multipliers: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass
class Model:
    name: str
    display_probability: np.ndarray                 # per second
    margin: object                                  # float, or per-second np.ndarray
    shared_parameters: SharedParameters
    internal_probability: np.ndarray | None = None
    margin_components: dict[str, object] = field(default_factory=dict)
    book_risk_parameters: BookRiskParameters | None = None

    @property
    def uses_book_risk_margin(self) -> bool:
        return self.book_risk_parameters is not None

    def margin_at(self, t: int) -> float:
        return float(self.margin) if np.isscalar(self.margin) else float(self.margin[t])

    def total_margin_at(self, t: int, extra_margin: float = 0.0) -> float:
        return float(np.clip(
            self.margin_at(t) + extra_margin,
            0.0,
            self.shared_parameters.maximum_total_margin,
        ))

    def quote(self, t: int, net_delta_imbalance: float, extra_margin: float = 0.0):
        return quote_odds(
            self.display_probability[t],
            self.total_margin_at(t, extra_margin),
            net_delta_imbalance,
            self.shared_parameters,
        )

    def flat_book_odds(self, index):
        """Vectorised odds on a balanced book (imbalance = 0), used for static evaluation."""
        maximum_total_margin = self.shared_parameters.maximum_total_margin
        maximum_odds = self.shared_parameters.maximum_odds
        probability = self.display_probability[index]
        margin = self.margin if np.isscalar(self.margin) else self.margin[index]
        margin = np.clip(margin, 0.0, maximum_total_margin)
        odds_up = np.minimum((1.0 - margin) / probability, maximum_odds)
        odds_down = np.minimum((1.0 - margin) / (1.0 - probability), maximum_odds)
        return odds_up, odds_down
