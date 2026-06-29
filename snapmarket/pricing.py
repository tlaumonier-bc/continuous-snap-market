"""Shared pricing math.

Every model prices through `quote_odds`: it converts a displayed probability and a
margin into decimal (odds_up, odds_down), skewed for inventory and clamped so no side
is ever offered above its fair odds. A model is just a way to fill the displayed
probability and the margin; the pricing is identical for all of them.
"""
from __future__ import annotations

import math

import numpy as np

from .parameters import SharedParameters

_PROBABILITY_EPSILON = 1e-6


def _skewed_up_probability(display_probability: float, net_delta_imbalance: float,
                           parameters: SharedParameters) -> float:
    """Shift the displayed probability toward the crowded side of the book."""
    fair_log_odds = math.log(display_probability / (1.0 - display_probability))
    skew = parameters.inventory_skew_sensitivity * net_delta_imbalance
    return 1.0 / (1.0 + math.exp(-(fair_log_odds + skew)))


def quote_odds(display_probability, margin, net_delta_imbalance,
               parameters: SharedParameters):
    """Decimal (odds_up, odds_down): skewed for inventory and clamped at each side's fair odds.

    Works for any model: pass the model's displayed probability and margin for the second.
    """
    display_probability = float(np.clip(display_probability,
                                        _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON))
    margin = float(np.clip(margin, 0.0, parameters.maximum_total_margin))

    skewed_up_probability = _skewed_up_probability(display_probability,
                                                   net_delta_imbalance, parameters)
    odds_up = min((1.0 - margin) / skewed_up_probability,
                  1.0 / display_probability,
                  parameters.maximum_odds)
    odds_down = min((1.0 - margin) / (1.0 - skewed_up_probability),
                    1.0 / (1.0 - display_probability),
                    parameters.maximum_odds)
    return odds_up, odds_down
