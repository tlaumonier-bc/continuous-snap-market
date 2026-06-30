"""Convert hidden signal strength into a symmetric margin add-on."""
from __future__ import annotations

import numpy as np

from ...parameters import SharedParameters
from .parameters import HiddenSymmetricMarginParameters

_PROBABILITY_EPSILON = 1e-6


def information_margin_over_display(hidden_probability, display_probability,
                                   shared_parameters: SharedParameters,
                                   parameters: HiddenSymmetricMarginParameters) -> np.ndarray:
    """Extra symmetric margin so an attacker who knows `hidden_probability` is non-positive.

    On a balanced book the displayed odds are (1 - margin) / display on the up side and
    (1 - margin) / (1 - display) on the down side. An attacker with true probability p breaks
    even on the up side at margin = 1 - display / p, and on the down side at
    margin = 1 - (1 - display) / (1 - p). We charge the larger so neither side is positive,
    add a buffer, and subtract the vig the house already takes.
    """
    p = np.clip(np.asarray(hidden_probability, dtype=float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    q = np.clip(np.asarray(display_probability, dtype=float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)

    up_side_margin = 1.0 - q / p
    down_side_margin = 1.0 - (1.0 - q) / (1.0 - p)
    required_margin = np.maximum(np.maximum(up_side_margin, down_side_margin), 0.0)

    add_on = required_margin + parameters.information_margin_buffer - shared_parameters.house_margin
    return np.maximum(add_on, 0.0)


def information_margin_from_probability(internal_probability,
                                       shared_parameters: SharedParameters,
                                       parameters: HiddenSymmetricMarginParameters) -> np.ndarray:
    """Symmetric margin for a flat displayed P = 0.50 (the hidden-signal model's special case)."""
    probability = np.asarray(internal_probability, dtype=float)
    flat_display = np.full(probability.shape, 0.5)
    return information_margin_over_display(probability, flat_display, shared_parameters, parameters)
