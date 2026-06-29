"""Convert hidden signal strength into a symmetric margin add-on."""
from __future__ import annotations

import numpy as np

from ...parameters import SharedParameters
from .parameters import HiddenSymmetricMarginParameters


def information_margin_from_probability(internal_probability,
                                       shared_parameters: SharedParameters,
                                       parameters: HiddenSymmetricMarginParameters) -> np.ndarray:
    """Symmetric margin needed to make the stronger hidden side non-positive expected value.

    With displayed P=0.50, decimal odds are 2 * (1 - margin). A bettor with true win
    probability q breaks even at margin = 1 - 1 / (2q).
    """
    probability = np.asarray(internal_probability, dtype=float)
    strongest_side_probability = np.maximum(probability, 1.0 - probability)
    required_margin = 1.0 - 1.0 / (2.0 * strongest_side_probability)
    add_on = required_margin + parameters.information_margin_buffer - shared_parameters.house_margin
    return np.maximum(add_on, 0.0)
