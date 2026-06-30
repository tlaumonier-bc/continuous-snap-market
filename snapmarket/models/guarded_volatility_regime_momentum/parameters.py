"""Parameters for the guarded volatility-regime momentum model.

Defence in depth: the `display` block prices the direction (volatility-regime momentum), and
the `guard` block adds an information margin computed from a richer hidden logistic estimate,
plus live-book risk margin. Reusing the two existing parameter dataclasses keeps every knob in
one familiar place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..hidden_symmetric_margin import HiddenSymmetricMarginParameters
from ..volatility_regime_momentum import VolatilityRegimeMomentumParameters


@dataclass(frozen=True)
class GuardedVolatilityRegimeMomentumParameters:
    display: VolatilityRegimeMomentumParameters = field(
        default_factory=VolatilityRegimeMomentumParameters)
    guard: HiddenSymmetricMarginParameters = field(
        default_factory=HiddenSymmetricMarginParameters)
