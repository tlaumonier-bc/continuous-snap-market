"""Volatility-regime momentum model package. Importing it registers the model."""
from __future__ import annotations

from .builder import MODEL_NAME, build
from .parameters import VolatilityRegimeMomentumParameters

__all__ = ["MODEL_NAME", "build", "VolatilityRegimeMomentumParameters"]
