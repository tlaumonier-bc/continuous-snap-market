"""Hidden-signal symmetric-margin model package. Importing it registers the model."""
from __future__ import annotations

from .builder import MODEL_NAME, build
from .information_margin import information_margin_from_probability
from .internal_probability import build_internal_probability
from .parameters import HiddenSymmetricMarginParameters

__all__ = [
    "MODEL_NAME",
    "build",
    "information_margin_from_probability",
    "build_internal_probability",
    "HiddenSymmetricMarginParameters",
]
