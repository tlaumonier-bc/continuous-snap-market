"""Model catalogue.

Importing this package imports every model module, which registers each model in the
shared registry. Notebooks then use `build_model(name, ...)` or loop over
`available_model_names()` to run the same experiment on every model.

Adding a model: create `snapmarket/models/<name>/` with its own parameters and builder
that calls `register_model(...)`, then import it below.
"""
from __future__ import annotations

from ..registry import (
    ModelSpecification,
    available_model_names,
    build_model,
    get_model_specification,
    register_model,
)
from . import hidden_symmetric_margin, momentum_lookup, momentum_lookup_rolling

__all__ = [
    "ModelSpecification",
    "available_model_names",
    "build_model",
    "get_model_specification",
    "register_model",
    "momentum_lookup",
    "momentum_lookup_rolling",
    "hidden_symmetric_margin",
]
