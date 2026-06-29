"""Run the same experiment across several models.

An experiment is any callable that takes a built `Model` and returns a result. This
keeps notebooks short: define the experiment once, then compare every model on it.
"""
from __future__ import annotations

from typing import Callable

from .features import Features
from .model import Model
from .parameters import SharedParameters
from .registry import available_model_names, build_model


def run_across_models(model_names: list[str] | None, features: Features,
                      shared_parameters: SharedParameters,
                      experiment: Callable[[Model], object],
                      model_parameters: dict[str, object] | None = None) -> dict[str, object]:
    """Build each named model and run `experiment` on it.

    `model_names` defaults to every registered model. `model_parameters` optionally maps a
    model name to a custom parameter dataclass; models absent from it use their defaults.
    """
    if model_names is None:
        model_names = available_model_names()
    model_parameters = model_parameters or {}

    results: dict[str, object] = {}
    for name in model_names:
        model = build_model(name, features, shared_parameters, model_parameters.get(name))
        results[name] = experiment(model)
    return results
