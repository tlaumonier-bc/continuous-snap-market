"""Model registry.

Each model registers a `ModelSpecification` describing how to build it and what its
default parameters are. Notebooks then select a model by name and build it, or loop
over `available_model_names()` to run the same experiment on every model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .features import Features
from .model import Model
from .parameters import SharedParameters


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    description: str
    default_parameters: Callable[[], object]
    build: Callable[[Features, SharedParameters, object], Model]


_REGISTRY: dict[str, ModelSpecification] = {}


def register_model(specification: ModelSpecification) -> None:
    if specification.name in _REGISTRY:
        raise ValueError(f"a model named '{specification.name}' is already registered")
    _REGISTRY[specification.name] = specification


def available_model_names() -> list[str]:
    return sorted(_REGISTRY)


def get_model_specification(name: str) -> ModelSpecification:
    if name not in _REGISTRY:
        known = ", ".join(available_model_names()) or "none"
        raise KeyError(f"unknown model '{name}'. Registered models: {known}")
    return _REGISTRY[name]


def build_model(name: str, features: Features, shared_parameters: SharedParameters,
                model_parameters: object | None = None) -> Model:
    """Build a registered model by name, using its default parameters when none are given."""
    specification = get_model_specification(name)
    if model_parameters is None:
        model_parameters = specification.default_parameters()
    return specification.build(features, shared_parameters, model_parameters)
