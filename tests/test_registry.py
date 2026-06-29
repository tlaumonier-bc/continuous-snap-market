"""Tests for the model registry and the cross-model experiment helper."""
from __future__ import annotations

from snapmarket.experiments import run_across_models
from snapmarket.registry import available_model_names, build_model

from .synthetic import shared_parameters, synthetic_features


def test_both_models_are_registered():
    names = available_model_names()
    assert "momentum_lookup" in names
    assert "hidden_symmetric_margin" in names


def test_unknown_model_name_raises():
    features = synthetic_features()
    try:
        build_model("does_not_exist", features, shared_parameters())
    except KeyError:
        return
    raise AssertionError("expected a KeyError for an unknown model name")


def test_run_across_models_runs_the_experiment_on_each_model():
    features = synthetic_features()
    results = run_across_models(
        ["momentum_lookup"], features, shared_parameters(),
        experiment=lambda model: model.name,
    )
    assert results == {"momentum_lookup": "momentum_lookup"}
