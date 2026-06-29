"""Tests for the momentum-lookup model."""
from __future__ import annotations

import numpy as np

from snapmarket.models import build_model
from snapmarket.models.momentum_lookup import MomentumLookupParameters

from .synthetic import shared_parameters, synthetic_features


def test_momentum_lookup_is_registered_and_builds():
    features = synthetic_features()
    model = build_model("momentum_lookup", features, shared_parameters())
    assert model.name == "momentum_lookup"
    assert len(model.display_probability) == features.number_of_seconds


def test_displayed_probability_stays_within_unit_interval():
    features = synthetic_features()
    model = build_model("momentum_lookup", features, shared_parameters())
    assert model.display_probability.min() >= 0.0
    assert model.display_probability.max() <= 1.0


def test_displayed_probability_is_shrunk_toward_one_half():
    features = synthetic_features()
    parameters = MomentumLookupParameters(calibration_shrinkage=0.0)
    model = build_model("momentum_lookup", features, shared_parameters(), parameters)
    assert np.allclose(model.display_probability, 0.5)


def test_momentum_lookup_uses_a_constant_margin_and_no_book_risk():
    features = synthetic_features()
    model = build_model("momentum_lookup", features, shared_parameters())
    assert np.isscalar(model.margin)
    assert model.book_risk_parameters is None
