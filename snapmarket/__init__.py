"""Snap Market continuous-snap trading toolkit.

Public surface:
  - SharedParameters            shared settings for features, pricing, and the engine
  - build_features              per-second features from a price series
  - build_model / available_model_names / get_model_specification   model selection
  - Model                       the runtime model object
  - simulate / build_contract_table / pocket_edge   evaluation harness
  - run_across_models           run one experiment on several models

Importing this package registers every model in `snapmarket.models`.
"""
from __future__ import annotations

from .data import PriceSeries, load_fast_feed, load_oracle_prices
from .experiments import common_evaluation_start, run_across_models
from .features import Features, build_features, contract_entries, quantile_bins
from .model import BookRiskParameters, Model
from .parameters import SharedParameters
from .registry import available_model_names, build_model, get_model_specification
from .signals import regime_conditional_probability, walk_forward_logistic_probability
from . import models  # noqa: F401  (imported for model registration side effects)

__all__ = [
    "PriceSeries",
    "load_oracle_prices",
    "load_fast_feed",
    "run_across_models",
    "common_evaluation_start",
    "Features",
    "build_features",
    "contract_entries",
    "quantile_bins",
    "BookRiskParameters",
    "Model",
    "SharedParameters",
    "available_model_names",
    "build_model",
    "get_model_specification",
    "walk_forward_logistic_probability",
    "regime_conditional_probability",
]
