"""Evaluation harness: a dynamic book simulation and a static contract-level view.

Both entry points are model-agnostic; any registered model runs through them unchanged.
"""
from __future__ import annotations

from .book_risk import book_risk_margin
from .contract_table import ContractTable, build_contract_table, pocket_edge
from .simulation import BettorResult, SimulationResult, simulate

__all__ = [
    "book_risk_margin",
    "ContractTable",
    "build_contract_table",
    "pocket_edge",
    "BettorResult",
    "SimulationResult",
    "simulate",
]
