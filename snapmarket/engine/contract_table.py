"""Static contract-level view.

`build_contract_table(...)` produces per-contract outcomes and flat-book payouts on the
non-overlapping entry grid; `pocket_edge(...)` measures the expected value of betting a
chosen side inside any subset (a "pocket"). Use this to find where the displayed curve is
mispriced, with no engine involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import Features, contract_entries
from ..model import Model


@dataclass
class ContractTable:
    entries: np.ndarray
    up_wins: np.ndarray
    down_wins: np.ndarray
    is_push: np.ndarray
    realized_up_rate: np.ndarray
    house_probability: np.ndarray            # displayed probability at entry
    payout_betting_up: np.ndarray            # payout per $1 staked (push refunds 1.0)
    payout_betting_down: np.ndarray
    training_mask: np.ndarray
    test_mask: np.ndarray

    def feature_at_entry(self, feature_array: np.ndarray) -> np.ndarray:
        return feature_array[self.entries]


def _split_masks(entry_count: int, training_fraction: float):
    split_index = int(training_fraction * entry_count)
    training_mask = np.zeros(entry_count, bool)
    training_mask[:split_index] = True
    return training_mask, ~training_mask


def build_contract_table(model: Model, features: Features,
                         training_fraction: float = 0.40) -> ContractTable:
    """Per-contract outcomes and flat-book payouts on the non-overlapping entry grid."""
    shared = model.shared_parameters
    horizon = shared.horizon_seconds
    price = features.price
    entries = contract_entries(features.number_of_seconds, shared)

    price_at_entry = price[entries]
    price_at_settlement = price[entries + horizon]
    up_wins = price_at_settlement > price_at_entry
    down_wins = price_at_settlement < price_at_entry
    is_push = price_at_settlement == price_at_entry

    house_probability = model.display_probability[entries]
    odds_up, odds_down = model.flat_book_odds(entries)
    payout_betting_up = np.where(up_wins, odds_up, np.where(is_push, 1.0, 0.0))
    payout_betting_down = np.where(down_wins, odds_down, np.where(is_push, 1.0, 0.0))

    training_mask, test_mask = _split_masks(len(entries), training_fraction)

    return ContractTable(
        entries=entries, up_wins=up_wins, down_wins=down_wins, is_push=is_push,
        realized_up_rate=up_wins.astype(float), house_probability=house_probability,
        payout_betting_up=payout_betting_up, payout_betting_down=payout_betting_down,
        training_mask=training_mask, test_mask=test_mask,
    )


def pocket_edge(table: ContractTable, mask: np.ndarray, side: str) -> dict:
    """Bettor expected value and house edge for betting `side` ('up'/'down') inside `mask`."""
    payout = table.payout_betting_up if side == "up" else table.payout_betting_down
    count = int(mask.sum())
    if count == 0:
        return dict(count=0, realized_up_rate=np.nan, house_probability=np.nan,
                    bettor_edge=np.nan, house_edge=np.nan)
    bettor_edge = payout[mask].mean() - 1
    return dict(
        count=count,
        realized_up_rate=float(table.realized_up_rate[mask].mean()),
        house_probability=float(table.house_probability[mask].mean()),
        bettor_edge=float(bettor_edge),
        house_edge=float(-bettor_edge),
    )
