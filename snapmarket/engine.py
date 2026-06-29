"""Evaluation harness.

Two entry points, both model-agnostic:

  simulate(...)            the dynamic book — runs one or more bettors through the live
                           engine (settlement, inventory skew, net-delta cap) and isolates
                           each bettor's PnL. Use for house edge by flow type, and for the
                           "attacker inside a background pool" / capacity tests.

  build_contract_table(...) + pocket_edge(...)
                           the static view — per-contract outcomes and the expected value
                           of betting a chosen side inside any subset (a "pocket"). Use to
                           measure where the displayed curve is mispriced, with no engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .features import Features, contract_entries
from .models import Model


# --------------------------------------------------------------------------- #
#  Dynamic book                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class BettorResult:
    pnl: float = 0.0
    stake: float = 0.0

    @property
    def edge(self) -> float:
        return self.pnl / self.stake if self.stake else 0.0


@dataclass
class SimulationResult:
    house_pnl: float
    total_volume: float
    house_edge: float
    max_absolute_net_delta: float
    refused_seconds: int
    per_bettor: dict
    net_delta_series: np.ndarray = field(repr=False)
    pnl_series: np.ndarray = field(repr=False)
    risk_margin_series: np.ndarray = field(repr=False)


def _house_profit_vector(up, down, odds_up, odds_down, strike_price, settlement_price):
    return np.where(
        settlement_price > strike_price,
        down - up * (odds_up - 1.0),
        np.where(settlement_price < strike_price, up - down * (odds_down - 1.0), 0.0),
    )


def book_risk_margin(model: Model, features: Features, start_index: int, step: int,
                     total_up, total_down, odds_up_recorded, odds_down_recorded) -> float:
    """Extra margin from live-book loss-at-risk and near-expiry ATM concentration."""
    parameters = model.parameters
    if not model.use_book_risk_margin:
        return 0.0

    horizon = parameters.horizon_seconds
    open_start = max(0, step - horizon + 1)
    if open_start >= step:
        return 0.0

    open_steps = np.arange(open_start, step)
    open_up = total_up[open_steps]
    open_down = total_down[open_steps]
    live = (open_up + open_down) > 1e-12
    if not live.any():
        return 0.0

    open_steps = open_steps[live]
    open_up = open_up[live]
    open_down = open_down[live]
    odds_up = odds_up_recorded[open_steps]
    odds_down = odds_down_recorded[open_steps]

    now = start_index + step
    entry_times = start_index + open_steps
    strike_price = features.price[entry_times]
    current_price = features.price[now]
    remaining = np.maximum(horizon - (step - open_steps), 1)
    current_volatility = max(float(features.per_second_volatility[now]), 1e-12)

    stress_losses = []
    for multiplier in parameters.stress_sigma_multipliers:
        settlement_price = current_price * np.exp(
            multiplier * current_volatility * np.sqrt(remaining)
        )
        profit = _house_profit_vector(open_up, open_down, odds_up, odds_down,
                                      strike_price, settlement_price).sum()
        stress_losses.append(max(0.0, -float(profit)))
    loss_at_risk = max(stress_losses) if stress_losses else 0.0

    scale = parameters.maximum_net_delta
    book_margin = (
        parameters.book_risk_margin_sensitivity
        * loss_at_risk
        / (loss_at_risk + scale)
        if loss_at_risk > 0 else 0.0
    )

    distance = np.abs(np.log(current_price / strike_price))
    sigma_distance = current_volatility * np.sqrt(remaining)
    atm_weight = np.exp(-0.5 * (distance / np.maximum(sigma_distance, 1e-12)) ** 2)
    time_weight = 1.0 / np.sqrt(remaining)
    gamma_notional = float(((open_up + open_down) * atm_weight * time_weight).sum())
    gamma_margin = (
        parameters.terminal_gamma_margin_sensitivity
        * gamma_notional
        / (gamma_notional + scale)
        if gamma_notional > 0 else 0.0
    )

    headroom = parameters.maximum_total_margin - model.margin_at(now)
    return float(np.clip(book_margin + gamma_margin, 0.0, max(0.0, headroom)))


def simulate(model: Model, features: Features, bettors: dict[str, Callable],
             start_index: int, number_of_steps: int, seed: int = 42) -> SimulationResult:
    """Run the live book for `number_of_steps` seconds from `start_index`.

    `bettors` maps a name to a `bettor(t, random_generator) -> (up_stake, down_stake)`.
    All bettors are filled at the same odds each second (one quote off the shared book).
    Returns total house PnL/edge and each bettor's isolated PnL/edge.
    """
    parameters = model.parameters
    horizon = parameters.horizon_seconds
    price = features.price
    random_generator = np.random.default_rng(seed)
    names = list(bettors)

    total_up = np.zeros(number_of_steps)
    total_down = np.zeros(number_of_steps)
    odds_up_recorded = np.zeros(number_of_steps)
    odds_down_recorded = np.zeros(number_of_steps)
    bettor_up = {name: np.zeros(number_of_steps) for name in names}
    bettor_down = {name: np.zeros(number_of_steps) for name in names}

    open_up_exposure = open_down_exposure = house_pnl = 0.0
    net_delta_series = np.empty(number_of_steps)
    pnl_series = np.empty(number_of_steps)
    risk_margin_series = np.zeros(number_of_steps)
    refused_seconds = 0
    results = {name: BettorResult() for name in names}

    def house_profit(up, down, odds_up, odds_down, price_now, price_then):
        if price_now > price_then:
            return down - up * (odds_up - 1)
        if price_now < price_then:
            return up - down * (odds_down - 1)
        return 0.0  # push refunds the stake

    for step in range(number_of_steps):
        now = start_index + step

        # settle the cohort struck horizon seconds ago
        if step - horizon >= 0:
            settle_step = step - horizon
            settle_time = start_index + settle_step
            odds_up = odds_up_recorded[settle_step]
            odds_down = odds_down_recorded[settle_step]
            house_pnl += house_profit(total_up[settle_step], total_down[settle_step],
                                      odds_up, odds_down, price[now], price[settle_time])
            for name in names:
                up, down = bettor_up[name][settle_step], bettor_down[name][settle_step]
                if up + down > 1e-12:
                    results[name].pnl += -house_profit(up, down, odds_up, odds_down, price[now], price[settle_time])
                    results[name].stake += up + down
            open_up_exposure -= total_up[settle_step]
            open_down_exposure -= total_down[settle_step]

        # one quote off the current book imbalance
        net_delta_imbalance = (
            (open_up_exposure - open_down_exposure)
            / (open_up_exposure + open_down_exposure + parameters.maximum_net_delta)
        )
        extra_margin = book_risk_margin(model, features, start_index, step, total_up, total_down,
                                        odds_up_recorded, odds_down_recorded)
        odds_up, odds_down = model.quote(now, net_delta_imbalance, extra_margin=extra_margin)

        # collect every bettor's flow for this second
        this_up = {name: 0.0 for name in names}
        this_down = {name: 0.0 for name in names}
        for name in names:
            up_stake, down_stake = bettors[name](now, random_generator)
            this_up[name], this_down[name] = up_stake, down_stake

        # enforce the net-delta cap: refuse the crowded side across all bettors this second
        current_net_delta = open_up_exposure - open_down_exposure
        second_up = sum(this_up.values())
        second_down = sum(this_down.values())
        refused = False
        if current_net_delta >= parameters.maximum_net_delta and second_up > 0:
            this_up = {name: 0.0 for name in names}; refused = True
        if current_net_delta <= -parameters.maximum_net_delta and second_down > 0:
            this_down = {name: 0.0 for name in names}; refused = True
        if refused:
            refused_seconds += 1

        # record
        odds_up_recorded[step] = odds_up
        odds_down_recorded[step] = odds_down
        for name in names:
            bettor_up[name][step] = this_up[name]
            bettor_down[name][step] = this_down[name]
        total_up[step] = sum(this_up.values())
        total_down[step] = sum(this_down.values())
        open_up_exposure += total_up[step]
        open_down_exposure += total_down[step]
        net_delta_series[step] = open_up_exposure - open_down_exposure
        pnl_series[step] = house_pnl
        risk_margin_series[step] = extra_margin

    total_volume = total_up.sum() + total_down.sum()
    return SimulationResult(
        house_pnl=house_pnl,
        total_volume=total_volume,
        house_edge=house_pnl / total_volume if total_volume else 0.0,
        max_absolute_net_delta=float(np.abs(net_delta_series).max()),
        refused_seconds=refused_seconds,
        per_bettor=results,
        net_delta_series=net_delta_series,
        pnl_series=pnl_series,
        risk_margin_series=risk_margin_series,
    )


# --------------------------------------------------------------------------- #
#  Static contract-level view                                                 #
# --------------------------------------------------------------------------- #
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


def build_contract_table(model: Model, features: Features) -> ContractTable:
    """Per-contract outcomes and flat-book payouts on the non-overlapping entry grid."""
    parameters = model.parameters
    horizon = parameters.horizon_seconds
    price = features.price
    entries = contract_entries(features.number_of_seconds, parameters)

    price_at_entry = price[entries]
    price_at_settlement = price[entries + horizon]
    up_wins = price_at_settlement > price_at_entry
    down_wins = price_at_settlement < price_at_entry
    is_push = price_at_settlement == price_at_entry

    house_probability = model.display_probability[entries]
    odds_up, odds_down = model.flat_book_odds(entries)
    payout_betting_up = np.where(up_wins, odds_up, np.where(is_push, 1.0, 0.0))
    payout_betting_down = np.where(down_wins, odds_down, np.where(is_push, 1.0, 0.0))

    split_index = int(parameters.training_fraction * len(entries))
    training_mask = np.zeros(len(entries), bool)
    training_mask[:split_index] = True
    test_mask = ~training_mask

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
