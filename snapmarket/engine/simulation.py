"""Dynamic book simulation.

`simulate(...)` runs one or more bettors through the live engine (settlement, inventory
skew, net-delta cap) and isolates each bettor's PnL. Use it for house edge by flow type
and for the "attacker inside a background pool" / capacity tests. It is model-agnostic:
any model quotes through the same `Model.quote`.

A bettor is `bettor(t, random_generator, odds_up, odds_down) -> (up_stake, down_stake)`.
The current quoted odds are passed in so informed bettors can size on expected value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..features import Features
from ..model import Model
from .book_risk import book_risk_margin


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


def _house_profit(up, down, odds_up, odds_down, price_now, price_then) -> float:
    if price_now > price_then:
        return down - up * (odds_up - 1)
    if price_now < price_then:
        return up - down * (odds_down - 1)
    return 0.0  # push refunds the stake


def simulate(model: Model, features: Features, bettors: dict[str, Callable],
             start_index: int, number_of_steps: int, seed: int = 42) -> SimulationResult:
    """Run the live book for `number_of_steps` seconds from `start_index`.

    `bettors` maps a name to a
    `bettor(t, random_generator, odds_up, odds_down) -> (up_stake, down_stake)`.
    All bettors are filled at the same odds each second (one quote off the shared book).
    Returns total house PnL/edge and each bettor's isolated PnL/edge.
    """
    shared = model.shared_parameters
    horizon = shared.horizon_seconds
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

    for step in range(number_of_steps):
        now = start_index + step

        # settle the cohort struck horizon seconds ago
        if step - horizon >= 0:
            settle_step = step - horizon
            settle_time = start_index + settle_step
            odds_up = odds_up_recorded[settle_step]
            odds_down = odds_down_recorded[settle_step]
            house_pnl += _house_profit(total_up[settle_step], total_down[settle_step],
                                       odds_up, odds_down, price[now], price[settle_time])
            for name in names:
                up, down = bettor_up[name][settle_step], bettor_down[name][settle_step]
                if up + down > 1e-12:
                    results[name].pnl += -_house_profit(up, down, odds_up, odds_down,
                                                        price[now], price[settle_time])
                    results[name].stake += up + down
            open_up_exposure -= total_up[settle_step]
            open_down_exposure -= total_down[settle_step]

        # one quote off the current book imbalance
        net_delta_imbalance = (
            (open_up_exposure - open_down_exposure)
            / (open_up_exposure + open_down_exposure + shared.maximum_net_delta)
        )
        extra_margin = book_risk_margin(model, features, start_index, step, total_up, total_down,
                                        odds_up_recorded, odds_down_recorded)
        odds_up, odds_down = model.quote(now, net_delta_imbalance, extra_margin=extra_margin)

        # collect every bettor's flow for this second
        this_up = {name: 0.0 for name in names}
        this_down = {name: 0.0 for name in names}
        for name in names:
            up_stake, down_stake = bettors[name](now, random_generator, odds_up, odds_down)
            this_up[name], this_down[name] = up_stake, down_stake

        # enforce the net-delta cap: refuse the crowded side across all bettors this second
        current_net_delta = open_up_exposure - open_down_exposure
        second_up = sum(this_up.values())
        second_down = sum(this_down.values())
        refused = False
        if current_net_delta >= shared.maximum_net_delta and second_up > 0:
            this_up = {name: 0.0 for name in names}; refused = True
        if current_net_delta <= -shared.maximum_net_delta and second_down > 0:
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
