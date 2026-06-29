"""Live-book risk margin.

Generic over models: a model opts in by attaching `BookRiskParameters`. The margin
combines a stressed loss-at-risk term and a near-expiry at-the-money concentration
(gamma) term. Models without book-risk parameters pay nothing here.
"""
from __future__ import annotations

import numpy as np

from ..features import Features
from ..model import Model


def _house_profit_vector(up, down, odds_up, odds_down, strike_price, settlement_price):
    return np.where(
        settlement_price > strike_price,
        down - up * (odds_up - 1.0),
        np.where(settlement_price < strike_price, up - down * (odds_down - 1.0), 0.0),
    )


def _open_positions(step: int, horizon: int, total_up, total_down):
    """Indices and stakes of contracts still open at `step` (None if the book is empty)."""
    open_start = max(0, step - horizon + 1)
    if open_start >= step:
        return None
    open_steps = np.arange(open_start, step)
    live = (total_up[open_steps] + total_down[open_steps]) > 1e-12
    if not live.any():
        return None
    open_steps = open_steps[live]
    return open_steps, total_up[open_steps], total_down[open_steps]


def _loss_at_risk(open_up, open_down, odds_up, odds_down, strike_price, current_price,
                  remaining, current_volatility, stress_sigma_multipliers) -> float:
    stress_losses = []
    for multiplier in stress_sigma_multipliers:
        settlement_price = current_price * np.exp(
            multiplier * current_volatility * np.sqrt(remaining)
        )
        profit = _house_profit_vector(open_up, open_down, odds_up, odds_down,
                                      strike_price, settlement_price).sum()
        stress_losses.append(max(0.0, -float(profit)))
    return max(stress_losses) if stress_losses else 0.0


def _saturating_margin(quantity: float, sensitivity: float, scale: float) -> float:
    return sensitivity * quantity / (quantity + scale) if quantity > 0 else 0.0


def _terminal_gamma_notional(open_up, open_down, strike_price, current_price,
                             remaining, current_volatility) -> float:
    distance = np.abs(np.log(current_price / strike_price))
    sigma_distance = current_volatility * np.sqrt(remaining)
    atm_weight = np.exp(-0.5 * (distance / np.maximum(sigma_distance, 1e-12)) ** 2)
    time_weight = 1.0 / np.sqrt(remaining)
    return float(((open_up + open_down) * atm_weight * time_weight).sum())


def book_risk_margin(model: Model, features: Features, start_index: int, step: int,
                     total_up, total_down, odds_up_recorded, odds_down_recorded) -> float:
    """Extra margin from live-book loss-at-risk and near-expiry at-the-money concentration."""
    book_risk = model.book_risk_parameters
    if book_risk is None:
        return 0.0

    shared = model.shared_parameters
    horizon = shared.horizon_seconds
    open_positions = _open_positions(step, horizon, total_up, total_down)
    if open_positions is None:
        return 0.0
    open_steps, open_up, open_down = open_positions

    now = start_index + step
    strike_price = features.price[start_index + open_steps]
    current_price = features.price[now]
    remaining = np.maximum(horizon - (step - open_steps), 1)
    current_volatility = max(float(features.per_second_volatility[now]), 1e-12)
    scale = shared.maximum_net_delta

    loss_at_risk = _loss_at_risk(
        open_up, open_down, odds_up_recorded[open_steps], odds_down_recorded[open_steps],
        strike_price, current_price, remaining, current_volatility,
        book_risk.stress_sigma_multipliers,
    )
    loss_margin = _saturating_margin(loss_at_risk, book_risk.book_risk_margin_sensitivity, scale)

    gamma_notional = _terminal_gamma_notional(open_up, open_down, strike_price,
                                              current_price, remaining, current_volatility)
    gamma_margin = _saturating_margin(gamma_notional,
                                      book_risk.terminal_gamma_margin_sensitivity, scale)

    headroom = shared.maximum_total_margin - model.margin_at(now)
    return float(np.clip(loss_margin + gamma_margin, 0.0, max(0.0, headroom)))
