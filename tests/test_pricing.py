"""Data-independent invariants of the shared pricing math."""
from __future__ import annotations

from snapmarket.parameters import SharedParameters
from snapmarket.pricing import quote_odds


def _approximately_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(a - b) < tolerance


def test_balanced_book_is_symmetric_with_expected_overround():
    parameters = SharedParameters()
    margin = parameters.house_margin
    odds = quote_odds(0.50, margin, 0.0, parameters)
    assert _approximately_equal(odds[0], 2 * (1 - margin))
    assert _approximately_equal(odds[1], 2 * (1 - margin))
    assert _approximately_equal(1 / odds[0] + 1 / odds[1], 1 / (1 - margin))


def test_no_side_is_ever_quoted_above_its_fair_odds():
    parameters = SharedParameters()
    margin = parameters.house_margin
    for probability in [0.40, 0.50, 0.58, 0.62]:
        for imbalance in [-0.9, -0.3, 0.0, 0.3, 0.9]:
            odds = quote_odds(probability, margin, imbalance, parameters)
            assert odds[0] <= 1 / probability + 1e-9
            assert odds[1] <= 1 / (1 - probability) + 1e-9


def test_inventory_skew_shortens_the_crowded_side():
    parameters = SharedParameters()
    margin = parameters.house_margin
    odds = quote_odds(0.50, margin, 0.3, parameters)
    assert odds[0] < 2 * (1 - margin) < odds[1]


def test_uptrend_shortens_the_up_side():
    parameters = SharedParameters()
    margin = parameters.house_margin
    assert (quote_odds(0.58, margin, 0.0, parameters)[0]
            < quote_odds(0.50, margin, 0.0, parameters)[0])


def test_per_bet_house_edge_at_the_fair_probability_equals_the_vig():
    parameters = SharedParameters()
    margin = parameters.house_margin
    for probability in [0.45, 0.50, 0.58]:
        odds = quote_odds(probability, margin, 0.0, parameters)
        assert _approximately_equal(1 - probability * odds[0], margin, 1e-3)
        assert _approximately_equal(1 - (1 - probability) * odds[1], margin, 1e-3)


def test_inventory_skew_is_the_only_displayed_directional_asymmetry():
    parameters = SharedParameters()
    odds_balanced = quote_odds(0.50, parameters.house_margin, 0.0, parameters)
    odds_long_up = quote_odds(0.50, parameters.house_margin, 0.4, parameters)
    assert odds_balanced[0] == odds_balanced[1]
    assert odds_long_up[0] < odds_balanced[0]
    assert odds_long_up[1] > odds_balanced[1]
