import numpy as np

from snapmarket.data import PriceSeries
from snapmarket.engine import book_risk_margin, simulate
from snapmarket.features import build_features
from snapmarket.models import (
    Model,
    ModelParameters,
    build_hidden_signal_symmetric_margin_model,
    information_margin_from_probability,
    quote_odds,
    run_golden_tests,
    run_model3_golden_tests,
)
from snapmarket.strategies import noise_pool


def synthetic_features(number_of_seconds=5_000):
    t = np.arange(number_of_seconds)
    price = 100_000 + 25 * np.sin(t / 19) + 0.02 * t
    prices = PriceSeries(price=price, log_price=np.log(price),
                         first_timestamp=0, last_timestamp=number_of_seconds - 1)
    return build_features(prices, ModelParameters(
        volatility_ewma_span=60,
        internal_minimum_training_contracts=50,
        internal_training_window_contracts=100,
        internal_retrain_contracts=50,
        internal_logistic_iterations=10,
        minimum_samples_per_bin=10,
    ))


def test_existing_pricing_golden_tests_still_pass():
    run_golden_tests()


def test_model3_information_margin_is_symmetric():
    parameters = ModelParameters()
    margin = information_margin_from_probability(np.array([0.40, 0.50, 0.60]), parameters)
    assert margin[0] == margin[2]
    assert margin[1] == 0.0
    run_model3_golden_tests(parameters)


def test_model3_hides_internal_direction_from_display_probability():
    features = synthetic_features()
    parameters = ModelParameters(
        volatility_ewma_span=60,
        internal_minimum_training_contracts=50,
        internal_training_window_contracts=100,
        internal_retrain_contracts=50,
        internal_logistic_iterations=10,
        minimum_samples_per_bin=10,
    )
    model = build_hidden_signal_symmetric_margin_model(features, parameters)

    assert model.internal_probability is not None
    assert np.allclose(model.display_probability, 0.5)
    assert np.nanmax(model.margin) >= model.parameters.house_margin


def test_inventory_skew_remains_the_only_displayed_directional_asymmetry():
    parameters = ModelParameters()
    odds_balanced = quote_odds(0.50, parameters.house_margin, 0.0, parameters)
    odds_long_up = quote_odds(0.50, parameters.house_margin, 0.4, parameters)

    assert odds_balanced[0] == odds_balanced[1]
    assert odds_long_up[0] < odds_balanced[0]
    assert odds_long_up[1] > odds_balanced[1]


def test_book_risk_margin_activates_only_for_book_risk_model():
    features = synthetic_features(1_000)
    parameters = ModelParameters(
        volatility_ewma_span=60,
        book_risk_margin_sensitivity=0.10,
        terminal_gamma_margin_sensitivity=0.10,
    )
    model = Model(
        name="risk_model",
        display_probability=np.full(features.number_of_seconds, 0.5),
        margin=parameters.house_margin,
        parameters=parameters,
        use_book_risk_margin=True,
    )

    start_index = 100
    step = 20
    total_up = np.zeros(60)
    total_down = np.zeros(60)
    total_up[10:20] = 1_000
    odds_up = np.full(60, 1.75)
    odds_down = np.full(60, 1.75)

    margin = book_risk_margin(model, features, start_index, step, total_up, total_down,
                              odds_up, odds_down)
    assert margin > 0.0


def test_simulate_accepts_model3_extra_margin_series():
    features = synthetic_features()
    model = build_hidden_signal_symmetric_margin_model(features, ModelParameters(
        volatility_ewma_span=60,
        internal_minimum_training_contracts=50,
        internal_training_window_contracts=100,
        internal_retrain_contracts=50,
        internal_logistic_iterations=10,
        minimum_samples_per_bin=10,
    ))

    result = simulate(model, features, {"noise": noise_pool(base_stake=10)}, 500, 200, seed=1)
    assert len(result.risk_margin_series) == 200
    assert result.total_volume > 0
