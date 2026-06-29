"""Pricing math and model objects.

A fitted model is just two per-second series plus the shared pricing math:
  - display_probability : the probability the displayed odds are built around
  - margin              : the vig (scalar, or a per-second array for models that widen it)

Every model prices through the same `quote_odds`, runs through the same engine, and is
attacked by the same strategies. A new model = a new way to fill those two series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .features import Features, contract_entries


@dataclass(frozen=True)
class ModelParameters:
    # --- contract ---
    horizon_seconds: int = 30               # contract life; fixed for every bet

    # --- fair price (calibration) ---
    momentum_lookback_seconds: int = 5      # trailing window that defines the momentum state
    volatility_ewma_span: int = 600         # EWMA span for per-second realised volatility
    calibration_bin_count: int = 25         # number of quantile bins in the p_up lookup
    calibration_shrinkage: float = 0.40     # lambda: keep this fraction of each bin's edge over 0.5
    training_fraction: float = 0.40         # share of history used to calibrate the fair price
    minimum_samples_per_bin: int = 50       # bins below this fall back to 0.5
    monotonic_pooling_passes: int = 200     # pool-adjacent-violators sweeps enforcing monotonicity

    # --- pricing ---
    house_margin: float = 0.125             # the vig
    inventory_skew_sensitivity: float = 3.0 # how hard a book imbalance skews the odds
    maximum_odds: float = 5.0               # cap on decimal odds offered on either side
    maximum_total_margin: float = 0.49      # keep quoted odds at or above roughly 1.0x

    # --- risk ---
    maximum_net_delta: float = 2.0e4        # hard cap on |net delta| (USDT)
    information_margin_buffer: float = 0.02 # extra cushion above hidden-signal break-even
    book_risk_margin_sensitivity: float = 0.0
    terminal_gamma_margin_sensitivity: float = 0.0
    stress_sigma_multipliers: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

    # --- hidden internal model ---
    internal_minimum_training_contracts: int = 5_000
    internal_training_window_contracts: int = 20_000
    internal_retrain_contracts: int = 5_000
    internal_logistic_iterations: int = 40
    internal_logistic_learning_rate: float = 0.08
    internal_logistic_l2: float = 0.02
    internal_probability_clip: float = 0.02

    @property
    def warmup_seconds(self) -> int:
        return self.volatility_ewma_span + self.momentum_lookback_seconds


def quote_odds(display_probability, margin, net_delta_imbalance, parameters: ModelParameters):
    """Decimal (odds_up, odds_down): skewed for inventory and clamped at each side's fair odds.

    Works for any model: pass the model's display probability and margin for the second.
    """
    display_probability = float(np.clip(display_probability, 1e-6, 1.0 - 1e-6))
    margin = float(np.clip(margin, 0.0, parameters.maximum_total_margin))
    fair_log_odds = math.log(display_probability / (1.0 - display_probability))
    skewed_up_probability = 1.0 / (1.0 + math.exp(
        -(fair_log_odds + parameters.inventory_skew_sensitivity * net_delta_imbalance)
    ))
    odds_up = min((1.0 - margin) / skewed_up_probability,
                  1.0 / display_probability,
                  parameters.maximum_odds)
    odds_down = min((1.0 - margin) / (1.0 - skewed_up_probability),
                    1.0 / (1.0 - display_probability),
                    parameters.maximum_odds)
    return odds_up, odds_down


@dataclass
class Model:
    name: str
    display_probability: np.ndarray         # per second
    margin: object                          # float, or per-second np.ndarray
    parameters: ModelParameters
    internal_probability: np.ndarray | None = None
    margin_components: dict[str, object] = field(default_factory=dict)
    use_book_risk_margin: bool = False

    def margin_at(self, t) -> float:
        return float(self.margin) if np.isscalar(self.margin) else float(self.margin[t])

    def total_margin_at(self, t: int, extra_margin: float = 0.0) -> float:
        return float(np.clip(
            self.margin_at(t) + extra_margin,
            0.0,
            self.parameters.maximum_total_margin,
        ))

    def quote(self, t: int, net_delta_imbalance: float, extra_margin: float = 0.0):
        return quote_odds(
            self.display_probability[t],
            self.total_margin_at(t, extra_margin),
            net_delta_imbalance,
            self.parameters,
        )

    def flat_book_odds(self, index):
        """Vectorised odds on a balanced book (imbalance = 0), used for static evaluation."""
        probability = self.display_probability[index]
        margin = self.margin if np.isscalar(self.margin) else self.margin[index]
        margin = np.clip(margin, 0.0, self.parameters.maximum_total_margin)
        odds_up = np.minimum((1.0 - margin) / probability, self.parameters.maximum_odds)
        odds_down = np.minimum((1.0 - margin) / (1.0 - probability), self.parameters.maximum_odds)
        return odds_up, odds_down


def calibrate_fair_probability(training_momentum, training_outcomes, parameters: ModelParameters):
    """Return (bin_edges, fair_probability_per_bin): a monotone, shrunk lookup of p_up vs the state."""
    bin_count = parameters.calibration_bin_count
    bin_edges = np.quantile(training_momentum, np.linspace(0, 1, bin_count + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bin_index = np.digitize(training_momentum, bin_edges) - 1

    fair_probability_per_bin = np.array([
        training_outcomes[bin_index == i].mean()
        if (bin_index == i).sum() > parameters.minimum_samples_per_bin else 0.5
        for i in range(bin_count)
    ])
    for _ in range(parameters.monotonic_pooling_passes):
        for i in range(bin_count - 1):
            if fair_probability_per_bin[i] > fair_probability_per_bin[i + 1]:
                pooled = (fair_probability_per_bin[i] + fair_probability_per_bin[i + 1]) / 2
                fair_probability_per_bin[i] = fair_probability_per_bin[i + 1] = pooled

    shrunk_probability = 0.5 + parameters.calibration_shrinkage * (fair_probability_per_bin - 0.5)
    return bin_edges, shrunk_probability


def build_momentum_lookup_model(features: Features, parameters: ModelParameters,
                                name: str = "momentum_lookup_v1") -> Model:
    """The current production model (v1): calibrate p_up on the oracle's own momentum,
    fixed train/test split, constant margin."""
    price = features.price
    horizon = parameters.horizon_seconds
    entries = contract_entries(features.number_of_seconds, parameters)
    split_index = int(parameters.training_fraction * len(entries))
    training_entries = entries[:split_index]
    training_outcomes = (price[training_entries + horizon] > price[training_entries]).astype(float)

    bin_edges, probability_per_bin = calibrate_fair_probability(
        features.standardized_momentum[training_entries], training_outcomes, parameters
    )
    index = np.clip(np.digitize(features.standardized_momentum, bin_edges) - 1, 0, len(probability_per_bin) - 1)
    display_probability = probability_per_bin[index]

    return Model(name=name, display_probability=display_probability,
                 margin=parameters.house_margin, parameters=parameters)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _internal_feature_matrix(features: Features, index=slice(None)) -> np.ndarray:
    """Feature matrix for the hidden probability model.

    The scaling keeps the hand-written logistic optimiser independent of sklearn.
    """
    return np.column_stack([
        features.standardized_momentum[index],
        features.absolute_standardized_momentum[index],
        features.momentum_2_seconds[index] * 1e4,
        features.momentum_5_seconds[index] * 1e4,
        features.momentum_15_seconds[index] * 1e4,
        features.momentum_30_seconds[index] * 1e4,
        features.momentum_60_seconds[index] * 1e4,
        features.annualized_volatility[index],
        features.position_in_range_60_seconds[index],
        features.position_in_range_120_seconds[index],
        features.path_efficiency_30_seconds[index],
        features.acceleration[index] * 1e4,
    ]).astype(float)


def _fit_logistic_probability(training_features, training_outcomes, parameters: ModelParameters):
    finite = np.isfinite(training_features).all(axis=1) & np.isfinite(training_outcomes)
    x = training_features[finite]
    y = training_outcomes[finite].astype(float)
    if len(y) < parameters.minimum_samples_per_bin or y.min() == y.max():
        return None

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale

    weights = np.zeros(x.shape[1] + 1)
    for _ in range(parameters.internal_logistic_iterations):
        linear = weights[0] + x @ weights[1:]
        error = _sigmoid(linear) - y
        gradient = np.empty_like(weights)
        gradient[0] = error.mean()
        gradient[1:] = (x.T @ error) / len(y) + parameters.internal_logistic_l2 * weights[1:]
        weights -= parameters.internal_logistic_learning_rate * gradient
    return weights, mean, scale


def _predict_logistic_probability(feature_matrix, fitted, parameters: ModelParameters):
    if fitted is None:
        return np.full(len(feature_matrix), 0.5)
    weights, mean, scale = fitted
    x = np.nan_to_num((feature_matrix - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    probability = _sigmoid(weights[0] + x @ weights[1:])
    clip = parameters.internal_probability_clip
    return np.clip(probability, clip, 1.0 - clip)


def _fill_internal_probability(internal_probability, features: Features, start: int, stop: int,
                               fitted, parameters: ModelParameters, chunk_size: int = 500_000):
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(stop, chunk_start + chunk_size)
        internal_probability[chunk_start:chunk_stop] = _predict_logistic_probability(
            _internal_feature_matrix(features, slice(chunk_start, chunk_stop)),
            fitted,
            parameters,
        )


def build_internal_probability(features: Features, parameters: ModelParameters,
                               rolling: bool = True) -> np.ndarray:
    """Walk-forward hidden estimate of P(S[t+horizon] > S[t]).

    This probability is diagnostic/internal only. Model 3 converts its strength into a
    symmetric margin instead of displaying the direction.
    """
    price = features.price
    horizon = parameters.horizon_seconds
    entries = contract_entries(features.number_of_seconds, parameters)
    outcomes = (price[entries + horizon] > price[entries]).astype(float)

    internal_probability = np.full(features.number_of_seconds, 0.5)
    if len(entries) == 0:
        return internal_probability

    if not rolling:
        split_index = max(1, int(parameters.training_fraction * len(entries)))
        fitted = _fit_logistic_probability(_internal_feature_matrix(features, entries[:split_index]),
                                           outcomes[:split_index], parameters)
        _fill_internal_probability(internal_probability, features, 0,
                                   features.number_of_seconds, fitted, parameters)
        return internal_probability

    minimum = min(parameters.internal_minimum_training_contracts, len(entries))
    if minimum < parameters.minimum_samples_per_bin:
        return internal_probability

    retrain_every = max(1, parameters.internal_retrain_contracts)
    window_size = max(minimum, parameters.internal_training_window_contracts)

    for start in range(minimum, len(entries), retrain_every):
        train_start = max(0, start - window_size)
        train_entries = entries[train_start:start]
        fitted = _fit_logistic_probability(_internal_feature_matrix(features, train_entries),
                                           outcomes[train_start:start], parameters)

        segment_start = int(entries[start])
        next_start = min(len(entries), start + retrain_every)
        segment_stop = int(entries[next_start]) if next_start < len(entries) else features.number_of_seconds
        _fill_internal_probability(internal_probability, features, segment_start,
                                   segment_stop, fitted, parameters)

    return internal_probability


def information_margin_from_probability(internal_probability, parameters: ModelParameters) -> np.ndarray:
    """Symmetric margin needed to make the stronger hidden side non-positive EV.

    With displayed P=0.50, decimal odds are 2 * (1 - margin). A bettor with true
    win probability q breaks even at margin = 1 - 1 / (2q).
    """
    p = np.asarray(internal_probability, dtype=float)
    strongest_side_probability = np.maximum(p, 1.0 - p)
    required_margin = 1.0 - 1.0 / (2.0 * strongest_side_probability)
    add_on = required_margin + parameters.information_margin_buffer - parameters.house_margin
    return np.maximum(add_on, 0.0)

# MODEL 3
def build_hidden_signal_symmetric_margin_model(
    features: Features,
    parameters: ModelParameters,
    name: str = "hidden_signal_symmetric_margin_v1",
    rolling: bool = True,
    enable_book_risk_margin: bool = True,
) -> Model:
    """Model 3: hide directional P(up), charge symmetric margin when predictability is high."""
    if enable_book_risk_margin and (
        parameters.book_risk_margin_sensitivity == 0.0
        and parameters.terminal_gamma_margin_sensitivity == 0.0
    ):
        parameters = replace(parameters,
                             book_risk_margin_sensitivity=0.08,
                             terminal_gamma_margin_sensitivity=0.06)

    internal_probability = build_internal_probability(features, parameters, rolling=rolling)
    information_margin = information_margin_from_probability(internal_probability, parameters)
    total_static_margin = np.clip(
        parameters.house_margin + information_margin,
        0.0,
        parameters.maximum_total_margin,
    )
    display_probability = np.full(features.number_of_seconds, 0.5)

    return Model(
        name=name,
        display_probability=display_probability,
        margin=total_static_margin,
        parameters=parameters,
        internal_probability=internal_probability,
        margin_components={"information": information_margin},
        use_book_risk_margin=enable_book_risk_margin,
    )


def run_golden_tests(parameters: ModelParameters = ModelParameters()) -> None:
    """Invariants the pricing must satisfy, independent of any data."""
    margin = parameters.house_margin

    def approximately_equal(a, b, tolerance=1e-6):
        return abs(a - b) < tolerance

    # 1. balanced book, no momentum => symmetric base odds, overround 1/(1-margin)
    odds = quote_odds(0.50, margin, 0.0, parameters)
    assert approximately_equal(odds[0], 2 * (1 - margin)) and approximately_equal(odds[1], 2 * (1 - margin))
    assert approximately_equal(1 / odds[0] + 1 / odds[1], 1 / (1 - margin))

    # 2. EV frontier: no side is ever quoted above its fair odds (house EV >= 0)
    for probability in [0.40, 0.50, 0.58, 0.62]:
        for imbalance in [-0.9, -0.3, 0.0, 0.3, 0.9]:
            odds = quote_odds(probability, margin, imbalance, parameters)
            assert odds[0] <= 1 / probability + 1e-9 and odds[1] <= 1 / (1 - probability) + 1e-9

    # 3. skew direction: long UP (imbalance > 0) shortens UP, lengthens DOWN
    odds = quote_odds(0.50, margin, 0.3, parameters)
    assert odds[0] < 2 * (1 - margin) < odds[1]

    # 4. momentum direction: uptrend (p_up > 0.5) shortens UP vs the symmetric quote
    assert quote_odds(0.58, margin, 0.0, parameters)[0] < quote_odds(0.50, margin, 0.0, parameters)[0]

    # 5. per-bet house edge at the fair probability equals the vig, both sides
    for probability in [0.45, 0.50, 0.58]:
        odds = quote_odds(probability, margin, 0.0, parameters)
        assert approximately_equal(1 - probability * odds[0], margin, 1e-3)
        assert approximately_equal(1 - (1 - probability) * odds[1], margin, 1e-3)

    print("all golden tests pass")


def run_model3_golden_tests(parameters: ModelParameters = ModelParameters()) -> None:
    p = np.array([0.40, 0.50, 0.60])
    margin = information_margin_from_probability(p, parameters)
    assert abs(margin[0] - margin[2]) < 1e-12
    assert margin[1] == 0.0

    total_margin = parameters.house_margin + margin[2]
    odds = quote_odds(0.50, total_margin, 0.0, parameters)
    assert 1.0 - p[2] * odds[0] >= parameters.information_margin_buffer - 1e-9
    assert 1.0 - (1.0 - p[0]) * odds[1] >= parameters.information_margin_buffer - 1e-9

    print("all model3 golden tests pass")
