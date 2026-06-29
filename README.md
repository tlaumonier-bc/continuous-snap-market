# Snap Market

Continuous-snap trading research toolkit. The house quotes second-by-second up/down odds
on a price feed; models decide the displayed probability and the margin, and a shared
engine settles bets, skews odds for inventory, and caps net delta. Every model prices,
simulates, and is attacked through the same code, so models are directly comparable.

## Layout

```
snap-market/
  data/
    btc_pyth_prices.parquet
    binance_btcusdt_1s_aligned.parquet
  snapmarket/
    parameters.py            # SharedParameters: settings every model needs
    pricing.py               # quote_odds: the shared pricing math
    model.py                 # Model runtime object + BookRiskParameters
    registry.py              # register/build/list models by name
    features.py              # per-second features from a price series
    data.py                  # price loaders
    strategies.py            # bettor strategies (attackers and pools)
    experiments.py           # run one experiment across several models
    models/                  # one self-contained package per model
      momentum_lookup/
      hidden_symmetric_margin/
    engine/
      simulation.py          # dynamic book simulation
      book_risk.py           # live-book risk margin (opt-in per model)
      contract_table.py      # static contract-level view
  tests/
  notebooks/
    01_model_v1.ipynb
    02_attack.ipynb
```

## Selecting models

```python
from snapmarket import SharedParameters, build_features, load_oracle_prices
from snapmarket import build_model, available_model_names

shared_parameters = SharedParameters()
features = build_features(load_oracle_prices(), shared_parameters)

model = build_model("momentum_lookup", features, shared_parameters)
print(available_model_names())   # ['hidden_symmetric_margin', 'momentum_lookup']
```

## Running one experiment on several models

```python
from snapmarket import run_across_models
from snapmarket.engine import simulate
from snapmarket.strategies import noise_pool

def experiment(model):
    return simulate(model, features, {"noise": noise_pool()}, start_index=500_000,
                    number_of_steps=100_000).house_edge

edges = run_across_models(None, features, shared_parameters, experiment)
```

## Parameters

- `SharedParameters` (in `snapmarket/parameters.py`) holds everything features, pricing,
  and the engine need regardless of the model: the contract horizon, the feature windows,
  the pricing settings, and the net-delta cap.
- Each model has its own parameter dataclass under
  `snapmarket/models/<name>/parameters.py` for settings only that model uses.

`build_model(name, features, shared_parameters)` uses the model's default parameters; pass
a model parameter instance as the fourth argument to override them.

## Adding a model

1. Create `snapmarket/models/<name>/` with:
   - `parameters.py`: a frozen dataclass of the settings only your model uses.
   - one or more small modules holding the model math.
   - `builder.py`: a `build(features, shared_parameters, model_parameters) -> Model`
     function that calls `register_model(...)`.
2. Import the new package in `snapmarket/models/__init__.py` so it registers on import.
3. The model is now selectable by name everywhere; no other file changes.

## Tests

```
python -m pytest tests/
```
