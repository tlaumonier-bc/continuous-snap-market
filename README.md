Architecture of this repo:

snap-market/
├── README.md
├── data/
│   └── binance_btcusdt_1s_aligned.parquet
│   └── btc_pyth_prices.parquet
├── snapmarket/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── strategies.py
│   └── engine.py
│   └── ...
└── notebooks/
    ├── 01_model_v1.ipynb
    └── 02_attack.ipynb