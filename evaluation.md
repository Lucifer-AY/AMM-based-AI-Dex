# RL Predictor Evaluation

This document explains how RL evaluation works in this project, where data comes from, which data fields are used by the RL predictor, and where the model is saved.

## 1) Evaluation Script

Use the dedicated script:

- `evaluate_rl_model.py`

Example command:

```bash
python evaluate_rl_model.py --symbol BTC --days 180 --warmup 30
```

Outputs:

- Model file: `models/rl_q_table.json`
- Summary report: `reports/rl_evaluation_summary.json`

---

## 2) Where the data is fetched from

The evaluator uses `data_fetcher.py`:

- `get_historical_data(symbol, days, interval='1d')`
- Source API: CoinGecko OHLC endpoint
- Endpoint pattern: `/coins/{coin_id}/ohlc`
- Fields retrieved and used in evaluation candles:
  - `timestamp`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume` (set to `0` in current free OHLC flow)

In online/live inference (`trading_agent.py`), RL-related market context is also built from `get_coin_details(...)` (CoinGecko market_data fields).

---

## 3) What all data is used in RL predictor

The RL predictor (`rl_predictor.py`) consumes:

### A) Market data input

- `price`
- `change_24h`
- `price_change_percentage_7d` (or fallback from `change_24h`)
- `volume_24h`
- `volume_ratio`
- `high_24h`
- `low_24h`

### B) Derived features inside `_extract_features`

- `current_price`
- `momentum_7d`
- `volatility` = `abs(high_24h - low_24h) / price`
- `volume_ratio`
- `rsi` (from technical analysis)
- `ema_trend` (from EMA 9 vs EMA 21 when available)
- `bb_position` (Bollinger position when available)

### C) Technical analysis context (optional but used in evaluation)

Built from `technical_analysis.py` and passed into RL:

- RSI block
- EMA block
- Bollinger Bands block

The state key used by Q-learning is a discretized combination of:

- momentum bucket (`up`, `neutral`, `down`)
- volatility bucket (`high`, `low`)
- RSI bucket (`overbought`, `normal`, `oversold`)
- volume bucket (`high`, `normal`, `low`)

---

## 4) Where the model is saved

The RL “model” is the Q-table (`state -> action values`).

Persistence is implemented in `rl_predictor.py`:

- `save_model(file_path='models/rl_q_table.json')`
- `load_model(file_path='models/rl_q_table.json')`

Default save location:

- `models/rl_q_table.json`

This JSON stores:

- metadata (`learning_rate`, `discount_factor`, `epsilon`, actions)
- learned `q_table`

---

## 5) Evaluation logic summary

`evaluate_rl_model.py` uses a walk-forward loop:

1. Fetch historical OHLC data.
2. Build market + technical features candle by candle.
3. Predict action via RL policy (`predict_signal`).
4. Compute next-candle reward from realized return and action strength.
5. Update Q-table (`update_from_outcome`).
6. Track metrics (directional accuracy, strategy return, reward stats).
7. Save model and JSON report.

---

## 6) Notes

- The RL predictor is lightweight Q-learning (table-based), not a deep neural network.
- Without explicit `save_model(...)`, Q-table remains in memory for that process only.
- You can reuse learned states by calling `load_model(...)` before inference/evaluation.
