"""RL Predictor Model Evaluation Script.

Runs a walk-forward evaluation for the Q-learning predictor using historical
CoinGecko OHLC data, updates Q-values online, and saves the trained Q-table.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer
from rl_predictor import RLPredictor


def _safe_pct_change(current_value: float, previous_value: float) -> float:
    if previous_value == 0:
        return 0.0
    return ((current_value - previous_value) / previous_value) * 100.0


def _build_market_data(df: pd.DataFrame, index: int) -> Dict:
    current = df.iloc[index]
    current_price = float(current["close"])

    prev_price = float(df.iloc[index - 1]["close"]) if index > 0 else current_price
    change_24h = _safe_pct_change(current_price, prev_price)

    if index >= 7:
        base_7d = float(df.iloc[index - 7]["close"])
        momentum_7d = _safe_pct_change(current_price, base_7d)
    else:
        momentum_7d = change_24h * 3

    volume_series = df.iloc[max(0, index - 6): index + 1]["volume"]
    avg_volume_7d = float(volume_series.mean()) if len(volume_series) > 0 else 0.0
    current_volume = float(current.get("volume", 0.0))
    volume_ratio = (current_volume / avg_volume_7d) if avg_volume_7d > 0 else 1.0

    return {
        "price": current_price,
        "change_24h": change_24h,
        "price_change_percentage_7d": momentum_7d,
        "volume_24h": current_volume,
        "volume_ratio": volume_ratio,
        "high_24h": float(current.get("high", current_price)),
        "low_24h": float(current.get("low", current_price)),
    }


def _build_technical_analysis(df: pd.DataFrame, index: int, window: int = 90) -> Dict:
    start = max(0, index - window + 1)
    window_df = df.iloc[start: index + 1].copy()
    analyzer = TechnicalAnalyzer(window_df)
    return analyzer.get_complete_analysis()


def evaluate_predictor(
    symbol: str,
    days: int,
    warmup: int,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    model_output: str,
    summary_output: str,
) -> Tuple[Dict, RLPredictor]:
    fetcher = get_data_fetcher()
    data = fetcher.get_historical_data(symbol, days=days, interval="1d")

    if data.empty:
        raise ValueError("No historical data returned from data_fetcher.")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if len(data) < warmup + 2:
        raise ValueError(
            f"Not enough data for evaluation: got {len(data)} rows, "
            f"need at least {warmup + 2}."
        )

    predictor = RLPredictor(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )

    rewards = []
    realized_returns = []
    strategy_returns = []
    signal_counter = Counter()
    directional_hits = 0
    evaluation_steps = 0

    for index in range(warmup, len(data) - 1):
        market_data = _build_market_data(data, index)
        technical_analysis = _build_technical_analysis(data, index)

        signal = predictor.predict_signal(market_data, technical_analysis)
        action = signal["signal"]
        action_idx = predictor.actions.index(action)
        action_strength = predictor.action_values[action_idx] / 2.0

        current_price = market_data["price"]
        next_price = float(data.iloc[index + 1]["close"])
        realized_return = _safe_pct_change(next_price, current_price)

        reward = realized_return * action_strength

        next_market_data = _build_market_data(data, index + 1)
        next_technical = _build_technical_analysis(data, index + 1)
        next_features = predictor._extract_features(next_market_data, next_technical)
        next_state_key = predictor._get_state_key(next_features)

        predictor.update_from_outcome(signal["state"], action_idx, reward, next_state_key)

        predicted_direction = np.sign(action_strength)
        actual_direction = np.sign(realized_return)
        if predicted_direction == 0:
            if abs(realized_return) < 0.25:
                directional_hits += 1
        elif predicted_direction == actual_direction:
            directional_hits += 1

        signal_counter[action] += 1
        rewards.append(reward)
        realized_returns.append(realized_return)
        strategy_returns.append(realized_return * action_strength)
        evaluation_steps += 1

    strategy_curve = np.cumprod(1 + (np.array(strategy_returns) / 100.0))
    buy_hold_curve = np.cumprod(1 + (np.array(realized_returns) / 100.0))

    summary = {
        "symbol": symbol.upper(),
        "data_points": int(len(data)),
        "evaluation_steps": evaluation_steps,
        "warmup_period": warmup,
        "data_range": {
            "start": str(data.iloc[0]["timestamp"]),
            "end": str(data.iloc[-1]["timestamp"]),
        },
        "model_hyperparameters": {
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
        },
        "metrics": {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "median_reward": float(np.median(rewards)) if rewards else 0.0,
            "directional_accuracy_pct": round((directional_hits / evaluation_steps) * 100, 2) if evaluation_steps else 0.0,
            "strategy_return_pct": round((strategy_curve[-1] - 1) * 100, 2) if len(strategy_curve) else 0.0,
            "buy_hold_return_pct": round((buy_hold_curve[-1] - 1) * 100, 2) if len(buy_hold_curve) else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
        },
        "signals": dict(signal_counter),
        "state_count": len(predictor.q_table),
        "saved_model_path": str(Path(model_output).as_posix()),
    }

    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    predictor.save_model(model_output)
    return summary, predictor


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL predictor with walk-forward backtest.")
    parser.add_argument("--symbol", type=str, default="BTC", help="Crypto symbol, e.g., BTC, ETH, SOL")
    parser.add_argument("--days", type=int, default=180, help="Number of historical days to fetch")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup candles before evaluation starts")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--model-output", type=str, default="models/rl_q_table.json")
    parser.add_argument("--summary-output", type=str, default="reports/rl_evaluation_summary.json")

    args = parser.parse_args()

    summary, _ = evaluate_predictor(
        symbol=args.symbol,
        days=args.days,
        warmup=args.warmup,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        model_output=args.model_output,
        summary_output=args.summary_output,
    )

    print("\n✅ RL evaluation complete")
    print(f"Symbol: {summary['symbol']}")
    print(f"Data points: {summary['data_points']}")
    print(f"Directional accuracy: {summary['metrics']['directional_accuracy_pct']}%")
    print(f"Strategy return: {summary['metrics']['strategy_return_pct']}%")
    print(f"Buy & hold return: {summary['metrics']['buy_hold_return_pct']}%")
    print(f"Q-table states learned: {summary['state_count']}")
    print(f"Model saved to: {summary['saved_model_path']}")


if __name__ == "__main__":
    main()
