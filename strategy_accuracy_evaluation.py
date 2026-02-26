"""Strategy Accuracy Evaluation for the Crypto Trading Agent.

Purpose:
Evaluate how accurate/trustworthy the strategy signals are by backtesting the
agent's core signal engine (Momentum + RL combined logic) against future
historical returns.

What this script measures:
1) Directional accuracy:
   - BUY/STRONG_BUY is correct if future return > hold_band
   - SELL/STRONG_SELL is correct if future return < -hold_band
   - HOLD is correct if |future return| <= hold_band
2) Strategy return (%):
   - BUY = +1 position, SELL = -1, HOLD = 0
   - Compares strategy cumulative return vs buy-and-hold
3) Confidence profile:
   - Accuracy split by confidence bucket (high/medium/low)

Data source:
- CoinGecko OHLC from data_fetcher.get_historical_data(...)

Usage:
    python strategy_accuracy_evaluation.py
    python strategy_accuracy_evaluation.py --symbols BTC ETH SOL ADA --days 365 --horizon 7
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer
from rl_predictor import RLPredictor


@dataclass
class EvalPoint:
    symbol: str
    index: int
    signal: str
    confidence: str
    actual_return_pct: float
    is_correct: bool
    position: float
    strategy_return_pct: float


def pct_change(current_value: float, previous_value: float) -> float:
    if previous_value == 0:
        return 0.0
    return ((current_value - previous_value) / previous_value) * 100.0


def classify_signal(score: float) -> tuple[str, str]:
    if score >= 1.5:
        return "strong_buy", "high"
    if score >= 0.5:
        return "buy", "medium"
    if score <= -1.5:
        return "strong_sell", "high"
    if score <= -0.5:
        return "sell", "medium"
    return "hold", "low"


def momentum_signal_from_window(window_df: pd.DataFrame) -> tuple[str, float]:
    current_price = float(window_df.iloc[-1]["close"])

    prev_7 = float(window_df.iloc[-8]["close"]) if len(window_df) >= 8 else float(window_df.iloc[0]["close"])
    prev_30 = float(window_df.iloc[-31]["close"]) if len(window_df) >= 31 else float(window_df.iloc[0]["close"])

    change_7d = pct_change(current_price, prev_7)
    change_30d = pct_change(current_price, prev_30)

    # Estimate 1y annualized trend from available window if < 365 rows.
    if len(window_df) >= 365:
        prev_1y = float(window_df.iloc[-365]["close"])
        change_1y = pct_change(current_price, prev_1y)
    else:
        change_1y = change_30d * 12.0

    momentum_7d = change_7d * 0.5
    momentum_30d = change_30d * 0.3
    momentum_1y = (change_1y / 52.0) * 0.2
    combined_momentum = momentum_7d + momentum_30d + momentum_1y

    # Match trading_agent signal thresholds using expected horizon return estimate.
    expected_return = combined_momentum
    if expected_return > 10:
        return "strong_buy", expected_return
    if expected_return > 5:
        return "buy", expected_return
    if expected_return < -10:
        return "strong_sell", expected_return
    if expected_return < -5:
        return "sell", expected_return
    return "hold", expected_return


def run_symbol_backtest(
    symbol: str,
    df: pd.DataFrame,
    horizon: int,
    warmup: int,
    hold_band: float,
    rl_predictor: RLPredictor,
) -> List[EvalPoint]:
    signal_scores = {
        "strong_sell": -2,
        "sell": -1,
        "hold": 0,
        "buy": 1,
        "strong_buy": 2,
    }

    points: List[EvalPoint] = []

    max_index = len(df) - horizon - 1
    for idx in range(warmup, max_index + 1):
        window_df = df.iloc[: idx + 1].copy()
        current = window_df.iloc[-1]
        current_price = float(current["close"])

        # Build market data for RL feature extraction.
        prev_price = float(window_df.iloc[-2]["close"]) if len(window_df) > 1 else current_price
        change_24h = pct_change(current_price, prev_price)

        if len(window_df) >= 8:
            prev_7d = float(window_df.iloc[-8]["close"])
            momentum_7d = pct_change(current_price, prev_7d)
        else:
            momentum_7d = change_24h * 3

        recent_volume = window_df.iloc[-7:]["volume"]
        avg_vol_7d = float(recent_volume.mean()) if len(recent_volume) > 0 else 0.0
        curr_vol = float(current.get("volume", 0.0))
        volume_ratio = (curr_vol / avg_vol_7d) if avg_vol_7d > 0 else 1.0

        market_data = {
            "price": current_price,
            "change_24h": change_24h,
            "price_change_percentage_7d": momentum_7d,
            "volume_24h": curr_vol,
            "volume_ratio": volume_ratio,
            "high_24h": float(current.get("high", current_price)),
            "low_24h": float(current.get("low", current_price)),
        }

        tech = TechnicalAnalyzer(window_df).get_complete_analysis()
        rl_result = rl_predictor.predict_signal(market_data, tech)

        mom_signal, _ = momentum_signal_from_window(window_df)
        combined_score = (signal_scores[mom_signal] + signal_scores[rl_result["signal"]]) / 2.0
        final_signal, final_confidence = classify_signal(combined_score)

        future_price = float(df.iloc[idx + horizon]["close"])
        forward_return = pct_change(future_price, current_price)

        if final_signal in ("buy", "strong_buy"):
            is_correct = forward_return > hold_band
            position = 1.0
        elif final_signal in ("sell", "strong_sell"):
            is_correct = forward_return < -hold_band
            position = -1.0
        else:
            is_correct = abs(forward_return) <= hold_band
            position = 0.0

        strategy_return = position * forward_return

        points.append(
            EvalPoint(
                symbol=symbol,
                index=idx,
                signal=final_signal,
                confidence=final_confidence,
                actual_return_pct=forward_return,
                is_correct=is_correct,
                position=position,
                strategy_return_pct=strategy_return,
            )
        )

        # Online RL update to mimic learning-through-time.
        action_idx = rl_predictor.actions.index(rl_result["signal"])
        reward = (rl_predictor.action_values[action_idx] / 2.0) * forward_return

        next_idx = min(idx + 1, len(df) - 1)
        next_window = df.iloc[: next_idx + 1].copy()
        next_current = next_window.iloc[-1]
        next_price = float(next_current["close"])
        next_prev_price = float(next_window.iloc[-2]["close"]) if len(next_window) > 1 else next_price
        next_change_24h = pct_change(next_price, next_prev_price)
        next_market_data = {
            "price": next_price,
            "change_24h": next_change_24h,
            "price_change_percentage_7d": next_change_24h * 3,
            "volume_24h": float(next_current.get("volume", 0.0)),
            "volume_ratio": 1.0,
            "high_24h": float(next_current.get("high", next_price)),
            "low_24h": float(next_current.get("low", next_price)),
        }
        next_tech = TechnicalAnalyzer(next_window).get_complete_analysis()
        next_features = rl_predictor._extract_features(next_market_data, next_tech)
        next_state = rl_predictor._get_state_key(next_features)
        rl_predictor.update_from_outcome(rl_result["state"], action_idx, reward, next_state)

    return points


def summarize(points: List[EvalPoint], horizon: int, hold_band: float) -> Dict:
    if not points:
        return {
            "total_signals": 0,
            "accuracy_pct": 0.0,
            "strategy_mean_return_pct": 0.0,
            "strategy_median_return_pct": 0.0,
            "market_mean_forward_return_pct": 0.0,
            "profit_factor": 0.0,
            "horizon_days": horizon,
            "hold_band_pct": hold_band,
            "signals_distribution": {},
            "accuracy_by_confidence": {},
        }

    total = len(points)
    correct = sum(1 for p in points if p.is_correct)
    accuracy = (correct / total) * 100.0

    strategy_returns_pct = np.array([p.strategy_return_pct for p in points], dtype=float)
    market_returns_pct = np.array([p.actual_return_pct for p in points], dtype=float)

    strategy_mean = float(np.mean(strategy_returns_pct)) if len(strategy_returns_pct) else 0.0
    strategy_median = float(np.median(strategy_returns_pct)) if len(strategy_returns_pct) else 0.0
    market_mean = float(np.mean(market_returns_pct)) if len(market_returns_pct) else 0.0

    positive_sum = float(strategy_returns_pct[strategy_returns_pct > 0].sum())
    negative_sum = float(np.abs(strategy_returns_pct[strategy_returns_pct < 0].sum()))
    profit_factor = (positive_sum / negative_sum) if negative_sum > 0 else 0.0

    distribution = Counter(p.signal for p in points)

    conf_stats = defaultdict(list)
    for p in points:
        conf_stats[p.confidence].append(1 if p.is_correct else 0)

    accuracy_by_conf = {
        conf: round((sum(vals) / len(vals)) * 100.0, 2)
        for conf, vals in conf_stats.items()
    }

    return {
        "total_signals": total,
        "accuracy_pct": round(accuracy, 2),
        "strategy_mean_return_pct": round(strategy_mean, 4),
        "strategy_median_return_pct": round(strategy_median, 4),
        "market_mean_forward_return_pct": round(market_mean, 4),
        "profit_factor": round(profit_factor, 4),
        "horizon_days": horizon,
        "hold_band_pct": hold_band,
        "signals_distribution": dict(distribution),
        "accuracy_by_confidence": accuracy_by_conf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy output accuracy using historical backtest.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "ADA"])
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--hold-band", type=float, default=1.5, help="Percent band for HOLD correctness")
    parser.add_argument("--output-json", default="reports/strategy_accuracy_report.json")
    args = parser.parse_args()

    fetcher = get_data_fetcher()

    all_points: List[EvalPoint] = []
    by_symbol_summary: Dict[str, Dict] = {}

    for symbol in args.symbols:
        df = fetcher.get_historical_data(symbol, days=args.days, interval="1d")
        if df.empty or len(df) < args.warmup + args.horizon + 5:
            continue

        df = df.sort_values("timestamp").reset_index(drop=True)
        rl_predictor = RLPredictor(learning_rate=0.1, discount_factor=0.95, epsilon=0.05)

        points = run_symbol_backtest(
            symbol=symbol,
            df=df,
            horizon=args.horizon,
            warmup=args.warmup,
            hold_band=args.hold_band,
            rl_predictor=rl_predictor,
        )
        all_points.extend(points)
        by_symbol_summary[symbol] = summarize(points, args.horizon, args.hold_band)

    overall_summary = summarize(all_points, args.horizon, args.hold_band)

    verdict = "good" if overall_summary["accuracy_pct"] >= 60 else "moderate" if overall_summary["accuracy_pct"] >= 50 else "weak"

    report = {
        "overall": {
            **overall_summary,
            "trustworthiness_verdict": verdict,
            "interpretation": (
                "Strategy appears reasonably reliable in historical directional calls."
                if verdict == "good"
                else "Strategy has mixed reliability; use with strict risk controls."
                if verdict == "moderate"
                else "Strategy reliability is weak on backtest; treat output as low-confidence guidance."
            ),
        },
        "by_symbol": by_symbol_summary,
        "config": {
            "symbols": args.symbols,
            "days": args.days,
            "horizon": args.horizon,
            "warmup": args.warmup,
            "hold_band": args.hold_band,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n✅ Strategy accuracy evaluation completed")
    print(f"Total signals evaluated: {overall_summary['total_signals']}")
    print(f"Accuracy: {overall_summary['accuracy_pct']}%")
    print(f"Mean strategy return / signal: {overall_summary['strategy_mean_return_pct']}%")
    print(f"Mean market forward return: {overall_summary['market_mean_forward_return_pct']}%")
    print(f"Profit factor: {overall_summary['profit_factor']}")
    print(f"Trustworthiness verdict: {verdict}")
    print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
