"""Prediction feedback loop for live self-improvement.

Flow:
1) Store each generated prediction in a local JSON file.
2) When prediction horizon is reached, compare predicted direction vs actual move.
3) Update RL Q-table with reward from real outcome.
4) Persist updated RL model.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from config import settings
from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer
from rl_predictor import get_rl_predictor


def _store_path() -> Path:
    return Path(settings.prediction_feedback_store_path)


def _load_records() -> List[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    if isinstance(payload, dict):
        return payload.get("predictions", [])
    if isinstance(payload, list):
        return payload
    return []


def _save_records(records: List[Dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": records,
    }

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _signal_position(signal: str) -> float:
    signal = (signal or "hold").lower()
    if signal == "strong_buy":
        return 1.0
    if signal == "buy":
        return 0.5
    if signal == "sell":
        return -0.5
    if signal == "strong_sell":
        return -1.0
    return 0.0


def _is_signal_correct(signal: str, realized_return_pct: float, hold_band: float) -> bool:
    signal = (signal or "hold").lower()

    if signal in ("buy", "strong_buy"):
        return realized_return_pct > hold_band
    if signal in ("sell", "strong_sell"):
        return realized_return_pct < -hold_band
    return abs(realized_return_pct) <= hold_band


def store_prediction(
    symbol: str,
    entry_price: float,
    horizon_days: int,
    final_signal: str,
    confidence: str,
    predicted_price: float,
    query_text: str,
    rl_state: str | None = None,
    rl_action_idx: int | None = None,
    rl_action_strength: float | None = None,
) -> None:
    """Persist a new prediction record to local file storage."""
    if not settings.enable_prediction_feedback_loop:
        return

    now = datetime.now(timezone.utc)
    due_at = now + timedelta(days=max(1, int(horizon_days)))

    safe_action_idx = int(rl_action_idx) if rl_action_idx is not None else None
    safe_action_strength = float(rl_action_strength) if rl_action_strength is not None else None

    record = {
        "id": str(uuid4()),
        "created_at": now.isoformat(),
        "due_at": due_at.isoformat(),
        "status": "pending",
        "symbol": symbol.upper(),
        "query_text": query_text,
        "entry_price": float(entry_price),
        "horizon_days": int(horizon_days),
        "final_signal": final_signal,
        "confidence": confidence,
        "predicted_price": float(predicted_price),
        "rl_state": rl_state,
        "rl_action_idx": safe_action_idx,
        "rl_action_strength": safe_action_strength,
    }

    records = _load_records()
    records.append(record)
    _save_records(records)


def process_due_predictions(max_items: int = 20) -> Dict[str, Any]:
    """Evaluate due predictions and update RL model from real outcomes."""
    if not settings.enable_prediction_feedback_loop:
        return {"processed": 0, "updated": 0}

    now = datetime.now(timezone.utc)
    hold_band = float(settings.prediction_feedback_hold_band_pct)

    records = _load_records()
    if not records:
        return {"processed": 0, "updated": 0}

    fetcher = get_data_fetcher()
    rl_predictor = get_rl_predictor()

    processed = 0
    updated = 0

    for record in records:
        if processed >= max_items:
            break
        if record.get("status") != "pending":
            continue

        due_at_raw = record.get("due_at")
        if not due_at_raw:
            continue

        due_at = _parse_iso(due_at_raw)
        if due_at > now:
            continue

        symbol = record.get("symbol", "").upper()
        entry_price = float(record.get("entry_price", 0) or 0)
        if not symbol or entry_price <= 0:
            record["status"] = "invalid"
            processed += 1
            continue

        try:
            coin_details = fetcher.get_coin_details(symbol)
            current_price = float(coin_details.get("price", 0) or 0)
            if current_price <= 0:
                continue

            realized_return_pct = ((current_price - entry_price) / entry_price) * 100
            final_signal = record.get("final_signal", "hold")
            is_correct = _is_signal_correct(final_signal, realized_return_pct, hold_band)

            # RL update (if state/action available)
            rl_state = record.get("rl_state")
            rl_action_idx = record.get("rl_action_idx")
            rl_action_strength = float(record.get("rl_action_strength", 0) or 0)

            if rl_state and isinstance(rl_action_idx, int):
                historical_data = fetcher.get_historical_data(symbol, days=90, interval="1d")
                technical_analysis = (
                    TechnicalAnalyzer(historical_data).get_complete_analysis()
                    if not historical_data.empty
                    else None
                )

                market_data = {
                    "price": current_price,
                    "change_24h": coin_details.get("price_change_percentage_24h", 0),
                    "price_change_percentage_7d": coin_details.get("price_change_percentage_7d", 0),
                    "volume_24h": coin_details.get("volume_24h", 0),
                    "high_24h": coin_details.get("high_24h", current_price),
                    "low_24h": coin_details.get("low_24h", current_price),
                }

                next_features = rl_predictor._extract_features(market_data, technical_analysis)
                next_state = rl_predictor._get_state_key(next_features)

                position = rl_action_strength / 2.0 if rl_action_strength else _signal_position(final_signal)
                reward = realized_return_pct * position

                rl_predictor.update_from_outcome(rl_state, rl_action_idx, reward, next_state)
                updated += 1

            record["status"] = "evaluated"
            record["evaluated_at"] = now.isoformat()
            record["actual_price"] = current_price
            record["actual_return_pct"] = round(realized_return_pct, 4)
            record["is_correct"] = is_correct
            record["hold_band_pct"] = hold_band
            processed += 1

        except Exception as exc:
            record["status"] = "evaluation_error"
            record["evaluated_at"] = now.isoformat()
            record["error"] = str(exc)
            processed += 1

    if updated > 0:
        rl_predictor.save_model(settings.rl_model_path)

    _save_records(records)
    return {"processed": processed, "updated": updated}
