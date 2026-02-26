"""Local query logging utilities for real-time user request persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings


class QueryLogger:
    """SQLite-backed query logger."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            mode TEXT NOT NULL,
            query_text TEXT NOT NULL,
            symbol TEXT,
            amount REAL,
            risk_tolerance TEXT,
            investment_horizon INTEGER,
            success INTEGER NOT NULL,
            latency_seconds REAL,
            error_message TEXT,
            response_excerpt TEXT
        );
        """
        with self._connect() as conn:
            conn.execute(create_sql)
            conn.commit()

    def log_query(
        self,
        mode: str,
        query_text: str,
        success: bool,
        latency_seconds: Optional[float] = None,
        symbol: Optional[str] = None,
        amount: Optional[float] = None,
        risk_tolerance: Optional[str] = None,
        investment_horizon: Optional[int] = None,
        error_message: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> None:
        insert_sql = """
        INSERT INTO query_logs (
            timestamp_utc, mode, query_text, symbol, amount,
            risk_tolerance, investment_horizon, success,
            latency_seconds, error_message, response_excerpt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        excerpt = (response_text or "")[:1000] if response_text else None

        with self._connect() as conn:
            conn.execute(
                insert_sql,
                (
                    timestamp,
                    mode,
                    query_text,
                    symbol,
                    amount,
                    risk_tolerance,
                    investment_horizon,
                    1 if success else 0,
                    latency_seconds,
                    error_message,
                    excerpt,
                ),
            )
            conn.commit()


_query_logger: Optional[QueryLogger] = None


def get_query_logger() -> Optional[QueryLogger]:
    """Get singleton logger if query logging is enabled."""
    global _query_logger

    if not settings.enable_query_logging:
        return None

    if _query_logger is None:
        _query_logger = QueryLogger(settings.query_log_db_path)
    return _query_logger


def log_query_event(
    mode: str,
    query_text: str,
    success: bool,
    latency_seconds: Optional[float] = None,
    symbol: Optional[str] = None,
    amount: Optional[float] = None,
    risk_tolerance: Optional[str] = None,
    investment_horizon: Optional[int] = None,
    error_message: Optional[str] = None,
    response_text: Optional[str] = None,
) -> None:
    """Safe logging wrapper (never raises)."""
    try:
        logger = get_query_logger()
        if logger is None:
            return
        logger.log_query(
            mode=mode,
            query_text=query_text,
            success=success,
            latency_seconds=latency_seconds,
            symbol=symbol,
            amount=amount,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            error_message=error_message,
            response_text=response_text,
        )
    except Exception:
        # Never break agent flow because of logging failures.
        pass
