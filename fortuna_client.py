"""
fortuna_client.py
Talks to the Fortuna API on behalf of the bot engine.
Fetches active users + their configs, posts trade results and equity updates.

Everything is scoped by `strategy_family` so Strat 1 and Portfolio Strategy
workers hit their own rows on the backend.
"""

import os
import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

API_URL    = os.environ.get("FORTUNA_API_URL", "")
BOT_SECRET = os.environ.get("BOT_ENGINE_SECRET", "")

_HEADERS = {
    "x-bot-secret": BOT_SECRET,
    "Content-Type":  "application/json",
}


def _get(path: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(API_URL + path, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Fortuna API GET %s failed: %s", path, exc)
        return None


def _post(path: str, data: dict, params: Optional[dict] = None) -> bool:
    try:
        resp = requests.post(API_URL + path, json=data, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Fortuna API POST %s failed: %s", path, exc)
        return False


def get_active_users(strategy_family: str = "strat_1") -> List[Dict]:
    """
    Returns list of users with an active bot in the given family:
      [{user_id, strategy_family, capital, strategy_mode, risk_per_trade}, ...]
    Empty list if API is unavailable (bot falls back to single-user mode).
    """
    if not API_URL:
        return []
    data = _get("/api/bot/internal/active-users", params={"strategy_family": strategy_family})
    return data if isinstance(data, list) else []


def get_user_config(user_id: str, strategy_family: str = "strat_1") -> Optional[Dict]:
    """
    Returns {user_id, strategy_family, capital, api_key, api_secret, passphrase,
             strategy_mode, risk_per_trade} for one user + family.
    Returns None if user not found or API unavailable.
    """
    if not API_URL:
        return None
    return _get(
        f"/api/bot/internal/user-config/{user_id}",
        params={"strategy_family": strategy_family},
    )


def post_trade(
    user_id: str,
    trade: dict,
    equity_after: float,
    exit_reason: str,
    strategy_family: str = "strat_1",
) -> bool:
    """Post a completed trade so the dashboard shows it (attributed to family)."""
    if not API_URL:
        return False
    payload = {
        "user_id":         user_id,
        "strategy_family": strategy_family,
        "pair":            trade.get("pair", ""),
        "slot_key":        trade.get("slot_key", ""),
        "side":            trade.get("side", ""),
        "entry_price":     trade.get("entry_price", 0),
        "exit_price":      trade.get("exit_price", 0),
        "quantity":        trade.get("quantity", 0),
        "leverage":        trade.get("leverage", 1),
        "confidence":      trade.get("confidence", 0),
        "pnl_pct":         trade.get("pnl_pct", 0),
        "pnl_usdt":        trade.get("pnl_usdt", 0),
        "candles_held":    trade.get("candles_held", 0),
        "exit_reason":     exit_reason,
        "equity_after":    equity_after,
        "mae_pct":         trade.get("mae_pct", 0),
        "mfe_pct":         trade.get("mfe_pct", 0),
        "wick_breach":     trade.get("wick_breach", 0),
    }
    return _post("/api/trades/internal", payload)


def post_equity(
    user_id: str,
    equity: float,
    hwm: float,
    strategy_family: str = "strat_1",
) -> bool:
    """Update a user's equity and HWM in the dashboard for the given family."""
    if not API_URL:
        return False
    return _post(
        f"/api/bot/internal/equity/{user_id}",
        {"equity": equity, "hwm": hwm},
        params={"strategy_family": strategy_family},
    )
