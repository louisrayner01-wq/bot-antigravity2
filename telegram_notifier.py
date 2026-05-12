"""
telegram_notifier.py — Fortuna trade alerts via Telegram Bot API.

Environment variables (set in Railway):
  TELEGRAM_TOKEN   — bot token from @BotFather
  TELEGRAM_CHAT_ID — target chat/channel ID
  TELEGRAM_ENABLED — "true" / "false"  (default: true)
"""

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"


def _send(text: str):
    if not _ENABLED or not _TOKEN or not _CHAT_ID:
        return
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("Telegram send failed: %s", resp.text)
    except Exception as exc:
        logger.warning("Telegram error: %s", exc)


def _asset_name(raw: str) -> str:
    """Strip exchange suffixes: BTCUSDT_UMCBL → BTC, BTCUSDT → BTC."""
    return raw.replace("USDT_UMCBL", "").replace("USDT_SPBL", "").replace("USDT", "")


def notify_startup(pairs: list, equity: float):
    """Ping on (re)start — confirms Telegram env vars are set."""
    text = (
        f"<b>🚀 Fortuna Bot Started</b>\n"
        f"Pairs  : {', '.join(pairs)}\n"
        f"Equity : £{equity:.2f}"
    )
    _send(text)


def notify_open(symbol: str, side: str, timeframe_label: str,
                entry: float, sl: float, tp: float,
                risk_amount: float, equity: float):
    side_str = "LONG" if side == "long" else "SHORT"
    emoji    = "🟢" if side == "long" else "🔴"
    name     = _asset_name(symbol)
    text = (
        f"<b>{emoji} {side_str} — {name}</b>\n"
        f"Entry : <b>${entry:,.4f}</b>\n"
        f"SL    : ${sl:,.4f}\n"
        f"TP    : ${tp:,.4f}"
    )
    _send(text)


def notify_close(symbol: str, side: str, timeframe_label: str,
                 entry: float, exit_price: float,
                 pnl_usdt: float, equity: float, reason: str):
    won       = pnl_usdt >= 0
    emoji     = "✅" if won else "❌"
    label     = "Take Profit" if "take_profit" in reason else "Stop Loss"
    side_str  = "LONG" if side == "long" else "SHORT"
    name      = _asset_name(symbol)
    sign      = "+" if pnl_usdt >= 0 else ""
    text = (
        f"<b>{emoji} {label} — {name}  {side_str}</b>\n"
        f"Entry : ${entry:,.4f}\n"
        f"Exit  : ${exit_price:,.4f}\n"
        f"P&L   : <b>{sign}£{pnl_usdt:.2f}</b>"
    )
    _send(text)



def notify_model_alert(slot: str, alert_type: str, detail: str):
    icons = {
        "signal_drought":   "⚠️",
        "confidence_drift": "📉",
        "monthly_retrain":  "🔄",
    }
    icon  = icons.get(alert_type, "ℹ️")
    label = alert_type.replace("_", " ").title()
    text = (
        f"<b>{icon} MODEL HEALTH  {label}</b>\n"
        f"Slot   : {slot}\n"
        f"Detail : {detail}"
    )
    _send(text)
