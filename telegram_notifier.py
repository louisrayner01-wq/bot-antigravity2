"""
telegram_notifier.py — Trade alert notifications via Telegram Bot API.

Environment variables (set in Railway):
  TELEGRAM_TOKEN   — bot token from @BotFather
  TELEGRAM_CHAT_ID — target chat/channel ID
  TELEGRAM_ENABLED — "true" / "false"  (default: true)
"""

import os
import logging

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


def notify_open(symbol: str, side: str, timeframe_label: str,
                entry: float, sl: float, tp: float,
                risk_amount: float, equity: float):
    """Send alert when a trade is opened."""
    pct_risked = (risk_amount / equity * 100) if equity > 0 else 0.0
    side_str   = "LONG" if side == "long" else "SHORT"
    name       = symbol.replace("USDT_UMCBL", "")

    text = (
        f"<b>{'🟢' if side == 'long' else '🔴'} {side_str}  {name}  [{timeframe_label}]</b>\n"
        f"Entry  : <b>${entry:,.4f}</b>\n"
        f"SL     : ${sl:,.4f}\n"
        f"TP     : ${tp:,.4f}\n"
        f"Risked : {pct_risked:.2f}% of account"
    )
    _send(text)


def notify_close(symbol: str, side: str, timeframe_label: str,
                 entry: float, exit_price: float,
                 pnl_usdt: float, equity: float, reason: str):
    """Send alert when a trade is closed."""
    pct_account = (pnl_usdt / equity * 100) if equity > 0 else 0.0
    won         = pnl_usdt >= 0
    emoji       = "✅" if won else "❌"
    name        = symbol.replace("USDT_UMCBL", "")
    reason_str  = reason.replace("_", " ").title()

    text = (
        f"<b>{emoji} CLOSED  {name}  [{timeframe_label}]  ({reason_str})</b>\n"
        f"Entry  : ${entry:,.4f}\n"
        f"Exit   : ${exit_price:,.4f}\n"
        f"P&L    : <b>{pct_account:+.2f}% of account</b>"
    )
    _send(text)
