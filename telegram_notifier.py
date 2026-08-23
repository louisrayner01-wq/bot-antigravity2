"""
telegram_notifier.py — Fortuna trade alerts via Telegram Bot API.

Environment variables (set in Railway):
  TELEGRAM_TOKEN   — bot token from @BotFather
  TELEGRAM_CHAT_ID — target chat/channel ID
  TELEGRAM_ENABLED — "true" / "false"  (default: true)
"""

import logging
import os

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
    return raw.replace("USDT_UMCBL", "").replace("USDT_SPBL", "").replace("USDT", "")


def _planned_rr(entry: float, sl: float, tp: float, side: str) -> float:
    if side == "long":
        risk   = entry - sl
        reward = tp - entry
    else:
        risk   = sl - entry
        reward = entry - tp
    return round(reward / risk, 2) if risk > 0 else 0.0


def notify_startup(pairs: list, equity: float):
    _send(
        f"<b>🚀 Fortuna Bot Started</b>\n"
        f"Pairs : {', '.join(pairs)}"
    )


def _strategy_line(strategy_label: str) -> str:
    """Optional header line so the reader can tell which family fired.
    Kept short — the user only needs to know 'Strat 1' vs 'Portfolio Strategy',
    not which sub-strategy of the portfolio."""
    return f"Strategy : <b>{strategy_label}</b>\n" if strategy_label else ""


def notify_open(symbol: str, side: str, timeframe_label: str,
                entry: float, sl: float, tp: float,
                strategy_label: str = ""):
    side_str = "LONG" if side == "long" else "SHORT"
    emoji    = "🟢" if side == "long" else "🔴"
    name     = _asset_name(symbol)
    rr       = _planned_rr(entry, sl, tp, side)
    _send(
        f"<b>{emoji} {side_str} — {name} [{timeframe_label}]</b>\n"
        f"{_strategy_line(strategy_label)}"
        f"Entry : <b>${entry:,.4f}</b>\n"
        f"SL    : ${sl:,.4f}\n"
        f"TP    : ${tp:,.4f}\n"
        f"RR    : 1:{rr}"
    )


def notify_close(symbol: str, side: str, timeframe_label: str,
                 entry: float, exit_price: float,
                 pnl_usdt: float, reason: str,
                 sl: float = 0.0, tp: float = 0.0,
                 strategy_label: str = ""):
    won      = pnl_usdt >= 0
    emoji    = "✅" if won else "❌"
    result   = "WIN" if won else "LOSS"
    side_str = "LONG" if side == "long" else "SHORT"
    name     = _asset_name(symbol)
    sign     = "+" if pnl_usdt >= 0 else ""

    rr_line = ""
    if sl and tp:
        planned = _planned_rr(entry, sl, tp, side)
        risk    = abs(entry - sl)
        actual  = round(abs(exit_price - entry) / risk, 2) if risk > 0 else 0.0
        rr_line = f"\nRR    : planned 1:{planned}  actual 1:{actual}"

    _send(
        f"<b>{emoji} {result} — {name} [{timeframe_label}]  {side_str}</b>\n"
        f"{_strategy_line(strategy_label)}"
        f"Entry : ${entry:,.4f}\n"
        f"Exit  : ${exit_price:,.4f}\n"
        f"P&L   : <b>{sign}£{pnl_usdt:.2f}</b>"
        f"{rr_line}"
    )


def notify_model_alert(slot: str, alert_type: str, detail: str):
    icons = {
        "signal_drought":   "⚠️",
        "confidence_drift": "📉",
        "monthly_retrain":  "🔄",
    }
    icon  = icons.get(alert_type, "ℹ️")
    label = alert_type.replace("_", " ").title()
    _send(
        f"<b>{icon} MODEL HEALTH — {label}</b>\n"
        f"Slot   : {slot}\n"
        f"Detail : {detail}"
    )
