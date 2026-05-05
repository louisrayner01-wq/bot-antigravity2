"""
telegram_notifier.py — Trade alert notifications via Telegram Bot API.

Environment variables (set in Railway):
  TELEGRAM_TOKEN   — bot token from @BotFather
  TELEGRAM_CHAT_ID — target chat/channel ID
  TELEGRAM_ENABLED — "true" / "false"  (default: true)
"""

import os
import csv
import logging
from datetime import datetime, timezone, timedelta

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


def notify_daily_summary(trades_csv: str):
    """
    Send a 24-hour trading summary to Telegram.
    Reads the trades CSV for the past 24 hours.
    If no trades were taken, reads the skipped-signals CSV to explain why.
    Called once per day at 02:29 UK time from the main loop.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    # ── Read completed trades from the last 24 h ──────────────────────────────
    trades = []
    if os.path.exists(trades_csv):
        try:
            with open(trades_csv, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts >= cutoff:
                            trades.append(row)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Daily summary: could not read trades CSV: %s", exc)

    now_str = datetime.now(timezone.utc).strftime("%d %b %Y")

    # ── No trades taken ───────────────────────────────────────────────────────
    if not trades:
        reason = _no_trade_reason(trades_csv, cutoff)
        text = (
            f"<b>📊 Daily Summary — {now_str}</b>\n\n"
            f"No trades taken in the last 24 hours.\n\n"
            f"<i>{reason}</i>"
        )
        _send(text)
        return

    # ── Build stats ───────────────────────────────────────────────────────────
    total    = len(trades)
    wins     = [t for t in trades if float(t.get("pnl_usdt", 0)) > 0]
    losses   = [t for t in trades if float(t.get("pnl_usdt", 0)) <= 0]
    total_pnl = sum(float(t.get("pnl_usdt", 0)) for t in trades)
    win_rate  = len(wins) / total * 100 if total else 0

    best  = max(trades, key=lambda t: float(t.get("pnl_usdt", 0)))
    worst = min(trades, key=lambda t: float(t.get("pnl_usdt", 0)))

    pnl_emoji  = "📈" if total_pnl >= 0 else "📉"
    result_str = f"+£{total_pnl:.2f}" if total_pnl >= 0 else f"-£{abs(total_pnl):.2f}"

    # ── Per-trade breakdown ───────────────────────────────────────────────────
    lines = []
    for t in trades:
        pnl   = float(t.get("pnl_usdt", 0))
        emoji = "✅" if pnl > 0 else "❌"
        pair  = t.get("pair", "").replace("USDT_UMCBL", "")
        side  = t.get("side", "").upper()
        sign  = "+" if pnl >= 0 else ""
        lines.append(f"  {emoji} {pair} {side}  {sign}£{pnl:.2f}")

    trades_block = "\n".join(lines)

    text = (
        f"<b>{pnl_emoji} Daily Summary — {now_str}</b>\n\n"
        f"Trades  : <b>{total}</b>  ({len(wins)}W / {len(losses)}L)\n"
        f"Win rate: <b>{win_rate:.0f}%</b>\n"
        f"P&L     : <b>{result_str}</b>\n\n"
        f"<b>Breakdown:</b>\n{trades_block}\n\n"
        f"Best  : {best.get('pair','').replace('USDT_UMCBL','')} "
        f"+£{float(best.get('pnl_usdt',0)):.2f}\n"
        f"Worst : {worst.get('pair','').replace('USDT_UMCBL','')} "
        f"£{float(worst.get('pnl_usdt',0)):.2f}"
    )
    _send(text)


def _no_trade_reason(trades_csv: str, cutoff: datetime) -> str:
    """Inspect skipped-signals CSV to explain why no trades were taken."""
    skipped_csv = trades_csv.replace(".csv", "_skipped.csv")
    if not os.path.exists(skipped_csv):
        return "Market conditions did not produce any qualifying signals."

    skipped = []
    try:
        with open(skipped_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        skipped.append(row)
                except Exception:
                    pass
    except Exception:
        pass

    if not skipped:
        return "Market in consolidation — no signals fired across any pairs."

    # Check most common skip reason
    reasons = [r.get("skip_reason", "") for r in skipped]
    if any("position" in r.lower() for r in reasons):
        return (
            f"{len(skipped)} signal(s) fired but were skipped — "
            "an existing position was already open."
        )

    # Low confidence signals fired but didn't pass threshold
    confidences = []
    for r in skipped:
        try:
            confidences.append(float(r.get("confidence", 0)))
        except Exception:
            pass
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        return (
            f"{len(skipped)} signal(s) fired but confidence was too low "
            f"(avg {avg_conf:.0%}) — market likely in consolidation."
        )

    return f"{len(skipped)} signal(s) fired but did not meet entry criteria."


def notify_model_alert(slot: str, alert_type: str, detail: str):
    """
    Send a model health alert. Called for:
      • signal_drought  — slot has been silent for too long
      • confidence_drift — model confidence collapsing vs baseline
      • monthly_retrain  — scheduled retrain completed
    """
    icons = {
        "signal_drought":   "⚠️",
        "confidence_drift": "📉",
        "monthly_retrain":  "🔄",
    }
    icon = icons.get(alert_type, "ℹ️")
    label = alert_type.replace("_", " ").title()
    text = (
        f"<b>{icon} MODEL HEALTH  {label}</b>\n"
        f"Slot   : {slot}\n"
        f"Detail : {detail}"
    )
    _send(text)
