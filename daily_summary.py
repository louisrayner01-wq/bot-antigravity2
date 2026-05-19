"""
daily_summary.py — Combined Telegram daily summary for all running bots.

Both M1 and M2 call send_if_due() at 02:29 UK time.
A shared date-stamp file ensures only one sends per calendar day regardless
of which bot reaches 02:29 first.

CSV field conventions:
  M1 (bot-antigravity2): asset column = "pair"  (e.g. BTCUSDT_UMCBL)
  M2 (bot-fvg):          asset column = "asset" (e.g. BTCUSDT)
"""

import csv
import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Persist the sent-flag on the Railway /data volume so redeploys don't reset it.
# Fall back to the local directory when running outside Railway.
_DATA_DIR  = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
SENT_FLAG  = os.path.join(_DATA_DIR, "last_summary_date.txt")

M1_CSV = "/data/trades.csv"                             # Railway volume path
M2_CSV = "/Users/Louis/bot-fvg/data/trades_m2.csv"     # local path


def _asset_name(raw: str) -> str:
    return raw.replace("USDT_UMCBL", "").replace("USDT_SPBL", "").replace("USDT", "")


def _already_sent_today() -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not os.path.exists(SENT_FLAG):
        return False
    with open(SENT_FLAG) as f:
        return f.read().strip() == today


def _mark_sent():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(SENT_FLAG, "w") as f:
        f.write(today)


def _read_trades(csv_path: str, asset_field: str) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        rows.append({
                            "asset": _asset_name(row.get(asset_field, "")),
                            "side":  row.get("side", "").upper(),
                            "pnl":   float(row.get("pnl_usdt", 0)),
                            "rr":    float(row.get("rr", 0) or 0),
                        })
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("daily_summary: could not read %s: %s", csv_path, exc)
    return rows


def _send(text: str):
    import os as _os
    token   = _os.getenv("TELEGRAM_TOKEN", "")
    chat_id = _os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = _os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
    if not enabled or not token or not chat_id:
        return
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("Telegram daily summary failed: %s", resp.text)
    except Exception as exc:
        logger.warning("Telegram daily summary error: %s", exc)


def send_if_due(m1_csv: str = M1_CSV, m2_csv: str = M2_CSV) -> bool:
    """
    Send the combined daily summary if it hasn't been sent today.
    Returns True if the summary was sent this call.
    """
    if _already_sent_today():
        return False

    _mark_sent()   # mark before sending — prevents a second bot racing in

    trades = _read_trades(m1_csv, "pair") + _read_trades(m2_csv, "asset")
    now_str = datetime.now(timezone.utc).strftime("%d %b %Y")

    if not trades:
        _send(
            f"<b>📊 Daily Summary — {now_str}</b>\n\n"
            f"No trades closed in the last 24 hours."
        )
        return True

    total     = len(trades)
    wins      = sum(1 for t in trades if t["pnl"] > 0)
    losses    = total - wins
    total_pnl = sum(t["pnl"] for t in trades)
    wr        = wins / total * 100
    avg_rr    = sum(t["rr"] for t in trades) / total if total else 0

    lines = []
    for t in trades:
        em   = "✅" if t["pnl"] > 0 else "❌"
        sign = "+" if t["pnl"] >= 0 else ""
        rr_s = f"  RR {t['rr']:+.2f}R" if t["rr"] != 0 else ""
        lines.append(f"  {em} {t['asset']} {t['side']}  {sign}£{t['pnl']:.2f}{rr_s}")

    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    pnl_str   = f"+£{total_pnl:.2f}" if total_pnl >= 0 else f"-£{abs(total_pnl):.2f}"

    _send(
        f"<b>{pnl_emoji} Daily Summary — {now_str}</b>\n\n"
        f"Trades   : <b>{total}</b>  ({wins}W / {losses}L)\n"
        f"Win rate : <b>{wr:.0f}%</b>\n"
        f"Avg RR   : <b>{avg_rr:+.2f}R</b>\n"
        f"P&L      : <b>{pnl_str}</b>\n\n"
        + "\n".join(lines)
    )
    return True
