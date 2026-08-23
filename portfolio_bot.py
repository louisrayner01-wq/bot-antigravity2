"""
portfolio_bot.py
────────────────
Live paper worker for the Portfolio v1 strategy family.

Runs concurrently with bot_rules.py (Strat 1 family). Both workers share the
same repo but poll separate Fortuna endpoints (strategy_family="portfolio"
vs "strat_1"), maintain independent state files, and post trades tagged
with their own family.

Run with: python portfolio_bot.py

Environment variables (Railway):
  FORTUNA_API_URL            — API base URL for multi-user mode
  BOT_ENGINE_SECRET          — shared secret with the API
  PORTFOLIO_POLL_SECONDS     — override the default 5 min tick cadence
  PORTFOLIO_PAPER_STATE_DIR  — where per-user portfolio state files go
                               (default: ./portfolio_state)
"""

from __future__ import annotations
import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from weex_client import WeexClient
from portfolio.config import (
    STRATEGIES, DEFAULT_POLL_SECONDS, DEFAULT_STARTING_EQUITY,
    DEFAULT_RISK_PER_TRADE, ENTRY_TF, BIAS_TF, DEFAULT_LEVERAGE,
)
from portfolio.engine import PortfolioEngine, ClosedTrade

try:
    import fortuna_client
except Exception:
    fortuna_client = None

try:
    from telegram_notifier import notify_open, notify_close, notify_startup  # type: ignore
except Exception:
    notify_open = notify_close = notify_startup = lambda *a, **k: None


FAMILY = "portfolio"

# CSV log of closed portfolio trades. Read by dashboard.py's /api/trades
# endpoint when the user selects the Portfolio family in the family
# dropdown. Path is env-tunable so Railway can point it at /data.
PORTFOLIO_TRADES_FILE = os.environ.get(
    "PORTFOLIO_TRADES_FILE",
    "/data/portfolio_trades.csv" if os.path.isdir("/data") else "./portfolio_trades.csv",
)

_TRADE_CSV_COLUMNS = [
    "timestamp", "user_id", "strategy_key", "pair", "side",
    "entry_price", "exit_price", "quantity", "leverage",
    "pnl_pct", "pnl_usdt", "candles_held", "exit_reason",
]


def _append_trade_row(user_id: str, trade) -> None:
    """Append one closed trade to the portfolio trades CSV so dashboard can display it."""
    path = Path(PORTFOLIO_TRADES_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(_TRADE_CSV_COLUMNS)
            w.writerow([
                trade.exit_ts,
                user_id,
                trade.strategy_key,
                trade.asset,
                "long" if trade.side > 0 else "short",
                f"{trade.entry_price:.8f}",
                f"{trade.exit_price:.8f}",
                f"{abs(trade.quantity):.8f}",
                trade.leverage,
                f"{trade.pnl_pct:.6f}",
                f"{trade.pnl_usdt:.6f}",
                trade.bars_held,
                trade.exit_reason,
            ])
    except Exception as exc:
        logging.getLogger("portfolio_bot").warning("trade CSV append failed: %s", exc)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    level = os.environ.get("PORTFOLIO_LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)


# ── Candle fetch helper ───────────────────────────────────────────────────────

# Weex granularity: minutes as string.
_TF_TO_GRAN = {
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "4h":  "240",
    "1d":  "1440",
}
# How many bars to pull. Enough history for the 200-bar VPT window, 90d
# tercile rolling on daily, and 60-bar entry tercile. 500 bars of 4h data
# covers ~83 days.
_CANDLE_LIMIT = 500


def make_candle_fetcher(weex: WeexClient):
    """Return a fetch_candles(asset, tf) → DataFrame callable."""
    def fetch(asset: str, tf: str) -> pd.DataFrame:
        gran = _TF_TO_GRAN.get(tf)
        if gran is None:
            return pd.DataFrame()
        try:
            raw = weex.get_candles(asset, granularity=gran, limit=_CANDLE_LIMIT)
        except Exception as exc:
            logging.getLogger("portfolio").warning("get_candles(%s,%s) failed: %s", asset, tf, exc)
            return pd.DataFrame()
        if not raw:
            return pd.DataFrame()
        # Weex sometimes returns >6 columns per candle (quote volume, turnover,
        # buy/sell volume, etc). We only need OHLCV — trim to the first 6.
        rows = [r[:6] for r in raw]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("ts").astype({"open": "float64", "high": "float64",
                                        "low": "float64", "close": "float64",
                                        "volume": "float64"})
        return df
    return fetch


# ── Trade posting to Fortuna ─────────────────────────────────────────────────

def _post_closed_trade(user_id: str, trade: ClosedTrade, equity_after: float, hwm: float) -> None:
    if fortuna_client is None or not os.environ.get("FORTUNA_API_URL"):
        return
    payload = {
        "pair":         trade.asset,
        "slot_key":     f"{trade.strategy_key}::{trade.asset}",
        "side":         "long" if trade.side > 0 else "short",
        "entry_price":  trade.entry_price,
        "exit_price":   trade.exit_price,
        "quantity":     abs(trade.quantity),
        "leverage":     trade.leverage,
        "pnl_pct":      trade.pnl_pct,
        "pnl_usdt":     trade.pnl_usdt,
        "candles_held": trade.bars_held,
    }
    try:
        fortuna_client.post_trade(
            user_id, payload, equity_after, trade.exit_reason,
            strategy_family=FAMILY,
        )
        fortuna_client.post_equity(user_id, equity_after, hwm, strategy_family=FAMILY)
    except Exception as exc:
        logging.getLogger("portfolio").debug("post_trade failed: %s", exc)


# ── Per-user runner ───────────────────────────────────────────────────────────

def make_state_dir() -> Path:
    # Prefer the Railway persistent volume (/data) so state survives deploys.
    # Falls back to a repo-relative dir for local dev. Must stay in lock-step
    # with dashboard.py's PORTFOLIO_PAPER_STATE_DIR default — otherwise the
    # dashboard looks in one place while the bot writes to another.
    default = "/data/portfolio_state" if os.path.isdir("/data") else "./portfolio_state"
    d = Path(os.environ.get("PORTFOLIO_PAPER_STATE_DIR", default))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_weex_client() -> WeexClient:
    """Portfolio bot is paper-only. Public candle endpoints work without keys,
    but WeexClient signs private endpoints when needed. Env-var keys are
    optional for the portfolio bot since we never place real orders — only
    read prices."""
    return WeexClient(
        api_key=os.environ.get("WEEX_API_KEY", ""),
        api_secret=os.environ.get("WEEX_API_SECRET", ""),
        passphrase=os.environ.get("WEEX_PASSPHRASE", ""),
        base_url=os.environ.get("WEEX_BASE_URL", "https://api-spot.weex.com"),
    )


def run_single_user_tick(engine: PortfolioEngine, log: logging.Logger) -> None:
    closed = engine.tick()
    for tr in closed:
        _append_trade_row(engine.user_id, tr)
        _post_closed_trade(engine.user_id, tr, engine.state.equity, engine.state.hwm)
        try:
            notify_close(
                tr.asset, "long" if tr.side > 0 else "short", ENTRY_TF,
                tr.entry_price, tr.exit_price, tr.pnl_usdt, tr.exit_reason,
                sl=0.0, tp=0.0,
            )
        except Exception:
            pass

    # Equity heartbeat even when no trades this tick (so dashboard equity is fresh)
    if fortuna_client is not None and os.environ.get("FORTUNA_API_URL"):
        try:
            fortuna_client.post_equity(
                engine.user_id, engine.state.equity, engine.state.hwm,
                strategy_family=FAMILY,
            )
        except Exception:
            pass

    log.info("[%s] tick done — equity=%.2f open_positions=%d",
             engine.user_id[:8], engine.state.equity, len(engine.state.positions))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    log = logging.getLogger("portfolio_bot")
    poll = int(os.environ.get("PORTFOLIO_POLL_SECONDS", DEFAULT_POLL_SECONDS))
    api_url = os.environ.get("FORTUNA_API_URL", "")

    log.info("🚀 PortfolioBot starting — %d strategies, poll=%ds",
             len(STRATEGIES), poll)
    log.info("Strategies: %s", ", ".join(s.key for s in STRATEGIES))

    weex = build_weex_client()
    fetch_candles = make_candle_fetcher(weex)
    state_dir = make_state_dir()

    # Startup heartbeat — proves the bot at least booted this far. Read by
    # dashboard.py's /api/portfolio-debug so we can distinguish "bot never
    # started" from "bot started but hasn't ticked yet" from "bot crashed
    # mid-tick".
    try:
        hb = state_dir / "_heartbeat.json"
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        hb.write_text(_json.dumps({
            "phase":        "started",
            "state_dir":    str(state_dir),
            "poll_seconds": poll,
            "fortuna_api":  bool(api_url),
            "started_at":   _dt.now(_tz.utc).isoformat(),
        }))
    except Exception as exc:
        log.warning("could not write startup heartbeat: %s", exc)

    # Cache one PortfolioEngine per user across ticks so state stays in-memory.
    engines: Dict[str, PortfolioEngine] = {}

    def get_engine(user_id: str, capital: float, risk: float) -> PortfolioEngine:
        eng = engines.get(user_id)
        if eng is None:
            eng = PortfolioEngine(
                user_id=user_id,
                starting_equity=capital,
                risk_per_trade=risk,
                state_dir=state_dir,
                fetch_candles=fetch_candles,
            )
            engines[user_id] = eng
        else:
            # Keep risk in sync with user's latest slider
            eng.risk_per_trade = risk
        return eng

    def local_tick():
        """Fallback single-user tick — mirrors bot_rules.py behaviour when
        no active Fortuna users exist. Keeps the dashboard populated so we
        can see the strategy scanning locally."""
        uid = os.environ.get("LOCAL_USER_ID", "local")
        cap = float(os.environ.get("LOCAL_CAPITAL", DEFAULT_STARTING_EQUITY))
        risk = float(os.environ.get("LOCAL_RISK", DEFAULT_RISK_PER_TRADE))
        eng = get_engine(uid, cap, risk)
        run_single_user_tick(eng, log)

    import traceback as _tb
    last_error: dict = {}

    while True:
        try:
            users: list = []
            if api_url and fortuna_client is not None:
                users = fortuna_client.get_active_users(strategy_family=FAMILY) or []

            if users:
                log.info("Active portfolio users: %d", len(users))
                for u in users:
                    uid  = u.get("user_id")
                    cap  = float(u.get("capital") or DEFAULT_STARTING_EQUITY)
                    risk = float(u.get("risk_per_trade") or DEFAULT_RISK_PER_TRADE)
                    if not uid:
                        continue
                    eng = get_engine(uid, cap, risk)
                    try:
                        run_single_user_tick(eng, log)
                    except Exception as exc:
                        log.exception("[%s] tick error: %s", uid[:8], exc)
                        last_error = {
                            "where": f"user_tick[{uid[:8]}]",
                            "type":  type(exc).__name__,
                            "msg":   str(exc),
                            "trace": _tb.format_exc(),
                        }
                # Drop engines whose users are no longer active
                active_ids = {u.get("user_id") for u in users}
                for uid in list(engines.keys()):
                    if uid not in active_ids and uid != "local":
                        log.info("[%s] user inactive — dropping engine", uid[:8])
                        engines.pop(uid, None)
            else:
                # No active portfolio users (or Fortuna unreachable) — fall
                # back to a local single-user tick so the bot keeps scanning
                # and the dashboard has data to show.
                log.info("No active portfolio users — running local fallback tick")
                try:
                    local_tick()
                except Exception as exc:
                    log.exception("local_tick error: %s", exc)
                    last_error = {
                        "where": "local_tick",
                        "type":  type(exc).__name__,
                        "msg":   str(exc),
                        "trace": _tb.format_exc(),
                    }
        except KeyboardInterrupt:
            log.info("Interrupted — exiting")
            return
        except Exception as exc:
            log.exception("Loop error (will continue): %s", exc)
            last_error = {
                "where": "outer_loop",
                "type":  type(exc).__name__,
                "msg":   str(exc),
                "trace": _tb.format_exc(),
            }

        # Per-tick heartbeat: written whether or not the tick raised, and
        # includes the last exception (if any) so /api/portfolio-debug can
        # surface it without needing Railway logs.
        try:
            from datetime import datetime as _dt, timezone as _tz
            import json as _json
            (state_dir / "_heartbeat.json").write_text(_json.dumps({
                "phase":       "ticking",
                "last_tick":   _dt.now(_tz.utc).isoformat(),
                "engines":     list(engines.keys()),
                "fortuna_api": bool(api_url),
                "last_error":  last_error or None,
            }))
        except Exception:
            pass

        time.sleep(poll)


if __name__ == "__main__":
    main()
