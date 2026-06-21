"""
bot_rules.py
────────────
Standalone trading bot that runs ONE of the rule-based strategies:
  - conservative  (max 1 position, priority ETH→SOL→BTC)
  - balanced      (max 2 positions, priority SOL→ETH→BTC)
  - aggressive    (take all signals, up to 3 positions)

How to run:
  python bot_rules.py

Strategy and risk are configured via the `strategy_rules:` section of
config.yaml, with environment variable overrides:
  STRATEGY_MODE = conservative | balanced | aggressive
  RISK_PER_TRADE = 0.005 .. 0.02
  PAPER_TRADING  = true | false

This bot is independent from bot.py — it has its own state file
(`risk_state_rules.json`) so the two can never interfere.
"""

from __future__ import annotations
import logging
import os
import sys
import time
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

# Existing infra (shared with the ML bot)
from weex_client     import WeexClient
from indicators      import candles_to_df
from trade_logger    import TradeLogger

# New rule-based components
from strategy_rules    import (
    StrategyController, evaluate_signal, compute_daily_btc_bias,
    Signal, LONG, SHORT, FLAT, normalise_symbol,
)
from risk_manager_rules import RuleRiskManager, RulePosition

# Optional integrations — gracefully no-op if unavailable
try:
    import fortuna_client
except Exception:
    fortuna_client = None

try:
    from telegram_notifier import notify_open, notify_close, notify_startup
except Exception:
    notify_open = notify_close = notify_startup = lambda *a, **k: None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Config + logging
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Exchange creds — env overrides for Railway
    if os.getenv("WEEX_API_KEY"):
        cfg["exchange"]["api_key"]    = os.environ["WEEX_API_KEY"]
    if os.getenv("WEEX_API_SECRET"):
        cfg["exchange"]["api_secret"] = os.environ["WEEX_API_SECRET"]
    if os.getenv("WEEX_PASSPHRASE"):
        cfg["exchange"]["passphrase"] = os.environ["WEEX_PASSPHRASE"]
    if os.getenv("BASE_URL"):
        cfg["exchange"]["base_url"]   = os.environ["BASE_URL"]

    # Paper trading toggle
    if os.getenv("PAPER_TRADING") is not None:
        cfg["trading"]["paper_trading"] = (
            os.getenv("PAPER_TRADING", "true").lower() == "true"
        )

    # Rule-strategy section — created lazily if missing
    cfg.setdefault("strategy_rules", {})
    sr = cfg["strategy_rules"]
    sr.setdefault("strategy_mode",    "conservative")
    sr.setdefault("risk_per_trade",   0.01)          # 1 % default — user-recommended start
    sr.setdefault("max_leverage",     10)
    sr.setdefault("daily_loss_pct",   0.05)
    sr.setdefault("hwm_drawdown_pct", 0.25)
    sr.setdefault("poll_seconds",     300)            # 5 minutes
    sr.setdefault("min_holding_bars", 1)
    sr.setdefault("assets", [
        {"base": "BTC", "symbol": "BTCUSDT_UMCBL"},
        {"base": "ETH", "symbol": "ETHUSDT_UMCBL"},
        {"base": "SOL", "symbol": "SOLUSDT_UMCBL"},
    ])

    # Env overrides (highest priority)
    if os.getenv("STRATEGY_MODE"):
        sr["strategy_mode"] = os.environ["STRATEGY_MODE"].lower()
    if os.getenv("RISK_PER_TRADE"):
        sr["risk_per_trade"] = float(os.environ["RISK_PER_TRADE"])

    return cfg


def setup_logging(cfg: dict) -> None:
    log_file = cfg["logging"].get("log_file", "logs/bot_rules.log")
    log_file = log_file.replace("bot.log", "bot_rules.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    level = getattr(logging, cfg["logging"].get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main bot
# ─────────────────────────────────────────────────────────────────────────────

class RuleBot:

    def __init__(self, config_path: str = "config.yaml",
                 user_id: str = "", user_override: Optional[dict] = None):
        self.cfg = load_config(config_path)
        setup_logging(self.cfg)
        self.log = logging.getLogger(f"RuleBot[{user_id[:8]}]" if user_id else "RuleBot")
        self.user_id = user_id

        # Apply per-user Fortuna overrides
        if user_override:
            if user_override.get("api_key"):
                self.cfg["exchange"]["api_key"]    = user_override["api_key"]
                self.cfg["exchange"]["api_secret"] = user_override["api_secret"]
                self.cfg["exchange"]["passphrase"] = user_override.get("passphrase", "")
            if user_override.get("capital"):
                self.cfg["risk"]["initial_capital"] = float(user_override["capital"])
            # Per-user strategy / risk overrides from dashboard
            if user_override.get("strategy_mode"):
                self.cfg["strategy_rules"]["strategy_mode"] = user_override["strategy_mode"]
            if user_override.get("risk_per_trade") is not None:
                self.cfg["strategy_rules"]["risk_per_trade"] = float(user_override["risk_per_trade"])

        ec = self.cfg["exchange"]
        rc = self.cfg["risk"]
        sr = self.cfg["strategy_rules"]
        tc = self.cfg["trading"]

        self.paper = bool(tc.get("paper_trading", True))
        self.assets: List[Dict] = sr["assets"]                 # [{base,symbol}, ...]
        self.poll_seconds = int(sr.get("poll_seconds", 300))
        self.min_holding_bars = int(sr.get("min_holding_bars", 1))

        # Exchange
        self.client = WeexClient(
            api_key=ec["api_key"], api_secret=ec["api_secret"],
            passphrase=ec["passphrase"], base_url=ec["base_url"],
        )

        # Strategy + risk
        mode = str(sr["strategy_mode"]).lower()
        risk_pct = float(sr["risk_per_trade"])
        risk_pct = max(0.005, min(0.02, risk_pct))   # clamp to 0.5%–2%

        self.controller = StrategyController.from_preset(mode)
        self.risk = RuleRiskManager(
            initial_capital=float(rc.get("initial_capital", 100.0)),
            risk_per_trade=risk_pct,
            max_concurrent=self.controller.max_concurrent,
            max_leverage=int(sr.get("max_leverage", 10)),
            daily_loss_pct=float(sr.get("daily_loss_pct", 0.05)),
            hwm_drawdown_pct=float(sr.get("hwm_drawdown_pct", 0.25)),
            data_dir=os.path.join(rc.get("data_dir", "/data"), user_id) if user_id
                     else self.cfg.get("data", {}).get("data_dir", "/data"),
        )

        # Trade logger reuses the existing CSV format so the dashboard works
        trades_path = self.cfg["logging"].get("trades_file", "/data/trades.csv")
        self.logger = TradeLogger(trades_path.replace("trades.csv", "trades_rules.csv"))

        # Last 4h candle timestamp seen per asset — used to gate entry checks
        self._last_4h_ts: Dict[str, pd.Timestamp] = {}

        mode_emoji = "🟡 PAPER" if self.paper else "🔴 LIVE"
        self.log.info(
            "%s | Strategy=%s | Risk=%.2f%% | Max concurrent=%d | "
            "Assets=%s | Account=£%.2f",
            mode_emoji, mode.upper(), risk_pct * 100,
            self.controller.max_concurrent,
            ",".join(a["base"] for a in self.assets),
            self.risk.equity,
        )

        try:
            notify_startup(mode_emoji, self.risk.equity, len(self.assets))
        except Exception:
            pass

    # ── Data helpers ─────────────────────────────────────────────────────────

    def fetch_4h(self, symbol: str, limit: int = 120) -> Optional[pd.DataFrame]:
        """4h candles (granularity=240 minutes)."""
        raw = self.client.get_candles(symbol, granularity="240", limit=limit)
        if not raw:
            return None
        return candles_to_df(raw)

    def fetch_1d(self, symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
        """1d candles (granularity=1440 minutes)."""
        raw = self.client.get_candles(symbol, granularity="1440", limit=limit)
        if not raw:
            return None
        return candles_to_df(raw)

    def live_price(self, symbol: str, fallback_df: Optional[pd.DataFrame] = None) -> Optional[float]:
        """Mark/last price for the symbol, with candle close as a fallback."""
        tick = self.client.get_ticker(symbol)
        if tick:
            for field in ("markPrice", "lastPr", "last", "close"):
                v = tick.get(field)
                if v:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
        if fallback_df is not None and not fallback_df.empty:
            return float(fallback_df["close"].iloc[-1])
        return None

    # ── Execution ────────────────────────────────────────────────────────────

    def _futures_order(self, symbol: str, side: str, qty: float, price: float) -> Optional[str]:
        """side: 'open_long'|'open_short'|'close_long'|'close_short'"""
        if self.paper:
            self.log.info("[PAPER] %s %s qty=%.6f @ £%.4f", side.upper(), symbol, qty, price)
            return f"paper-{side}-{symbol}-{int(time.time())}"
        try:
            resp = self.client.futures_order(symbol, side, qty)
            return (resp.get("data") or {}).get("orderId")
        except Exception as exc:
            self.log.error("Futures order failed (%s %s): %s", side, symbol, exc)
            return None

    def _enter(self, signal: Signal) -> None:
        ok, reason = self.risk.can_open(signal.base)
        if not ok:
            self.log.info("Skip %s entry: %s", signal.base, reason)
            return

        qty, leverage = self.risk.calc_position(signal.entry_price, signal.sl_price)
        if qty <= 0:
            self.log.warning("Skip %s — calc_position returned 0 qty", signal.base)
            return

        # Set leverage on exchange (live only)
        if not self.paper:
            try:
                self.client.set_leverage(signal.symbol, leverage,
                                         "long" if signal.side == LONG else "short")
            except Exception as exc:
                self.log.warning("set_leverage failed for %s: %s", signal.symbol, exc)

        side_word = "long" if signal.side == LONG else "short"
        order_side = "open_long" if signal.side == LONG else "open_short"
        order_id = self._futures_order(signal.symbol, order_side, qty, signal.entry_price)

        pos = RulePosition(
            base=signal.base,
            symbol=signal.symbol,
            side=side_word,
            entry_price=signal.entry_price,
            quantity=qty,
            stop_loss=signal.sl_price,
            take_profit=signal.tp_price,
            leverage=leverage,
            entry_time=utcnow().isoformat(),
            atr=signal.atr,
            order_id=order_id,
        )
        self.risk.open_position(pos)

        # Place exchange-side TP/SL plan order so it triggers even if bot is down
        if not self.paper:
            try:
                self.client.place_tpsl(signal.symbol, side_word,
                                       signal.sl_price, signal.tp_price, size=qty)
            except Exception as exc:
                self.log.warning("place_tpsl failed for %s: %s", signal.symbol, exc)

        try:
            notify_open(signal.symbol, side_word, signal.entry_price,
                        signal.sl_price, signal.tp_price, qty, leverage)
        except Exception:
            pass

    def _exit(self, base: str, exit_price: float, reason: str) -> None:
        pos = self.risk.open_positions.get(base)
        if not pos:
            return
        order_side = "close_long" if pos.side == "long" else "close_short"
        self._futures_order(pos.symbol, order_side, pos.quantity, exit_price)

        trade = self.risk.close_position(base, exit_price, reason)
        if trade is None:
            return

        # Persist + report
        try:
            self.logger.log_trade(trade, self.risk.equity, exit_reason=reason)
        except Exception as exc:
            self.log.warning("trade_logger.log_trade failed: %s", exc)

        if self.user_id and fortuna_client is not None:
            try:
                fortuna_client.post_trade(self.user_id, trade, self.risk.equity, reason)
                fortuna_client.post_equity(self.user_id, self.risk.equity, self.risk.hwm)
            except Exception as exc:
                self.log.debug("fortuna_client post failed: %s", exc)

        try:
            notify_close(trade["pair"], trade["side"], trade["entry_price"],
                         trade["exit_price"], trade["pnl_usdt"], reason)
        except Exception:
            pass

    # ── Tick logic ───────────────────────────────────────────────────────────

    def tick(self) -> None:
        # 1) BTC daily bias — driver for ALL assets' direction
        btc_symbol = next((a["symbol"] for a in self.assets if a["base"] == "BTC"), None)
        btc_bias = 0
        if btc_symbol:
            btc_d = self.fetch_1d(btc_symbol, limit=30)
            if btc_d is not None and not btc_d.empty:
                # Drop today's still-forming daily candle so we only use closed bars
                today = pd.Timestamp.utcnow().tz_localize(None).normalize()
                btc_d = btc_d[btc_d["timestamp"] < today]
                btc_bias = compute_daily_btc_bias(btc_d)
        self.log.info("BTC daily bias = %s | %s",
                      {1: "BULL ✅", -1: "BEAR ⚠️", 0: "UNKNOWN"}.get(btc_bias, "?"),
                      self.risk.summary())

        # 2) Per-asset: fetch 4h, evaluate signal, check exits
        signals: List[Signal] = []
        for asset in self.assets:
            symbol, base = asset["symbol"], asset["base"]
            df4 = self.fetch_4h(symbol, limit=120)
            if df4 is None or df4.empty:
                self.log.warning("No 4h data for %s — skipping", symbol)
                continue

            # Live exits — check current price against any open position
            if base in self.risk.open_positions:
                price = self.live_price(symbol, df4)
                if price is not None:
                    # Pass last-bar wick to capture moves between polls
                    last = df4.iloc[-1]
                    reason = self.risk.should_exit(
                        base, price,
                        candle_high=float(last["high"]),
                        candle_low=float(last["low"]),
                    )
                    if reason:
                        self._exit(base, price, reason)

            # Entry evaluation — only on a NEW closed 4h candle
            if len(df4) >= 2:
                latest_closed_ts = pd.Timestamp(df4.iloc[-2]["timestamp"])
                seen = self._last_4h_ts.get(symbol)
                if seen is not None and latest_closed_ts <= seen:
                    continue   # already evaluated this 4h candle
                self._last_4h_ts[symbol] = latest_closed_ts

                sig = evaluate_signal(symbol, df4, btc_bias, use_last_closed=True)
                if sig.is_actionable():
                    self.log.info("Signal %s %s @ %.2f | SL=%.2f TP=%.2f RR=%.2f | %s",
                                  sig.base,
                                  "LONG" if sig.side == LONG else "SHORT",
                                  sig.entry_price, sig.sl_price, sig.tp_price,
                                  sig.rr, sig.reason)
                    signals.append(sig)
                else:
                    self.log.debug("No signal %s: %s", base, sig.reason)

        # 3) Concurrency-aware entry selection
        if signals:
            chosen = self.controller.select_entries(signals, self.risk.open_bases())
            for sig in chosen:
                self._enter(sig)

        # 4) Equity heartbeat → Fortuna (for the dashboard)
        if self.user_id and fortuna_client is not None:
            try:
                fortuna_client.post_equity(self.user_id, self.risk.equity, self.risk.hwm)
            except Exception:
                pass

    # ── Main loop ────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        self.log.info("🚀 RuleBot starting — poll interval %ds", self.poll_seconds)
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.log.info("Interrupted — saving state and exiting cleanly")
                self.risk._save_state()
                return
            except Exception as exc:
                self.log.exception("Tick error (will continue): %s", exc)
            time.sleep(self.poll_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Multi-user mode (Fortuna dashboard) if FORTUNA_API_URL is set; else single-user
    api_url = os.environ.get("FORTUNA_API_URL", "")
    if not api_url or fortuna_client is None:
        bot = RuleBot()
        bot.run_forever()
        return

    # Multi-user: one tick across all users per cycle
    cfg = load_config()
    setup_logging(cfg)
    log = logging.getLogger("RuleBot.multi")
    log.info("📡 Multi-user mode — polling Fortuna for active users")

    bots: Dict[str, RuleBot] = {}
    while True:
        try:
            users = fortuna_client.get_active_users()
            log.info("Active users: %d", len(users))
            for u in users:
                uid = u.get("user_id")
                if not uid:
                    continue
                user_cfg = fortuna_client.get_user_config(uid) or {}
                # Rebuild bot if strategy_mode or risk changed
                key = (uid, user_cfg.get("strategy_mode", ""),
                       user_cfg.get("risk_per_trade", ""))
                existing = bots.get(uid)
                if existing is None or getattr(existing, "_key", None) != key:
                    bots[uid] = RuleBot(user_id=uid, user_override=user_cfg)
                    bots[uid]._key = key
                bots[uid].tick()
        except KeyboardInterrupt:
            return
        except Exception as exc:
            log.exception("Multi-user loop error: %s", exc)
        time.sleep(int(cfg["strategy_rules"].get("poll_seconds", 300)))


if __name__ == "__main__":
    main()
