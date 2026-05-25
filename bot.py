"""
bot.py  —  Main trading loop

Before every trade the bot prints a full trade quality card:

  ┌─ TRADE QUALITY  BTC ─────────────────────────────┐
  │  Signal     : BUY  (confidence 71.3 %)            │
  │  Win rate   : 58.2 %  (from 34 past trades)       │
  │  Avg R/R    : 2.14                                 │
  │  Expect.Val : +0.041 %  per trade  ✅             │
  │  Actual R/R : 2.10  (SL=£1.5  TP=£3.15)          │
  │  HTF trend  : 1h BULLISH ✅ (confluence confirmed) │
  │  Position   : 0.00021 BTC  (risking £5.00)        │
  │  VERDICT    : TAKE THE TRADE                       │
  └──────────────────────────────────────────────────┘

Startup sequence
────────────────
1. Data collection  — pulls up to 2 years of multi-TF candles from Weex
                      (skipped if CSVs are fresh, ~2 min on first run)
2. Analysis         — backtests every timeframe + confluence combination
                      and saves recommendations to /data/analysis_results.json
                      (skipped if results are < 7 days old)
3. Strategy init    — loads analysis recommendations, pre-trains model on
                      the full historical dataset
4. Trading loop     — evaluates signals every N minutes, applies HTF confluence
                      filter before executing any trade

Run with: python bot.py
"""

import time
import logging
import os
import json
import yaml
import shutil
import urllib.request
from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)

from typing import Dict, Optional

import pandas as pd

from weex_client   import WeexClient
from indicators    import candles_to_df, compute_features
from strategy      import TradingStrategy, BUY, SELL, HOLD
from risk_manager  import RiskManager, Position
from trade_logger  import TradeLogger
from data_collector import DataCollector, TF_LABELS
from analysis      import Analyzer
from mae_analyser   import MAEAnalyser
from historical_mae import run_historical_mae
from news_calendar  import (entries_blocked, stops_should_tighten,
                             next_event, NEWS_TIGHTEN_PCT)
from telegram_notifier import notify_open, notify_close, notify_startup
from daily_summary import send_if_due as send_daily_summary
from zoneinfo import ZoneInfo


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Railway / cloud deployment: override with environment variables ────────
    if os.getenv("WEEX_API_KEY"):
        cfg["exchange"]["api_key"]    = os.environ["WEEX_API_KEY"]
    if os.getenv("WEEX_API_SECRET"):
        cfg["exchange"]["api_secret"] = os.environ["WEEX_API_SECRET"]
    if os.getenv("WEEX_PASSPHRASE"):
        cfg["exchange"]["passphrase"] = os.environ["WEEX_PASSPHRASE"]

    if os.getenv("PAPER_TRADING") is not None:
        cfg["trading"]["paper_trading"] = os.getenv("PAPER_TRADING", "true").lower() == "true"

    if os.getenv("LEVERAGE"):
        cfg["trading"]["leverage"] = int(os.getenv("LEVERAGE"))
    if os.getenv("RISK_PER_TRADE"):
        cfg["risk"]["risk_per_trade_abs"] = float(os.getenv("RISK_PER_TRADE"))

    # BASE_URL — override the Weex API base URL if the primary domain changes.
    # e.g.  BASE_URL=https://api-spot.weex.com
    if os.getenv("BASE_URL"):
        cfg["exchange"]["base_url"] = os.environ["BASE_URL"]

    # FORCE_RETRAIN=true  — wipe saved models and retrain from scratch on next startup.
    # Remove the variable once models are generating healthy signals.
    # (Consumed in startup(), not stored in cfg — just documented here.)

    return cfg


def setup_logging(cfg: dict):
    log_file = cfg["logging"]["log_file"]
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    level = getattr(logging, cfg["logging"].get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trade quality card
# ─────────────────────────────────────────────────────────────────────────────

def print_trade_card(pair: str, signal: int, confidence: float,
                     ev: Optional[float], win_rate: Optional[float],
                     avg_rr: Optional[float], actual_rr: float,
                     qty: float, risk_amount: float,
                     ev_reason: str, verdict: bool,
                     htf_label: str = "", htf_direction: int = 0):
    sig_str = {BUY: "BUY", SELL: "SELL", HOLD: "HOLD"}.get(signal, "?")

    if ev is not None:
        ev_str = f"{ev*100:+.3f}%  {'✅' if ev > 0 else '❌'}"
    else:
        ev_str = "N/A (still gathering data)"

    wr_str  = f"{win_rate*100:.1f}%" if win_rate is not None else "N/A"
    rr_str  = f"{avg_rr:.2f}"        if avg_rr   is not None else "N/A"
    verdict_str = "✅ TAKE THE TRADE" if verdict else "❌ SKIP (R/R or EV gate)"

    htf_str = ""
    if htf_label:
        trend_word = {1: "BULLISH ✅", -1: "BEARISH ⚠️", 0: "NEUTRAL"}.get(htf_direction, "N/A")
        htf_str = f"{htf_label} {trend_word}"

    border = "─" * 52
    logging.getLogger("TradeCard").info(
        "\n┌%s┐\n"
        "│  %-50s│\n"
        "│  Signal      : %-4s  (confidence %5.1f %%)       │\n"
        "│  Win rate    : %-44s│\n"
        "│  Avg R/R     : %-44s│\n"
        "│  Expect.Val  : %-44s│\n"
        "│  Actual R/R  : %-5.2f  (ATR-derived SL/TP)         │\n"
        "│  HTF trend   : %-44s│\n"
        "│  Position    : %.5f %s  (risking £%.2f)  │\n"
        "│  EV detail   : %-44s│\n"
        "│  VERDICT     : %-44s│\n"
        "└%s┘",
        border,
        f"TRADE QUALITY  {pair}",
        sig_str, confidence * 100,
        wr_str,
        rr_str,
        ev_str,
        actual_rr,
        htf_str or "N/A (no HTF data)",
        qty, pair.replace("USDT_UMCBL", "").replace("USDT_SPBL", ""), risk_amount,
        ev_reason[:44],
        verdict_str,
        border,
    )


# ─────────────────────────────────────────────────────────────────────────────
class TradingBot:

    def __init__(self, config_path: str = "config.yaml",
                 user_id: str = "",
                 user_override: dict = None):
        """
        user_id:       Fortuna user ID — used to isolate state and report trades.
        user_override: {api_key, api_secret, passphrase, capital} from Fortuna API.
                       When provided, overrides config.yaml values for this user.
        """
        self.cfg  = load_config(config_path)
        setup_logging(self.cfg)
        self.log  = logging.getLogger(f"Bot[{user_id[:8]}]" if user_id else "Bot")
        self.user_id = user_id

        # Apply per-user overrides from the Fortuna API
        if user_override:
            if user_override.get("api_key"):
                self.cfg["exchange"]["api_key"]    = user_override["api_key"]
                self.cfg["exchange"]["api_secret"] = user_override["api_secret"]
                self.cfg["exchange"]["passphrase"] = user_override.get("passphrase", "")
            if user_override.get("capital"):
                self.cfg["risk"]["initial_capital"] = float(user_override["capital"])

        ec = self.cfg["exchange"]
        self.paper    = self.cfg["trading"]["paper_trading"]
        self.pairs    = [p for p in self.cfg["trading"]["pairs"] if p["enabled"]]
        self.tf       = str(self.cfg["trading"]["timeframe"])
        self.lookback = self.cfg["trading"]["lookback_candles"]
        base_data_dir = self.cfg.get("data", {}).get("data_dir", "/data")
        # Per-user state isolation — each user gets their own subdirectory
        self.data_dir = os.path.join(base_data_dir, user_id) if user_id else base_data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Higher-timeframe for confluence filter
        # Analysis recommendations override this after the analysis runs
        data_cfg       = self.cfg.get("data", {})
        self.htf_tf    = str(data_cfg.get("confluence_timeframe", "60"))  # default 1h
        self.htf_label = TF_LABELS.get(self.htf_tf, "1h")

        # Lower-timeframe for reversal exit signal (1h when trading 4h)
        tc_cfg          = self.cfg.get("trading", {})
        self.ltf_tf     = str(tc_cfg.get("ltf_reversal_tf", "60"))
        self.ltf_label  = TF_LABELS.get(self.ltf_tf, "1h")

        # Per-symbol signal TF and HTF — populated by _apply_analysis_recommendations()
        # Falls back to self.tf / self.htf_tf when not set for a given symbol.
        self.symbol_tf:  Dict[str, str] = {}   # symbol → signal TF (minutes string)
        self.symbol_htf: Dict[str, str] = {}   # symbol → HTF (minutes string)

        # Active strategies — list of dicts: {symbol, tf_label, tf_min, htf_min, slot_key}
        # Each entry is an independent (symbol, timeframe) strategy the bot will trade.
        # Populated from analysis profitable_strategies after analysis runs.
        # Falls back to one entry per enabled pair using the default TF.
        self.active_strategies: list = []

        mode = "🟡 PAPER FUTURES" if self.paper else "🔴 LIVE FUTURES"
        self.log.info("%s MODE  |  Account: £%.2f  |  Risk/trade: HWM×day%%  |  Lev: dynamic (max %dx)",
                      mode,
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"].get("max_leverage", 20))

        self.client    = WeexClient(
            api_key=ec["api_key"], api_secret=ec["api_secret"],
            passphrase=ec["passphrase"], base_url=ec["base_url"],
        )
        self.collector = DataCollector(self.client, data_dir=self.data_dir)
        self.analyzer  = Analyzer(data_dir=self.data_dir, results_dir=self.data_dir)
        self.strategy  = TradingStrategy(self.cfg)
        self.risk      = RiskManager(self.cfg, data_dir=self.data_dir)
        self.logger    = TradeLogger(self.cfg["logging"]["trades_file"])

        # ── One-shot model backup restore (hobby accounts have no console) ──────
        # Set RESTORE_MODEL_BACKUP=true in Railway Variables, redeploy, then remove it.
        # Copies the most recent backup_* folder back over the live models so the
        # original well-trained models are restored after a bad retrain.
        if os.getenv("RESTORE_MODEL_BACKUP", "").lower() == "true":
            try:
                _models_dir = self.cfg.get("logging", {}).get("models_dir", "/data/models/")
                _backups = sorted([
                    d for d in os.listdir(_models_dir)
                    if d.startswith("backup_") and os.path.isdir(os.path.join(_models_dir, d))
                ], reverse=True)
                if _backups:
                    _src_dir = os.path.join(_models_dir, _backups[0])
                    self.log.warning("⚠️  RESTORE_MODEL_BACKUP: restoring from %s", _src_dir)
                    for _fname in os.listdir(_src_dir):
                        if _fname.endswith(".joblib"):
                            shutil.copy2(
                                os.path.join(_src_dir, _fname),
                                os.path.join(_models_dir, _fname)
                            )
                    # Reset last_monthly_retrain so health check knows models are fresh
                    self._last_monthly_retrain = None
                    self._save_health_state()
                    self.log.warning("✅ Models restored from backup. Remove RESTORE_MODEL_BACKUP now.")
                else:
                    self.log.error("RESTORE_MODEL_BACKUP: no backup folders found in %s", _models_dir)
            except Exception as _e:
                self.log.error("RESTORE_MODEL_BACKUP failed: %s", _e)

        # ── One-shot equity override (hobby accounts have no console) ─────────
        # Set EQUITY_OVERRIDE=82.90 in Railway Variables, redeploy, then remove it.
        _eq_override = os.getenv("EQUITY_OVERRIDE")
        if _eq_override:
            try:
                _eq_val = float(_eq_override)
                self.risk.equity           = _eq_val
                self.risk.day_start_equity = _eq_val
                self.risk._save_state()
                self.log.warning(
                    "⚠️  EQUITY_OVERRIDE applied — equity set to £%.2f. "
                    "Remove EQUITY_OVERRIDE from Railway Variables now.",
                    _eq_val,
                )
            except ValueError:
                self.log.error("EQUITY_OVERRIDE value '%s' is not a valid number — ignored.", _eq_override)
        self._tick_count = 0
        # slot_key → original SL price for positions tightened ahead of news
        # (breakeven moves are NOT stored here — they stay at breakeven after)
        self._news_tightened_sl: Dict[str, float] = {}

        # ── Model health tracking ─────────────────────────────────────────────
        # last_signal_ts: slot → datetime of last prediction with confidence
        #   >= ConfidenceTracker.MIN_SIGNAL_CONFIDENCE (not just opened trades)
        self._last_signal_ts:       Dict[str, datetime] = {}
        # When the bot last ran a full forced retrain of all models
        self._last_monthly_retrain: Optional[datetime]  = None
        self._health_path = os.path.join(
            self.cfg.get("data", {}).get("data_dir", "/data"), "model_health.json"
        )
        self._load_health_state()

    # ── Dashboard state ───────────────────────────────────────────────────────

    def _report_trade(self, trade: dict, exit_reason: str) -> None:
        """Post completed trade to Fortuna API and update equity. No-op if no user_id."""
        if not self.user_id:
            return
        try:
            import fortuna_client
            fortuna_client.post_trade(self.user_id, trade, self.risk.equity, exit_reason)
            fortuna_client.post_equity(self.user_id, self.risk.equity, self.risk.hwm)
        except Exception as exc:
            self.log.warning("Could not report trade to Fortuna API: %s", exc)

    def _write_state(self) -> None:
        """Write current bot state to /data/state.json for the dashboard."""
        try:
            positions = {}
            for slot_key, pos in self.risk.open_positions.items():
                positions[slot_key] = {
                    "symbol":     pos.pair,
                    "tf":         slot_key.split("_")[-1] if "_" in slot_key else self.tf,
                    "side":       pos.side,
                    "entry_price": pos.entry_price,
                    "sl":         pos.stop_loss,
                    "tp":         pos.take_profit,
                    "tp1_price":  pos.tp1_price,
                    "tp1_hit":    pos.tp1_hit,
                    "qty":        pos.quantity,
                    "qty_original": pos.quantity_original,
                    "leverage":   pos.leverage,
                    "entry_time": pos.entry_time,
                }
            state = {
                "updated_at":      utcnow().isoformat(),
                "paper":           self.paper,
                "equity":          round(self.risk.equity, 4),
                "hwm":             round(self.risk.hwm, 4),
                "initial_capital": self.risk.initial_capital,
                "max_open":        self.risk.max_open,
                "positions":       positions,
            }
            state_path = os.path.join(self.data_dir, "state.json")
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            # Always write to base /data/state.json so the Railway dashboard can read it
            base_state_path = os.path.join(
                self.cfg.get("data", {}).get("data_dir", "/data"), "state.json"
            )
            if base_state_path != state_path:
                with open(base_state_path, "w") as f:
                    json.dump(state, f, indent=2)
        except Exception as exc:
            self.log.debug("Could not write state.json: %s", exc)

    # ── Data ──────────────────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str,
                      tf: Optional[str] = None,
                      limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        raw = self.client.get_candles(
            symbol,
            granularity = tf or self.tf,
            limit       = limit or self.lookback,
        )
        if not raw:
            self.log.error("No candle data for %s", symbol)
            return None
        return candles_to_df(raw)

    def live_price(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Returns the current live mark price from the Weex ticker.

        For futures (_UMCBL) symbols this is the contract markPrice — the same
        price shown on the Weex futures UI and used for PnL/liquidation.
        Falls back to the last closed candle's close price if the ticker call
        fails (e.g. network blip), so the bot keeps running rather than
        crashing.

        Using the live price means entry price, SL, TP, and position sizing
        are all anchored to the real market price at signal time — not a
        candle that may have closed up to 15 minutes ago.
        """
        tick = self.client.get_ticker(symbol)
        if tick:
            # Prefer markPrice for futures (fair value); fall back to lastPr / last
            for field in ("markPrice", "lastPr", "last", "close"):
                val = tick.get(field)
                if val:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        # Fallback: last closed candle close
        self.log.debug("Ticker unavailable for %s — using candle close as price", symbol)
        return float(df["close"].iloc[-1])

    def get_equity(self) -> float:
        if self.paper:
            return self.risk.equity
        try:
            return self.client.get_futures_balance()
        except Exception:
            return self.risk.equity

    def _sym_tf(self, symbol: str) -> str:
        """Signal TF (minutes string) for this symbol, e.g. '15' or '240'."""
        return self.symbol_tf.get(symbol, self.tf)

    def _sym_htf(self, symbol: str) -> str:
        """HTF (minutes string) for this symbol."""
        return self.symbol_htf.get(symbol, self.htf_tf)

    def _sym_tf_label(self, symbol: str) -> str:
        """Human-readable TF label for this symbol, e.g. '15m' or '4h'."""
        return TF_LABELS.get(self._sym_tf(symbol), self.tf)

    # ── Futures order execution ────────────────────────────────────────────────

    def _futures_order(self, symbol: str, side: str,
                       qty: float, price: float) -> Optional[str]:
        """
        Place a futures market order and return the order ID.
        side: 'open_long' | 'open_short' | 'close_long' | 'close_short'
        """
        if self.paper:
            self.log.info("[PAPER] %s  %s  qty=%.5f @ £%.4f",
                          side.upper(), symbol, qty, price)
            return f"paper-{side}-{symbol}-{int(time.time())}"
        resp = self.client.futures_order(symbol, side, qty)
        oid  = (resp.get("data") or {}).get("orderId")
        if not oid:
            self.log.error("Futures order failed (%s %s): %s", side, symbol, resp)
        return oid

    def _open_long(self, s: str, qty: float, p: float) -> Optional[str]:
        return self._futures_order(s, "open_long",  qty, p)

    def _open_short(self, s: str, qty: float, p: float) -> Optional[str]:
        return self._futures_order(s, "open_short", qty, p)

    def _place_exchange_tpsl(self, symbol: str, side: str,
                              sl: float, tp: float, qty: float) -> None:
        """
        Place a native TP/SL plan order on WEEX so the exchange will close the
        position even if the bot process is offline.
        Skipped in paper mode. Failures are logged but never raise — software
        monitoring in monitor_exits() remains as a backup.
        """
        if self.paper:
            self.log.info("[PAPER] Would place exchange TPSL: %s %s  SL=%.4f  TP=%.4f",
                          symbol, side, sl, tp)
            return
        hold_side = "long" if side == "long" else "short"
        try:
            resp = self.client.place_tpsl(symbol, hold_side, sl, tp, size=qty)
            plan_id = (resp.get("data") or {}).get("orderId") or (resp.get("data") or {}).get("planOrderId")
            if plan_id:
                self.log.info("✅ Exchange TPSL placed: %s %s  SL=%.4f  TP=%.4f  plan_id=%s",
                              symbol, hold_side, sl, tp, plan_id)
            else:
                self.log.warning("⚠️  Exchange TPSL response missing order ID — may not have been placed: %s", resp)
        except Exception as exc:
            self.log.error("⚠️  Failed to place exchange TPSL for %s: %s — software monitoring active",
                           symbol, exc)

    def _close_long(self, s: str, qty: float, p: float) -> Optional[str]:
        return self._futures_order(s, "close_long",  qty, p)

    def _close_short(self, s: str, qty: float, p: float) -> Optional[str]:
        return self._futures_order(s, "close_short", qty, p)

    def _close_pos(self, symbol: str, qty: float,
                   price: float, side: str) -> Optional[str]:
        """Close any position correctly based on its side."""
        if side == "long":
            return self._close_long(symbol, qty, price)
        return self._close_short(symbol, qty, price)

    def _setup_leverage(self):
        """
        Set leverage for every trading pair (both long and short sides).
        Called once during startup before the trading loop begins.
        In paper mode this is skipped (no real account to configure).
        """
        if self.paper:
            self.log.info("⚙️  [PAPER] Leverage setup skipped (paper trading mode)")
            return
        lev = self.cfg["risk"].get("max_leverage", 20)
        for pair_cfg in self.pairs:
            symbol = pair_cfg["symbol"]
            try:
                self.client.set_leverage(symbol, lev, "long")
                self.client.set_leverage(symbol, lev, "short")
                self.log.info("⚙️  Leverage set: %s  %dx (long + short)", symbol, lev)
            except Exception as exc:
                self.log.warning("⚠️  Could not set leverage for %s: %s", symbol, exc)

    # ── Startup sequence ──────────────────────────────────────────────────────

    def startup(self):
        """
        Full startup pipeline:
          1. Collect historical data (skipped if CSVs are fresh)
          2. Run analysis (skipped if results are < 7 days old)
          3. Apply analysis recommendations
          4. Train initial model on historical data
        """
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.info("  STARTUP PIPELINE  (FUTURES mode)")
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ── Step 0: Load confidence tracker history ──────────────────────────
        conf_log_path = os.path.join(self.data_dir, "confidence_log.json")
        self.strategy.conf_tracker.load(conf_log_path)

        # ── Step 0a: Set leverage for all pairs ─────────────────────────────
        # NOTE: equity/HWM are restored by RiskManager.__init__ from risk_state.json.
        # A second restoration from state.json was removed — it caused silent equity
        # corruption when state.json was stale (e.g. after a bad trade write).
        self._setup_leverage()

        # ── Step 0b: Download external models from GitHub release ─────────────
        use_external = self.cfg.get("strategy", {}).get("use_external_models", False)
        if use_external:
            self._download_models_from_github()

        # ── Step 1: Data collection ───────────────────────────────────────────
        self.log.info("STEP 1/3  Data collection")
        try:
            self.collector.collect_all()
        except Exception as exc:
            self.log.error("Data collection error (continuing anyway): %s", exc)

        # ── Step 2: Analysis ──────────────────────────────────────────────────
        self.log.info("STEP 2/3  Strategy analysis")
        use_external = self.cfg.get("strategy", {}).get("use_external_models", False)
        if use_external:
            self.log.info("  Skipped — use_external_models=true (strategies and models provided externally)")
        else:
            try:
                stale_days = self.cfg.get("data", {}).get("analysis_stale_days", 7)
                if not self.analyzer.results_are_fresh(max_age_days=stale_days):
                    self.analyzer.run()
                else:
                    self.log.info("  Analysis results are fresh — skipping re-run.")
            except Exception as exc:
                self.log.error("Analysis error (continuing anyway): %s", exc)

        # ── Step 3: Apply recommendations + train ─────────────────────────────
        self.log.info("STEP 3/3  Initial model training")
        try:
            # ── FORCE_RETRAIN: wipe stale models so the bot retrains from scratch ──
            # Set env var FORCE_RETRAIN=true on Railway to trigger a clean retrain.
            # Remove the env var once you've confirmed the models are generating signals.
            if os.getenv("FORCE_RETRAIN", "false").lower() == "true":
                self.log.info("♻️  FORCE_RETRAIN=true — clearing saved models for clean retrain")
                models_dir = self.cfg["logging"]["models_dir"]
                deleted = 0
                if os.path.isdir(models_dir):
                    for fname in os.listdir(models_dir):
                        if fname.endswith(".joblib"):
                            try:
                                os.remove(os.path.join(models_dir, fname))
                                deleted += 1
                            except Exception as e:
                                self.log.warning("Could not delete %s: %s", fname, e)
                # Reset in-memory state so _initial_train starts clean
                self.strategy.model = None
                self.strategy.symbol_models.clear()
                self.strategy.symbol_scalers.clear()
                self.strategy.symbol_features.clear()
                self.log.info("♻️  Deleted %d model file(s) — will train fresh on historical data", deleted)

            # Reload analysis so strategy picks up fresh recommendations
            self.strategy.reload_analysis()
            self._apply_analysis_recommendations()
            self._initial_train()
            # Mark retrain timestamp so _check_model_health() doesn't fire
            # another retrain immediately at the first 4h tick.  The 28-day
            # interval starts from this deployment, not from "never".
            if self._last_monthly_retrain is None:
                self._last_monthly_retrain = utcnow()
                self._save_health_state()
        except Exception as exc:
            self.log.error("Training error (continuing anyway): %s", exc)

        # ── Step 4a: Historical MAE backtest — calibrate SL/TP before trade #1 ───
        # Runs a walk-forward simulation on historical OHLCV data so the bot
        # starts with an evidence-based SL rather than a hardcoded default.
        try:
            self._run_historical_mae()
        except Exception as exc:
            self.log.warning("Historical MAE error (non-fatal): %s", exc)

        # ── Step 4b: Live MAE analysis — refines SL from real closed trades ────
        try:
            self._run_mae_analysis()
        except Exception as exc:
            self.log.warning("MAE analysis error (non-fatal): %s", exc)

        # ── Step 5: Catch missed TP/SL wicks on restored positions ──────────────
        # Positions restored from risk_state.json have seen_high/seen_low = 0.
        # Do a one-time deep fetch (200 candles ≈ 16h) for each restored position
        # so any TP/SL wick that occurred while the bot was down is caught now.
        try:
            self._catch_missed_exits()
        except Exception as exc:
            self.log.error("Startup wick-check error (non-fatal): %s", exc)

        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.info("  STARTUP COMPLETE — entering trading loop")
        _ne = next_event()
        if _ne:
            self.log.info("  📰 Next news event: %s in %.0f min", _ne[0], _ne[1])
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _apply_analysis_recommendations(self):
        """
        Read analysis results and update bot settings accordingly:
          • Switch the signal timeframe to the best one found by analysis
          • Use the recommended higher-TF filter (or auto-pick the next TF up)
          • Update the loop interval to match the new signal TF
        """
        if not self.strategy.analysis:
            return
        recs     = self.strategy.analysis.get("recommendations", {})
        rev_map  = {v: k for k, v in TF_LABELS.items()}
        data_cfg = self.cfg.get("data", {})

        # ── Auto-select HTF based on signal TF (next timeframe up) ────────────
        HTF_UP       = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}
        TF_INTERVALS = {"5": 300, "15": 900, "60": 3600, "240": 14400, "1440": 86400}

        # ── Switch signal timeframe if analysis found a better one ─────────────
        best_signal = recs.get("best_signal_timeframe")  # e.g. "4h"
        if best_signal:
            sig_min = rev_map.get(best_signal)
            # Respect min_signal_tf_minutes — never let analysis override below our floor
            _min_tf = data_cfg.get("min_signal_tf_minutes")
            if sig_min and _min_tf and int(sig_min) < int(_min_tf):
                self.log.info("📊 Analysis recommended %s but min_signal_tf=%s — keeping %s",
                              best_signal, _min_tf, TF_LABELS.get(self.tf, self.tf))
                sig_min = None
            if sig_min and sig_min != self.tf:
                old_tf   = self.tf
                self.tf  = sig_min
                # Update loop interval so the run() loop uses the right cadence
                self.cfg["trading"]["loop_interval_s"] = TF_INTERVALS.get(sig_min,
                                                         self.cfg["trading"]["loop_interval_s"])
                # Auto-set HTF to one step up from new signal TF
                htf_min        = HTF_UP.get(sig_min, "1440")
                self.htf_tf    = htf_min
                self.htf_label = TF_LABELS.get(htf_min, "1d")
                self.log.info("📊 Analysis switched signal TF: %s → %s  (HTF=%s  loop=%ds)",
                              TF_LABELS.get(old_tf, old_tf), best_signal,
                              self.htf_label,
                              self.cfg["trading"]["loop_interval_s"])

        # ── Override HTF if analysis explicitly recommends a filter ───────────
        best_filter = recs.get("best_filter_timeframe")  # e.g. "1h"
        if best_filter and best_filter != "None":
            htf_min = rev_map.get(best_filter)
            if htf_min:
                self.htf_tf    = htf_min
                self.htf_label = best_filter
                self.log.info("📊 Analysis recommends HTF filter: %s", best_filter)

        # ── Per-symbol signal TF (the main Option-1 change) ──────────────────
        # analysis.py emits {"BTCUSDT": "4h", "ETHUSDT": "15m", "SOLUSDT": "4h"}
        per_sym = recs.get("per_symbol_best_tf", {})
        if per_sym:
            for pair_cfg in self.pairs:
                sym      = pair_cfg["symbol"]                              # "ETHUSDT_SPBL"
                base     = sym.replace("_SPBL", "").replace("_UMCBL", "") # "ETHUSDT"
                tf_label = per_sym.get(base)                               # "15m"
                if tf_label:
                    min_tf = rev_map.get(tf_label)
                    if min_tf:
                        htf_min = HTF_UP.get(min_tf, "1440")
                        self.symbol_tf[sym]  = min_tf
                        self.symbol_htf[sym] = htf_min

            # Log the full per-symbol TF assignment table
            assignments = {
                p["name"]: f"{TF_LABELS.get(self.symbol_tf.get(p['symbol'], self.tf), '?')}"
                           f" (HTF: {TF_LABELS.get(self.symbol_htf.get(p['symbol'], self.htf_tf), '?')})"
                for p in self.pairs
            }
            self.log.info("📊 Per-symbol TF assignments: %s", assignments)

        # ── Enforce min/max signal TF bounds ─────────────────────────────────────
        # min_signal_tf_minutes: floor — respect the configured minimum TF
        # max_signal_tf_minutes: ceiling — 1d has only 730 candles, too thin
        # 15m (~70k candles), 1h (~17k), 4h (~4.4k) all compete freely on CV.
        min_tf_cfg = data_cfg.get("min_signal_tf_minutes")
        max_tf_cfg = data_cfg.get("max_signal_tf_minutes")

        def _clamp_tf(tf_str: str) -> str:
            tf_int = int(tf_str)
            if min_tf_cfg and tf_int < int(min_tf_cfg):
                return str(int(min_tf_cfg))
            if max_tf_cfg and tf_int > int(max_tf_cfg):
                return str(int(max_tf_cfg))
            return tf_str

        # Apply to per-symbol TFs
        for sym in list(self.symbol_tf.keys()):
            old_tf_str  = self.symbol_tf[sym]
            new_tf_str  = _clamp_tf(old_tf_str)
            if new_tf_str != old_tf_str:
                new_htf = HTF_UP.get(new_tf_str, "1440")
                self.symbol_tf[sym]  = new_tf_str
                self.symbol_htf[sym] = new_htf
                self.log.info("📊 %s TF adjusted: %s → %s (bounds: %s–%s min)",
                              sym,
                              TF_LABELS.get(old_tf_str, old_tf_str),
                              TF_LABELS.get(new_tf_str, new_tf_str),
                              min_tf_cfg or "any", max_tf_cfg or "any")

        # Apply to global TF + update loop interval
        clamped_global = _clamp_tf(self.tf)
        if clamped_global != self.tf:
            old_tf     = TF_LABELS.get(self.tf, self.tf)
            self.tf    = clamped_global
            new_interval = TF_INTERVALS.get(clamped_global,
                           self.cfg["trading"]["loop_interval_s"])
            self.cfg["trading"]["loop_interval_s"] = new_interval
            self.log.info("📊 Global TF adjusted: %s → %s  (loop=%ds)",
                          old_tf, TF_LABELS.get(clamped_global, clamped_global),
                          new_interval)

        # Log top features so they're visible in logs
        top_feats = recs.get("top_features", [])[:5]
        if top_feats:
            self.log.info("📊 Top features from analysis: %s", top_feats)

        # ── Build active_strategies from analysis profitable_strategies ────────
        self._build_active_strategies(recs)

    def _build_active_strategies(self, recs: dict):
        """
        Populate self.active_strategies from analysis recommendations.

        Each strategy is a dict:
          symbol        — exchange symbol e.g. "BTCUSDT_UMCBL"
          tf_label      — human label e.g. "4h"
          tf_min        — minutes string e.g. "240"
          htf_min       — higher-TF minutes string e.g. "1440"
          strategy_type — "single" or "mtf"
          dir_tf_min    — direction TF minutes (MTF only) e.g. "240"
          slot_key      — unique position key e.g. "BTCUSDT_UMCBL_4h"

        If fixed_strategies is defined in config, those are used directly and
        analysis discovery is skipped entirely.
        Falls back to one entry per enabled pair if analysis produced nothing.
        """
        HTF_UP      = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}

        # ── Fixed strategies: bypass analysis entirely ────────────────────────
        fixed = self.cfg.get("strategy", {}).get("fixed_strategies", [])
        if fixed:
            strategies = []
            for s in fixed:
                tf_min  = str(s["tf_min"])
                htf_min = str(s["htf_min"])
                strategies.append({
                    "symbol":        s["symbol"],
                    "name":          s["name"],
                    "tf_label":      TF_LABELS.get(tf_min, tf_min + "m"),
                    "tf_min":        tf_min,
                    "htf_min":       htf_min,
                    "dir_tf_min":    None,
                    "strategy_type": s.get("strategy_type", "confluence"),
                    "filter_tf":     TF_LABELS.get(htf_min, htf_min + "m"),
                    "slot_key":      s["slot_key"],
                    "cv_accuracy":   1.0,
                })
            self.active_strategies = strategies
            self.log.info("📊 Fixed strategies (%d):", len(strategies))
            for s in strategies:
                self.log.info("   %s [%s+%s]  slot=%s",
                              s["name"], s["tf_label"],
                              TF_LABELS.get(s["htf_min"], s["htf_min"]),
                              s["slot_key"])
            return
        rev_map     = {v: k for k, v in TF_LABELS.items()}
        min_acc       = self.cfg.get("strategy", {}).get("min_strategy_accuracy", 0.52)
        blocked_slots = set(self.cfg.get("strategy", {}).get("blocked_slots", []))
        data_cfg      = self.cfg.get("data", {})
        min_tf_cfg    = data_cfg.get("min_signal_tf_minutes")
        max_tf_cfg    = data_cfg.get("max_signal_tf_minutes")

        def _clamp(tf_min_str: str) -> str:
            n = int(tf_min_str)
            if min_tf_cfg and n < int(min_tf_cfg):
                return str(int(min_tf_cfg))
            if max_tf_cfg and n > int(max_tf_cfg):
                return str(int(max_tf_cfg))
            return tf_min_str

        # Build a lookup: base symbol → enabled pair config
        sym_lookup = {}
        for p in self.pairs:
            base = p["symbol"].replace("_UMCBL", "").replace("_SPBL", "")
            sym_lookup[base] = p

        strategies = []
        profitable = recs.get("profitable_strategies", [])
        for entry in profitable:
            base    = entry["symbol"]       # e.g. "BTCUSDT"
            tf_lbl  = entry["timeframe"]    # e.g. "4h"
            accuracy = entry.get("cv_accuracy", 0)
            stype   = entry.get("strategy_type", "single")

            if accuracy < min_acc:
                continue

            pair_cfg = sym_lookup.get(base)
            if pair_cfg is None:
                continue   # symbol not in our enabled pairs list

            tf_min   = _clamp(rev_map.get(tf_lbl, self.tf))
            symbol   = pair_cfg["symbol"]

            # Confluence strategies get a slot_key that includes the filter TF
            # so BTC_15m+1d and BTC_15m+4h are independent slots.
            if stype == "confluence":
                filter_lbl = entry.get("filter_tf", "")
                slot_key   = f"{symbol}_{tf_lbl}+{filter_lbl}" if filter_lbl else f"{symbol}_{tf_lbl}"
            else:
                filter_lbl = ""
                slot_key   = f"{symbol}_{tf_lbl}"

            if slot_key in blocked_slots:
                self.log.info("📊 Skipping blocked slot: %s", slot_key)
                continue

            if stype == "mtf":
                dir_lbl    = entry.get("direction_tf", "")
                dir_tf_min = rev_map.get(dir_lbl, HTF_UP.get(tf_min, "1440"))
                htf_min    = dir_tf_min   # for exit/ATR sizing — same TF
            elif stype == "confluence":
                dir_tf_min  = None
                htf_min     = rev_map.get(filter_lbl, HTF_UP.get(tf_min, "1440"))
            else:
                dir_tf_min = None
                htf_min    = HTF_UP.get(tf_min, "1440")

            strategies.append({
                "symbol":        symbol,
                "name":          pair_cfg["name"],
                "tf_label":      TF_LABELS.get(tf_min, tf_lbl),
                "tf_min":        tf_min,
                "htf_min":       htf_min,
                "dir_tf_min":    dir_tf_min,
                "strategy_type": stype,
                "filter_tf":     filter_lbl,
                "slot_key":      slot_key,
                "cv_accuracy":   accuracy,
            })

        if not strategies:
            # Fallback: one strategy per enabled pair using current default TF
            self.log.info("📊 No profitable strategies from analysis — using default TF per pair")
            for p in self.pairs:
                tf_min   = _clamp(self.tf)
                htf_min  = HTF_UP.get(tf_min, "1440")
                tf_lbl   = TF_LABELS.get(tf_min, self.tf)
                slot_key = f"{p['symbol']}_{tf_lbl}"
                strategies.append({
                    "symbol":   p["symbol"],
                    "name":     p["name"],
                    "tf_label": tf_lbl,
                    "tf_min":   tf_min,
                    "htf_min":  htf_min,
                    "slot_key": slot_key,
                    "cv_accuracy": 0.0,
                })

        # Deduplicate by slot_key — keep highest cv_accuracy entry per slot
        seen: dict = {}
        for s in strategies:
            key = s["slot_key"]
            if key not in seen or s.get("cv_accuracy", 0) > seen[key].get("cv_accuracy", 0):
                seen[key] = s
        strategies = sorted(seen.values(), key=lambda x: x.get("cv_accuracy", 0), reverse=True)

        self.active_strategies = strategies
        self.log.info("📊 Active strategies (%d):", len(strategies))
        for s in strategies:
            self.log.info("     %s [%s]  cv=%.3f  slot=%s",
                          s["name"], s["tf_label"], s.get("cv_accuracy", 0), s["slot_key"])

    def _download_models_from_github(self):
        """
        Download trained .joblib models from the GitHub release 'models-v1' into
        /data/models/ if the directory is empty. Requires GITHUB_TOKEN env var.
        No-op if models already present.
        """
        models_dir = self.cfg["logging"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)

        existing = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
        if existing:
            self.log.info("📦 External models already present (%d files) — skipping download.", len(existing))
            return

        token = os.getenv("GITHUB_TOKEN", "")
        repo  = os.getenv("GITHUB_REPO", "louisrayner01-wq/bot-antigravity2")
        tag   = os.getenv("MODEL_RELEASE_TAG", "models-v1")

        self.log.info("📥 Downloading external models from GitHub release '%s'...", tag)

        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        req = urllib.request.Request(api_url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                release = json.loads(resp.read())
        except Exception as exc:
            self.log.error("Could not fetch GitHub release info: %s", exc)
            return

        assets = release.get("assets", [])
        if not assets:
            self.log.error("No assets found in release '%s'.", tag)
            return

        downloaded = 0
        for asset in assets:
            name = asset["name"]
            if not name.endswith(".joblib"):
                continue
            dl_url = asset["url"]
            dest   = os.path.join(models_dir, name)
            try:
                dl_req = urllib.request.Request(dl_url, headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/octet-stream",
                })
                with urllib.request.urlopen(dl_req, timeout=120) as resp, \
                     open(dest, "wb") as fout:
                    shutil.copyfileobj(resp, fout)
                self.log.info("  ✅ %s  (%.1f MB)", name, os.path.getsize(dest) / 1e6)
                downloaded += 1
            except Exception as exc:
                self.log.error("  ❌ Failed to download %s: %s", name, exc)

        self.log.info("📦 Downloaded %d model file(s) to %s", downloaded, models_dir)

    def _initial_train(self):
        """
        Train a separate model per active strategy (symbol × timeframe).
        When use_external_models=true, skips training entirely if a model is
        already loaded from disk — models are trained locally and uploaded.
        """
        use_external = self.cfg.get("strategy", {}).get("use_external_models", False)

        strategies = self.active_strategies or [
            {
                "symbol":   p["symbol"],
                "name":     p["name"],
                "tf_label": self._sym_tf_label(p["symbol"]),
                "tf_min":   self._sym_tf(p["symbol"]),
            }
            for p in self.pairs
        ]

        for strat in strategies:
            symbol    = strat["symbol"]
            name      = strat["name"]
            sig_label = strat["tf_label"]
            tf_min    = strat["tf_min"]
            sym_key   = self.strategy._sym_key(symbol, sig_label)

            if use_external and sym_key in self.strategy.symbol_models:
                self.log.info("✅ %s [%s] using external model — skipping retrain.", name, sig_label)
                continue

            self.log.info("⏳ Training model for %s [%s]…", name, sig_label)
            hist = self.strategy.load_historical_candles(symbol, sig_label)
            if hist is not None and len(hist) >= self.strategy.min_samples:
                self.strategy.train(hist, symbol=symbol, timeframe_label=sig_label)
                self.log.info("✅ %s [%s] trained on %d historical candles.",
                              name, sig_label, len(hist))
            else:
                df = self.fetch_candles(symbol, tf=tf_min)
                if df is not None and len(df) >= self.strategy.min_samples:
                    self.strategy.train(df, symbol=symbol, timeframe_label=sig_label)
                    self.log.info("✅ %s [%s] trained on %d live candles (CSV not ready).",
                                  name, sig_label, len(df))
                else:
                    n = len(df) if df is not None else 0
                    self.log.info("⚠️  %s [%s]: only %d candles — will train once data builds up.",
                                  name, sig_label, n)

    def _run_mae_analysis(self):
        """
        Run MAE/MFE stop-loss optimisation analysis against the trade log.
        Logs suggestions.  If confidence is high (≥50 trades) and the suggested
        sl_atr_mult differs meaningfully from the current one, auto-updates it.
        """
        trades_file = self.cfg["logging"].get("trades_file", "/data/trades.csv")
        analyser    = MAEAnalyser(trades_file)
        current_mult = self.risk.sl_atr_mult

        report = analyser.analyse(current_sl_atr_mult=current_mult)
        if not report:
            return

        suggested = report.get("suggested_sl_atr_mult", current_mult)
        n_trades  = report.get("total_trades", 0)

        # Auto-apply only when we have enough data and a meaningful difference
        from mae_analyser import MIN_TRADES_FOR_AUTO, AUTO_APPLY_MARGIN
        diff = abs(suggested - current_mult)
        if n_trades >= MIN_TRADES_FOR_AUTO and diff > AUTO_APPLY_MARGIN * current_mult:
            self.risk.sl_atr_mult = suggested
            self.log.info("⚙️  MAE auto-applied: sl_atr_mult %.2f → %.2f "
                          "(%d trades, diff %.2f)",
                          current_mult, suggested, n_trades, diff)
        elif n_trades > 0:
            self.log.info("📐 MAE suggestion: sl_atr_mult %.2f (current %.2f) — "
                          "need %d+ trades to auto-apply (have %d)",
                          suggested, current_mult, MIN_TRADES_FOR_AUTO, n_trades)

    def _apply_mae_result(self, result: dict) -> None:
        """Apply SL/TP multipliers from a historical MAE result dict."""
        if not result or result.get("simulated_trades", 0) < 15:
            self.log.info("📐 Historical MAE: insufficient data — "
                          "will calibrate from live trades once they accumulate.")
            return

        suggested_sl = result.get("suggested_sl_atr_mult")
        suggested_tp = result.get("suggested_tp_atr_mult")
        conf         = result.get("confidence", "low")

        sl_tol = 0.05 if conf == "high" else 0.15
        if suggested_sl and abs(suggested_sl - self.risk.sl_atr_mult) > sl_tol:
            old = self.risk.sl_atr_mult
            self.risk.sl_atr_mult = suggested_sl
            self.log.info("⚙️  Historical MAE → sl_atr_mult: %.2f → %.2f  [%s confidence]",
                          old, suggested_sl, conf)
        else:
            self.log.info("⚙️  Historical MAE: SL unchanged (%.2f) — already well-calibrated",
                          self.risk.sl_atr_mult)

        if suggested_tp and suggested_tp >= self.risk.sl_atr_mult * 1.5:
            old = self.risk.tp_atr_mult
            self.risk.tp_atr_mult = suggested_tp
            self.log.info("⚙️  Historical MAE → tp_atr_mult: %.2f → %.2f  [%s confidence]",
                          old, suggested_tp, conf)

    def _run_historical_mae(self):
        """
        Walk-forward MAE/MFE backtest on historical CSV data.

        Results are cached to mae_backtest_results.json and re-used on every
        startup until they are older than analysis_stale_days (default 7 days).
        This means the expensive backtest only re-runs when the model is
        genuinely stale — not on every Railway redeploy.

        The cache is skipped and a fresh run is forced if:
          • the JSON file does not exist
          • the file is older than analysis_stale_days
          • the file is unreadable / corrupt
        """
        cache_path = os.path.join(self.data_dir, "mae_backtest_results.json")
        stale_days = self.cfg.get("data", {}).get("analysis_stale_days", 7)

        # ── FORCE_MAE_RERUN: delete cache and re-run fresh ────────────────────
        if os.getenv("FORCE_MAE_RERUN", "false").lower() == "true":
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    self.log.info("♻️  FORCE_MAE_RERUN=true — deleted MAE cache, will re-run backtest")
                except Exception as exc:
                    self.log.warning("Could not delete MAE cache: %s", exc)

        # ── Try cache first ───────────────────────────────────────────────────
        if os.path.exists(cache_path):
            age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
            if age_days < stale_days:
                try:
                    with open(cache_path) as f:
                        cached = json.load(f)
                    self.log.info(
                        "📐 Historical MAE: using cached results "
                        "(%.1f days old — re-runs after %d days)",
                        age_days, stale_days,
                    )
                    self._apply_mae_result(cached)
                    return
                except Exception as exc:
                    self.log.warning("MAE cache unreadable (%s) — re-running backtest", exc)

        # ── Run backtest ──────────────────────────────────────────────────────
        self.log.info("📐 Running historical MAE backtest…")
        # Read thresholds from config so backtest and live bot always match
        strat_cfg   = self.cfg.get("strategy", {})
        buy_thresh  = strat_cfg.get("buy_threshold",  0.45)
        sell_thresh = strat_cfg.get("sell_threshold", 0.35)
        HTF_UP     = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}
        htf_tf_min = HTF_UP.get(self.tf, "60")

        # Pass active_strategies to the backtest so it runs on every
        # (symbol × TF) combo, not just one TF per symbol
        self.strategy._backtest_strategies = self.active_strategies or []

        result = run_historical_mae(
            strategy    = self.strategy,
            symbol_tf   = self.symbol_tf,
            pairs       = self.pairs,
            data_dir    = self.data_dir,
            sl_mult     = self.risk.sl_atr_mult,
            tp_mult     = self.risk.tp_atr_mult,
            buy_thresh  = buy_thresh,
            sell_thresh = sell_thresh,
            htf_tf_min  = htf_tf_min,
        )

        # ── Cache results so future startups skip the heavy computation ───────
        if result and result.get("simulated_trades", 0) >= 15:
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(result, f, indent=2)
                self.log.info("📐 MAE results saved → %s", cache_path)
            except Exception as exc:
                self.log.warning("Could not save MAE cache: %s", exc)

        self._apply_mae_result(result or {})

    # ── LTF reversal detector ─────────────────────────────────────────────────

    def _ltf_reversal(self, df_ltf: pd.DataFrame, side: str) -> bool:
        """
        Returns True if the lower-timeframe (1h) shows a reversal signal
        that suggests the current trend is running out of steam.

        For a LONG position (looking for bearish reversal):
          • WaveTrend WT1 crossed BELOW WT2 while coming from positive territory, OR
          • MACD histogram flipped from positive to negative this candle

        For a SHORT position (looking for bullish reversal):
          • WaveTrend WT1 crossed ABOVE WT2 while coming from negative territory, OR
          • MACD histogram flipped from negative to positive
        """
        if df_ltf is None or len(df_ltf) < 30:
            return False
        try:
            from indicators import compute_features
            df = compute_features(df_ltf.copy())
            if len(df) < 2:
                return False
            last = df.iloc[-1]
            prev = df.iloc[-2]

            if side == "long":
                wt_cross_bear  = float(last.get("vmcb_wt_cross_bear", 0)) == 1.0
                wt_was_pos     = float(last.get("vmcb_wt1", 0))           >  0
                macd_flip_bear = (float(last.get("macd_diff", 0)) < 0 and
                                  float(prev.get("macd_diff", 0)) >= 0)
                return (wt_cross_bear and wt_was_pos) or macd_flip_bear
            else:
                wt_cross_bull  = float(last.get("vmcb_wt_cross_bull", 0)) == 1.0
                wt_was_neg     = float(last.get("vmcb_wt1", 0))           <  0
                macd_flip_bull = (float(last.get("macd_diff", 0)) > 0 and
                                  float(prev.get("macd_diff", 0)) <= 0)
                return (wt_cross_bull and wt_was_neg) or macd_flip_bull
        except Exception as exc:
            self.log.debug("LTF reversal check error: %s", exc)
            return False

    # ── Shared entry logic ────────────────────────────────────────────────────

    def _try_enter(self, symbol: str, df: pd.DataFrame,
                   price: float, atr: float, htf_direction: int,
                   slot_key: str = "", timeframe_label: str = "",
                   _signal: int = None, _buy_p: float = None,
                   _sell_p: float = None) -> bool:
        """
        Evaluate a potential entry for `symbol` given current candle data.
        slot_key   — position registry key e.g. "BTCUSDT_UMCBL_4h"
        timeframe_label — signal TF label e.g. "4h" (selects the right model)
        _signal/_buy_p/_sell_p — pre-computed signal (skips internal predict).
        Called from both tick() and scan_entries().
        Returns True if a position was opened.
        """
        if not slot_key:
            slot_key = symbol

        # ── Time-of-day gate ─────────────────────────────────────────────────
        current_hour = utcnow().hour
        if current_hour in self.risk.blocked_hours_utc:
            self.log.info("  ⛔ %s[%s]  entries blocked — weak hour %02d:00 UTC",
                          symbol, timeframe_label, current_hour)
            return False

        # ── News calendar gate ───────────────────────────────────────────────
        blocked_event = entries_blocked(utcnow())
        if blocked_event:
            self.log.info("  ⛔ %s[%s]  entries blocked — news event: %s",
                          symbol, timeframe_label, blocked_event)
            return False

        # ── Day-of-week gate ──────────────────────────────────────────────────
        today_dow = utcnow().weekday()  # 0=Mon … 6=Sun
        sc        = self.cfg.get("strategy", {})
        day_blocks = sc.get("day_blocks", {})
        blocked_today = day_blocks.get(today_dow, [])
        if slot_key in blocked_today:
            self.log.info("  ⛔ %s[%s]  blocked on day %d (day_blocks config)",
                          symbol, timeframe_label, today_dow)
            return False

        # ── Thursday threshold + R/R overrides ───────────────────────────────
        is_thursday = (today_dow == 3)
        if is_thursday:
            buy_thresh  = sc.get("thursday_buy_threshold",  self.strategy.buy_threshold)
            sell_thresh = sc.get("thursday_sell_threshold", self.strategy.sell_threshold)
            min_rr      = sc.get("thursday_min_rr",         self.risk.min_rr)
        else:
            buy_thresh  = self.strategy.buy_threshold
            sell_thresh = self.strategy.sell_threshold
            min_rr      = self.risk.min_rr

        if _signal is not None:
            signal, buy_p, sell_p = _signal, _buy_p, _sell_p
        else:
            # Temporarily override thresholds for prediction on Thursdays
            orig_buy  = self.strategy.buy_threshold
            orig_sell = self.strategy.sell_threshold
            self.strategy.buy_threshold  = buy_thresh
            self.strategy.sell_threshold = sell_thresh
            signal, buy_p, sell_p = self.strategy.predict(df, symbol=symbol,
                                                           timeframe_label=timeframe_label)
            self.strategy.buy_threshold  = orig_buy
            self.strategy.sell_threshold = orig_sell
            signal, buy_p, sell_p = self.strategy.apply_confluence(
                signal, buy_p, sell_p, htf_direction
            )

        confidence = buy_p if signal == BUY else sell_p if signal == SELL else max(buy_p, sell_p)

        if signal == HOLD:
            self.log.info("  %s  → HOLD  (buy_p=%.2f  sell_p=%.2f  htf=%+d)",
                          symbol, buy_p, sell_p, htf_direction)
            return False

        side = "long" if signal == BUY else "short"
        sl   = self.risk.stop_loss_price(price, atr, side)
        tp   = self.risk.take_profit_price(price, atr, side)
        rr_ok, actual_rr = self.risk.rr_acceptable(price, sl, tp, side, min_rr=min_rr)

        # EV stats are tracked per slot_key so each strategy has independent stats
        ev, win_rate, avg_rr = self.strategy.stats.ev_and_winrate(slot_key)
        ev_ok, ev_reason     = self.strategy.trade_is_worth_it(slot_key)

        qty, leverage = self.risk.calc_position(price, atr, win_rate)

        # Use per-symbol HTF label for the trade card
        htf_lbl = TF_LABELS.get(self._sym_htf(symbol), self.htf_label)

        verdict = rr_ok and ev_ok and signal != HOLD
        print_trade_card(
            pair=f"{symbol}[{timeframe_label}]" if timeframe_label else symbol,
            signal=signal, confidence=confidence,
            ev=ev, win_rate=win_rate, avg_rr=avg_rr,
            actual_rr=actual_rr, qty=qty,
            risk_amount=self.risk.risk_amount_today(),
            ev_reason=ev_reason, verdict=verdict,
            htf_label=htf_lbl, htf_direction=htf_direction,
        )

        sl_pct = abs(price - sl) / price
        if sl_pct > self.risk.max_sl_pct:
            self.log.info("  ⛔ %s[%s]  SL too wide %.2f%% > max %.2f%% — skipped",
                          symbol, timeframe_label, sl_pct * 100, self.risk.max_sl_pct * 100)
            return False

        if not rr_ok:
            self.log.info("  ⛔ %s[%s]  R/R %.2f < min %.2f — skipped",
                          symbol, timeframe_label, actual_rr, min_rr)
            return False
        if not ev_ok:
            self.log.info("  ⛔ %s[%s]  EV gate — %s", symbol, timeframe_label, ev_reason)
            return False

        can_open, reason = self.risk.can_open(slot_key, side=side)
        if not can_open:
            self.log.info("  ⛔ %s[%s]  %s", symbol, timeframe_label, reason)
            # Record skipped signals caused by the per-symbol position limit so we
            # can replay them later and decide whether allowing stacked positions
            # would have improved overall performance.
            if "already has an open" in reason or "already have an open position" in reason:
                blocking = next(
                    (k for k in self.risk.open_positions
                     if self.risk._base_symbol(k) == self.risk._base_symbol(slot_key)),
                    ""
                )
                sig_str = "BUY" if signal == BUY else "SELL"
                ev, _, _ = self.strategy.stats.ev_and_winrate(slot_key)
                self.logger.log_skipped(
                    slot_key     = slot_key,
                    symbol       = symbol,
                    timeframe    = timeframe_label,
                    signal       = sig_str,
                    confidence   = confidence,
                    entry_price  = price,
                    sl_price     = sl,
                    tp_price     = tp,
                    rr           = actual_rr,
                    ev_pct       = ev * 100 if ev is not None else None,
                    skip_reason  = reason,
                    blocking_slot = blocking,
                )
            return False
        if qty <= 0:
            self.log.warning("  ⚠️  %s[%s]  Position size is 0 — check ATR/equity",
                             symbol, timeframe_label)
            return False

        # Capture entry candle's wick levels for MAE wick-breach detection
        entry_candle_low  = float(df["low"].iloc[-1])  if "low"  in df.columns else 0.0
        entry_candle_high = float(df["high"].iloc[-1]) if "high" in df.columns else 0.0

        # Re-fetch live price at the moment of order placement — the price from
        # the top of the scan loop may be several seconds stale by this point.
        price = self.live_price(symbol, df)

        if signal == BUY:
            order_id = self._open_long(symbol, qty, price)
            if order_id:
                pos = Position(
                    pair=symbol, side="long",
                    entry_price=price, quantity=qty,
                    stop_loss=sl, take_profit=tp,
                    tp1_price=0.0, quantity_original=qty,  # TP1 disabled — pure SL/TP only
                    leverage=leverage,
                    entry_time=utcnow().isoformat(),
                    order_id=order_id,
                    entry_candle_low=entry_candle_low,
                    entry_candle_high=entry_candle_high,
                    confidence=confidence,
                )
                self.risk.open_position(pos, slot_key=slot_key)
                self._place_exchange_tpsl(symbol, "long", sl, tp, qty)
                self.log.info("🟢 LONG  %s[%s]  qty=%.5f @ £%.4f  SL=£%.4f  TP=£%.4f  conf=%.2f",
                              symbol, timeframe_label, qty, price, sl, tp, confidence)
                notify_open(symbol, "long", timeframe_label, price, sl, tp)
                return True

        elif signal == SELL:
            order_id = self._open_short(symbol, qty, price)
            if order_id:
                pos = Position(
                    pair=symbol, side="short",
                    entry_price=price, quantity=qty,
                    stop_loss=sl, take_profit=tp,
                    tp1_price=0.0, quantity_original=qty,  # TP1 disabled — pure SL/TP only
                    leverage=leverage,
                    entry_time=utcnow().isoformat(),
                    order_id=order_id,
                    entry_candle_low=entry_candle_low,
                    entry_candle_high=entry_candle_high,
                    confidence=confidence,
                )
                self.risk.open_position(pos, slot_key=slot_key)
                self._place_exchange_tpsl(symbol, "short", sl, tp, qty)
                self.log.info("🔴 SHORT %s[%s]  qty=%.5f @ £%.4f  SL=£%.4f  TP=£%.4f  conf=%.2f",
                              symbol, timeframe_label, qty, price, sl, tp, confidence)
                notify_open(symbol, "short", timeframe_label, price, sl, tp)
                return True

        return False

    # ── Model health state persistence ───────────────────────────────────────

    def _load_health_state(self) -> None:
        try:
            if os.path.exists(self._health_path):
                with open(self._health_path) as f:
                    state = json.load(f)
                lmr = state.get("last_monthly_retrain")
                if lmr:
                    self._last_monthly_retrain = datetime.fromisoformat(lmr)
                sig_ts = state.get("last_signal_ts", {})
                for slot, ts_str in sig_ts.items():
                    self._last_signal_ts[slot] = datetime.fromisoformat(ts_str)
                self.log.info("💾 Health state loaded (last retrain: %s)",
                              self._last_monthly_retrain or "never")
        except Exception as exc:
            self.log.debug("Could not load health state: %s", exc)

    def _save_health_state(self) -> None:
        try:
            state = {
                "last_monthly_retrain": (
                    self._last_monthly_retrain.isoformat()
                    if self._last_monthly_retrain else None
                ),
                "last_signal_ts": {
                    slot: ts.isoformat()
                    for slot, ts in self._last_signal_ts.items()
                },
            }
            with open(self._health_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            self.log.debug("Could not save health state: %s", exc)

    # ── Model health checks ───────────────────────────────────────────────────

    def _monthly_retrain(self) -> None:
        """
        Force a full retrain of all strategy models on the latest data.
        Called automatically every 28 days from _check_model_health().
        The CSVs are kept fresh by accumulate_data() so training data is
        always current — no extra fetch needed.

        Old model files are moved to a timestamped backup folder rather than
        deleted.  If the new models turn out to be worse, copy the backup
        .joblib files back into models_dir and restart the bot — it will load
        them on startup via _try_load_symbol_models().
        """
        self.log.info("🔄 Monthly retrain — backing up models and retraining on latest data…")

        models_dir = self.cfg["logging"]["models_dir"]
        timestamp  = utcnow().strftime("%Y-%m-%d_%H%M")
        backup_dir = os.path.join(models_dir, f"backup_{timestamp}")

        # ── Back up existing .joblib files ────────────────────────────────────
        backed_up = 0
        if os.path.isdir(models_dir):
            joblib_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
            if joblib_files:
                os.makedirs(backup_dir, exist_ok=True)
                for fname in joblib_files:
                    try:
                        import shutil
                        shutil.copy2(
                            os.path.join(models_dir, fname),
                            os.path.join(backup_dir, fname),
                        )
                        backed_up += 1
                    except Exception as exc:
                        self.log.warning("Could not back up model %s: %s", fname, exc)
                self.log.info("💾 Backed up %d model file(s) → %s", backed_up, backup_dir)

                # ── Prune old backups — keep only the 3 most recent ──────────
                # Prevents the volume filling up with months of backups.
                try:
                    all_backups = sorted([
                        d for d in os.listdir(models_dir)
                        if d.startswith("backup_") and
                        os.path.isdir(os.path.join(models_dir, d))
                    ])
                    for old in all_backups[:-3]:   # keep last 3
                        import shutil
                        shutil.rmtree(os.path.join(models_dir, old), ignore_errors=True)
                        self.log.info("🗑️  Pruned old backup: %s", old)
                except Exception as exc:
                    self.log.debug("Could not prune old backups: %s", exc)

        # ── Clear in-memory models so _initial_train() trains from scratch ────
        self.strategy.symbol_models.clear()
        self.strategy.symbol_scalers.clear()
        self.strategy.symbol_features.clear()
        self.strategy.model = None

        self._initial_train()
        self._last_monthly_retrain = utcnow()
        self._save_health_state()
        self.log.info("✅ Monthly retrain complete  (backup: %s)", backup_dir if backed_up else "none")

    def _check_model_health(self) -> None:
        """
        Run three health checks on every 4h tick:

        1. Monthly retrain — if it has been 28+ days since the last forced
           retrain, wipe and retrain all models on the fresh rolling CSV data.
           This prevents regime-drift where a model trained months ago stops
           generating signals in a changed market.

        2. Signal drought — alert via Telegram if any active strategy slot
           hasn't generated a prediction with confidence >=
           ConfidenceTracker.MIN_SIGNAL_CONFIDENCE in the past 14 days.
           A drought means the model has effectively gone silent.

        3. Confidence drift — alert if a slot's recent 14-day average
           confidence is < 85% of its 90-day baseline.  This is an early
           warning before a full drought develops.
        """
        now = utcnow()
        slots = [s["slot_key"] for s in self.active_strategies]

        # ── 1. Monthly retrain ────────────────────────────────────────────────
        use_external = self.cfg.get("strategy", {}).get("use_external_models", False)
        if use_external:
            self.log.info("📅 Monthly retrain skipped — use_external_models=true")
        else:
            retrain_interval_days = self.cfg.get("data", {}).get("retrain_interval_days", 28)
            if self._last_monthly_retrain is None:
                days_since = retrain_interval_days  # force on first check if never done
            else:
                days_since = (now.replace(tzinfo=None) -
                              self._last_monthly_retrain.replace(tzinfo=None)).days
            if days_since >= retrain_interval_days:
                self.log.info("📅 Monthly retrain due (%d days since last — interval %d days)",
                              days_since, retrain_interval_days)
                try:
                    self._monthly_retrain()
                except Exception as exc:
                    self.log.error("Monthly retrain failed: %s", exc)
            else:
                self.log.info("📅 Monthly retrain in %d day(s)",
                              retrain_interval_days - days_since)

        # ── 2. Signal drought check ───────────────────────────────────────────
        drought_days = self.cfg.get("data", {}).get("signal_drought_days", 14)
        for slot in slots:
            last_ts_str = self.strategy.conf_tracker.last_signal_ts(slot)
            if last_ts_str is None:
                # Never recorded — only alert if bot has been running long enough
                first_seen = self._last_signal_ts.get(slot)
                if first_seen and (now.replace(tzinfo=None) -
                                   first_seen.replace(tzinfo=None)).days >= drought_days:
                    msg = f"No signal recorded ever (running {drought_days}+ days)"
                    self.log.warning("⚠️  Signal drought  %s — %s", slot, msg)
                continue

            last_ts = datetime.fromisoformat(last_ts_str).replace(tzinfo=None)
            silent_days = (now.replace(tzinfo=None) - last_ts).days
            if silent_days >= drought_days:
                msg = f"No signal for {silent_days} days (last: {last_ts.strftime('%Y-%m-%d')})"
                self.log.warning("⚠️  Signal drought  %s — %s", slot, msg)
            else:
                self.log.info("💓 %s  last signal %d day(s) ago", slot, silent_days)

        # ── 3. Confidence drift check ─────────────────────────────────────────
        for slot in slots:
            is_drifting, recent_avg, baseline_avg = \
                self.strategy.conf_tracker.detect_drift(slot)
            if is_drifting:
                msg = (f"Confidence collapsing: recent 14d avg={recent_avg:.3f} "
                       f"vs 90d baseline={baseline_avg:.3f} "
                       f"({recent_avg/baseline_avg*100:.0f}% of baseline)")
                self.log.warning("📉 Confidence drift  %s — %s", slot, msg)
            elif baseline_avg > 0:
                self.log.info("📊 %s  conf: recent=%.3f  baseline=%.3f",
                              slot, recent_avg, baseline_avg)

        # Persist confidence log so it survives restarts
        conf_log_path = os.path.join(self.data_dir, "confidence_log.json")
        self.strategy.conf_tracker.save(conf_log_path)

    # ── Exit monitor (runs every 5 min regardless of signal TF) ──────────────

    def _manage_news_stops(self):
        """
        Called from monitor_exits() every 5 min.

        T-5 min → T+5 min window:
          • Position in profit  → move SL to breakeven (stays there after)
          • Position not in profit → tighten SL to 0.8% from current price
                                     but never worse than original SL
        After T+5 min:
          • Tightened (not-in-profit) stops restored to original SL
          • Breakeven stops remain at breakeven
        """
        now        = utcnow()
        tighten_event = stops_should_tighten(now)

        if tighten_event:
            # Inside the tighten window — apply to any position not yet tightened
            for slot_key, pos in list(self.risk.open_positions.items()):
                if slot_key in self._news_tightened_sl:
                    continue   # already handled this position

                symbol = pos.pair
                df     = self.fetch_candles(symbol, limit=5)
                if df is None or df.empty:
                    continue
                price = self.live_price(symbol, df)

                in_profit = (price > pos.entry_price if pos.side == "long"
                             else price < pos.entry_price)

                if in_profit:
                    if pos.stop_loss != pos.entry_price:
                        self.log.info(
                            "📰 NEWS [%s]  %s → SL moved to breakeven £%.4f",
                            tighten_event, slot_key, pos.entry_price,
                        )
                        pos.stop_loss = pos.entry_price
                    # Don't add to _news_tightened_sl — breakeven stays after news
                else:
                    original_sl = pos.stop_loss
                    if pos.side == "long":
                        tightened = price * (1 - NEWS_TIGHTEN_PCT)
                        new_sl    = max(tightened, original_sl)  # never worse than original
                    else:
                        tightened = price * (1 + NEWS_TIGHTEN_PCT)
                        new_sl    = min(tightened, original_sl)  # never worse than original

                    if new_sl != original_sl:
                        self.log.info(
                            "📰 NEWS [%s]  %s → SL tightened £%.4f → £%.4f (0.8%% from £%.4f)",
                            tighten_event, slot_key, original_sl, new_sl, price,
                        )
                        self._news_tightened_sl[slot_key] = original_sl
                        pos.stop_loss = new_sl
                    else:
                        # Tightened SL would be worse than original — leave unchanged
                        self._news_tightened_sl[slot_key] = original_sl  # mark as handled

        elif self._news_tightened_sl:
            # Window has passed — restore tightened stops
            for slot_key, original_sl in list(self._news_tightened_sl.items()):
                pos = self.risk.open_positions.get(slot_key)
                if pos:
                    self.log.info(
                        "📰 NEWS over  %s → SL restored £%.4f → £%.4f",
                        slot_key, pos.stop_loss, original_sl,
                    )
                    pos.stop_loss = original_sl
            self._news_tightened_sl.clear()

    def _catch_missed_exits(self):
        """
        Called once on startup. For every restored position, fetch 200 5m
        candles (~16 h) and immediately run the exit check so any TP/SL wick
        that occurred while the bot was down is caught before entering the
        trading loop.  Always re-checks on startup — the old guard that skipped
        positions with existing wick history was wrong because SL/TP could have
        been hit between the last save and this restart.
        """
        if not self.risk.open_positions:
            return
        for slot_key, pos in list(self.risk.open_positions.items()):
            symbol   = pos.pair
            tf_label = slot_key.replace(symbol + "_", "") if "_" in slot_key else self._sym_tf_label(symbol)
            # Extract signal TF from slot label (e.g. "5m+4h" → "5")
            entry_tf_min = tf_label.split("+")[0].replace("m", "").replace("h", "")
            try:
                int(entry_tf_min)
            except ValueError:
                entry_tf_min = "5"   # safe default
            self.log.info("🔍 Startup wick-check  %s  entry=£%.4f  SL=£%.4f  TP=£%.4f",
                          slot_key, pos.entry_price, pos.stop_loss, pos.take_profit)
            # Always fetch 5m candles — finest resolution for wick detection
            df = self.fetch_candles(symbol, tf="5", limit=200)
            if df is None or df.empty:
                self.log.warning("  No candles returned for %s — skipping", symbol)
                continue
            price = self.live_price(symbol, df)
            entry_dt = (pd.Timestamp(pos.entry_time).tz_convert(None)
                        if pd.Timestamp(pos.entry_time).tzinfo is not None
                        else pd.Timestamp(pos.entry_time))
            post_entry = df[df["timestamp"] > entry_dt]
            self.log.info("  Live price=£%.4f  post-entry candles=%d  "
                          "window_high=£%.4f  window_low=£%.4f",
                          price,
                          len(post_entry),
                          float(post_entry["high"].max()) if not post_entry.empty else 0,
                          float(post_entry["low"].min())  if not post_entry.empty else 0)
            if not post_entry.empty:
                candle_high = float(post_entry["high"].max())
                candle_low  = float(post_entry["low"].min())
            else:
                candle_high, candle_low = 0.0, 0.0
            self.risk.update_excursion(slot_key, price,
                                       candle_high=candle_high,
                                       candle_low=candle_low)
            exit_reason = self.risk.should_exit(slot_key, price,
                                                candle_high=candle_high,
                                                candle_low=candle_low)
            self.log.info("  Exit check result: %s", exit_reason or "hold")
            if exit_reason in ("stop_loss", "take_profit"):
                exit_price = pos.stop_loss if exit_reason == "stop_loss" else pos.take_profit

                # Find the actual candle where the TP/SL was breached so we can
                # log the real exit time rather than the startup detection time.
                actual_exit_ts = None
                if not post_entry.empty:
                    if exit_reason == "take_profit" and pos.side == "long":
                        hit = post_entry[post_entry["high"] >= pos.take_profit]
                    elif exit_reason == "take_profit" and pos.side == "short":
                        hit = post_entry[post_entry["low"] <= pos.take_profit]
                    elif exit_reason == "stop_loss" and pos.side == "long":
                        hit = post_entry[post_entry["low"] <= pos.stop_loss]
                    else:
                        hit = post_entry[post_entry["high"] >= pos.stop_loss]
                    if not hit.empty:
                        actual_exit_ts = hit.iloc[0]["timestamp"]

                delay_mins = None
                if actual_exit_ts is not None:
                    delay_mins = int((pd.Timestamp.utcnow() - pd.Timestamp(actual_exit_ts)).total_seconds() / 60)
                    self.log.info("  Actual exit candle: %s (~%d min ago)", actual_exit_ts, delay_mins)

                self.log.info("🚨 Startup catch: %s  %s[%s]  @ £%.4f",
                              exit_reason.upper(), symbol, tf_label, exit_price)
                self._close_pos(symbol, pos.quantity, exit_price, pos.side)
                trade = self.risk.close_position(slot_key, exit_price)
                if trade:
                    trade["slot_key"] = slot_key
                    self.logger.log_trade(trade, self.risk.equity, exit_reason)
                    self._report_trade(trade, exit_reason)
                    # Only notify via Telegram if the exit was recent (≤ 60 min).
                    # Older catches are logged internally but suppressed from Telegram
                    # to avoid stale signals appearing live.
                    if delay_mins is None or delay_mins <= 60:
                        notify_close(symbol, trade["side"], tf_label,
                                     trade["entry_price"], trade["exit_price"],
                                     trade["pnl_usdt"], exit_reason,
                                     sl=trade.get("stop_loss", 0.0),
                                     tp=trade.get("take_profit", 0.0))
                    else:
                        self.log.info("  Telegram suppressed — exit was ~%d min ago (stale)", delay_mins)

    def monitor_exits(self):
        """
        Runs every 5 min. Pure SL/TP exits only — matching the backtest methodology.
        TP1 partial close and LTF reversal exit are disabled so live behaviour
        matches exactly what was proven profitable in the walk-forward backtest.
        """
        if not self.risk.open_positions:
            return

        self._manage_news_stops()

        for slot_key, pos in list(self.risk.open_positions.items()):
            symbol = pos.pair
            tf_label = slot_key.replace(symbol + "_", "") if "_" in slot_key else self._sym_tf_label(symbol)

            df = self.fetch_candles(symbol, limit=10)
            if df is None or df.empty:
                continue
            price = self.live_price(symbol, df)

            # Always check candle wicks (paper and live) so SL/TP hits aren't missed
            # between 5-min monitor cycles.  Only use candles closed AFTER entry —
            # pre-entry wicks must be ignored.
            entry_dt = (pd.Timestamp(pos.entry_time).tz_convert(None)
                        if pd.Timestamp(pos.entry_time).tzinfo is not None
                        else pd.Timestamp(pos.entry_time))
            post_entry = df[df["timestamp"] > entry_dt] if len(df) >= 2 else pd.DataFrame()
            candle_high = float(post_entry["high"].max()) if not post_entry.empty else 0.0
            candle_low  = float(post_entry["low"].min())  if not post_entry.empty else 0.0

            self.risk.update_excursion(slot_key, price,
                                       candle_high=candle_high, candle_low=candle_low)
            exit_reason = self.risk.should_exit(slot_key, price,
                                                candle_high=candle_high, candle_low=candle_low)

            # Exchange reconciliation (live only) ─────────────────────────────
            # If the software check didn't catch an exit, verify the position
            # still exists on WEEX.  When the native TPSL order fires on the
            # exchange the position is closed there but the bot is unaware until
            # this check.  Price may have bounced before the next poll, so the
            # wick check above also missed it.
            exchange_reconciled = False
            if not self.paper and exit_reason not in ("stop_loss", "take_profit"):
                try:
                    exchange_pos = self.client.get_futures_position(symbol)
                    if exchange_pos is None:
                        # Exchange already closed the position — infer which level was hit
                        if pos.side == "long":
                            if candle_low > 0 and candle_low <= pos.stop_loss:
                                exit_reason = "stop_loss"
                            elif candle_high > 0 and candle_high >= pos.take_profit:
                                exit_reason = "take_profit"
                            else:
                                exit_reason = ("stop_loss"
                                               if abs(price - pos.stop_loss) <= abs(price - pos.take_profit)
                                               else "take_profit")
                        else:
                            if candle_high > 0 and candle_high >= pos.stop_loss:
                                exit_reason = "stop_loss"
                            elif candle_low > 0 and candle_low <= pos.take_profit:
                                exit_reason = "take_profit"
                            else:
                                exit_reason = ("stop_loss"
                                               if abs(price - pos.stop_loss) <= abs(price - pos.take_profit)
                                               else "take_profit")
                        exchange_reconciled = True
                        self.log.warning(
                            "🔄 Reconciled: %s[%s] TPSL fired on exchange (%s) — re-syncing bot state",
                            symbol, tf_label, exit_reason)
                except Exception as exc:
                    self.log.warning("Exchange position check failed for %s: %s", symbol, exc)

            if exit_reason in ("stop_loss", "take_profit"):
                # Paper or reconciled: use exact SL/TP level (exchange filled at that price).
                # Software-caught live: use current market price.
                if self.paper or exchange_reconciled:
                    exit_price = pos.stop_loss if exit_reason == "stop_loss" else pos.take_profit
                else:
                    exit_price = price
                self.log.info("🚨 %s  %s[%s]  @ £%.4f", exit_reason.upper(), symbol, tf_label, exit_price)
                # Only send a close order if the bot is initiating the close —
                # if the exchange already fired TPSL, the position is gone.
                if not exchange_reconciled:
                    self._close_pos(symbol, pos.quantity, exit_price, pos.side)
                trade = self.risk.close_position(slot_key, exit_price)
                if trade:
                    trade["slot_key"] = slot_key
                    self.logger.log_trade(trade, self.risk.equity, exit_reason)
                    self._report_trade(trade, exit_reason)
                    notify_close(symbol, trade["side"], tf_label,
                                 trade["entry_price"], trade["exit_price"],
                                 trade["pnl_usdt"], exit_reason,
                                 sl=trade.get("stop_loss", 0.0),
                                 tp=trade.get("take_profit", 0.0))
                    full_df = self.fetch_candles(symbol)
                    if full_df is not None:
                        self.strategy.record_outcome(
                            trade, full_df, symbol=symbol,
                            timeframe_label=tf_label,
                        )

    # ── Background data accumulation ─────────────────────────────────────────

    def accumulate_data(self):
        """
        Called on every 5-min monitor cycle.
        Appends any newly closed candles to all 15 CSV files.
        Each timeframe's collector skips the call if the data is already
        fresh (i.e. no new candle has closed since the last append).
        Over time this builds up unlimited history — critical for 5m/15m
        analysis which only has ~25 h / ~3 days from the initial API batch.
        """
        try:
            self.collector.collect_all(quiet=True)
        except Exception as exc:
            # Raised to WARNING so storage errors are visible in Railway logs
            self.log.warning("Background data accumulation error: %s", exc)

    def log_data_sizes(self):
        """
        Print a one-line summary of how many candles are stored per CSV.
        Called periodically from the run loop so growth is visible in logs.
        """
        from data_collector import SYMBOLS, TF_LABELS, TIMEFRAMES
        parts = []
        total_kb = 0.0
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                label = TF_LABELS[tf]
                fp = os.path.join(self.data_dir, f"{sym}_{label}.csv")
                if os.path.exists(fp):
                    try:
                        rows = sum(1 for _ in open(fp)) - 1   # subtract header
                        kb   = os.path.getsize(fp) / 1024
                        total_kb += kb
                        parts.append(f"{sym[:3]}/{label}={rows}")
                    except Exception:
                        pass
        self.log.info("📦 Data store  %.1f KB total  |  %s", total_kb, "  ".join(parts))

    # ── 15-min entry scan (runs when flat — no open positions) ───────────────

    def scan_entries(self):
        """
        Lightweight entry scan — runs every entry_scan_interval_s (15 min).
        Iterates over ALL active strategies (not just pairs), so BTC-4h and
        BTC-15m are evaluated independently.  Each strategy slot is skipped
        only if that specific slot already has an open position.
        """
        if self.risk.trading_halted():
            return

        strategies = self.active_strategies or [
            {
                "symbol":   p["symbol"],
                "name":     p["name"],
                "tf_label": self._sym_tf_label(p["symbol"]),
                "tf_min":   self._sym_tf(p["symbol"]),
                "htf_min":  self._sym_htf(p["symbol"]),
                "slot_key": p["symbol"],
            }
            for p in self.pairs
        ]

        results = []

        for strat in strategies:
            symbol    = strat["symbol"]
            name      = strat["name"]
            tf_label  = strat["tf_label"]
            tf_min    = strat["tf_min"]
            htf_min   = strat["htf_min"]
            slot_key  = strat["slot_key"]

            if slot_key in self.risk.open_positions:
                results.append(f"{name}[{tf_label}]→OPEN")
                continue

            df = self.fetch_candles(symbol, tf=tf_min)
            if df is None or len(df) < 60:
                results.append(f"{name}[{tf_label}]→NO_DATA")
                continue

            df    = compute_features(df)
            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            stype      = strat.get("strategy_type", "single")
            dir_tf_min = strat.get("dir_tf_min")

            htf_df = None
            if stype == "mtf" and dir_tf_min:
                dir_df = self.fetch_candles(symbol, tf=dir_tf_min, limit=100)
                htf_df = dir_df
                dir_lbl = TF_LABELS.get(dir_tf_min, dir_tf_min)
                signal, buy_p, sell_p = self.strategy.predict_mtf(
                    df, dir_df, symbol=symbol,
                    entry_tf_label=tf_label,
                    direction_tf_label=dir_lbl,
                )
                htf_direction = self.strategy.htf_trend(dir_df) if dir_df is not None else 0
            else:
                htf_df = self.fetch_candles(symbol, tf=htf_min, limit=100) if htf_min != tf_min else None
                htf_direction = self.strategy.htf_trend(htf_df) if htf_df is not None else 0
                signal, buy_p, sell_p = self.strategy.predict(df, symbol=symbol,
                                                               timeframe_label=tf_label)
                signal, buy_p, sell_p = self.strategy.apply_confluence(
                    signal, buy_p, sell_p, htf_direction
                )

            # Record raw confidence on every prediction — used for drift detection
            # and signal drought monitoring, regardless of whether signal passed.
            raw_confidence = max(buy_p, sell_p)
            self.strategy.conf_tracker.record(slot_key, raw_confidence)
            if raw_confidence >= self.strategy.conf_tracker.MIN_SIGNAL_CONFIDENCE:
                self._last_signal_ts[slot_key] = utcnow()

            if signal == 0:
                results.append(f"{name}[{tf_label}]→HOLD(b{buy_p:.2f}/s{sell_p:.2f})")
                continue

            # Per-slot confidence gate (e.g. XRP requires 0.60)
            slot_conf_thresholds = self.cfg.get("risk", {}).get("slot_conf_threshold", {})
            slot_min_conf = slot_conf_thresholds.get(slot_key, 0.0)
            conf_val = buy_p if signal == BUY else sell_p
            if slot_min_conf > 0 and conf_val < slot_min_conf:
                self.log.info("  ⛔ %s[%s]  conf %.3f < slot min %.2f — skipped",
                              symbol, tf_label, conf_val, slot_min_conf)
                results.append(f"{name}[{tf_label}]→CONF_GATE")
                continue

            # ADX regime filter — skip entry if 4h is choppy (ADX < threshold)
            adx_threshold = self.cfg.get("risk", {}).get("adx_threshold", 0)
            if adx_threshold > 0 and htf_df is not None and not htf_df.empty:
                adx_val = self.strategy.htf_adx(htf_df)
                if adx_val is not None and adx_val < adx_threshold:
                    self.log.info("  ⛔ %s[%s]  4h ADX %.1f < %.0f (choppy) — skipped",
                                  symbol, tf_label, adx_val, adx_threshold)
                    results.append(f"{name}[{tf_label}]→ADX_BLOCK(adx={adx_val:.1f})")
                    continue

            entered = self._try_enter(symbol, df, price, atr, htf_direction,
                                      slot_key=slot_key, timeframe_label=tf_label,
                                      _signal=signal, _buy_p=buy_p, _sell_p=sell_p)
            results.append(f"{name}[{tf_label}]→{'ENTERED' if entered else 'BLOCKED'}")

        self.log.info("🔍 Scan @ %s  |  %s",
                      utcnow().strftime("%H:%M UTC"), "  ".join(results))

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self):
        equity = self.get_equity()
        self.risk.update_equity(equity)
        self.log.info("══ Tick #%d @ %s  │  Equity: £%.2f ══",
                      self._tick_count, utcnow().strftime("%H:%M UTC"), equity)

        if self.risk.trading_halted():
            return

        strategies = self.active_strategies or [
            {
                "symbol":   p["symbol"],
                "name":     p["name"],
                "tf_label": self._sym_tf_label(p["symbol"]),
                "tf_min":   self._sym_tf(p["symbol"]),
                "htf_min":  self._sym_htf(p["symbol"]),
                "slot_key": p["symbol"],
            }
            for p in self.pairs
        ]

        for strat in strategies:
            symbol   = strat["symbol"]
            tf_min   = strat["tf_min"]
            tf_label = strat["tf_label"]
            htf_min  = strat["htf_min"]
            slot_key = strat["slot_key"]

            df = self.fetch_candles(symbol, tf=tf_min)
            if df is None or len(df) < 60:
                continue

            df    = compute_features(df)
            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            stype      = strat.get("strategy_type", "single")
            dir_tf_min = strat.get("dir_tf_min")

            if slot_key in self.risk.open_positions:
                continue

            if stype == "mtf" and dir_tf_min:
                dir_df = self.fetch_candles(symbol, tf=dir_tf_min, limit=100)
                dir_lbl = TF_LABELS.get(dir_tf_min, dir_tf_min)
                htf_direction = self.strategy.htf_trend(dir_df) if dir_df is not None else 0
                signal, buy_p, sell_p = self.strategy.predict_mtf(
                    df, dir_df, symbol=symbol,
                    entry_tf_label=tf_label,
                    direction_tf_label=dir_lbl,
                )
                self._try_enter(symbol, df, price, atr, htf_direction,
                                slot_key=slot_key, timeframe_label=tf_label,
                                _signal=signal, _buy_p=buy_p, _sell_p=sell_p)
            else:
                htf_df = self.fetch_candles(symbol, tf=htf_min, limit=100) if htf_min != tf_min else None
                htf_direction = self.strategy.htf_trend(htf_df) if htf_df is not None else 0
                self._try_enter(symbol, df, price, atr, htf_direction,
                                slot_key=slot_key, timeframe_label=tf_label)

        # Performance summary every 5 ticks
        if self._tick_count % 5 == 0:
            self.logger.print_summary()
            top_pairs = self.strategy.best_pairs()
            if top_pairs:
                self.log.info("📊 Strategy ranking by EV: %s",
                              {k: f"{v*100:+.3f}%" for k, v in top_pairs.items()})

        # MAE/MFE stop-loss optimisation — re-run every 10 ticks (~40h)
        if self._tick_count % 10 == 0:
            try:
                self._run_mae_analysis()
            except Exception as exc:
                self.log.warning("MAE analysis error: %s", exc)

        # Model health: drought, drift, monthly retrain — runs every 4h tick
        try:
            self._check_model_health()
        except Exception as exc:
            self.log.warning("Model health check error: %s", exc)

        # Push live trade report to GitHub so trades are visible without Railway access
        try:
            from push_reports import push_reports
            push_reports(self.data_dir)
        except Exception as exc:
            self.log.debug("push_reports error (non-fatal): %s", exc)

        self._tick_count += 1

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        self.log.info("🚀 Weex Futures Trading Bot  v3  starting…")
        self.log.info("   Pairs      : %s", [p["name"] for p in self.pairs])
        self.log.info("   Signal TF  : %s min (per-symbol TFs applied after analysis)", self.tf)
        self.log.info("   Filter TF  : %s (HTF confluence)", self.htf_label)
        self.log.info("   Leverage   : dynamic — auto-set per trade (max %dx)",
                      self.cfg["risk"].get("max_leverage", 20))
        self.log.info("   Account    : £%.2f  |  Risk/trade: HWM×day%% (Thu=7%% Mon/Wed/Fri=5%% Sat=4%% Tue/Sun=2.5%%)",
                      self.cfg["risk"]["initial_capital"])

        self.startup()
        notify_startup(
            [p["name"] for p in self.pairs],
            self.cfg["risk"]["initial_capital"],
        )

        # Three-speed loop:
        #   Every 5 min  → monitor_exits()   — SL/TP/TP1/LTF reversal check
        #   Every 15 min → scan_entries()    — ML entry hunt when flat
        #   Every 4 h    → tick()            — full tick: retrain, summaries, entries
        tc = self.cfg["trading"]
        MONITOR_INTERVAL = tc.get("monitor_interval_s",    300)    # 5 min
        SCAN_INTERVAL    = tc.get("entry_scan_interval_s", 900)    # 15 min
        SIGNAL_INTERVAL  = tc["loop_interval_s"]                   # 4 h

        last_scan_time   = 0.0   # force an entry scan almost immediately
        last_signal_time = 0.0   # force a full tick on startup
        monitor_cycle    = 0     # counts 5-min cycles for periodic reporting
        last_summary_date = None

        while True:
            try:
                now = time.time()

                # 1. Always check SL / TP / TP1 / LTF reversal + accumulate data
                self.monitor_exits()
                self.accumulate_data()
                self._write_state()
                monitor_cycle += 1

                # Log data store sizes every hour (12 × 5-min cycles)
                if monitor_cycle % 12 == 0:
                    self.log_data_sizes()

                # Daily Telegram summary at 02:29 UK time
                uk_now = datetime.now(ZoneInfo("Europe/London"))
                if uk_now.hour == 2 and uk_now.minute == 29 and last_summary_date != uk_now.date():
                    self.log.info("📊 Sending daily Telegram summary (%s UK)", uk_now.strftime("%H:%M"))
                    send_daily_summary(
                        m1_csv=self.logger.trades_file,
                        m2_csv=self.cfg["logging"].get("m2_trades_file", ""),
                    )
                    last_summary_date = uk_now.date()

                # 2. Full 4h tick — retraining, performance summary, entries
                if now - last_signal_time >= SIGNAL_INTERVAL:
                    self.tick()
                    last_signal_time = time.time()
                    last_scan_time   = time.time()   # reset so scan doesn't fire immediately after

                # 3. 15-min entry scan — only runs when no position is open
                elif now - last_scan_time >= SCAN_INTERVAL:
                    self.scan_entries()
                    last_scan_time = time.time()

            except KeyboardInterrupt:
                self.log.info("Shutdown requested by user.")
                self.logger.print_summary()
                break
            except Exception as exc:
                self.log.exception("Unexpected error in main loop: %s", exc)

            time.sleep(MONITOR_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.getenv("RUN_BACKTEST", "false").lower() == "true":
        # ── Backtest mode ─────────────────────────────────────────────────────
        # Set RUN_BACKTEST=true on Railway to run a walk-forward backtest over
        # all profitable strategies from analysis, then exit.
        # Remove the env var to resume normal trading.
        import backtest as bt
        import yaml as _yaml
        import json as _json

        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(levelname)-8s  %(message)s")
        _log = logging.getLogger("backtest-main")
        _log.info("═══ RUN_BACKTEST=true — entering backtest mode ═══")

        _cfg = {}
        if os.path.exists("config.yaml"):
            with open("config.yaml") as _f:
                _cfg = _yaml.safe_load(_f)

        # Step 1: collect fresh data
        _log.info("Step 1/3  Collecting data…")
        try:
            from data_collector import DataCollector
            from weex_client import WeexClient as _WeexClient
            _ec = _cfg.get("exchange", {})
            _client = _WeexClient(
                api_key=_ec["api_key"], api_secret=_ec["api_secret"],
                passphrase=_ec["passphrase"], base_url=_ec["base_url"],
            )
            _data_dir = _cfg.get("data", {}).get("data_dir", "/data")
            _collector = DataCollector(_client, data_dir=_data_dir)
            _collector.collect_all()
        except Exception as _exc:
            _log.warning("Data collection error (continuing): %s", _exc)

        # Step 2: run analysis so profitable_strategies + confluence_results exist
        _log.info("Step 2/3  Running analysis…")
        try:
            from analysis import Analyzer
            _data_dir = _cfg.get("data", {}).get("data_dir", "/data")
            _analyzer = Analyzer(data_dir=_data_dir, results_dir=_data_dir)
            _analyzer.run()
        except Exception as _exc:
            _log.warning("Analysis error (continuing with existing results): %s", _exc)

        # Step 3: backtest every profitable strategy identified by analysis
        _log.info("Step 3/3  Running walk-forward backtests…")
        _data_dir = _cfg.get("data", {}).get("data_dir", "/data")
        _rc       = _cfg.get("risk", {})
        _sl_mult  = _rc.get("stop_loss_atr_mult",   2.2)
        _tp_mult  = _rc.get("take_profit_atr_mult", 3.5)

        _models_dir  = _cfg.get("logging", {}).get("models_dir", "/data/models/")
        _all_results = bt.run_all(_data_dir, _sl_mult, _tp_mult, models_dir=_models_dir)
        bt.print_report(_all_results)

        _out = os.path.join(_data_dir, "backtest_results.json")
        with open(_out, "w") as _f:
            _json.dump({"results": _all_results}, _f, indent=2, default=str)
        _log.info("Full results saved → %s", _out)

        # Push summary reports to GitHub so they're readable without Railway access
        try:
            from push_reports import push_reports
            push_reports(_data_dir)
        except Exception as _exc:
            _log.warning("Could not push reports to GitHub: %s", _exc)

        _log.info("═══ Backtest complete — exiting ═══")
        raise SystemExit(0)

    if os.getenv("PRINT_TEMPORAL", "false").lower() == "true":
        # ── Temporal report mode ──────────────────────────────────────────────
        # Set PRINT_TEMPORAL=true on Railway to print day-of-week and monthly
        # win-rate tables from the saved backtest_results.json, then exit.
        # Remove the env var to resume normal trading.
        from assess import analyse_backtest_temporal
        import yaml as _yaml
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(levelname)-8s  %(message)s")
        _cfg      = {}
        if os.path.exists("config.yaml"):
            with open("config.yaml") as _f:
                _cfg = _yaml.safe_load(_f)
        _data_dir = _cfg.get("data", {}).get("data_dir", "/data")
        analyse_backtest_temporal(_data_dir)
        raise SystemExit(0)

    bot = TradingBot("config.yaml")
    bot.run()
