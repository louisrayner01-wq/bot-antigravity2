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
from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)

from typing import Dict, Optional

import pandas as pd

from weex_client   import WeexClient
from indicators    import candles_to_df
from strategy      import TradingStrategy, BUY, SELL, HOLD
from risk_manager  import RiskManager, Position
from trade_logger  import TradeLogger
from data_collector import DataCollector, TF_LABELS
from analysis      import Analyzer
from mae_analyser  import MAEAnalyser
from historical_mae import run_historical_mae


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
    verdict_str = "✅ TAKE THE TRADE" if verdict else "❌ SKIP (EV/Win-rate gate)"

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

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg  = load_config(config_path)
        setup_logging(self.cfg)
        self.log  = logging.getLogger("Bot")

        ec = self.cfg["exchange"]
        self.paper    = self.cfg["trading"]["paper_trading"]
        self.pairs    = [p for p in self.cfg["trading"]["pairs"] if p["enabled"]]
        self.tf       = str(self.cfg["trading"]["timeframe"])
        self.lookback = self.cfg["trading"]["lookback_candles"]
        self.data_dir = self.cfg.get("data", {}).get("data_dir", "/data")

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
        self.log.info("%s MODE  |  Account: £%.2f  |  Risk/trade: £%.2f  |  Lev: dynamic (max %dx)",
                      mode,
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"],
                      self.cfg["risk"].get("max_leverage", 20))

        self.client    = WeexClient(
            api_key=ec["api_key"], api_secret=ec["api_secret"],
            passphrase=ec["passphrase"], base_url=ec["base_url"],
        )
        self.collector = DataCollector(self.client, data_dir=self.data_dir)
        self.analyzer  = Analyzer(data_dir=self.data_dir, results_dir=self.data_dir)
        self.strategy  = TradingStrategy(self.cfg)
        self.risk      = RiskManager(self.cfg)
        self.logger    = TradeLogger(self.cfg["logging"]["trades_file"])
        self._tick_count = 0

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

        # ── Step 0: Set leverage for all pairs ───────────────────────────────
        self._setup_leverage()

        # ── Step 1: Data collection ───────────────────────────────────────────
        self.log.info("STEP 1/3  Data collection")
        try:
            self.collector.collect_all()
        except Exception as exc:
            self.log.error("Data collection error (continuing anyway): %s", exc)

        # ── Step 2: Analysis ──────────────────────────────────────────────────
        self.log.info("STEP 2/3  Strategy analysis")
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

        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.info("  STARTUP COMPLETE — entering trading loop")
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
        recs = self.strategy.analysis.get("recommendations", {})
        rev_map = {v: k for k, v in TF_LABELS.items()}

        # ── Auto-select HTF based on signal TF (next timeframe up) ────────────
        HTF_UP       = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}
        TF_INTERVALS = {"5": 300, "15": 900, "60": 3600, "240": 14400, "1440": 86400}

        # ── Switch signal timeframe if analysis found a better one ─────────────
        best_signal = recs.get("best_signal_timeframe")  # e.g. "4h"
        if best_signal:
            sig_min = rev_map.get(best_signal)
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
        # min_signal_tf_minutes: floor — 5m is too noisy/frequent for signals
        # max_signal_tf_minutes: ceiling — 1d has only 730 candles, too thin
        # 15m (~70k candles), 1h (~17k), 4h (~4.4k) all compete freely on CV.
        data_cfg   = self.cfg.get("data", {})
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

        Strategy list comes from analysis profitable_strategies (all (symbol, TF)
        combos that scored >= min_strategy_accuracy in backtesting).
        Falls back to one entry per enabled pair if analysis produced nothing.
        """
        HTF_UP      = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}
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

    def _initial_train(self):
        """
        Train a separate model per active strategy (symbol × timeframe).
        Skips training if a backtest-saved model is already loaded for that
        slot — the backtest model is preferred as it was trained on the full
        70% split and its confidence scores are directly comparable to the
        0.55 threshold proven in walk-forward testing.
        """
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

            # Skip if backtest already saved a model for this slot
            sym_key = self.strategy._sym_key(symbol, sig_label)
            if sym_key in self.strategy.symbol_models:
                self.log.info("✅ %s [%s] — backtest model loaded, skipping retrain.", name, sig_label)
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
        if _signal is not None:
            signal, buy_p, sell_p = _signal, _buy_p, _sell_p
        else:
            signal, buy_p, sell_p = self.strategy.predict(df, symbol=symbol,
                                                           timeframe_label=timeframe_label)
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
        rr_ok, actual_rr = self.risk.rr_acceptable(price, sl, tp, side)

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
            risk_amount=self.risk.risk_per_trade_abs,
            ev_reason=ev_reason, verdict=verdict,
            htf_label=htf_lbl, htf_direction=htf_direction,
        )

        if not rr_ok:
            self.log.info("  ⛔ %s[%s]  R/R %.2f < min %.2f — skipped",
                          symbol, timeframe_label, actual_rr, self.risk.min_rr)
            return False
        if not ev_ok:
            self.log.info("  ⛔ %s[%s]  EV gate — %s", symbol, timeframe_label, ev_reason)
            return False

        can_open, reason = self.risk.can_open(slot_key)
        if not can_open:
            self.log.info("  ⛔ %s[%s]  %s", symbol, timeframe_label, reason)
            # Record skipped signals caused by the per-symbol position limit so we
            # can replay them later and decide whether allowing stacked positions
            # would have improved overall performance.
            if "symbol already has an open position" in reason or "already have an open position" in reason:
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
                )
                self.risk.open_position(pos, slot_key=slot_key)
                self.log.info("🟢 LONG  %s[%s]  qty=%.5f @ £%.4f  SL=£%.4f  TP=£%.4f",
                              symbol, timeframe_label, qty, price, sl, tp)
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
                )
                self.risk.open_position(pos, slot_key=slot_key)
                self.log.info("🔴 SHORT %s[%s]  qty=%.5f @ £%.4f  SL=£%.4f  TP=£%.4f",
                              symbol, timeframe_label, qty, price, sl, tp)
                return True

        return False

    # ── Exit monitor (runs every 5 min regardless of signal TF) ──────────────

    def monitor_exits(self):
        """
        Runs every 5 min. Pure SL/TP exits only — matching the backtest methodology.
        TP1 partial close and LTF reversal exit are disabled so live behaviour
        matches exactly what was proven profitable in the walk-forward backtest.
        """
        if not self.risk.open_positions:
            return

        for slot_key, pos in list(self.risk.open_positions.items()):
            symbol = pos.pair
            tf_label = slot_key.replace(symbol + "_", "") if "_" in slot_key else self._sym_tf_label(symbol)

            df = self.fetch_candles(symbol, limit=10)
            if df is None or df.empty:
                continue
            price = self.live_price(symbol, df)

            self.risk.update_excursion(slot_key, price)

            exit_reason = self.risk.should_exit(slot_key, price)

            if exit_reason in ("stop_loss", "take_profit"):
                self.log.info("🚨 %s  %s[%s]  @ £%.4f", exit_reason.upper(), symbol, tf_label, price)
                self._close_pos(symbol, pos.quantity, price, pos.side)
                trade = self.risk.close_position(slot_key, price)
                if trade:
                    trade["slot_key"] = slot_key
                    self.logger.log_trade(trade, self.risk.equity, exit_reason)
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

            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            stype      = strat.get("strategy_type", "single")
            dir_tf_min = strat.get("dir_tf_min")

            if stype == "mtf" and dir_tf_min:
                dir_df = self.fetch_candles(symbol, tf=dir_tf_min, limit=100)
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

            if signal == 0:
                results.append(f"{name}[{tf_label}]→HOLD(b{buy_p:.2f}/s{sell_p:.2f})")
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

        self._tick_count += 1

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        self.log.info("🚀 Weex Futures Trading Bot  v3  starting…")
        self.log.info("   Pairs      : %s", [p["name"] for p in self.pairs])
        self.log.info("   Signal TF  : %s min (per-symbol TFs applied after analysis)", self.tf)
        self.log.info("   Filter TF  : %s (HTF confluence)", self.htf_label)
        self.log.info("   Leverage   : dynamic — auto-set per trade (max %dx)",
                      self.cfg["risk"].get("max_leverage", 20))
        self.log.info("   Account    : £%.2f  |  Risk/trade: £%.2f",
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"])

        self.startup()

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

        while True:
            try:
                now = time.time()

                # 1. Always check SL / TP / TP1 / LTF reversal + accumulate data
                self.monitor_exits()
                self.accumulate_data()
                monitor_cycle += 1

                # Log data store sizes every hour (12 × 5-min cycles)
                if monitor_cycle % 12 == 0:
                    self.log_data_sizes()

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
