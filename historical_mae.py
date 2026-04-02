"""
historical_mae.py  —  Historical MAE/MFE backtest for pre-trade SL calibration

Problem it solves
─────────────────
The live MAEAnalyser only learns from real closed trades, so it can't improve
the stop-loss BEFORE the first trade is taken.  This module runs a walk-forward
simulation over historical OHLCV data to produce the same MAE/MFE statistics
BEFORE any live position is ever opened.

By the time the bot takes trade #1, it already has a data-driven stop-loss
calibrated from hundreds of simulated historical entries.

How it works
────────────
1.  Load the historical CSV for each pair at its optimal signal timeframe
    (same data the model was trained on).

2.  To minimise look-ahead, simulate on a HOLDOUT window:
      • First 70% of candles → ignored (model trained on this)
      • Last 30% of candles → used for walk-forward simulation
    If the holdout produces fewer than MIN_SIM_TRADES results, the full
    dataset is used as a fallback (better noisy calibration than none).

3.  Pre-compute all technical features (indicators) in one pass rather than
    re-running on every candle — much faster.

4.  For each candle in the holdout:
      a. Pass the pre-computed feature row through the already-trained model.
      b. If the model emits a BUY or SELL signal above the confidence threshold,
         record an "entry" at that candle's close price.
      c. Walk forward through the next MAX_HOLD candles:
           – Track the running max/min price (highs/lows) to build MAE and MFE
           – Stop when price first hits the ATR-derived SL or TP
           – Record whether the entry candle's wick was breached

5.  Feed all simulated trade records into MAEAnalyser._find_optimal_threshold()
    to find the MAE% that keeps ≥85% of winners alive while cutting ≥70% of
    losers early.

6.  Convert optimal MAE% → sl_atr_mult via the observed ATR% at entry:
      sl_atr_mult = optimal_sl_pct / median(atr / entry_price × 100)

Output
──────
A dict with:
    suggested_sl_atr_mult  — ready to plug into config / RiskManager
    suggested_tp_atr_mult  — derived from observed MFE on winners
    simulated_trades       — count used for calibration
    per_pair               — per-pair breakdown
    confidence             — "low" / "medium" / "high" based on trade count
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from indicators import compute_features, FEATURE_COLS
from strategy   import BUY, SELL, HOLD
from mae_analyser import _percentile

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How many future candles to look at after each simulated entry
# (keyed by timeframe string e.g. "5", "15", "60", "240", "1440")
MAX_HOLD_CANDLES = {
    "5":    48,    # 4 h of 5-min bars
    "15":   32,    # 8 h of 15-min bars
    "60":   24,    # 1 day of 1-h bars
    "240":  18,    # 3 days of 4-h bars
    "1440": 10,    # 10 days of daily bars
}

# Fraction of history to keep as a holdout (avoid look-ahead when possible)
HOLDOUT_FRACTION = 0.30

# Minimum simulated trades needed for the result to be meaningful
MIN_SIM_TRADES   = 15

# Confidence thresholds for the historical backtest.
# Deliberately looser than the live bot thresholds (0.60/0.40) so that
# overfit models — which assign near-zero probabilities to everything
# in the holdout window — still generate enough signals for calibration.
DEFAULT_BUY_THRESHOLD  = 0.45
DEFAULT_SELL_THRESHOLD = 0.35

# Clamp on suggested multipliers
MIN_MULT = 0.5
MAX_MULT = 3.5   # tighter ceiling — 5x would mean a near-useless SL

# Sanity gate: if the optimal MAE% is more than this multiple of the
# median ATR%, the winner/loser distributions overlap too heavily
# (usually means the model is not providing useful signal separation).
# Skip calibration for that symbol and keep the current multiplier.
MAX_MAE_ATR_RATIO = 2.5

# Maximum single-deployment change to sl_atr_mult (50% up or down).
# Prevents a single noisy result from doubling or halving the stop.
MAX_MULT_CHANGE_FACTOR = 0.50


# ─────────────────────────────────────────────────────────────────────────────
def _simulate_trade(
    entry_price:       float,
    entry_candle_low:  float,
    entry_candle_high: float,
    atr:               float,
    side:              str,          # "long" | "short"
    future_ohlc:       pd.DataFrame, # rows after the entry candle
    sl_mult:           float,
    tp_mult:           float,
) -> Optional[dict]:
    """
    Simulate a single trade and return its MAE / MFE / outcome.

    Returns None if the future window is empty or entry data is invalid.

    The simulation walks candle-by-candle through `future_ohlc` and checks each
    candle's high and low against the ATR-derived SL and TP.  It stops at the
    first candle that touches either level, or at the end of the window.

    Stop-loss / take-profit levels:
      long  → SL = entry − atr × sl_mult,  TP = entry + atr × tp_mult
      short → SL = entry + atr × sl_mult,  TP = entry − atr × tp_mult
    """
    if future_ohlc is None or future_ohlc.empty or entry_price <= 0 or atr <= 0:
        return None

    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    if side == "long":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    mae_pct = 0.0
    mfe_pct = 0.0
    outcome = "timeout"   # neither SL nor TP hit within max_hold candles

    for _, candle in future_ohlc.iterrows():
        low  = float(candle["low"])
        high = float(candle["high"])

        if side == "long":
            adverse   = max(0.0, (entry_price - low)  / entry_price) * 100
            favorable = max(0.0, (high - entry_price) / entry_price) * 100
        else:
            adverse   = max(0.0, (high - entry_price) / entry_price) * 100
            favorable = max(0.0, (entry_price - low)  / entry_price) * 100

        mae_pct = max(mae_pct, adverse)
        mfe_pct = max(mfe_pct, favorable)

        # Check SL / TP hit (use candle's range — worst-first assumption)
        if side == "long":
            if low  <= sl_price:
                outcome = "loss"; break
            if high >= tp_price:
                outcome = "win";  break
        else:
            if high >= sl_price:
                outcome = "loss"; break
            if low  <= tp_price:
                outcome = "win";  break

    # Wick breach: did the worst adverse price go past the entry candle's wick?
    worst_price = (entry_price * (1 - mae_pct / 100) if side == "long"
                   else entry_price * (1 + mae_pct / 100))
    if side == "long":
        wick_breach = 1 if (entry_candle_low  > 0 and worst_price < entry_candle_low)  else 0
    else:
        wick_breach = 1 if (entry_candle_high > 0 and worst_price > entry_candle_high) else 0

    return {
        "mae_pct":     mae_pct,
        "mfe_pct":     mfe_pct,
        "win":         outcome == "win",
        "wick_breach": wick_breach,
        "outcome":     outcome,
        "atr_pct":     atr / entry_price * 100 if entry_price > 0 else 0.0,
        "entry_price": entry_price,
    }


# ─────────────────────────────────────────────────────────────────────────────
class HistoricalMAEBacktest:
    """
    Runs a walk-forward MAE/MFE simulation on historical OHLCV CSVs.

    Parameters
    ----------
    data_dir   : where the CSVs live (e.g. /data)
    sl_mult    : current SL ATR multiplier (from config)
    tp_mult    : current TP ATR multiplier (from config)
    buy_thresh : model buy confidence threshold
    sell_thresh: model sell confidence threshold
    """

    def __init__(self,
                 data_dir:    str   = "/data",
                 sl_mult:     float = 1.5,
                 tp_mult:     float = 3.0,
                 buy_thresh:  float = DEFAULT_BUY_THRESHOLD,
                 sell_thresh: float = DEFAULT_SELL_THRESHOLD,
                 htf_tf_min:  str   = "60"):
        self.data_dir    = data_dir
        self.sl_mult     = sl_mult
        self.tp_mult     = tp_mult
        self.buy_thresh  = buy_thresh
        self.sell_thresh = sell_thresh
        # Higher timeframe used for confluence filtering (same as live bot)
        self.htf_tf_min  = htf_tf_min

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, strategy, symbol_tf: Dict[str, str],
            pairs: List[dict]) -> dict:
        """
        Run historical MAE backtest for all enabled pairs.

        Parameters
        ----------
        strategy   : TradingStrategy instance (already trained)
        symbol_tf  : {symbol → tf_minutes_str}  e.g. {"ETHUSDT_UMCBL": "15"}
        pairs      : list of pair config dicts (from config.yaml)

        Returns dict with suggested multipliers and diagnostics.
        """
        all_trades: List[dict] = []
        per_pair:   Dict[str, dict] = {}

        # Build the list of (symbol, tf_min, name) to backtest.
        # If active_strategies is available (multi-strategy engine) use it so
        # every (symbol × TF) combo gets its own backtest result.
        # Falls back to the old pairs × symbol_tf approach for compatibility.
        active_strategies = getattr(strategy, "_backtest_strategies", None)
        if active_strategies:
            strategy_list = [
                (s["symbol"], s["tf_min"], f"{s['name']}[{s['tf_label']}]")
                for s in active_strategies
            ]
        else:
            strategy_list = [
                (p["symbol"],
                 symbol_tf.get(p["symbol"], "240"),
                 p.get("name", p["symbol"]))
                for p in pairs
            ]

        for symbol, tf_min, name in strategy_list:
            tf_label = _TF_LABEL.get(tf_min, f"{tf_min}m")

            logger.info("📐 Historical MAE backtest: %s [%s]…", name, tf_label)

            trades = self._backtest_pair(strategy, symbol, tf_min, tf_label)
            all_trades.extend(trades)

            if trades:
                w = [t for t in trades if t["win"]]
                l = [t for t in trades if not t["win"]]
                eq = self._equity_stats(trades)
                per_pair[name] = {
                    "simulated_trades": len(trades),
                    "wins":   len(w),
                    "losses": len(l),
                    "avg_mae_win":  round(sum(t["mae_pct"] for t in w) / max(len(w), 1), 3),
                    "avg_mae_loss": round(sum(t["mae_pct"] for t in l) / max(len(l), 1), 3),
                    "avg_mfe_win":  round(sum(t["mfe_pct"] for t in w) / max(len(w), 1), 3),
                    "tf":           tf_label,
                    "equity":       eq,
                }
            else:
                per_pair[name] = {"simulated_trades": 0, "tf": tf_label}

        if len(all_trades) < MIN_SIM_TRADES:
            logger.info("📐 Historical MAE: only %d simulated trades — need %d."
                        "  SL calibration will happen once live trades accumulate.",
                        len(all_trades), MIN_SIM_TRADES)
            return {"simulated_trades": len(all_trades), "per_pair": per_pair}

        # ── Aggregate and optimise across all pairs ───────────────────────────
        return self._optimise(all_trades, per_pair)

    # ── Per-pair simulation ───────────────────────────────────────────────────

    def _backtest_pair(self, strategy,
                       symbol: str, tf_min: str, tf_label: str) -> List[dict]:
        """Walk-forward simulation for one pair. Returns list of trade records."""

        # ── Load historical CSV ───────────────────────────────────────────────
        sym_clean = symbol.replace("_SPBL", "").replace("_UMCBL", "")
        csv_path  = os.path.join(self.data_dir, f"{sym_clean}_{tf_label}.csv")
        if not os.path.exists(csv_path):
            logger.warning("  No CSV for %s %s — skipping", sym_clean, tf_label)
            return []

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("  Could not load %s: %s", csv_path, exc)
            return []

        if len(df) < 120:
            logger.info("  %s %s: only %d candles — need 120+ for backtest",
                        sym_clean, tf_label, len(df))
            return []

        # Ensure low/high columns are present
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                logger.warning("  %s missing '%s' column — skipping", csv_path, col)
                return []

        # ── Pre-compute all features in one pass ──────────────────────────────
        try:
            df_feat = compute_features(df.copy())
        except Exception as exc:
            logger.warning("  compute_features failed for %s: %s", sym_clean, exc)
            return []

        # ── Resolve model + scaler for this symbol ────────────────────────────
        sym_key = symbol.replace("_SPBL", "").replace("_UMCBL", "")
        if sym_key in strategy.symbol_models:
            model  = strategy.symbol_models[sym_key]
            scaler = strategy.symbol_scalers[sym_key]
            feats  = strategy.symbol_features[sym_key]
        elif strategy.model is not None:
            model  = strategy.model
            scaler = strategy.scaler
            feats  = strategy.selected_features
        else:
            logger.warning("  No trained model for %s — skipping", sym_key)
            return []

        feats = [f for f in feats if f in df_feat.columns]
        if not feats:
            return []

        classes = list(model.classes_)

        # ── Determine simulation window (holdout = last 30%) ──────────────────
        max_hold  = MAX_HOLD_CANDLES.get(tf_min, 20)
        n_total   = len(df_feat)
        holdout_start = int(n_total * (1 - HOLDOUT_FRACTION))
        # We also need at least 100 candles of warm-up for features to stabilise
        sim_start = max(100, holdout_start)
        # Stop early enough that we have max_hold candles to look forward
        sim_end   = n_total - max_hold - 1

        if sim_end <= sim_start:
            logger.info("  %s %s: not enough holdout candles", sym_clean, tf_label)
            return []

        # ── Load HTF data for confluence filter (mirrors live bot behaviour) ────
        # For every signal candle at index i, we look up the corresponding
        # HTF candle and suppress signals that fight the HTF trend —
        # exactly what apply_confluence() does in the live bot.
        df_htf    = None
        htf_ratio = 1
        if self.htf_tf_min and self.htf_tf_min != tf_min:
            htf_label = _TF_LABEL.get(self.htf_tf_min, "1h")
            htf_csv   = os.path.join(self.data_dir, f"{sym_clean}_{htf_label}.csv")
            if os.path.exists(htf_csv):
                try:
                    df_htf    = pd.read_csv(htf_csv)
                    htf_ratio = max(1, int(self.htf_tf_min) // int(tf_min))
                except Exception as exc:
                    logger.debug("Could not load HTF CSV %s: %s", htf_csv, exc)
                    df_htf = None

        # ── Walk-forward through holdout window ───────────────────────────────
        trades: List[dict] = []

        for i in range(sim_start, sim_end):
            row = df_feat.iloc[[i]][feats].copy()

            # Skip if any feature is NaN / inf
            row.replace([np.inf, -np.inf], np.nan, inplace=True)
            if row.isnull().any().any():
                continue

            # Predict
            try:
                X_scaled = pd.DataFrame(scaler.transform(row), columns=feats)
                proba    = model.predict_proba(X_scaled)[0]
            except Exception:
                continue

            buy_p  = float(proba[classes.index(BUY)])  if BUY  in classes else 0.0
            sell_p = float(proba[classes.index(SELL)]) if SELL in classes else 0.0

            if buy_p >= self.buy_thresh:
                signal = "long"
            elif sell_p >= self.sell_thresh:
                signal = "short"
            else:
                continue   # HOLD

            # ── HTF confluence filter (matches live apply_confluence logic) ───
            if df_htf is not None:
                htf_i      = min(i // htf_ratio, len(df_htf) - 1)
                htf_window = df_htf.iloc[max(0, htf_i - 49): htf_i + 1]
                htf_dir    = strategy.htf_trend(htf_window)
                # Suppress signal if it contradicts the HTF trend
                if signal == "long"  and htf_dir == -1:
                    continue   # bearish 1h suppresses BUY — same as live
                if signal == "short" and htf_dir ==  1:
                    continue   # bullish 1h suppresses SELL — same as live

            # Entry data from the raw (non-feature) row
            entry_price = float(df.iloc[i]["close"])
            candle_low  = float(df.iloc[i]["low"])
            candle_high = float(df.iloc[i]["high"])
            atr_val     = float(df_feat.iloc[i].get("atr_14", entry_price * 0.01))

            # Simulate this trade
            future = df.iloc[i + 1: i + 1 + max_hold]
            result = _simulate_trade(
                entry_price       = entry_price,
                entry_candle_low  = candle_low,
                entry_candle_high = candle_high,
                atr               = atr_val,
                side              = signal,
                future_ohlc       = future,
                sl_mult           = self.sl_mult,
                tp_mult           = self.tp_mult,
            )
            if result is not None:
                result["symbol"]    = sym_clean
                result["tf"]        = tf_label
                result["entry_idx"] = i
                trades.append(result)

        # If holdout gives too few trades, fall back to full dataset
        if len(trades) < MIN_SIM_TRADES:
            logger.info("  %s %s: holdout gave %d trades — retrying on full history",
                        sym_clean, tf_label, len(trades))
            return self._backtest_full(strategy, df, df_feat, model, scaler, feats, classes,
                                       sym_clean, tf_label, tf_min, max_hold)

        logger.info("  %s %s: %d simulated trades  (%d W / %d L)  from holdout",
                    sym_clean, tf_label, len(trades),
                    sum(1 for t in trades if t["win"]),
                    sum(1 for t in trades if not t["win"]))
        return trades

    def _backtest_full(self, strategy, df, df_feat, model, scaler, feats, classes,
                       sym_clean, tf_label, tf_min, max_hold) -> List[dict]:
        """Fallback: run simulation over the whole dataset (with minor look-ahead)."""

        # Load HTF data for confluence (same as holdout path)
        df_htf    = None
        htf_ratio = 1
        if self.htf_tf_min and self.htf_tf_min != tf_min:
            htf_label = _TF_LABEL.get(self.htf_tf_min, "1h")
            htf_csv   = os.path.join(self.data_dir, f"{sym_clean}_{htf_label}.csv")
            if os.path.exists(htf_csv):
                try:
                    df_htf    = pd.read_csv(htf_csv)
                    htf_ratio = max(1, int(self.htf_tf_min) // int(tf_min))
                except Exception:
                    df_htf = None

        trades: List[dict] = []
        sim_end = len(df_feat) - max_hold - 1

        for i in range(100, sim_end):
            row = df_feat.iloc[[i]][feats].copy()
            row.replace([np.inf, -np.inf], np.nan, inplace=True)
            if row.isnull().any().any():
                continue
            try:
                X_scaled = pd.DataFrame(scaler.transform(row), columns=feats)
                proba    = model.predict_proba(X_scaled)[0]
            except Exception:
                continue

            buy_p  = float(proba[classes.index(BUY)])  if BUY  in classes else 0.0
            sell_p = float(proba[classes.index(SELL)]) if SELL in classes else 0.0

            if buy_p >= self.buy_thresh:
                signal = "long"
            elif sell_p >= self.sell_thresh:
                signal = "short"
            else:
                continue

            # HTF confluence filter
            if df_htf is not None:
                htf_i      = min(i // htf_ratio, len(df_htf) - 1)
                htf_window = df_htf.iloc[max(0, htf_i - 49): htf_i + 1]
                htf_dir    = strategy.htf_trend(htf_window)
                if signal == "long"  and htf_dir == -1:
                    continue
                if signal == "short" and htf_dir ==  1:
                    continue

            entry_price = float(df.iloc[i]["close"])
            candle_low  = float(df.iloc[i]["low"])
            candle_high = float(df.iloc[i]["high"])
            atr_val     = float(df_feat.iloc[i].get("atr_14", entry_price * 0.01))

            future = df.iloc[i + 1: i + 1 + max_hold]
            result = _simulate_trade(
                entry_price=entry_price, entry_candle_low=candle_low,
                entry_candle_high=candle_high, atr=atr_val, side=signal,
                future_ohlc=future, sl_mult=self.sl_mult, tp_mult=self.tp_mult,
            )
            if result is not None:
                result["symbol"]    = sym_clean
                result["tf"]        = tf_label
                result["entry_idx"] = i
                trades.append(result)

        logger.info("  %s %s: %d simulated trades (full history fallback)  (%d W / %d L)",
                    sym_clean, tf_label, len(trades),
                    sum(1 for t in trades if t["win"]),
                    sum(1 for t in trades if not t["win"]))
        return trades

    # ── Equity curve statistics ───────────────────────────────────────────────

    def _equity_stats(self, trades: List[dict],
                      initial_equity: float = 100.0,
                      risk_per_trade: float = 5.0) -> dict:
        """
        Simulate an equity curve from a chronological list of backtest trades.

        Assumes fixed-risk sizing:
          Win  →  +risk × (tp_mult / sl_mult)
          Loss →  −risk

        Returns peak, trough, max drawdown, win/loss streaks, final PnL, R/R.
        """
        if not trades:
            return {}

        rr      = self.tp_mult / self.sl_mult   # e.g. 2.0
        equity  = initial_equity
        peak    = initial_equity
        trough  = initial_equity
        running_peak = initial_equity

        win_streak  = loss_streak  = 0
        max_win_streak = max_loss_streak = 0
        max_drawdown_abs = 0.0
        peak_at_max_dd   = initial_equity   # running_peak snapshot when max_dd was set

        # Sort by entry_idx so equity curve is chronological
        ordered = sorted(trades, key=lambda t: t.get("entry_idx", 0))

        for t in ordered:
            if t["win"]:
                equity      += risk_per_trade * rr
                win_streak  += 1
                loss_streak  = 0
            else:
                equity      -= risk_per_trade
                loss_streak += 1
                win_streak   = 0

            max_win_streak  = max(max_win_streak,  win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)
            peak   = max(peak,   equity)
            trough = min(trough, equity)

            # Running drawdown — track the peak that produced it for a meaningful %
            running_peak = max(running_peak, equity)
            dd = running_peak - equity
            if dd > max_drawdown_abs:
                max_drawdown_abs = dd
                peak_at_max_dd   = running_peak

        wins   = sum(1 for t in trades if     t["win"])
        losses = sum(1 for t in trades if not t["win"])

        return {
            "initial_equity":    round(initial_equity, 2),
            "final_equity":      round(equity, 2),
            "total_pnl":         round(equity - initial_equity, 2),
            "total_pnl_pct":     round((equity - initial_equity) / initial_equity * 100, 1),
            "peak_equity":       round(peak, 2),
            "trough_equity":     round(trough, 2),
            "max_drawdown_abs":  round(max_drawdown_abs, 2),
            # % expressed relative to the peak at the time of the drawdown (not initial capital)
            "max_drawdown_pct":  round(max_drawdown_abs / peak_at_max_dd * 100, 1) if peak_at_max_dd > 0 else 0.0,
            "max_win_streak":    max_win_streak,
            "max_loss_streak":   max_loss_streak,
            "rr_ratio":          round(rr, 2),
            "win_rate_pct":      round(wins / max(len(trades), 1) * 100, 1),
            "total_trades":      len(trades),
            "wins":              wins,
            "losses":            losses,
        }

    # ── Optimisation ─────────────────────────────────────────────────────────

    def _optimise(self, all_trades: List[dict],
                  per_pair: Dict[str, dict]) -> dict:
        """
        Find optimal sl_atr_mult and tp_atr_mult by optimising PER SYMBOL,
        then taking the median across symbols that passed the sanity check.

        Why per-symbol rather than aggregate?
        Mixing 1d BTC trades (ATR ~2.5%) with 15m ETH trades (ATR ~0.15%)
        in one MAE distribution produces a nonsense threshold driven entirely
        by the highest-ATR pair.  Each pair calibrates its own SL using its
        own ATR as the denominator, then we take the median suggested multiplier
        across all symbols.

        Sanity check: if the optimal MAE% is > MAX_MAE_ATR_RATIO × median ATR%,
        the winner/loser distributions overlap too badly (usually a sign the
        model is not producing useful signal separation on that pair/TF).
        We skip that pair and log a warning rather than let it corrupt the result.
        """
        sym_sl_suggestions: List[float] = []
        sym_tp_suggestions: List[float] = []

        # Confidence is based on total trade count — compute early so it's
        # available in the early-return path (before the per-symbol loop).
        n = len(all_trades)
        confidence = "high" if n >= 100 else "medium" if n >= 30 else "low"

        # Group trades by symbol+TF
        groups: Dict[str, List[dict]] = {}
        for t in all_trades:
            key = f"{t['symbol']}_{t['tf']}"
            groups.setdefault(key, []).append(t)

        for key, trades in groups.items():
            winners = [t for t in trades if     t["win"]]
            losers  = [t for t in trades if not t["win"]]
            if len(winners) < 3 or len(losers) < 3:
                logger.info("  %s: too few W/L for per-symbol calibration (%dW/%dL) — skipping",
                            key, len(winners), len(losers))
                continue

            win_maes  = sorted(t["mae_pct"] for t in winners)
            loss_maes = sorted(t["mae_pct"] for t in losers)
            win_mfes  = [t["mfe_pct"] for t in winners]
            atr_pcts  = [t["atr_pct"] for t in trades if t["atr_pct"] > 0]

            optimal_sl_pct, keep, cut = _find_threshold(win_maes, loss_maes)
            median_atr_pct = _percentile(sorted(atr_pcts), 50) if atr_pcts else 0.0

            if median_atr_pct <= 0:
                logger.info("  %s: ATR%% is zero — skipping", key)
                continue

            # ── Sanity check: reject if distributions too mixed ────────────────
            ratio = optimal_sl_pct / median_atr_pct
            if ratio > MAX_MAE_ATR_RATIO:
                logger.info(
                    "  %s: optimal SL %.3f%% = %.1f× ATR (max %.1f×) — "
                    "winner/loser distributions overlap too much, skipping.",
                    key, optimal_sl_pct, ratio, MAX_MAE_ATR_RATIO
                )
                continue

            sl_mult_sym = round(
                max(MIN_MULT, min(MAX_MULT, optimal_sl_pct / median_atr_pct)), 2
            )
            p70_mfe = _percentile(sorted(win_mfes), 70) if win_mfes else 0.0
            tp_mult_sym = round(
                max(MIN_MULT, min(MAX_MULT, p70_mfe / median_atr_pct)), 2
            ) if p70_mfe > 0 else None

            logger.info(
                "  %s: opt_SL=%.3f%%  ATR%%=%.3f%%  → sl_mult=%.2f  "
                "(keeps %.1f%% W, cuts %.1f%% L)",
                key, optimal_sl_pct, median_atr_pct, sl_mult_sym,
                keep * 100, cut * 100
            )
            sym_sl_suggestions.append(sl_mult_sym)
            if tp_mult_sym is not None:
                sym_tp_suggestions.append(tp_mult_sym)

        if not sym_sl_suggestions:
            logger.info("📐 Historical MAE: no symbols passed the sanity check — "
                        "keeping current SL mult (%.2f).", self.sl_mult)
            overall_equity = self._equity_stats(all_trades)
            skip_result = {
                "simulated_trades":      len(all_trades),
                "wins":                  len([t for t in all_trades if t["win"]]),
                "losses":                len([t for t in all_trades if not t["win"]]),
                "suggested_sl_atr_mult": self.sl_mult,
                "suggested_tp_atr_mult": self.tp_mult,
                "current_sl_atr_mult":   self.sl_mult,
                "current_tp_atr_mult":   self.tp_mult,
                "per_pair":              per_pair,
                "overall_equity":        overall_equity,
                "skipped_calibration":   True,
                "per_symbol_sl_mults":   [],
                "win_mae_p50": 0, "win_mae_p80": 0, "loss_mae_p50": 0,
                "wick_breach_rate": 0, "wick_win_rate": 0, "wick_loss_rate": 0,
                "p70_mfe_winner": 0, "confidence": confidence,
                "symbols_calibrated": 0,
            }
            self._log_report(skip_result)
            return skip_result

        # ── Aggregate: median of per-symbol suggestions ───────────────────────
        sym_sl_suggestions.sort()
        suggested_sl_mult = round(_percentile(sym_sl_suggestions, 50), 2)

        # Apply the max-single-change cap (don't move more than 50% in one go)
        max_change  = self.sl_mult * MAX_MULT_CHANGE_FACTOR
        suggested_sl_mult = round(
            max(self.sl_mult - max_change,
                min(self.sl_mult + max_change, suggested_sl_mult)), 2
        )

        if sym_tp_suggestions:
            sym_tp_suggestions.sort()
            suggested_tp_mult = round(_percentile(sym_tp_suggestions, 50), 2)
            suggested_tp_mult = max(MIN_MULT, min(MAX_MULT, suggested_tp_mult))
        else:
            suggested_tp_mult = self.tp_mult

        # Ensure minimum R/R of 1.5:1
        if suggested_tp_mult < suggested_sl_mult * 1.5:
            suggested_tp_mult = round(suggested_sl_mult * 2.0, 2)

        # ── Wick breach across all trades ─────────────────────────────────────
        winners_all = [t for t in all_trades if     t["win"]]
        losers_all  = [t for t in all_trades if not t["win"]]
        wick_total  = sum(t["wick_breach"] for t in all_trades)
        wick_win    = sum(t["wick_breach"] for t in winners_all)
        wick_loss   = sum(t["wick_breach"] for t in losers_all)
        wick_rate   = wick_total / max(len(all_trades), 1) * 100
        wick_win_r  = wick_win   / max(len(winners_all), 1) * 100
        wick_loss_r = wick_loss  / max(len(losers_all),  1) * 100

        all_win_maes  = sorted(t["mae_pct"] for t in winners_all)
        all_loss_maes = sorted(t["mae_pct"] for t in losers_all)
        all_win_mfes  = [t["mfe_pct"] for t in winners_all]

        overall_equity = self._equity_stats(all_trades)

        result = {
            "simulated_trades":      n,
            "wins":                  len(winners_all),
            "losses":                len(losers_all),
            "symbols_calibrated":    len(sym_sl_suggestions),
            "win_mae_p50":           round(_percentile(all_win_maes,  50), 3) if all_win_maes  else 0,
            "win_mae_p80":           round(_percentile(all_win_maes,  80), 3) if all_win_maes  else 0,
            "loss_mae_p50":          round(_percentile(all_loss_maes, 50), 3) if all_loss_maes else 0,
            "per_symbol_sl_mults":   sym_sl_suggestions,
            "suggested_sl_atr_mult": suggested_sl_mult,
            "suggested_tp_atr_mult": suggested_tp_mult,
            "current_sl_atr_mult":   self.sl_mult,
            "current_tp_atr_mult":   self.tp_mult,
            "wick_breach_rate":      round(wick_rate, 1),
            "wick_win_rate":         round(wick_win_r, 1),
            "wick_loss_rate":        round(wick_loss_r, 1),
            "p70_mfe_winner":        round(_percentile(sorted(all_win_mfes), 70), 3) if all_win_mfes else 0,
            "confidence":            confidence,
            "per_pair":              per_pair,
            "overall_equity":        overall_equity,
        }

        self._log_report(result)
        return result

    def _log_report(self, r: dict):
        n    = r["simulated_trades"]
        conf = r.get("confidence", "?").upper()
        n_sym = r.get("symbols_calibrated", "?")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("  📐 HISTORICAL MAE BACKTEST  [confidence: %s]", conf)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("  Simulated trades  : %d  (%d W / %d L)  across %s symbol(s)",
                    n, r.get("wins", 0), r.get("losses", 0), n_sym)
        logger.info("  Per-symbol SL mults: %s",
                    [f"{v:.2f}" for v in r.get("per_symbol_sl_mults", [])])
        logger.info("")
        logger.info("  ── Winner MAE (adverse move before recovering) ──────")
        logger.info("  P50: %.3f%%   P80: %.3f%%",
                    r.get("win_mae_p50", 0), r.get("win_mae_p80", 0))
        logger.info("  ── Loser MAE ─────────────────────────────────────────")
        logger.info("  P50: %.3f%%", r.get("loss_mae_p50", 0))
        logger.info("")
        logger.info("  SL mult:  current=%.2f  →  suggested=%.2f",
                    r["current_sl_atr_mult"], r["suggested_sl_atr_mult"])
        logger.info("  TP mult:  current=%.2f  →  suggested=%.2f  (from P70 MFE=%.3f%%)",
                    r["current_tp_atr_mult"], r["suggested_tp_atr_mult"],
                    r.get("p70_mfe_winner", 0))

        sl_diff = r["suggested_sl_atr_mult"] - r["current_sl_atr_mult"]
        if abs(sl_diff) < 0.1:
            logger.info("  ✅ Current SL is already well-calibrated (Δ < 0.1)")
        elif sl_diff < 0:
            logger.info("  ⚡ TIGHTEN your stop — winners rarely go this deep before recovering")
        else:
            logger.info("  ⚡ WIDEN your stop — too many winners are being stopped out early")

        logger.info("")
        logger.info("  ── Entry candle wick breach ──────────────────────────")
        logger.info("  Overall: %.1f%%   Winners: %.1f%%   Losers: %.1f%%",
                    r["wick_breach_rate"], r["wick_win_rate"], r["wick_loss_rate"])
        if r["wick_win_rate"] < 20:
            logger.info("  ✅ Entry wick is a reliable SL anchor — few winners breach it")
        elif r["wick_win_rate"] > 50:
            logger.info("  ⚠️  Entry wick often breached by winners — placing SL there may be too tight")

        # ── Equity curve summary ──────────────────────────────────────────────
        eq = r.get("overall_equity", {})
        if eq:
            logger.info("")
            logger.info("  ── Equity simulation  (£%.0f start, £5 risk/trade, %.1f:1 R/R) ──",
                        eq.get("initial_equity", 100), eq.get("rr_ratio", 2.0))
            logger.info("  Win rate     : %.1f%%  (%dW / %dL  from %d trades)",
                        eq["win_rate_pct"], eq["wins"], eq["losses"], eq["total_trades"])
            logger.info("  Final equity : £%.2f  (P&L %+.2f  /  %+.1f%%)",
                        eq["final_equity"], eq["total_pnl"], eq["total_pnl_pct"])
            logger.info("  Peak         : £%.2f      Trough       : £%.2f",
                        eq["peak_equity"], eq["trough_equity"])
            logger.info("  Max drawdown : £%.2f  (%.1f%% from peak)",
                        eq["max_drawdown_abs"], eq["max_drawdown_pct"])
            logger.info("  Best streak  : %d wins in a row",  eq["max_win_streak"])
            logger.info("  Worst streak : %d losses in a row", eq["max_loss_streak"])

        logger.info("")
        logger.info("  ── Per-pair breakdown ────────────────────────────────")
        for name, ps in r["per_pair"].items():
            if ps.get("simulated_trades", 0) > 0:
                logger.info("  %-8s  tf=%-4s  trades=%-4d  W/L=%d/%d  MAE_W=%.3f%%  MFE_W=%.3f%%",
                            name, ps.get("tf", "?"), ps["simulated_trades"],
                            ps.get("wins", 0), ps.get("losses", 0),
                            ps.get("avg_mae_win", 0), ps.get("avg_mfe_win", 0))
                eq_p = ps.get("equity", {})
                if eq_p:
                    logger.info("            P&L=%+.2f (%.1f%%)  Peak=£%.2f  Trough=£%.2f  "
                                "MaxDD=£%.2f (%.1f%%)  BestRun=%d  WorstRun=%d",
                                eq_p.get("total_pnl", 0),
                                eq_p.get("total_pnl_pct", 0),
                                eq_p.get("peak_equity", 0),
                                eq_p.get("trough_equity", 0),
                                eq_p.get("max_drawdown_abs", 0),
                                eq_p.get("max_drawdown_pct", 0),
                                eq_p.get("max_win_streak", 0),
                                eq_p.get("max_loss_streak", 0))
            else:
                logger.info("  %-8s  tf=%-4s  no data", name, ps.get("tf", "?"))

        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ── Shared helpers ─────────────────────────────────────────────────────────────

_TF_LABEL = {"5": "5m", "15": "15m", "60": "1h", "240": "4h", "1440": "1d"}


def _find_threshold(win_maes: list, loss_maes: list,
                    keep_pct: float = 0.85,
                    cut_pct:  float = 0.70) -> Tuple[float, float, float]:
    """
    Find the lowest MAE threshold that keeps ≥keep_pct of winners and
    cuts ≥cut_pct of losers.  Returns (threshold, keep_fraction, cut_fraction).
    """
    candidates = sorted(set(win_maes + loss_maes))
    best_t, best_score, best_keep, best_cut = candidates[-1], -1.0, 1.0, 1.0

    for t in candidates:
        keep = sum(1 for m in win_maes  if m <= t) / max(len(win_maes),  1)
        cut  = sum(1 for m in loss_maes if m <= t) / max(len(loss_maes), 1)
        if keep >= keep_pct and cut >= cut_pct:
            score = -t   # prefer the tightest threshold that satisfies both constraints
            if score > best_score:
                best_t, best_score, best_keep, best_cut = t, score, keep, cut

    return best_t, best_keep, best_cut


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function called from bot.py
# ─────────────────────────────────────────────────────────────────────────────

def run_historical_mae(strategy, symbol_tf: dict, pairs: list,
                       data_dir:    str   = "/data",
                       sl_mult:     float = 1.5,
                       tp_mult:     float = 3.0,
                       buy_thresh:  float = DEFAULT_BUY_THRESHOLD,
                       sell_thresh: float = DEFAULT_SELL_THRESHOLD,
                       htf_tf_min:  str   = "60") -> dict:
    """
    Top-level helper. Instantiates HistoricalMAEBacktest and runs it.
    Returns the result dict (empty dict if not enough data).

    buy_thresh / sell_thresh should be passed from the live config so the
    backtest uses identical thresholds to the live bot.
    htf_tf_min is the higher timeframe used for confluence filtering
    (e.g. "60" for 1h when signal TF is 15m).
    """
    bt = HistoricalMAEBacktest(
        data_dir    = data_dir,
        sl_mult     = sl_mult,
        tp_mult     = tp_mult,
        buy_thresh  = buy_thresh,
        sell_thresh = sell_thresh,
        htf_tf_min  = htf_tf_min,
    )
    try:
        return bt.run(strategy, symbol_tf, pairs)
    except Exception as exc:
        logger.warning("Historical MAE backtest error: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from strategy import TradingStrategy
    from data_collector import TF_LABELS

    strategy   = TradingStrategy(cfg)
    pairs      = [p for p in cfg["trading"]["pairs"] if p["enabled"]]
    symbol_tf  = {}   # will use default TF for each pair

    result = run_historical_mae(
        strategy   = strategy,
        symbol_tf  = symbol_tf,
        pairs      = pairs,
        data_dir   = cfg.get("data", {}).get("data_dir", "/data"),
        sl_mult    = cfg["risk"]["stop_loss_atr_mult"],
        tp_mult    = cfg["risk"]["take_profit_atr_mult"],
    )

    if result:
        print(f"\nSuggested sl_atr_mult : {result.get('suggested_sl_atr_mult', 'N/A')}")
        print(f"Suggested tp_atr_mult : {result.get('suggested_tp_atr_mult', 'N/A')}")
