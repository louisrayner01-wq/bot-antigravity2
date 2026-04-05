"""
backtest.py
═══════════
Walk-forward backtest for every profitable strategy identified by analysis.

Covers three strategy types
────────────────────────────
  single-TF    — model trained on one timeframe, no filter
                 e.g. BTCUSDT 4h
  confluence   — model trained on entry_tf, only enters when filter_tf
                 EMA21 trend agrees (same as live apply_confluence logic)
                 e.g. BTCUSDT 15m signal, 1h filter
  MTF          — model trained on entry_tf, only enters when direction_tf
                 EMA21+MACD trend agrees (same as live predict_mtf logic)
                 e.g. BTCUSDT 4h direction, 15m entry

Methodology
───────────
  • Strategy source : analysis_results.json (produced by analysis.py)
                      Falls back to all symbol × TF combos if file missing
  • Train split     : first 70% of entry-TF candles  → model training
  • Test split      : last  30% of entry-TF candles  → simulated trading
  • No lookahead    : filter/direction TF data is forward-filled up to each
                      entry candle's timestamp — never uses future data
  • Entry           : candle close when model confidence ≥ min_confidence
                      AND (for confluence/MTF) filter trend agrees
  • SL / TP         : entry ± ATR × multiplier from config.yaml
  • Exit            : first subsequent candle whose high/low touches SL or TP
  • One trade at a time per slot (no pyramiding)
  • PnL             : % move from entry to SL or TP (no leverage, no fees)

Metrics reported per strategy
──────────────────────────────
  total trades  |  wins  |  losses  |  win rate
  avg win %  |  avg loss %  |  avg R/R
  EV % per trade  |  total PnL %
  longest win streak  |  longest loss streak
  max drawdown from peak
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import yaml
import joblib
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               ExtraTreesClassifier,
                               VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from indicators import compute_features, FEATURE_COLS
from data_collector import TF_LABELS, TIMEFRAMES, SYMBOLS
from analysis import compute_adaptive_threshold, label_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("backtest")

BUY   =  1
HOLD  =  0
SELL  = -1

TRAIN_SPLIT      = 0.70   # first 70% = train, last 30% = test
MIN_TRAIN_ROWS   = 200
MIN_TRAIN_LABELS = 50


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> CalibratedClassifierCV:
    rf  = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  min_samples_leaf=10, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=80, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    et  = ExtraTreesClassifier(n_estimators=100, max_depth=8,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)], voting="soft"
    )
    return CalibratedClassifierCV(voter, cv=3)


# ── HTF trend aligned to entry timestamps ─────────────────────────────────────

def _htf_trend_series(df_filter: pd.DataFrame,
                      entry_timestamps: pd.Series) -> pd.Series:
    """
    Compute EMA21 trend on df_filter and forward-fill it onto entry_timestamps.
    At each entry candle we only use filter data available up to that moment
    (no lookahead). Returns a Series of +1 / -1 / 0 by integer position.
    """
    df_f = df_filter[["timestamp", "close"]].copy()
    df_f = df_f.sort_values("timestamp").drop_duplicates("timestamp")
    df_f["ema21"] = df_f["close"].ewm(span=21, adjust=False).mean()
    df_f["trend"] = np.where(df_f["close"] > df_f["ema21"],  1,
                    np.where(df_f["close"] < df_f["ema21"], -1, 0))
    df_f = df_f.set_index("timestamp")["trend"]

    entry_ts = pd.to_datetime(entry_timestamps).values
    filter_ts = df_f.index

    # For each entry timestamp, find the most recent filter candle <= entry ts
    combined = df_f.reindex(filter_ts.union(pd.DatetimeIndex(entry_ts)))
    combined = combined.ffill()
    result = combined.reindex(pd.DatetimeIndex(entry_ts)).values
    return pd.Series(result, dtype=float).fillna(0)


# ── Streak computation ────────────────────────────────────────────────────────

def _streaks(outcomes: List[str]) -> Tuple[int, int]:
    """Return (longest_win_streak, longest_loss_streak)."""
    max_win = max_loss = cur_win = cur_loss = 0
    for o in outcomes:
        if o == "WIN":
            cur_win  += 1
            cur_loss  = 0
        else:
            cur_loss += 1
            cur_win   = 0
        max_win  = max(max_win,  cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


# ── Max drawdown from peak ────────────────────────────────────────────────────

def _max_drawdown(pnls: List[float]) -> float:
    equity = np.cumsum([0.0] + pnls)
    peak   = np.maximum.accumulate(equity)
    return float(np.min(equity - peak))


# ── Core trade simulation loop ────────────────────────────────────────────────

def _simulate(test: pd.DataFrame,
              probas: np.ndarray,
              classes: list,
              sl_mult: float,
              tp_mult: float,
              min_confidence: float,
              htf_trend: Optional[pd.Series] = None) -> List[Dict]:
    """
    Simulate trades on the test split.

    htf_trend — optional Series (integer-indexed, same length as test) giving
                the higher-TF trend (+1 bullish / -1 bearish / 0 neutral) at
                each test candle. When provided, only signals that agree with
                the trend are taken.
    """
    buy_idx  = classes.index(BUY)  if BUY  in classes else None
    sell_idx = classes.index(SELL) if SELL in classes else None

    trades: List[Dict] = []
    in_trade   = False
    direction  = 0
    entry_price = sl = tp = 0.0
    entry_ts   = None

    for i in range(len(test)):
        row   = test.iloc[i]
        high  = row["high"]
        low   = row["low"]
        close = row["close"]
        atr   = row["atr_14"]
        ts    = row["timestamp"]

        # ── Exit: check SL/TP on this candle ─────────────────────────────────
        if in_trade:
            hit_sl = (direction == BUY  and low  <= sl) or \
                     (direction == SELL and high >= sl)
            hit_tp = (direction == BUY  and high >= tp) or \
                     (direction == SELL and low  <= tp)

            if hit_tp or hit_sl:
                exit_price = tp if hit_tp else sl
                pnl_pct    = (exit_price - entry_price) / entry_price * direction * 100
                trades.append({
                    "entry_ts":    str(entry_ts),
                    "exit_ts":     str(ts),
                    "direction":   "BUY" if direction == BUY else "SELL",
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exit_price, 6),
                    "sl":          round(sl, 6),
                    "tp":          round(tp, 6),
                    "pnl_pct":     round(pnl_pct, 4),
                    "outcome":     "WIN" if hit_tp else "LOSS",
                })
                in_trade = False

        # ── Entry: model confidence + optional HTF gate ───────────────────────
        if not in_trade:
            prob      = probas[i]
            buy_conf  = prob[buy_idx]  if buy_idx  is not None else 0.0
            sell_conf = prob[sell_idx] if sell_idx is not None else 0.0

            if buy_conf >= min_confidence and buy_conf >= sell_conf:
                sig = BUY
            elif sell_conf >= min_confidence and sell_conf > buy_conf:
                sig = SELL
            else:
                continue

            # HTF filter gate
            if htf_trend is not None:
                trend = int(htf_trend.iloc[i]) if i < len(htf_trend) else 0
                if sig == BUY  and trend != 1:
                    continue
                if sig == SELL and trend != -1:
                    continue

            direction   = sig
            entry_price = close
            entry_ts    = ts
            sl = entry_price - atr * sl_mult if direction == BUY \
                 else entry_price + atr * sl_mult
            tp = entry_price + atr * tp_mult if direction == BUY \
                 else entry_price - atr * tp_mult
            in_trade = True

    return trades


# ── Temporal breakdown (day-of-week + monthly win rates) ─────────────────────

def _temporal_breakdown(trades: List[Dict]) -> Dict:
    """
    Slice a strategy's trade list by day-of-week and calendar month.
    Returns a dict with 'by_dow' and 'by_month' — both included in the
    result JSON and printed by print_temporal_report().
    """
    if not trades:
        return {"by_dow": {}, "by_month": {}}

    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    by_dow:   Dict[str, Dict] = {d: {"trades": 0, "wins": 0, "pnl": 0.0} for d in DOW}
    by_month: Dict[str, Dict] = {}

    for t in trades:
        try:
            ts = pd.Timestamp(t["entry_ts"])
        except Exception:
            continue

        win = t["outcome"] == "WIN"
        pnl = float(t["pnl_pct"])

        # Day of week
        dow = DOW[ts.dayofweek]
        by_dow[dow]["trades"] += 1
        by_dow[dow]["wins"]   += int(win)
        by_dow[dow]["pnl"]    += pnl

        # Calendar month  e.g. "2024-03"
        month = ts.strftime("%Y-%m")
        if month not in by_month:
            by_month[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_month[month]["trades"] += 1
        by_month[month]["wins"]   += int(win)
        by_month[month]["pnl"]    += pnl

    # Add win_rate field
    for bucket in list(by_dow.values()) + list(by_month.values()):
        n = bucket["trades"]
        bucket["win_rate"] = round(bucket["wins"] / n, 4) if n else None
        bucket["pnl"]      = round(bucket["pnl"], 4)

    return {"by_dow": by_dow, "by_month": dict(sorted(by_month.items()))}


def print_temporal_report(all_results: List[Dict]) -> None:
    """
    Print day-of-week and monthly win-rate tables for every strategy.
    Reads the 'temporal' key that _summarise() attaches to each result.
    """
    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗",
        "║                      TEMPORAL BREAKDOWN — win rate by day & month                           ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════╝",
    ]

    for r in sorted(all_results, key=lambda x: x["ev_pct"], reverse=True):
        temporal = r.get("temporal", {})
        if not temporal:
            continue

        sym   = r["symbol"]
        tf    = r["timeframe"]
        ftf   = r.get("filter_tf", "")
        label = f"{sym}  {tf}+{ftf}" if ftf else f"{sym}  {tf}"

        lines.append("")
        lines.append(f"  ── {label}  (total trades: {r['total_trades']}, overall WR: {r['win_rate']*100:.1f}%) ──")

        # Day-of-week table
        by_dow = temporal.get("by_dow", {})
        if by_dow:
            lines.append(f"  {'Day':<6}  {'Trades':>6}  {'Wins':>5}  {'WR%':>6}  {'PnL%':>8}  {'Signal':>8}")
            lines.append("  " + "─" * 48)
            for dow in DOW_ORDER:
                b  = by_dow.get(dow, {})
                n  = b.get("trades", 0)
                wr = b.get("win_rate")
                pnl = b.get("pnl", 0.0)
                if n == 0:
                    lines.append(f"  {dow:<6}  {'—':>6}")
                    continue
                wr_pct  = f"{wr*100:>5.1f}%"
                bar_val = wr * 100 if wr else 0
                bar     = "█" * int(bar_val / 10) + ("▌" if bar_val % 10 >= 5 else "")
                flag    = "✅" if wr and wr >= 0.50 else ("⚠️ " if wr and wr >= 0.44 else "❌")
                lines.append(f"  {dow:<6}  {n:>6}  {b['wins']:>5}  {wr_pct}  {pnl:>+7.2f}%  {flag} {bar}")

        # Monthly table
        by_month = temporal.get("by_month", {})
        if by_month:
            lines.append("")
            lines.append(f"  {'Month':<9}  {'Trades':>6}  {'Wins':>5}  {'WR%':>6}  {'PnL%':>8}")
            lines.append("  " + "─" * 42)
            for month, b in by_month.items():
                n   = b.get("trades", 0)
                wr  = b.get("win_rate")
                pnl = b.get("pnl", 0.0)
                if n == 0:
                    continue
                wr_pct = f"{wr*100:>5.1f}%" if wr is not None else "  N/A "
                flag   = "✅" if wr and wr >= 0.50 else ("⚠️ " if wr and wr >= 0.44 else "❌")
                lines.append(f"  {month:<9}  {n:>6}  {b['wins']:>5}  {wr_pct}  {pnl:>+7.2f}%  {flag}")

    lines.append("")
    for line in lines:
        log.info(line)


# ── Summarise trade list into result dict ─────────────────────────────────────

def _summarise(trades: List[Dict],
               symbol: str,
               label: str,
               train_rows: int,
               test_rows: int,
               threshold: float,
               strategy_type: str,
               filter_tf: str = "") -> Optional[Dict]:
    if not trades:
        return None

    pnls     = [t["pnl_pct"] for t in trades]
    outcomes = [t["outcome"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]

    win_rate  = len(wins) / len(pnls)
    avg_win   = float(np.mean(wins))   if wins   else 0.0
    avg_loss  = float(np.mean(losses)) if losses else 0.0
    total_pnl = float(np.sum(pnls))
    ev        = win_rate * avg_win + (1 - win_rate) * avg_loss
    avg_rr    = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    max_dd    = _max_drawdown(pnls)
    win_str, loss_str = _streaks(outcomes)

    result = {
        "symbol":            symbol,
        "timeframe":         label,
        "filter_tf":         filter_tf,
        "strategy_type":     strategy_type,
        "train_rows":        train_rows,
        "test_rows":         test_rows,
        "total_trades":      len(trades),
        "wins":              len(wins),
        "losses":            len(losses),
        "win_rate":          round(win_rate, 4),
        "avg_win_pct":       round(avg_win, 4),
        "avg_loss_pct":      round(avg_loss, 4),
        "avg_rr":            round(avg_rr, 4),
        "ev_pct":            round(ev, 4),
        "total_pnl_pct":     round(total_pnl, 4),
        "max_drawdown_pct":  round(max_dd, 4),
        "longest_win_streak":  win_str,
        "longest_loss_streak": loss_str,
        "threshold":         round(threshold, 4),
        "trades":            trades,
        "temporal":          _temporal_breakdown(trades),
    }

    tag = f"{filter_tf}→{label}" if filter_tf else label
    log.info(
        "  %-8s %-12s  trades=%3d  wins=%3d  losses=%3d  "
        "wr=%5.1f%%  rr=%4.2f  ev=%+.3f%%  pnl=%+7.2f%%  "
        "dd=%6.2f%%  W-streak=%d  L-streak=%d",
        symbol, tag, len(trades), len(wins), len(losses),
        win_rate * 100, avg_rr, ev, total_pnl,
        max_dd, win_str, loss_str,
    )
    return result


# ── Train model on entry_tf ───────────────────────────────────────────────────

def _train(df_raw: pd.DataFrame,
           symbol: str,
           tf_label: str,
           models_dir: Optional[str] = None) -> Optional[Tuple]:
    """
    Compute features, label, split 70/30, train model on train split.
    Returns (scaler, model, classes, train_df, test_df, threshold) or None.

    If models_dir is provided, saves the trained model + scaler in the exact
    format strategy.py expects — so the live bot loads and continues from this
    model directly without retraining.
    """
    df = compute_features(df_raw.copy())
    threshold = compute_adaptive_threshold(df)
    df["label"] = label_candles(df, threshold=threshold)

    feats = [c for c in FEATURE_COLS if c in df.columns]
    # Build column list without duplicates — atr_14 is already in FEATURE_COLS
    extra = [c for c in ["label", "high", "low", "close", "atr_14", "timestamp"]
             if c not in feats]
    cols  = feats + extra
    df    = df[cols].dropna(subset=feats + ["atr_14"])

    n     = len(df)
    split = int(n * TRAIN_SPLIT)

    if split < MIN_TRAIN_ROWS:
        log.warning("  %s %s — too few rows for train split (%d)", symbol, tf_label, split)
        return None

    train = df.iloc[:split]
    test  = df.iloc[split:].reset_index(drop=True)

    train_dir = train[train["label"] != HOLD]
    if len(train_dir) < MIN_TRAIN_LABELS:
        log.warning("  %s %s — too few directional labels in train (%d)",
                    symbol, tf_label, len(train_dir))
        return None

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(train_dir[feats].values)
    y_train   = train_dir["label"].values

    model = build_model()
    model.fit(X_train_s, y_train)

    # Save model in strategy.py format so the live bot loads it directly.
    # Key format: "BTCUSDT_15m" — same as TradingStrategy._sym_key()
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        base    = symbol.replace("_UMCBL", "").replace("_SPBL", "")
        sym_key = f"{base}_{tf_label}"
        model_path = os.path.join(models_dir, f"model_{sym_key}.joblib")
        meta_path  = os.path.join(models_dir, f"meta_{sym_key}.joblib")
        joblib.dump(model, model_path)
        joblib.dump({"scaler": scaler, "features": feats,
                     "total_trades": 0, "trade_history": []}, meta_path)
        log.info("💾 Saved backtest model → %s", model_path)

    return scaler, model, list(model.classes_), feats, train, test, threshold


# ── Strategy backtests ────────────────────────────────────────────────────────

def backtest_single(df_raw: pd.DataFrame,
                    symbol: str,
                    tf_label: str,
                    sl_mult: float,
                    tp_mult: float,
                    min_confidence: float = 0.55,
                    models_dir: Optional[str] = None) -> Optional[Dict]:
    """Single-timeframe walk-forward backtest."""
    result = _train(df_raw, symbol, tf_label, models_dir=models_dir)
    if result is None:
        return None
    scaler, model, classes, feats, train, test, threshold = result

    probas = model.predict_proba(scaler.transform(test[feats].fillna(0).values))
    trades = _simulate(test, probas, classes, sl_mult, tp_mult, min_confidence)

    return _summarise(trades, symbol, tf_label,
                      len(train), len(test), threshold, "single")


def backtest_confluence(df_entry: pd.DataFrame,
                        df_filter: pd.DataFrame,
                        symbol: str,
                        entry_tf: str,
                        filter_tf: str,
                        sl_mult: float,
                        tp_mult: float,
                        min_confidence: float = 0.55,
                        models_dir: Optional[str] = None) -> Optional[Dict]:
    """
    Confluence backtest: model trained on entry_tf, entries gated by
    filter_tf EMA21 trend agreement.
    """
    result = _train(df_entry, symbol, entry_tf, models_dir=models_dir)
    if result is None:
        return None
    scaler, model, classes, feats, train, test, threshold = result

    probas    = model.predict_proba(scaler.transform(test[feats].fillna(0).values))
    htf_trend = _htf_trend_series(df_filter, test["timestamp"])
    trades    = _simulate(test, probas, classes, sl_mult, tp_mult,
                          min_confidence, htf_trend=htf_trend)

    return _summarise(trades, symbol, entry_tf,
                      len(train), len(test), threshold,
                      "confluence", filter_tf=filter_tf)


def backtest_mtf(df_entry: pd.DataFrame,
                 df_direction: pd.DataFrame,
                 symbol: str,
                 entry_tf: str,
                 direction_tf: str,
                 sl_mult: float,
                 tp_mult: float,
                 min_confidence: float = 0.55,
                 models_dir: Optional[str] = None) -> Optional[Dict]:
    """
    MTF backtest: model trained on entry_tf, entries gated by direction_tf
    EMA21 trend (same gate as confluence — direction_tf provides trend bias).
    """
    result = _train(df_entry, symbol, entry_tf, models_dir=models_dir)
    if result is None:
        return None
    scaler, model, classes, feats, train, test, threshold = result

    probas    = model.predict_proba(scaler.transform(test[feats].fillna(0).values))
    htf_trend = _htf_trend_series(df_direction, test["timestamp"])
    trades    = _simulate(test, probas, classes, sl_mult, tp_mult,
                          min_confidence, htf_trend=htf_trend)

    return _summarise(trades, symbol, entry_tf,
                      len(train), len(test), threshold,
                      "mtf", filter_tf=direction_tf)


# ── Load analysis results ─────────────────────────────────────────────────────

def load_analysis(data_dir: str) -> Optional[Dict]:
    path = os.path.join(data_dir, "analysis_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Pretty report ─────────────────────────────────────────────────────────────

def print_report(all_results: List[Dict]) -> None:
    if not all_results:
        log.info("No backtest results to report.")
        return

    sorted_res = sorted(all_results, key=lambda x: x["ev_pct"], reverse=True)

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗",
        "║                          BACKTEST RESULTS — walk-forward, 2-year dataset                    ║",
        "╚══════════════════════════════════════════════════════════════════════════════════════════════╝",
        "",
    ]

    for stype in ["single", "confluence", "mtf"]:
        subset = [r for r in sorted_res if r["strategy_type"] == stype]
        if not subset:
            continue

        type_label = {
            "single":     "── Single-TF strategies ────────────────────────────────────────────────────────────",
            "confluence": "── Confluence strategies  (entry TF + filter TF) ───────────────────────────────────",
            "mtf":        "── MTF strategies  (direction TF → entry TF) ───────────────────────────────────────",
        }[stype]

        lines.append(f"  {type_label}")
        lines.append(
            f"  {'Symbol':<10}  {'Strategy':<12}  {'Trades':>6}  {'Wins':>5}  "
            f"{'Loss':>5}  {'WR%':>6}  {'AvgWin':>7}  {'AvgLoss':>8}  "
            f"{'R/R':>5}  {'EV%':>7}  {'TotalPnL':>9}  {'MaxDD':>8}  "
            f"{'W-Str':>6}  {'L-Str':>6}"
        )
        lines.append("  " + "─" * 100)

        for r in subset:
            if r["filter_tf"]:
                strat = f"{r['filter_tf']}→{r['timeframe']}" \
                        if stype == "mtf" \
                        else f"{r['timeframe']}+{r['filter_tf']}"
            else:
                strat = r["timeframe"]

            wr_flag = "✅" if r["win_rate"] >= 0.50 else "❌"
            ev_flag = "✅" if r["ev_pct"]   >  0    else "❌"

            lines.append(
                f"  {r['symbol']:<10}  {strat:<12}  {r['total_trades']:>6}  "
                f"{r['wins']:>5}  {r['losses']:>5}  "
                f"{r['win_rate']*100:>5.1f}%{wr_flag}  "
                f"{r['avg_win_pct']:>+6.2f}%  {r['avg_loss_pct']:>+7.2f}%  "
                f"{r['avg_rr']:>5.2f}  "
                f"{r['ev_pct']:>+6.3f}%{ev_flag}  "
                f"{r['total_pnl_pct']:>+8.2f}%  "
                f"{r['max_drawdown_pct']:>+7.2f}%  "
                f"{r['longest_win_streak']:>6}  {r['longest_loss_streak']:>6}"
            )
        lines.append("")

    lines += [
        "  Columns: WR% win rate | R/R avg_win÷avg_loss | EV% expected value per trade",
        "           TotalPnL sum of all % returns | MaxDD max peak-to-trough drawdown",
        "           W-Str longest win streak | L-Str longest loss streak",
        "           ✅ = above threshold | ❌ = below threshold",
        "",
    ]

    for line in lines:
        log.info(line)


# ── Run all strategies from analysis results ──────────────────────────────────

def run_all(data_dir: str,
            sl_mult: float,
            tp_mult: float,
            min_confidence: float = 0.55,
            min_confluence_gain: float = 0.05,
            models_dir: Optional[str] = None) -> List[Dict]:
    """
    Load analysis_results.json, backtest every profitable strategy it found.
    Falls back to all symbol × TF combos if the file is missing.
    """
    analysis = load_analysis(data_dir)
    all_results: List[Dict] = []

    # Cache loaded CSVs to avoid re-reading the same file multiple times
    _csv_cache: Dict[str, pd.DataFrame] = {}

    def _load(symbol: str, tf_label: str) -> Optional[pd.DataFrame]:
        key = f"{symbol}_{tf_label}"
        if key not in _csv_cache:
            fp = os.path.join(data_dir, f"{key}.csv")
            if not os.path.exists(fp):
                log.warning("    No CSV: %s", fp)
                _csv_cache[key] = None
            else:
                _csv_cache[key] = pd.read_csv(fp, parse_dates=["timestamp"])
        return _csv_cache[key]

    # ── 1. Single-TF strategies ───────────────────────────────────────────────
    if analysis:
        profitable = analysis.get("recommendations", {}).get("profitable_strategies", [])
        singles    = [s for s in profitable if s.get("strategy_type", "single") == "single"]
    else:
        # Fallback: test everything
        singles = [{"symbol": sym, "timeframe": TF_LABELS[tf]}
                   for sym in SYMBOLS for tf in TIMEFRAMES]

    log.info("── Single-TF backtests (%d strategies) ──", len(singles))
    for s in singles:
        symbol = s["symbol"]
        tf_lbl = s["timeframe"]
        df = _load(symbol, tf_lbl)
        if df is None:
            continue
        log.info("  %s %s  (%d candles)", symbol, tf_lbl, len(df))
        res = backtest_single(df, symbol, tf_lbl, sl_mult, tp_mult, min_confidence, models_dir=models_dir)
        if res:
            all_results.append(res)

    # ── 2. Confluence strategies ──────────────────────────────────────────────
    if analysis:
        conf_results = analysis.get("confluence_results", [])
        # Only test pairs where analysis found a meaningful gain
        conf_pairs = [c for c in conf_results
                      if c.get("accuracy_gain", 0) >= min_confluence_gain]
    else:
        conf_pairs = []

    log.info("── Confluence backtests (%d pairs) ──", len(conf_pairs))
    for c in conf_pairs:
        symbol    = c["symbol"]
        entry_tf  = c["signal_tf"]
        filter_tf = c["filter_tf"]
        df_entry  = _load(symbol, entry_tf)
        df_filter = _load(symbol, filter_tf)
        if df_entry is None or df_filter is None:
            continue
        log.info("  %s %s+%s  (gain=%.3f  cov=%.0f%%)",
                 symbol, entry_tf, filter_tf,
                 c.get("accuracy_gain", 0), c.get("coverage", 0) * 100)
        res = backtest_confluence(df_entry, df_filter, symbol,
                                  entry_tf, filter_tf,
                                  sl_mult, tp_mult, min_confidence,
                                  models_dir=models_dir)
        if res:
            all_results.append(res)

    # ── 3. MTF strategies ─────────────────────────────────────────────────────
    if analysis:
        profitable = analysis.get("recommendations", {}).get("profitable_strategies", [])
        mtf_list   = [s for s in profitable if s.get("strategy_type") == "mtf"]
    else:
        mtf_list = []

    log.info("── MTF backtests (%d strategies) ──", len(mtf_list))
    for s in mtf_list:
        symbol       = s["symbol"]
        entry_tf     = s["timeframe"]
        direction_tf = s.get("direction_tf", "")
        if not direction_tf:
            continue
        df_entry     = _load(symbol, entry_tf)
        df_direction = _load(symbol, direction_tf)
        if df_entry is None or df_direction is None:
            continue
        log.info("  %s %s→%s  (cv=%.3f  gain=%+.3f)",
                 symbol, direction_tf, entry_tf,
                 s.get("cv_accuracy", 0), s.get("accuracy_gain", 0))
        res = backtest_mtf(df_entry, df_direction, symbol,
                           entry_tf, direction_tf,
                           sl_mult, tp_mult, min_confidence,
                           models_dir=models_dir)
        if res:
            all_results.append(res)

    # Deduplicate — keep first occurrence of each (symbol, timeframe, filter_tf)
    seen_keys: set = set()
    deduped: List[Dict] = []
    for r in all_results:
        key = (r.get("symbol"), r.get("timeframe"), r.get("filter_tf", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)
    return deduped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--data-dir",    default=None,  help="Override data directory")
    parser.add_argument("--out",         default=None,  help="Save summary JSON to file")
    parser.add_argument("--confidence",  type=float, default=0.55,
                        help="Min model confidence to enter (default 0.55)")
    parser.add_argument("--min-conf-gain", type=float, default=0.05,
                        help="Min analysis accuracy_gain to include a confluence pair (default 0.05)")
    args = parser.parse_args()

    cfg      = load_config()
    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir", "/data")
    rc       = cfg.get("risk", {})
    sl_mult  = rc.get("stop_loss_atr_mult",   2.2)
    tp_mult  = rc.get("take_profit_atr_mult", 3.5)

    log.info("═══ Backtest starting ═══")
    log.info("  data_dir=%s  sl=%.1f×ATR  tp=%.1f×ATR  confidence≥%.2f",
             data_dir, sl_mult, tp_mult, args.confidence)

    all_results = run_all(data_dir, sl_mult, tp_mult,
                          args.confidence, args.min_conf_gain)

    print_report(all_results)
    print_temporal_report(all_results)

    # Save full results (with per-trade detail) to /data
    out_path = os.path.join(data_dir, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2, default=str)
    log.info("Full results saved → %s", out_path)

    # Optional summary-only file
    if args.out:
        summary = [{k: v for k, v in r.items() if k != "trades"} for r in all_results]
        with open(args.out, "w") as f:
            json.dump({"results": summary}, f, indent=2)
        log.info("Summary saved → %s", args.out)


if __name__ == "__main__":
    main()
