"""
analysis.py
═══════════
Reads historical OHLCV CSVs and answers three questions:

  1. Which single timeframe produces the most accurate signals?
  2. Which higher-timeframe filter (confluence) improves win rate most?
  3. Which features matter most — and should the bot focus on?

Outputs
───────
  /data/analysis_results.json   — machine-readable results loaded by strategy.py
  Console report                — human-readable summary printed at startup

How it works
────────────
For each timeframe we:
  • Compute all 22 technical indicators (same as live trading)
  • Label each candle BUY/HOLD/SELL using the same forward-return method
  • Train a Random Forest with 5-fold cross-validation
  • Record accuracy, per-class precision/recall, and feature importances

For each (signal_tf × filter_tf) confluence pair we:
  • Only count BUY/SELL signals where the higher-TF trend agrees
    (price above EMA21 = bullish; below = bearish)
  • Compare precision and win-rate with vs. without the filter
  • Record the improvement

Results are ranked and saved so strategy.py can auto-configure itself.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score

from indicators import compute_features, FEATURE_COLS
from data_collector import TF_LABELS, TIMEFRAMES, SYMBOLS

logger = logging.getLogger(__name__)

RESULTS_FILE  = "analysis_results.json"
LABEL_HORIZON = 4     # candles forward for labelling
BUY, HOLD, SELL = 1, 0, -1

# ── Timeframe-specific thresholds ─────────────────────────────────────────────
# A "meaningful" price move differs hugely between 5m and 1d candles.
# Using a single 0.5% threshold causes 5m/15m candles to almost never get
# labelled BUY or SELL — leaving too few samples to train on.
#
# These values are calibrated so roughly 30–40% of candles receive a
# directional label (BUY or SELL) at each timeframe.
TF_LABEL_THRESH: Dict[str, float] = {
    "5m":   0.003,   # 0.3 % — 5 candles × 5 min  ≈ 25 min lookahead
    "15m":  0.005,   # 0.5 % — 4 candles × 15 min ≈ 1 h  lookahead
    "1h":   0.008,   # 0.8 % — 4 candles × 1 h    ≈ 4 h  lookahead
    "4h":   0.015,   # 1.5 % — 4 candles × 4 h    ≈ 16 h lookahead
    "1d":   0.020,   # 2.0 % — 4 candles × 1 d    ≈ 4 d  lookahead
}

# Minimum number of labelled (BUY + SELL) rows needed after feature engineering.
# Lower for short TFs because we accumulate candles faster and the model
# still generalises well at lower sample counts.
TF_MIN_SAMPLES: Dict[str, int] = {
    "5m":   30,
    "15m":  50,
    "1h":   80,
    "4h":  120,
    "1d":  150,
}

# Minimum raw candle rows needed before we even try to load a CSV for analysis.
TF_MIN_CANDLES: Dict[str, int] = {
    "5m":   60,
    "15m":  80,
    "1h":  100,
    "4h":  100,
    "1d":  100,
}

# Fallback defaults for any unlisted label
_DEFAULT_THRESH      = 0.010
_DEFAULT_MIN_SAMPLES = 100
_DEFAULT_MIN_CANDLES = 100


# ── Labelling ─────────────────────────────────────────────────────────────────

def compute_adaptive_threshold(df: pd.DataFrame,
                                horizon: int = LABEL_HORIZON,
                                target_directional: float = 0.33) -> float:
    """
    Derive the label threshold directly from the data so that roughly
    `target_directional` fraction of candles receive a BUY or SELL label.

    Method: take the (1 − target_directional)th percentile of |forward returns|.
    At this cut-off, exactly target_directional × 100% of candles have a move
    large enough to be labelled — calibrated to *this* pair and *this* timeframe
    automatically, no guessing required.

    Clamped to [0.001, 0.15] so thin or extreme datasets can't produce nonsense.
    """
    future_ret = df["close"].pct_change(horizon).shift(-horizon).abs().dropna()
    if len(future_ret) < 50:
        return _DEFAULT_THRESH
    pct_point = (1.0 - target_directional) * 100   # e.g. 67.0 for 33% directional
    threshold = float(np.percentile(future_ret, pct_point))
    return round(max(0.001, min(threshold, 0.15)), 4)


def label_candles(df: pd.DataFrame,
                  horizon: int   = LABEL_HORIZON,
                  threshold: float = _DEFAULT_THRESH) -> pd.Series:
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(HOLD, index=df.index)
    labels[future_ret >  threshold] = BUY
    labels[future_ret < -threshold] = SELL
    return labels


# ── Single-timeframe analysis ─────────────────────────────────────────────────

def analyse_timeframe(df: pd.DataFrame,
                      tf_label: str,
                      symbol: str) -> Optional[Dict[str, Any]]:
    """
    Train and cross-validate a Random Forest on one timeframe.
    Returns a dict of metrics, or None if data is insufficient.
    """
    # Compute threshold from this pair's actual return distribution so that
    # ~33% of candles get a directional label — no per-pair guessing needed.
    # Falls back to the static TF table only if the dataset is too thin (<50 rows).
    min_samples = TF_MIN_SAMPLES.get(tf_label, _DEFAULT_MIN_SAMPLES)
    thresh      = compute_adaptive_threshold(df)

    df = compute_features(df.copy())
    labels = label_candles(df, threshold=thresh)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = df[available].copy()
    df_feat["label"] = labels

    # Drop rows with NaN features or the unlabellable tail
    df_feat = df_feat.dropna()
    df_feat = df_feat[df_feat["label"] != HOLD]   # binary: BUY vs SELL only

    if len(df_feat) < min_samples:
        logger.warning(
            "  Skipping %s %s — only %d labelled samples "
            "(need %d, thresh=%.3f%% — accumulating more data)",
            symbol, tf_label, len(df_feat), min_samples, thresh * 100)
        return None

    X = df_feat[available].values
    y = df_feat["label"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model — lean RF for speed
    rf = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 8,
        min_samples_leaf = 10,
        random_state = 42,
        n_jobs       = -1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated accuracy
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    accuracy  = float(np.mean(cv_scores))

    # Fit on full set for feature importances
    rf.fit(X_scaled, y)

    importances = dict(zip(available, rf.feature_importances_.tolist()))
    top_features = sorted(importances, key=importances.get, reverse=True)[:12]

    # Class balance
    buys  = int((y == BUY).sum())
    sells = int((y == SELL).sum())

    result = {
        "symbol":          symbol,
        "timeframe":       tf_label,
        "samples":         len(df_feat),
        "buy_count":       buys,
        "sell_count":      sells,
        "cv_accuracy":     round(accuracy, 4),
        "cv_std":          round(float(np.std(cv_scores)), 4),
        "label_threshold": thresh,   # adaptive — stored so strategy.py uses same value
        "feature_importance": {k: round(v, 6) for k, v in
                               sorted(importances.items(),
                                      key=lambda x: x[1], reverse=True)},
        "top_features": top_features,
    }
    logger.info("  %s %s  |  acc=%.3f±%.3f  |  samples=%d  |  thresh=%.2f%% (adaptive)  |  top=%s",
                symbol, tf_label, accuracy, np.std(cv_scores),
                len(df_feat), thresh * 100, top_features[:3])
    return result


# ── Confluence analysis ───────────────────────────────────────────────────────

def _higher_tf_trend(high_df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of trend labels (+1 bullish / -1 bearish / 0 neutral)
    for a higher-timeframe DataFrame, based on EMA21 and MACD direction.
    """
    high_df = compute_features(high_df.copy())

    trend = pd.Series(0, index=high_df.index)
    if "ema_21" in high_df.columns:
        bullish_ema = high_df["close"] > high_df["ema_21"]
        bearish_ema = high_df["close"] < high_df["ema_21"]
        trend[bullish_ema] =  1
        trend[bearish_ema] = -1

    # MACD confirmation (secondary vote — both must agree)
    if "macd_diff" in high_df.columns:
        macd_bull = high_df["macd_diff"] > 0
        macd_bear = high_df["macd_diff"] < 0
        # Require both EMA and MACD to agree for a strong trend
        strong_bull = bullish_ema & macd_bull
        strong_bear = bearish_ema & macd_bear
        trend[~strong_bull & ~strong_bear] = 0   # neutral if they disagree

    high_df["_trend"] = trend.values
    return high_df.set_index("timestamp")["_trend"]


def analyse_confluence(signal_df: pd.DataFrame,
                       filter_df: pd.DataFrame,
                       signal_tf: str,
                       filter_tf: str,
                       symbol: str) -> Optional[Dict[str, Any]]:
    """
    Compare win rate on signal_tf with and without a filter_tf trend gate.
    Returns metrics dict.
    """
    thresh      = compute_adaptive_threshold(signal_df)
    min_samples = TF_MIN_SAMPLES.get(signal_tf, _DEFAULT_MIN_SAMPLES)

    signal_df = compute_features(signal_df.copy())
    signal_labels = label_candles(signal_df, threshold=thresh)
    signal_df["label"] = signal_labels

    # Get higher-TF trend direction indexed by timestamp
    htrend = _higher_tf_trend(filter_df.copy())

    # Merge: for each signal candle, find the most recent higher-TF trend
    signal_df = signal_df.set_index("timestamp")
    # Forward-fill the higher-TF trend onto the signal-TF timestamps
    combined = signal_df.copy()
    combined["htf_trend"] = np.nan
    combined["htf_trend"] = (
        combined["htf_trend"]
        .combine_first(htrend.reindex(combined.index))
    )
    combined["htf_trend"] = combined["htf_trend"].ffill()
    combined = combined.reset_index().dropna(subset=["label", "htf_trend"])
    combined = combined[combined["label"] != HOLD]

    if len(combined) < min_samples:
        return None

    # Baseline: all signals regardless of HTF trend
    all_buys  = combined[combined["label"] == BUY]
    all_sells = combined[combined["label"] == SELL]

    # Confluence-filtered: only signals where HTF trend agrees
    conf_buys  = all_buys[all_buys["htf_trend"] == 1]
    conf_sells = all_sells[all_sells["htf_trend"] == -1]

    def win_rate(df_signals: pd.DataFrame, direction: int) -> float:
        if len(df_signals) == 0:
            return 0.0
        # "Win" = the label itself is BUY/SELL (already filtered)
        # We measure whether the labelled direction actually happened
        # (by definition it did — we use forward precision instead)
        return len(df_signals) / max(len(combined[combined["label"] == direction]), 1)

    baseline_coverage = (len(all_buys) + len(all_sells)) / max(len(combined), 1)
    filtered_coverage = (len(conf_buys) + len(conf_sells)) / max(len(combined), 1)

    # Precision: of BUY signals, how many were correct?
    # We approximate via the signal-TF model accuracy on filtered vs unfiltered
    def label_precision(subset: pd.DataFrame) -> float:
        """Fraction of signals that have a non-trivial feature set."""
        if len(subset) < 20:
            return 0.0
        feats = [c for c in FEATURE_COLS if c in subset.columns]
        X = subset[feats].dropna()
        if len(X) < 20:
            return 0.0
        y = subset.loc[X.index, "label"]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                    random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    baseline_acc = label_precision(combined)

    conf_subset = pd.concat([conf_buys, conf_sells])
    filtered_acc = label_precision(conf_subset) if len(conf_subset) >= 20 else baseline_acc

    improvement = round(filtered_acc - baseline_acc, 4)

    result = {
        "symbol":            symbol,
        "signal_tf":         signal_tf,
        "filter_tf":         filter_tf,
        "baseline_samples":  len(combined),
        "filtered_samples":  len(conf_subset),
        "baseline_accuracy": round(baseline_acc, 4),
        "filtered_accuracy": round(filtered_acc, 4),
        "accuracy_gain":     improvement,
        "signal_coverage":   round(filtered_coverage, 4),
    }
    logger.info("  Confluence %s→%s %s  |  base=%.3f → filtered=%.3f  (+%.3f)  coverage=%.1f%%",
                signal_tf, filter_tf, symbol,
                baseline_acc, filtered_acc, improvement,
                filtered_coverage * 100)
    return result


# ── Multi-timeframe entry confirmation analysis ───────────────────────────────

def analyse_mtf_entry(direction_df: pd.DataFrame,
                      entry_df:     pd.DataFrame,
                      direction_tf: str,
                      entry_tf:     str,
                      symbol:       str) -> Optional[Dict[str, Any]]:
    """
    Discover whether requiring a higher-TF direction to agree with a lower-TF
    entry signal improves predictive accuracy.

    Logic:
      • Label entry-TF candles with their own adaptive threshold.
      • Compute higher-TF trend (+1/0/-1) from EMA21+MACD on direction_df.
      • Forward-fill direction onto entry-TF timestamps.
      • Baseline: RF accuracy on all entry-TF directional labels.
      • Filtered: RF accuracy on only those labels where direction agrees.
      • A pair is useful if accuracy improves >= 2pp AND enough samples survive.

    Returns None if data is insufficient or the gain is negligible.
    """
    min_samples = TF_MIN_SAMPLES.get(entry_tf, _DEFAULT_MIN_SAMPLES)
    entry_thresh = compute_adaptive_threshold(entry_df)

    entry_df  = compute_features(entry_df.copy())
    entry_labels = label_candles(entry_df, threshold=entry_thresh)
    entry_df["label"] = entry_labels

    # Higher-TF trend indexed by timestamp
    htrend = _higher_tf_trend(direction_df.copy())

    entry_df = entry_df.set_index("timestamp")
    entry_df["dir_signal"] = np.nan
    entry_df["dir_signal"] = (
        entry_df["dir_signal"].combine_first(htrend.reindex(entry_df.index))
    )
    entry_df["dir_signal"] = entry_df["dir_signal"].ffill()
    entry_df = entry_df.reset_index().dropna(subset=["label", "dir_signal"])
    entry_df = entry_df[entry_df["label"] != HOLD]

    if len(entry_df) < min_samples:
        return None

    # Baseline accuracy on all directional labels
    def _acc(subset: pd.DataFrame) -> float:
        if len(subset) < 20:
            return 0.0
        feats = [c for c in FEATURE_COLS if c in subset.columns]
        X = subset[feats].dropna()
        if len(X) < 20:
            return 0.0
        y = subset.loc[X.index, "label"]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf  = RandomForestClassifier(n_estimators=50, max_depth=6,
                                      random_state=42, n_jobs=-1)
        cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    baseline_acc = _acc(entry_df)

    # Filtered: only signals where direction agrees with label
    conf_buys  = entry_df[(entry_df["label"] == BUY)  & (entry_df["dir_signal"] == 1)]
    conf_sells = entry_df[(entry_df["label"] == SELL) & (entry_df["dir_signal"] == -1)]
    filtered   = pd.concat([conf_buys, conf_sells])

    if len(filtered) < 20:
        return None

    filtered_acc  = _acc(filtered)
    accuracy_gain = round(filtered_acc - baseline_acc, 4)
    sample_ratio  = round(len(filtered) / max(len(entry_df), 1), 4)

    result = {
        "symbol":           symbol,
        "direction_tf":     direction_tf,
        "entry_tf":         entry_tf,
        "baseline_samples": len(entry_df),
        "filtered_samples": len(filtered),
        "baseline_accuracy": round(baseline_acc, 4),
        "mtf_accuracy":     round(filtered_acc, 4),
        "accuracy_gain":    accuracy_gain,
        "sample_ratio":     sample_ratio,
        "entry_label_threshold":     entry_thresh,
    }
    logger.info("  MTF %s→%s %s  |  base=%.3f → mtf=%.3f  (gain=%+.3f)  coverage=%.1f%%",
                direction_tf, entry_tf, symbol,
                baseline_acc, filtered_acc, accuracy_gain, sample_ratio * 100)
    return result


# ── Master analyser ───────────────────────────────────────────────────────────

class Analyzer:

    def __init__(self, data_dir: str = "/data", results_dir: str = "/data"):
        self.data_dir    = data_dir
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, RESULTS_FILE)

    def results_are_fresh(self, max_age_days: int = 7) -> bool:
        if not os.path.exists(self.results_path):
            return False
        age = (time.time() - os.path.getmtime(self.results_path)) / 86400
        return age < max_age_days

    def load_results(self) -> Optional[Dict]:
        if not os.path.exists(self.results_path):
            return None
        with open(self.results_path) as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline and save results.
        Returns the results dict.
        """
        t0 = time.time()
        logger.info("═══ Analysis pipeline starting ═══")

        tf_results:    List[Dict] = []
        conf_results:  List[Dict] = []
        mtf_results:   List[Dict] = []

        for symbol in SYMBOLS:
            logger.info("── Analysing %s ──", symbol)

            # Load all available timeframe data for this symbol
            tf_data: Dict[str, pd.DataFrame] = {}
            for tf in TIMEFRAMES:
                label = TF_LABELS[tf]
                fp = os.path.join(self.data_dir, f"{symbol}_{label}.csv")
                min_candles = TF_MIN_CANDLES.get(label, _DEFAULT_MIN_CANDLES)
                if os.path.exists(fp):
                    df = pd.read_csv(fp, parse_dates=["timestamp"])
                    if len(df) >= min_candles:
                        tf_data[tf] = df
                        logger.info("  Loaded %s %s — %d candles", symbol, label, len(df))
                    else:
                        logger.warning("  Skipping %s %s — insufficient data (%d rows, need %d)",
                                       symbol, label, len(df), min_candles)
                else:
                    logger.warning("  No CSV found for %s %s", symbol, label)

            if not tf_data:
                logger.warning("No data for %s — skipping", symbol)
                continue

            # ── Single-timeframe analysis ─────────────────────────────────────
            for tf, df in tf_data.items():
                res = analyse_timeframe(df, TF_LABELS[tf], symbol)
                if res:
                    tf_results.append(res)

            # ── Confluence analysis ───────────────────────────────────────────
            # Signal timeframes: 5m, 15m, 1h
            # Filter timeframes: always higher than signal
            signal_tfs = [t for t in ["5", "15", "60"] if t in tf_data]
            filter_tfs = [t for t in ["60", "240", "1440"] if t in tf_data]

            for sig_tf in signal_tfs:
                for flt_tf in filter_tfs:
                    # Filter must be a strictly higher timeframe
                    if int(flt_tf) <= int(sig_tf):
                        continue
                    res = analyse_confluence(
                        tf_data[sig_tf], tf_data[flt_tf],
                        TF_LABELS[sig_tf], TF_LABELS[flt_tf],
                        symbol,
                    )
                    if res:
                        conf_results.append(res)

            # ── MTF entry confirmation analysis ───────────────────────────────
            # Direction timeframes: 1h, 4h, 1d (higher — provide trend direction)
            # Entry timeframes: 5m, 15m, 1h (lower — provide the actual entry)
            # Tests whether requiring direction_tf to agree improves entry_tf accuracy.
            direction_tfs = [t for t in ["60", "240", "1440"] if t in tf_data]
            entry_tfs     = [t for t in ["5", "15", "60"]     if t in tf_data]

            for dir_tf in direction_tfs:
                for ent_tf in entry_tfs:
                    if int(dir_tf) <= int(ent_tf):
                        continue
                    res = analyse_mtf_entry(
                        tf_data[dir_tf], tf_data[ent_tf],
                        TF_LABELS[dir_tf], TF_LABELS[ent_tf],
                        symbol,
                    )
                    if res:
                        mtf_results.append(res)

        # ── Aggregate recommendations ─────────────────────────────────────────
        recommendations = self._build_recommendations(tf_results, conf_results,
                                                       mtf_results)

        results = {
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "duration_seconds":  round(time.time() - t0, 1),
            "timeframe_results": tf_results,
            "confluence_results": conf_results,
            "mtf_results":       mtf_results,
            "recommendations":   recommendations,
        }

        os.makedirs(self.results_dir, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved analysis → %s", self.results_path)

        self._print_report(recommendations, tf_results, conf_results, mtf_results)
        return results

    # ── Recommendations ───────────────────────────────────────────────────────

    def _build_recommendations(self,
                                tf_results: List[Dict],
                                conf_results: List[Dict],
                                mtf_results: List[Dict] = None) -> Dict[str, Any]:
        if not tf_results:
            return {}

        # Best signal timeframe (highest average CV accuracy)
        tf_acc: Dict[str, List[float]] = {}
        for r in tf_results:
            tf_acc.setdefault(r["timeframe"], []).append(r["cv_accuracy"])
        tf_avg = {tf: float(np.mean(accs)) for tf, accs in tf_acc.items()}
        best_signal_tf = max(tf_avg, key=tf_avg.get)

        # Best confluence pair (highest average accuracy gain)
        conf_by_pair: Dict[str, List[float]] = {}
        for r in conf_results:
            key = f"{r['signal_tf']}+{r['filter_tf']}"
            conf_by_pair.setdefault(key, []).append(r["accuracy_gain"])
        conf_avg = {k: float(np.mean(v)) for k, v in conf_by_pair.items()}
        best_confluence = max(conf_avg, key=conf_avg.get) if conf_avg else None

        # Global top features (average importance across all TF models)
        feat_scores: Dict[str, List[float]] = {}
        for r in tf_results:
            for feat, imp in r.get("feature_importance", {}).items():
                feat_scores.setdefault(feat, []).append(imp)
        global_importance = {
            feat: float(np.mean(scores))
            for feat, scores in feat_scores.items()
        }
        top_features = sorted(global_importance, key=global_importance.get,
                               reverse=True)[:15]

        # Per-symbol best timeframe
        per_symbol: Dict[str, str] = {}
        for symbol in SYMBOLS:
            sym_results = [r for r in tf_results if r["symbol"] == symbol]
            if sym_results:
                best = max(sym_results, key=lambda x: x["cv_accuracy"])
                per_symbol[symbol] = best["timeframe"]

        # Per-symbol per-TF adaptive label thresholds — stored so strategy.py
        # trains with exactly the same threshold that analysis evaluated each TF with.
        label_thresholds: Dict[str, Dict[str, float]] = {}
        for r in tf_results:
            sym = r["symbol"]
            tf  = r["timeframe"]
            label_thresholds.setdefault(sym, {})[tf] = r.get("label_threshold",
                                                              _DEFAULT_THRESH)

        # Confluence filter recommendation
        best_filter_tf = None
        if best_confluence:
            signal_part, filter_part = best_confluence.split("+")
            if signal_part == best_signal_tf:
                best_filter_tf = filter_part

        # All profitable (symbol, TF) strategies — above-chance threshold for binary
        # BUY/SELL classification.  The bot will train a separate model for each and
        # trade ALL of them simultaneously (one exchange position per symbol at a time).
        min_acc     = 0.52
        min_mtf_gain = 0.02   # minimum accuracy improvement for MTF to be worth adding

        # Single-TF strategies
        profitable_strategies = [
            {
                "symbol":        r["symbol"],
                "timeframe":     r["timeframe"],
                "cv_accuracy":   round(r["cv_accuracy"], 4),
                "strategy_type": "single",
            }
            for r in tf_results
            if r["cv_accuracy"] >= min_acc
        ]

        # MTF entry confirmation strategies — only added if analysis found a gain
        for r in (mtf_results or []):
            if (r["accuracy_gain"] >= min_mtf_gain
                    and r["sample_ratio"] >= 0.15
                    and r["mtf_accuracy"] >= min_acc):
                profitable_strategies.append({
                    "symbol":        r["symbol"],
                    "timeframe":     r["entry_tf"],      # entry TF — model + ATR sizing
                    "direction_tf":  r["direction_tf"],  # direction (higher) TF
                    "cv_accuracy":   round(r["mtf_accuracy"], 4),
                    "accuracy_gain": round(r["accuracy_gain"], 4),
                    "sample_ratio":  round(r["sample_ratio"], 4),
                    "strategy_type": "mtf",
                })

        profitable_strategies = sorted(profitable_strategies,
                                       key=lambda x: x["cv_accuracy"], reverse=True)

        single_count = sum(1 for s in profitable_strategies if s["strategy_type"] == "single")
        mtf_count    = sum(1 for s in profitable_strategies if s["strategy_type"] == "mtf")
        logger.info("Profitable strategies: %d single-TF + %d MTF = %d total",
                    single_count, mtf_count, len(profitable_strategies))
        for s in profitable_strategies:
            if s["strategy_type"] == "mtf":
                logger.info("  MTF  %s  %s→%s  cv=%.3f  gain=%+.3f  coverage=%.0f%%",
                            s["symbol"], s["direction_tf"], s["timeframe"],
                            s["cv_accuracy"], s.get("accuracy_gain", 0),
                            s.get("sample_ratio", 0) * 100)
            else:
                logger.info("  STF  %s  %s  cv=%.3f",
                            s["symbol"], s["timeframe"], s["cv_accuracy"])

        return {
            "best_signal_timeframe":  best_signal_tf,
            "best_filter_timeframe":  best_filter_tf,
            "best_confluence_pair":   best_confluence,
            "avg_confluence_gain":    conf_avg.get(best_confluence, 0.0) if best_confluence else 0.0,
            "top_features":           top_features,
            "global_feature_importance": {
                k: round(v, 6) for k, v in
                sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
            },
            "timeframe_accuracy":     {tf: round(v, 4) for tf, v in
                                       sorted(tf_avg.items(), key=lambda x: x[1], reverse=True)},
            "per_symbol_best_tf":     per_symbol,
            "label_thresholds":       label_thresholds,
            "profitable_strategies":  profitable_strategies,
            "min_strategy_accuracy":  min_acc,
        }

    # ── Report ────────────────────────────────────────────────────────────────

    def _print_report(self,
                      recs: Dict,
                      tf_results: List[Dict],
                      conf_results: List[Dict],
                      mtf_results: List[Dict] = None) -> None:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║              STRATEGY ANALYSIS REPORT                       ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "── Timeframe accuracy (avg across symbols) ─────────────────────",
        ]
        for tf, acc in recs.get("timeframe_accuracy", {}).items():
            bar = "█" * int(acc * 40)
            lines.append(f"  {tf:>5}   {acc:.3f}  {bar}")

        lines += [
            "",
            f"  ✅  Best signal timeframe : {recs.get('best_signal_timeframe', 'N/A')}",
            f"  ✅  Best filter timeframe : {recs.get('best_filter_timeframe', 'N/A')}",
            f"  ✅  Best confluence pair  : {recs.get('best_confluence_pair', 'N/A')}",
            f"  ✅  Avg confluence gain   : +{recs.get('avg_confluence_gain', 0):.3f}",
            "",
            "── Top features (ranked by importance) ─────────────────────────",
        ]
        for i, feat in enumerate(recs.get("top_features", [])[:10], 1):
            imp = recs.get("global_feature_importance", {}).get(feat, 0)
            lines.append(f"  {i:>2}.  {feat:<22}  {imp:.4f}")

        lines += [
            "",
            "── Per-symbol best timeframe ────────────────────────────────────",
        ]
        for sym, tf in recs.get("per_symbol_best_tf", {}).items():
            lines.append(f"  {sym:<12}  →  {tf}")

        lines += [
            "",
            "── Confluence improvement ───────────────────────────────────────",
        ]
        sorted_conf = sorted(conf_results,
                              key=lambda x: x.get("accuracy_gain", 0), reverse=True)[:6]
        for r in sorted_conf:
            gain = r.get("accuracy_gain", 0)
            sign = "+" if gain >= 0 else ""
            lines.append(f"  {r['signal_tf']:>4} + {r['filter_tf']:<4}  {r['symbol']:<12}"
                         f"  {sign}{gain:.3f}  "
                         f"({'▲ better' if gain > 0 else '▼ worse'})")

        # ── MTF entry confirmation discoveries ───────────────────────────────
        mtf_list = sorted(mtf_results or [],
                          key=lambda x: x.get("accuracy_gain", 0), reverse=True)[:8]
        if mtf_list:
            lines += [
                "",
                "── MTF entry confirmation (direction TF → entry TF) ─────────────",
            ]
            for r in mtf_list:
                gain = r.get("accuracy_gain", 0)
                ratio = r.get("sample_ratio", 0)
                traded = "✅ TRADED" if (gain >= 0.02 and ratio >= 0.15
                                         and r.get("mtf_accuracy", 0) >= 0.52) else "— below threshold"
                lines.append(
                    f"  {r['direction_tf']:>3} → {r['entry_tf']:<4}  {r['symbol']:<12}"
                    f"  base={r.get('baseline_accuracy', 0):.3f}"
                    f"  mtf={r.get('mtf_accuracy', 0):.3f}"
                    f"  gain={gain:+.3f}"
                    f"  cov={ratio*100:.0f}%"
                    f"  {traded}"
                )

        lines.append("")
        for line in lines:
            logger.info(line)


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("data", {}).get("data_dir", "/data")
    analyzer = Analyzer(data_dir=data_dir, results_dir=data_dir)
    analyzer.run()
