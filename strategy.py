"""
strategy.py  —  Self-improving ensemble ML strategy

Architecture
────────────
1. THREE models vote together (soft-vote majority):
     • Random Forest       — robust, handles non-linear patterns well
     • Gradient Boosting   — corrects errors iteratively, strong on tabular data
     • Extra-Trees         — adds randomness, reduces overfitting
   Raw soft-vote probabilities used directly (no CalibratedClassifierCV):
   calibration compressed signals below class prior (~0.07) making them
   unusable. Raw balanced-tree votes sit near 0.33 when uncertain and
   push toward 0.45-0.55 on genuine signals — workable with a 0.38 gate.

2. Expected Value gate
   Before every trade we calculate:
     EV = win_rate × avg_win_pct − loss_rate × avg_loss_pct
   We only enter if EV > min_ev AND historical win_rate > min_win_rate.
   This is the mathematical edge that separates systematic trading from gambling.

3. Dynamic retraining schedule
   The bot retrains MORE often early on so it learns fast:
     Trades  0-14  →  retrain every  5 completed trades
     Trades 15-39  →  retrain every 10
     Trades 40-99  →  retrain every 20
     Trades 100+   →  retrain every 40
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import sklearn
from typing import Tuple, Optional, List, Dict

from sklearn.ensemble import (RandomForestClassifier,
                               HistGradientBoostingClassifier,
                               ExtraTreesClassifier,
                               VotingClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

_EXPECTED_SKLEARN = "1.6.1"
if sklearn.__version__ != _EXPECTED_SKLEARN:
    import logging as _log
    _log.getLogger(__name__).error(
        "SKLEARN VERSION MISMATCH: expected %s but got %s — models may fail to load "
        "or produce silent wrong predictions. Pin scikit-learn==%s in requirements.txt.",
        _EXPECTED_SKLEARN, sklearn.__version__, _EXPECTED_SKLEARN,
    )

from indicators import FEATURE_COLS, compute_features

logger = logging.getLogger(__name__)

BUY  =  1
HOLD =  0
SELL = -1

# ─────────────────────────────────────────────────────────────────────────────
# Per-timeframe label thresholds
# ─────────────────────────────────────────────────────────────────────────────
# Must match analysis.py TF_LABEL_THRESH exactly — analysis picks the best
# timeframe using these thresholds, then strategy trains on the same labels.
# Mismatched thresholds would mean analysis scores TF A but strategy trains
# on TF A with different labels — defeating the purpose of the analysis.
_TF_LABEL_THRESHOLD = {
    "5m":  0.003,   # 0.3 % — 5-min candles, 25-min lookahead
    "15m": 0.005,   # 0.5 % — 15-min candles, 1-h lookahead
    "1h":  0.008,   # 0.8 % — 1-h candles, 4-h lookahead
    "4h":  0.015,   # 1.5 % — 4-h candles, 16-h lookahead
    "1d":  0.020,   # 2.0 % — daily candles, 4-day lookahead
}

# ─────────────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────────────

def label_candles(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.005) -> pd.Series:
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(HOLD, index=df.index)
    labels[future_return >  threshold] = BUY
    labels[future_return < -threshold] = SELL
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Trade outcome statistics
# ─────────────────────────────────────────────────────────────────────────────

class TradeStats:
    """Rolling statistics used for the Expected Value gate."""

    def __init__(self, min_trades: int = 10):
        self.min_trades = min_trades
        self._history: List[Dict] = []

    def record(self, outcome: dict):
        self._history.append(outcome)

    def for_pair(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol is None:
            return self._history
        return [t for t in self._history if t.get("pair") == symbol]

    def ev_and_winrate(self, symbol: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (expected_value, win_rate, avg_rr) for the given pair (or overall).
        Returns (None, None, None) if there is not enough data.
        EV = win_rate × avg_win − loss_rate × avg_loss  (in fractional terms)
        """
        history = self.for_pair(symbol)
        if len(history) < self.min_trades:
            return None, None, None

        pnls = [t["pnl_pct"] for t in history]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate  = len(wins)  / len(pnls)
        avg_win   = float(np.mean(wins))   if wins   else 0.0
        avg_loss  = float(np.mean(losses)) if losses else 0.0  # negative number

        ev = win_rate * avg_win + (1 - win_rate) * avg_loss   # avg_loss already negative

        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        return ev, win_rate, avg_rr

    def rank_pairs(self) -> Dict[str, float]:
        pairs = {t["pair"] for t in self._history}
        ranks = {}
        for pair in pairs:
            ev, wr, _ = self.ev_and_winrate(pair)
            if ev is not None:
                ranks[pair] = ev
        return dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))

    def summary_str(self, symbol: Optional[str] = None) -> str:
        ev, wr, rr = self.ev_and_winrate(symbol)
        history = self.for_pair(symbol)
        if ev is None:
            return f"({len(history)}/{self.min_trades} trades needed for EV calc)"
        return (f"trades={len(history)}  win_rate={wr*100:.1f}%  "
                f"avg_rr={rr:.2f}  EV={ev*100:+.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Confidence score tracker — detects model drift between market regimes
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceTracker:
    """
    Records raw model confidence (max of buy_p / sell_p) on every prediction,
    regardless of whether the signal crossed the entry threshold.

    Two uses:
      1. Drift detection — compare recent 14-day avg vs 90-day baseline.
         A meaningful drop signals the model has lost conviction in the current
         market regime and needs retraining.
      2. Signal drought — detect when a slot hasn't generated any signal
         (confidence >= min_signal_confidence) for too long.
    """

    # Minimum confidence to count as a "signal seen" for drought tracking.
    # Deliberately lower than the 0.55/0.70 entry threshold so we catch the
    # case where the model is generating weak signals vs generating nothing.
    MIN_SIGNAL_CONFIDENCE = 0.45

    def __init__(self):
        # slot → list of {"ts": ISO string, "confidence": float}
        self._log: Dict[str, List[Dict]] = {}

    def record(self, slot: str, confidence: float) -> None:
        """Call on every predict() call with max(buy_p, sell_p)."""
        from datetime import datetime, timezone
        if slot not in self._log:
            self._log[slot] = []
        self._log[slot].append({
            "ts":         datetime.now(timezone.utc).isoformat(),
            "confidence": round(float(confidence), 4),
        })
        # Keep at most 90 days of 15-min samples ≈ 8,640 entries per slot
        if len(self._log[slot]) > 9_000:
            self._log[slot] = self._log[slot][-9_000:]

    def last_signal_ts(self, slot: str) -> Optional[str]:
        """
        Returns ISO timestamp of the last time this slot had confidence >=
        MIN_SIGNAL_CONFIDENCE, or None if never recorded.
        """
        entries = self._log.get(slot, [])
        for entry in reversed(entries):
            if entry["confidence"] >= self.MIN_SIGNAL_CONFIDENCE:
                return entry["ts"]
        return None

    def detect_drift(self,
                     slot: str,
                     recent_days: int = 14,
                     baseline_days: int = 90,
                     threshold: float = 0.85) -> Tuple[bool, float, float]:
        """
        Compare recent average confidence vs long-term baseline.

        Returns (is_drifting, recent_avg, baseline_avg).
        is_drifting=True when recent_avg < baseline_avg * threshold.
        Requires at least 20 recent samples and 50 baseline samples to fire.
        """
        from datetime import datetime, timezone, timedelta
        entries = self._log.get(slot, [])
        if not entries:
            return False, 0.0, 0.0

        now      = datetime.now(timezone.utc)
        recent_cutoff   = (now - timedelta(days=recent_days)).isoformat()
        baseline_cutoff = (now - timedelta(days=baseline_days)).isoformat()

        recent   = [e["confidence"] for e in entries if e["ts"] >= recent_cutoff]
        baseline = [e["confidence"] for e in entries if e["ts"] >= baseline_cutoff]

        if len(recent) < 20 or len(baseline) < 50:
            return False, 0.0, 0.0

        recent_avg   = float(np.mean(recent))
        baseline_avg = float(np.mean(baseline))

        if baseline_avg == 0:
            return False, recent_avg, baseline_avg

        is_drifting = recent_avg < baseline_avg * threshold
        return is_drifting, recent_avg, baseline_avg

    def save(self, path: str) -> None:
        """Persist confidence log to JSON so it survives bot restarts."""
        try:
            import json
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._log, f)
        except Exception as exc:
            logger.warning("ConfidenceTracker: could not save to %s: %s", path, exc)

    def load(self, path: str) -> None:
        """Load a previously saved confidence log."""
        try:
            import json
            if os.path.exists(path):
                with open(path) as f:
                    self._log = json.load(f)
                total = sum(len(v) for v in self._log.values())
                logger.info("ConfidenceTracker: loaded %d entries across %d slots",
                            total, len(self._log))
        except Exception as exc:
            logger.warning("ConfidenceTracker: could not load from %s: %s", path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic retraining schedule
# ─────────────────────────────────────────────────────────────────────────────

def retrain_frequency(total_trades: int, stages: List[Dict]) -> int:
    """
    Look up how many completed trades between retrains,
    based on the `retrain_stages` list in config.
    """
    freq = stages[-1]["retrain_every"]
    for stage in reversed(stages):
        if total_trades >= stage["after_trades"]:
            freq = stage["retrain_every"]
            break
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> VotingClassifier:
    """
    RF + HistGradientBoosting + ExtraTrees, soft-vote ensemble.

    All three estimators use class_weight="balanced" so that the 83% HOLD
    label majority doesn't suppress BUY/SELL probabilities below the 0.38
    threshold. HistGradientBoostingClassifier replaces the old GradientBoosting
    which did not support class_weight — that caused GB to vote near class priors
    (0.09 BUY) and drag the soft-vote average from ~0.45 down to ~0.17.
    """
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=4,
        learning_rate=0.05, random_state=42, class_weight="balanced",
    )
    et = ExtraTreesClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb), ("et", et)], voting="soft"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main strategy class
# ─────────────────────────────────────────────────────────────────────────────

class TradingStrategy:

    def __init__(self, cfg: dict):
        sc = cfg["strategy"]
        self.buy_threshold   = sc["buy_threshold"]
        self.sell_threshold  = sc["sell_threshold"]
        self.retrain_stages  = sc["retrain_stages"]
        self.min_samples     = sc["min_training_samples"]
        self.label_horizon   = sc["label_horizon_candles"]
        self.label_threshold = sc["label_move_threshold"]
        self.min_ev          = sc["min_ev"]
        self.min_win_rate    = sc["min_win_rate"]
        self.min_ev_trades   = sc["min_ev_trades"]
        self.models_dir      = cfg["logging"]["models_dir"]
        self.data_dir        = cfg.get("data", {}).get("data_dir", "/data")

        os.makedirs(self.models_dir, exist_ok=True)

        # Default model (primary pair / backward-compat fallback)
        self.model: Optional[VotingClassifier] = None
        self.scaler = StandardScaler()
        self.selected_features: List[str] = list(FEATURE_COLS)

        # Per-strategy models — keyed by "{SYMBOL}_{TF}" e.g. "BTCUSDT_4h"
        # Allows multiple (symbol, timeframe) strategies to run simultaneously.
        self.symbol_models:   Dict[str, VotingClassifier] = {}
        self.symbol_scalers:  Dict[str, StandardScaler]         = {}
        self.symbol_features: Dict[str, List[str]]              = {}

        self.total_trades    = 0
        self.trades_since_retrain = 0
        self.stats = TradeStats(min_trades=self.min_ev_trades)
        self.feature_importance: Dict[str, float] = {}
        self.conf_tracker = ConfidenceTracker()

        # Analysis recommendations (loaded from /data/analysis_results.json)
        self.analysis: Optional[Dict] = None
        self._load_analysis_results()

        self._try_load_model()
        self._try_load_symbol_models()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _sym_key(symbol: str, timeframe_label: str = "") -> str:
        """
        Build a model key from symbol + optional timeframe.
        'ETHUSDT_UMCBL', '15m' → 'ETHUSDT_15m'
        'ETHUSDT_UMCBL', ''    → 'ETHUSDT'   (backward-compat)
        """
        base = symbol.replace("_SPBL", "").replace("_UMCBL", "")
        return f"{base}_{timeframe_label}" if timeframe_label else base

    def _model_path(self, sym_key: str = "") -> str:
        if sym_key:
            return os.path.join(self.models_dir, f"model_{sym_key}.joblib")
        return os.path.join(self.models_dir, "ensemble_model.joblib")

    def _meta_path(self, sym_key: str = "") -> str:
        if sym_key:
            return os.path.join(self.models_dir, f"meta_{sym_key}.joblib")
        return os.path.join(self.models_dir, "meta.joblib")

    def _try_load_model(self):
        if os.path.exists(self._model_path()):
            try:
                self.model             = joblib.load(self._model_path())
                meta                   = joblib.load(self._meta_path())
                self.scaler            = meta["scaler"]
                self.selected_features = meta["features"]
                self.total_trades      = meta.get("total_trades", 0)
                self.stats._history    = meta.get("trade_history", [])
                logger.info("✅ Loaded saved model (trained on %d trades)", self.total_trades)
            except Exception as exc:
                logger.warning("Could not load model (%s) — will train fresh.", exc)

    def _try_load_symbol_models(self):
        """Load any previously saved per-strategy models from disk."""
        try:
            for fname in os.listdir(self.models_dir):
                if not (fname.startswith("model_") and fname.endswith(".joblib")):
                    continue
                sym_key   = fname[len("model_"):-len(".joblib")]
                if not sym_key:
                    continue
                meta_path = self._meta_path(sym_key)
                if not os.path.exists(meta_path):
                    continue
                try:
                    model = joblib.load(self._model_path(sym_key))
                    meta  = joblib.load(meta_path)
                    self.symbol_models[sym_key]   = model
                    self.symbol_scalers[sym_key]  = meta["scaler"]
                    self.symbol_features[sym_key] = meta["features"]
                    logger.info("✅ Loaded strategy model: %s", sym_key)
                except Exception as exc:
                    logger.warning("Could not load model %s: %s", sym_key, exc)
        except Exception:
            pass   # models_dir may not exist yet on first run

    def _save_model(self, sym_key: str = ""):
        """Save model to disk. Pass sym_key to save a per-symbol model."""
        if sym_key:
            model   = self.symbol_models.get(sym_key, self.model)
            scaler  = self.symbol_scalers.get(sym_key, self.scaler)
            features = self.symbol_features.get(sym_key, self.selected_features)
        else:
            model, scaler, features = self.model, self.scaler, self.selected_features

        joblib.dump(model, self._model_path(sym_key))
        joblib.dump({
            "scaler":        scaler,
            "features":      features,
            "total_trades":  self.total_trades,
            "trade_history": self.stats._history,
        }, self._meta_path(sym_key))
        label = sym_key or "default"
        logger.info("💾 Model saved [%s] (%d total trades)", label, self.total_trades)

    # ── Analysis integration ──────────────────────────────────────────────────

    def _load_analysis_results(self) -> None:
        """Load the latest analysis results produced by analysis.py."""
        path = os.path.join(self.data_dir, "analysis_results.json")
        if not os.path.exists(path):
            logger.info("No analysis results yet — running with defaults.")
            return
        try:
            with open(path) as f:
                self.analysis = json.load(f)
            recs = self.analysis.get("recommendations", {})
            tf   = recs.get("best_signal_timeframe", "unknown")
            filt = recs.get("best_filter_timeframe", "none")
            top  = recs.get("top_features", [])[:5]
            logger.info("📊 Analysis loaded | best_tf=%s | filter=%s | top_features=%s",
                        tf, filt, top)
        except Exception as exc:
            logger.warning("Could not load analysis results: %s", exc)

    def reload_analysis(self) -> None:
        """Reload analysis results (call after a fresh analysis run)."""
        self._load_analysis_results()

    def recommended_features(self) -> List[str]:
        """
        Return the top features recommended by analysis, or fall back to
        the full FEATURE_COLS list if no analysis has been run.
        """
        if self.analysis:
            top = self.analysis.get("recommendations", {}).get("top_features", [])
            if top:
                return [f for f in top if f in FEATURE_COLS]
        return list(FEATURE_COLS)

    # ── Higher-timeframe confluence ───────────────────────────────────────────

    def htf_trend(self, higher_tf_df: pd.DataFrame) -> int:
        """
        Determine the trend direction from a higher-timeframe DataFrame.
        Returns:  +1 (bullish)  |  -1 (bearish)  |  0 (neutral / mixed)

        Logic (mirrors backtest htf_trend exactly):
          • EMA9 must be on the same side of EMA21 as price — confirms short-term
            momentum is aligned, not just that price briefly crossed EMA21.
          • MACD histogram must agree.
          All three must agree for a directional reading; any disagreement = neutral.
        """
        if higher_tf_df is None or len(higher_tf_df) < 30:
            return 0

        df = compute_features(higher_tf_df.copy())
        last = df.iloc[-1]

        ema9      = last.get("ema_9",    np.nan)
        ema21     = last.get("ema_21",   np.nan)
        macd_diff = last.get("macd_diff", np.nan)
        close     = last.get("close",    np.nan)

        if pd.isna(ema9) or pd.isna(ema21) or pd.isna(close):
            return 0

        bullish = ema9 > ema21 and close > ema21
        bearish = ema9 < ema21 and close < ema21

        if not pd.isna(macd_diff):
            if bullish and macd_diff > 0:
                return  1
            if bearish and macd_diff < 0:
                return -1
            return 0
        else:
            if bullish:
                return  1
            if bearish:
                return -1
            return 0

    def htf_adx(self, higher_tf_df: pd.DataFrame, period: int = 14) -> float | None:
        """
        Calculate ADX on the HTF DataFrame using Wilder's smoothing.
        Returns the most recent ADX value, or None if insufficient data.
        ADX < 25 = choppy/ranging; ADX >= 25 = trending.
        """
        if higher_tf_df is None or len(higher_tf_df) < period * 3:
            return None
        df = higher_tf_df.copy()
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        plus_dm  = (high - prev_high).clip(lower=0).where(
            (high - prev_high) > (prev_low - low), 0)
        minus_dm = (prev_low - low).clip(lower=0).where(
            (prev_low - low) > (high - prev_high), 0)
        alpha    = 1 / period
        atr_w    = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, float("nan"))
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, float("nan"))
        dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))).fillna(0)
        adx      = dx.ewm(alpha=alpha, adjust=False).mean()
        val = adx.iloc[-1]
        return float(val) if not pd.isna(val) else None

    def apply_confluence(self,
                         signal: int,
                         buy_p: float,
                         sell_p: float,
                         htf_direction: int) -> Tuple[int, float, float]:
        """
        Adjust a primary signal using the higher-timeframe trend direction.

        Rules:
          • BUY  + bullish HTF  → confirmed, slight confidence boost
          • BUY  + bearish HTF  → downgraded to HOLD (fighting the trend)
          • BUY  + neutral HTF  → kept as-is (no penalty, no boost)
          • SELL + bearish HTF  → confirmed
          • SELL + bullish HTF  → downgraded to HOLD
          • HOLD                → unchanged regardless of HTF
        """
        if signal == HOLD or htf_direction == 0:
            return signal, buy_p, sell_p

        if signal == BUY:
            if htf_direction == 1:
                # Confirmed — small boost to surface the trade card
                return BUY, min(buy_p * 1.08, 0.99), sell_p
            else:
                # Contradicted — suppress
                logger.debug("HTF bearish — suppressing BUY signal")
                return HOLD, buy_p, sell_p

        if signal == SELL:
            if htf_direction == -1:
                return SELL, buy_p, min(sell_p * 1.08, 0.99)
            else:
                logger.debug("HTF bullish — suppressing SELL signal")
                return HOLD, buy_p, sell_p

        return signal, buy_p, sell_p

    # ── Historical data loader ────────────────────────────────────────────────

    def load_historical_candles(self, symbol: str,
                                 timeframe_label: str = "15m") -> Optional[pd.DataFrame]:
        """
        Load saved historical CSV for a symbol/timeframe from the data dir.
        Returns None if not available.
        """
        # Strip _SPBL suffix for filename lookup
        sym = symbol.replace("_SPBL", "").replace("_UMCBL", "")
        path = os.path.join(self.data_dir, f"{sym}_{timeframe_label}.csv")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            logger.info("📂 Loaded historical data: %s %s — %d candles",
                        sym, timeframe_label, len(df))
            return df
        except Exception as exc:
            logger.warning("Could not load historical CSV %s: %s", path, exc)
            return None

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_features(df)
        available = [c for c in self.selected_features if c in df.columns]
        X = df[available].replace([np.inf, -np.inf], np.nan)
        # Forward-fill then back-fill so the final row always has values.
        # dropna() on the last row was silently returning HOLD when any indicator
        # lacked enough lookback candles (e.g. EMA200 on a short live window).
        return X.ffill().bfill().dropna()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, symbol: str = "", timeframe_label: str = "15m"):
        """
        Train the model. If historical CSV data is available for this symbol,
        it is prepended to the live candle data to give a much richer dataset.
        """
        # ── Merge historical + live data ──────────────────────────────────────
        if symbol:
            hist = self.load_historical_candles(symbol, timeframe_label)
            if hist is not None and len(hist) > 100:
                # Combine: historical first, then live candles on top
                combined = pd.concat([hist, df], ignore_index=True)
                combined = (combined
                            .drop_duplicates("timestamp")
                            .sort_values("timestamp")
                            .reset_index(drop=True))
                logger.info("📚 Training on %d historical + %d live candles = %d total",
                            len(hist), len(df), len(combined))
                df = combined

        df = compute_features(df)

        # Priority order for label threshold:
        #  1. Adaptive value computed by analysis.py from this symbol+TF's own
        #     return distribution (guarantees ~33% directional labels per pair)
        #  2. Static per-TF table (_TF_LABEL_THRESHOLD) — sensible defaults
        #  3. Global config fallback
        base_sym  = (symbol.replace("_UMCBL", "")
                           .replace("_SPBL", "")
                           .replace("_CMBL", ""))
        threshold = _TF_LABEL_THRESHOLD.get(timeframe_label, self.label_threshold)
        if self.analysis:
            stored = (self.analysis
                      .get("recommendations", {})
                      .get("label_thresholds", {})
                      .get(base_sym, {})
                      .get(timeframe_label))
            if stored is not None:
                threshold = stored
        logger.info("🏷️  Label threshold for %s [%s]: %.2f%% (adaptive from data)",
                    symbol or "?", timeframe_label, threshold * 100)
        labels = label_candles(df, self.label_horizon, threshold)

        available = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = df[available].replace([np.inf, -np.inf], np.nan)
        y = labels

        valid = X_raw.notna().all(axis=1) & y.notna()
        X_raw, y = X_raw[valid].iloc[:-self.label_horizon], y[valid].iloc[:-self.label_horizon]

        total_y = len(y)
        buys  = int((y == BUY).sum())
        sells = int((y == SELL).sum())
        holds = int((y == HOLD).sum())
        logger.info("📊 Label dist [%s %s]: BUY=%d (%.0f%%) SELL=%d (%.0f%%) HOLD=%d (%.0f%%)",
                    symbol or "?", timeframe_label,
                    buys,  buys  / total_y * 100,
                    sells, sells / total_y * 100,
                    holds, holds / total_y * 100)

        if len(X_raw) < self.min_samples:
            logger.info("⏳ Need %d samples to train (have %d). Skipping.",
                        self.min_samples, len(X_raw))
            return

        # ── Feature selection: rank importance on unscaled data ──────────────
        prelim_rf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                           class_weight="balanced", random_state=42, n_jobs=1)
        prelim_rf.fit(X_raw, y)
        importances = dict(zip(available, prelim_rf.feature_importances_))
        threshold   = np.percentile(list(importances.values()), 10)
        self.selected_features = [f for f, imp in importances.items() if imp >= threshold]
        self.feature_importance = importances
        logger.info("Feature selection: %d → %d features kept",
                    len(available), len(self.selected_features))

        # ── Scale ONLY the selected features — scaler must match predict() ───
        X_sel_raw = X_raw[self.selected_features]
        self.scaler = StandardScaler()
        X_sel = pd.DataFrame(
            self.scaler.fit_transform(X_sel_raw),
            columns=self.selected_features
        )

        # ── Build + fit lean calibrated model ─────────────────────────────────
        self.model = build_model()
        self.model.fit(X_sel, y)

        # ── Cross-val accuracy (3-fold to save memory) ────────────────────────
        scores = cross_val_score(self.model, X_sel, y, cv=3, scoring="accuracy")
        logger.info("🎯 Model trained | samples=%d | CV accuracy=%.2f±%.2f | features=%d",
                    len(X_sel), scores.mean(), scores.std(), len(self.selected_features))

        # ── Log top features ──────────────────────────────────────────────────
        top5 = sorted(self.selected_features,
                       key=lambda f: importances.get(f, 0), reverse=True)[:5]
        logger.info("Top features: %s",
                    {f: round(importances[f], 3) for f in top5})

        # ── Store model keyed by "SYMBOL_TF" (e.g. "BTCUSDT_4h") ─────────────
        if symbol:
            sym_key = self._sym_key(symbol, timeframe_label)
            self.symbol_models[sym_key]   = self.model
            self.symbol_scalers[sym_key]  = self.scaler
            self.symbol_features[sym_key] = self.selected_features
            self._save_model(sym_key)
        else:
            self._save_model()

        self.trades_since_retrain = 0

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, symbol: str = "",
                timeframe_label: str = "") -> Tuple[int, float, float]:
        """
        Returns (signal, buy_probability, sell_probability).
        signal: BUY (1), HOLD (0), SELL (-1)
        Probabilities are calibrated — 0.65 really does mean ~65% likely.

        Looks up the model keyed by "SYMBOL_TF" when timeframe_label is given,
        then falls back to plain "SYMBOL", then to the default model.
        """
        # Resolve which model/scaler/features to use
        sym_key_tf = self._sym_key(symbol, timeframe_label) if symbol and timeframe_label else ""
        sym_key    = self._sym_key(symbol) if symbol else ""

        if sym_key_tf and sym_key_tf in self.symbol_models:
            model   = self.symbol_models[sym_key_tf]
            scaler  = self.symbol_scalers[sym_key_tf]
            feats   = self.symbol_features[sym_key_tf]
        elif sym_key and sym_key in self.symbol_models:
            model   = self.symbol_models[sym_key]
            scaler  = self.symbol_scalers[sym_key]
            feats   = self.symbol_features[sym_key]
        elif self.model is not None:
            model   = self.model
            scaler  = self.scaler
            feats   = self.selected_features
        else:
            return HOLD, 0.0, 0.0

        X = self._prepare_X(df)
        feats = [c for c in feats if c in X.columns]
        if not feats or X.empty:
            return HOLD, 0.0, 0.0

        try:
            last_row = X[feats].iloc[[-1]]
            X_scaled = pd.DataFrame(
                scaler.transform(last_row),
                columns=feats
            )
        except Exception as exc:
            logger.debug("Prediction transform error: %s", exc)
            return HOLD, 0.0, 0.0

        proba   = model.predict_proba(X_scaled.values)[0]
        classes = list(model.classes_)

        buy_p  = float(proba[classes.index(BUY)])  if BUY  in classes else 0.0
        sell_p = float(proba[classes.index(SELL)]) if SELL in classes else 0.0

        if buy_p >= self.buy_threshold:
            return BUY,  buy_p, sell_p
        if sell_p >= self.sell_threshold:
            return SELL, buy_p, sell_p
        return HOLD, buy_p, sell_p

    def predict_mtf(self,
                    df_entry:           pd.DataFrame,
                    df_direction:       pd.DataFrame,
                    symbol:             str,
                    entry_tf_label:     str,
                    direction_tf_label: str) -> Tuple[int, float, float]:
        """
        Multi-timeframe entry confirmation signal.

        Step 1 — direction: use htf_trend() on df_direction (EMA21 + MACD rule).
        Step 2 — entry:     use the entry-TF model via predict().
        Step 3 — gate:      apply_confluence() suppresses entries that fight
                            the direction-TF trend.

        ATR for stop sizing is taken from df_entry (tighter stop → better R/R).
        No extra model training required — direction uses the rule-based trend,
        entry uses the already-trained entry-TF model.
        """
        direction = self.htf_trend(df_direction)
        signal, buy_p, sell_p = self.predict(df_entry, symbol=symbol,
                                              timeframe_label=entry_tf_label)
        signal, buy_p, sell_p = self.apply_confluence(signal, buy_p, sell_p, direction)
        return signal, buy_p, sell_p

    # ── Expected Value gate ───────────────────────────────────────────────────

    def trade_is_worth_it(self, symbol: str) -> Tuple[bool, str]:
        """
        EV gate — only blocks trades once min_ev_trades have been recorded.
        Below that threshold every signal is taken (warmup period).
        Thresholds: min_win_rate=0.40 (breakeven at 1.59 R/R is 38.6%),
                    min_ev_trades=30  (statistically meaningful sample).
        """
        ev, win_rate, avg_rr = self.stats.ev_and_winrate(symbol)
        if ev is None:
            # Not enough trades yet — warmup period, take all signals
            history = self.stats.for_pair(symbol)
            return True, f"warming up ({len(history)}/{self.min_ev_trades} trades)"

        if win_rate < self.min_win_rate:
            return False, (f"WR {win_rate*100:.1f}% < min {self.min_win_rate*100:.0f}%  "
                           f"EV={ev*100:+.3f}%  avg_rr={avg_rr:.2f}")
        if ev < self.min_ev:
            return False, (f"EV {ev*100:+.3f}% < min {self.min_ev*100:.3f}%  "
                           f"WR={win_rate*100:.1f}%  avg_rr={avg_rr:.2f}")

        return True, (f"EV={ev*100:+.3f}%  win_rate={win_rate*100:.1f}%  avg_rr={avg_rr:.2f}")

    # ── Outcome recording (triggers retraining) ───────────────────────────────

    def record_outcome(self, outcome: dict, df: pd.DataFrame,
                       symbol: str = "", timeframe_label: str = "15m"):
        """Call whenever a trade closes. Triggers retraining when due."""
        # Prefer slot_key (e.g. "BTCUSDT_UMCBL_4h") for per-strategy EV tracking;
        # fall back to pair so existing trade dicts without slot_key still work.
        stat_key = outcome.get("slot_key") or outcome.get("pair")
        outcome_for_stats = dict(outcome)
        outcome_for_stats["pair"] = stat_key   # TradeStats.for_pair() keys on "pair"
        self.stats.record(outcome_for_stats)
        self.total_trades        += 1
        self.trades_since_retrain += 1

        ev, wr, rr = self.stats.ev_and_winrate(stat_key)
        pnl = outcome.get("pnl_pct", 0)
        logger.info("📝 Trade #%d  %s  PnL=%+.2f%%  │  Overall: %s",
                    self.total_trades,
                    stat_key,
                    pnl * 100,
                    self.stats.summary_str(stat_key))

        freq = retrain_frequency(self.total_trades, self.retrain_stages)
        if self.trades_since_retrain >= freq:
            logger.info("🔄 Retraining triggered (every %d trades at this stage)…", freq)
            self.train(df, symbol=symbol, timeframe_label=timeframe_label)

    # ── Pair ranking for dynamic allocation ───────────────────────────────────

    def best_pairs(self) -> Dict[str, float]:
        return self.stats.rank_pairs()
