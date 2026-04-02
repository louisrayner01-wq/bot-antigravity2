"""
indicators.py
Calculates all technical indicators used as ML features.
Uses the 'ta' library (pip install ta) for standard indicators,
and custom implementations for VuManChu Cipher B + Divergences.

VuManChu Cipher B
─────────────────
The core of Cipher B is the WaveTrend Oscillator (WT1 / WT2):
  hlc3 = (high + low + close) / 3
  esa  = EMA(hlc3, n1=10)
  d    = EMA(|hlc3 - esa|, n1)
  ci   = (hlc3 - esa) / (0.015 × d)
  wt1  = EMA(ci, n2=21)     ← the "wave"
  wt2  = SMA(wt1, 4)        ← signal line

Divergences (as described by the user)
───────────────────────────────────────
  Bearish: price pivot-HIGH₂ > pivot-HIGH₁
           but  wt1  pivot-HIGH₂ < pivot-HIGH₁
           → momentum not confirming the new high → likely reversal down

  Bullish: price pivot-LOW₂  < pivot-LOW₁
           but  wt1  pivot-LOW₂  > pivot-LOW₁
           → selling pressure exhausting → likely reversal up

Divergence signals decay over 5 candles so the ML model can use them
even when the divergence candle itself is not the most recent row.
"""

import numpy as np
import pandas as pd
import ta
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Raw candle conversion
# ─────────────────────────────────────────────────────────────────────────────

def candles_to_df(raw_candles: List[List]) -> pd.DataFrame:
    """
    Convert raw Weex candle list to a clean DataFrame.
    Weex returns 11 fields per candle — we only need the first 6:
    [timestamp_ms, open, high, low, close, volume, ...]
    """
    if not raw_candles:
        return pd.DataFrame()

    trimmed = [row[:6] for row in raw_candles]
    df = pd.DataFrame(trimmed, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VuManChu Cipher B — WaveTrend Oscillator
# ─────────────────────────────────────────────────────────────────────────────

def _wavetrend(close: pd.Series,
               high:  pd.Series,
               low:   pd.Series,
               n1: int = 10,
               n2: int = 21) -> Tuple[pd.Series, pd.Series]:
    """
    Core WaveTrend calculation as used in VuManChu Cipher B.
    Returns (wt1, wt2).

    Parameters
    ──────────
    n1  Channel length  (default 10) — smoothing period for hlc3
    n2  Average length  (default 21) — EMA period for the final oscillator
    """
    hlc3 = (high + low + close) / 3.0
    esa  = hlc3.ewm(span=n1, adjust=False).mean()
    d    = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()

    # Avoid division by zero
    d_safe = d.replace(0.0, np.nan)
    ci   = (hlc3 - esa) / (0.015 * d_safe)

    wt1  = ci.ewm(span=n2, adjust=False).mean()
    wt2  = wt1.rolling(window=4, min_periods=1).mean()
    return wt1, wt2


def _cipher_b_mfi(close: pd.Series,
                  open_: pd.Series,
                  high:  pd.Series,
                  low:   pd.Series,
                  vol:   pd.Series,
                  length: int = 60) -> pd.Series:
    """
    Money Flow component of VuManChu Cipher B.

    Logic (mirrors the Pine Script):
      • Green candles (close >= open): upper shadow = max(high-open, close-low)
      • Red candles  (close <  open): lower shadow = max(open-low,  high-close)
    The result is normalised to [-1, +1] using a rolling sum.
    Positive values = buying pressure; negative = selling pressure.
    """
    green = close >= open_

    # Directional volume
    upper = pd.Series(0.0, index=close.index)
    lower = pd.Series(0.0, index=close.index)

    upper[green]  = np.maximum(
        (high  - open_)[green],
        (close - low  )[green]
    ) * vol[green]

    lower[~green] = np.maximum(
        (open_ - low  )[~green],
        (high  - close)[~green]
    ) * vol[~green]

    mf_upper = upper.rolling(length).sum()
    mf_lower = lower.rolling(length).sum()
    total    = (mf_upper + mf_lower).replace(0, np.nan)

    return (mf_upper - mf_lower) / total   # normalised to [-1, +1]


# ─────────────────────────────────────────────────────────────────────────────
# Pivot detection
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_highs_lows(series: pd.Series,
                      left:  int = 5,
                      right: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Identify pivot highs and pivot lows in a Series.

    A pivot HIGH at index i: series[i] is the maximum of the window
        [i-left … i+right].
    A pivot LOW  at index i: series[i] is the minimum of the same window.

    Note: because we require `right` candles to the right, pivots are only
    confirmed `right` candles after they occur.  This is intentional — it
    mirrors how divergences appear on a real chart: you see the divergence
    only once the right side of the pivot has formed.

    Returns
    ───────
    ph : boolean Series — True at pivot high locations
    pl : boolean Series — True at pivot low  locations
    """
    arr = series.values
    n   = len(arr)

    ph_flags = np.zeros(n, dtype=bool)
    pl_flags = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window = arr[i - left: i + right + 1]
        val    = arr[i]
        # Strict maximum / minimum — no ties (use >= for highs, <= for lows)
        if val >= np.max(window):
            ph_flags[i] = True
        if val <= np.min(window):
            pl_flags[i] = True

    return (
        pd.Series(ph_flags, index=series.index),
        pd.Series(pl_flags, index=series.index),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Divergence detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_divergences(price:    pd.Series,
                        wt1:      pd.Series,
                        left:     int = 5,
                        right:    int = 5,
                        decay:    int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Detect regular bullish and bearish divergences between price and WT1.

    Bearish divergence
    ──────────────────
    price pivot-HIGH₂ > price pivot-HIGH₁   (price making a higher high)
    AND
    wt1   pivot-HIGH₂ < wt1   pivot-HIGH₁   (wave making a lower high)
    → wave not confirming the breakout → bearish signal

    Bullish divergence
    ──────────────────
    price pivot-LOW₂  < price pivot-LOW₁    (price making a lower low)
    AND
    wt1   pivot-LOW₂  > wt1   pivot-LOW₁    (wave making a higher low)
    → selling exhaustion → bullish signal

    Both signals decay over `decay` candles via a rolling max so the ML
    model can read the divergence even one or two bars after detection.

    Returns
    ───────
    bull_div : float Series  (1.0 within decay window of bullish divergence, else 0)
    bear_div : float Series  (1.0 within decay window of bearish divergence, else 0)
    """
    ph, pl = _pivot_highs_lows(price, left, right)

    bear_instant = pd.Series(0.0, index=price.index)
    bull_instant = pd.Series(0.0, index=price.index)

    # ── Bearish: higher price high + lower wave high ───────────────────────────
    ph_idx = price.index[ph].tolist()
    for k in range(1, len(ph_idx)):
        i1, i2 = ph_idx[k - 1], ph_idx[k]
        if price[i2] > price[i1] and wt1[i2] < wt1[i1]:
            bear_instant[i2] = 1.0

    # ── Bullish: lower price low + higher wave low ─────────────────────────────
    pl_idx = price.index[pl].tolist()
    for k in range(1, len(pl_idx)):
        i1, i2 = pl_idx[k - 1], pl_idx[k]
        if price[i2] < price[i1] and wt1[i2] > wt1[i1]:
            bull_instant[i2] = 1.0

    # Decay: keep the signal alive for `decay` candles after detection
    bull_div = bull_instant.rolling(window=decay, min_periods=1).max()
    bear_div = bear_instant.rolling(window=decay, min_periods=1).max()

    return bull_div, bear_div


# ─────────────────────────────────────────────────────────────────────────────
# Main feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to the OHLCV DataFrame.
    Returns the df with extra feature columns (NaN rows at start are normal).
    """
    if len(df) < 60:
        logger.warning("Need at least 60 candles for reliable indicators (got %d)", len(df))
        return df

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]
    vol   = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"]  = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)

    df["ema_9_21_cross"]  = df["ema_9"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]

    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_obj.adx()

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["roc_5"]  = ta.momentum.ROCIndicator(close, window=5).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(close, window=20).roc()

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["vol_sma_20"] = vol.rolling(20).mean()
    df["vol_ratio"]  = vol / df["vol_sma_20"].replace(0, np.nan)

    df["obv"]       = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_slope"] = df["obv"].diff(5)

    # ── Price momentum ─────────────────────────────────────────────────────────
    df["ret_1"]  = close.pct_change(1)
    df["ret_5"]  = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)

    df["close_vs_ema21"] = (close - df["ema_21"]) / df["ema_21"].replace(0, np.nan)
    df["close_vs_bb"]    = df["bb_pct"] - 0.5

    # ── VuManChu Cipher B — WaveTrend ─────────────────────────────────────────
    wt1, wt2 = _wavetrend(close, high, low, n1=10, n2=21)

    df["vmcb_wt1"]  = wt1
    df["vmcb_wt2"]  = wt2
    df["vmcb_wt_diff"] = wt1 - wt2   # positive = WT1 above signal line (bullish lean)

    # Crossover signals (1 on the crossing candle, 0 otherwise)
    df["vmcb_wt_cross_bull"] = (
        (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    ).astype(float)
    df["vmcb_wt_cross_bear"] = (
        (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    ).astype(float)

    # Overbought / oversold zones (Cipher B thresholds: ±60)
    df["vmcb_wt_ob"] = (wt1 >  60).astype(float)   # overbought
    df["vmcb_wt_os"] = (wt1 < -60).astype(float)   # oversold

    # ── VuManChu Cipher B — Money Flow ────────────────────────────────────────
    df["vmcb_mfi"] = _cipher_b_mfi(close, open_, high, low, vol, length=60)

    # ── VuManChu Cipher B — Divergences ───────────────────────────────────────
    # Only run if we have enough candles for pivot detection (need left+right margin)
    if len(df) >= 60:
        try:
            bull_div, bear_div = _detect_divergences(
                close, wt1,
                left=5, right=5, decay=5,
            )
            df["vmcb_bull_div"] = bull_div
            df["vmcb_bear_div"] = bear_div
        except Exception as exc:
            logger.warning("Divergence detection failed: %s", exc)
            df["vmcb_bull_div"] = 0.0
            df["vmcb_bear_div"] = 0.0
    else:
        df["vmcb_bull_div"] = 0.0
        df["vmcb_bear_div"] = 0.0

    return df


# ── Feature list used by the ML model ─────────────────────────────────────────
FEATURE_COLS = [
    # Trend
    "ema_9_21_cross", "ema_21_50_cross",
    "macd", "macd_signal", "macd_diff",
    "adx",
    # Momentum
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "roc_5", "roc_20",
    # Volatility
    "bb_pct", "bb_width",
    "atr_14",
    # Volume
    "vol_ratio", "obv_slope",
    # Price momentum
    "ret_1", "ret_5", "ret_20",
    "close_vs_ema21", "close_vs_bb",
    # ── VuManChu Cipher B + Divergences ──────────────────────────────────────
    "vmcb_wt1",           # WaveTrend oscillator value
    "vmcb_wt2",           # WaveTrend signal line
    "vmcb_wt_diff",       # WT1 − WT2  (momentum of the wave)
    "vmcb_wt_cross_bull", # WT1 crossed above WT2 this candle
    "vmcb_wt_cross_bear", # WT1 crossed below WT2 this candle
    "vmcb_wt_ob",         # WaveTrend in overbought zone  (> +60)
    "vmcb_wt_os",         # WaveTrend in oversold zone    (< -60)
    "vmcb_mfi",           # Money Flow Index  (−1 = strong sell, +1 = strong buy)
    "vmcb_bull_div",      # Bullish divergence active  (lower low price, higher low wave)
    "vmcb_bear_div",      # Bearish divergence active  (higher high price, lower high wave)
]
