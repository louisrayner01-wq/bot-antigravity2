"""Signal computations ported from strategy-lab.

Every function takes an OHLCV DataFrame indexed by UTC timestamps and
returns a pandas Series aligned to that index. All computations use
`shift(1)` on close/volume where appropriate so the value at bar `t`
depends only on data up to bar `t-1` — no look-ahead.

BTC-network signals (difficulty, active_addresses) fetch live from
blockchain.info and cache per-process. They index on daily UTC timestamps.
"""
from __future__ import annotations
import logging
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests


logger = logging.getLogger(__name__)


# ── Price/volume signals ──────────────────────────────────────────────────────

def rsi_14(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI. Uses close up to bar t-1."""
    close = candles["close"].shift(1)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).rename(f"rsi_{period}")


def bollinger_bandwidth_20(candles: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """(upper − lower) / mid on closes shifted by 1."""
    close = candles["close"].shift(1)
    mid = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return ((upper - lower) / mid).rename(f"bb_bw_{period}")


def vpt_z_200(candles: pd.DataFrame, z_window: int = 200) -> pd.Series:
    """Volume Price Trend, z-scored over `z_window`."""
    close  = candles["close"].shift(1)
    volume = candles["volume"].shift(1)
    pct    = close.pct_change()
    vpt    = (pct * volume).cumsum()
    z = (vpt - vpt.rolling(z_window).mean()) / vpt.rolling(z_window).std()
    return z.rename(f"vpt_z_{z_window}")


def chaikin_vol_10(candles: pd.DataFrame, ema_period: int = 10, roc_period: int = 10) -> pd.Series:
    """Chaikin volatility — ROC of EMA(high − low)."""
    high = candles["high"].shift(1)
    low  = candles["low"].shift(1)
    hl   = high - low
    ema  = hl.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    roc  = (ema - ema.shift(roc_period)) / ema.shift(roc_period)
    return roc.rename(f"chaikin_vol_{ema_period}_{roc_period}")


def atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    high = candles["high"]
    low  = candles["low"]
    close_prev = candles["close"].shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ── BTC-network signals from blockchain.info ──────────────────────────────────
# Charts endpoint returns JSON like: {"values":[{"x":1622505600,"y":123.4}, ...]}
# where x is unix seconds. Very small payload; refresh every ~1h at most.

_BC_INFO_CACHE: Dict[str, tuple[float, pd.Series]] = {}
_BC_INFO_TTL_SECONDS = 3600
_BC_INFO_URL = "https://api.blockchain.info/charts/{series}"


def _fetch_blockchain_info(series: str, timespan_days: int = 180) -> pd.Series:
    """Fetch a blockchain.info chart series as a daily-indexed pd.Series.

    Cached in-process for _BC_INFO_TTL_SECONDS to avoid hammering the free
    endpoint. Returns an empty Series if the endpoint fails (caller can then
    skip this bias signal for the cycle).
    """
    cached = _BC_INFO_CACHE.get(series)
    if cached and time.time() - cached[0] < _BC_INFO_TTL_SECONDS:
        return cached[1]

    url = _BC_INFO_URL.format(series=series)
    try:
        resp = requests.get(url, params={
            "timespan":     f"{timespan_days}days",
            "format":       "json",
            "sampled":      "false",
        }, timeout=15)
        resp.raise_for_status()
        raw = resp.json().get("values", [])
    except Exception as exc:
        logger.warning("blockchain.info fetch failed (%s): %s", series, exc)
        return pd.Series(dtype="float64")

    if not raw:
        return pd.Series(dtype="float64")

    idx = pd.to_datetime([r["x"] for r in raw], unit="s", utc=True)
    ser = pd.Series([float(r["y"]) for r in raw], index=idx).sort_index()
    _BC_INFO_CACHE[series] = (time.time(), ser)
    return ser


def difficulty_chg_30(daily_index: pd.DatetimeIndex, n: int = 30) -> pd.Series:
    """30-day BTC difficulty change, reindexed onto `daily_index` (1d bars).

    Aligned as of bar t-1 via .shift(1) so bar t sees only data available up
    to t-1's close.
    """
    d = _fetch_blockchain_info("difficulty", timespan_days=max(180, n * 4))
    if d.empty:
        return pd.Series(np.nan, index=daily_index, name=f"diff_chg_{n}")
    chg = (d / d.shift(n) - 1).shift(1)
    return chg.reindex(daily_index, method="ffill").rename(f"diff_chg_{n}")


def active_addr_z_90(daily_index: pd.DatetimeIndex, z_window: int = 90) -> pd.Series:
    """90d rolling z-score of BTC active addresses."""
    a = _fetch_blockchain_info("n-unique-addresses", timespan_days=max(365, z_window * 4))
    if a.empty:
        return pd.Series(np.nan, index=daily_index, name=f"active_addr_z_{z_window}")
    z = ((a - a.rolling(z_window).mean()) / a.rolling(z_window).std()).shift(1)
    return z.reindex(daily_index, method="ffill").rename(f"active_addr_z_{z_window}")


# ── Registry ──────────────────────────────────────────────────────────────────
# Bias signals fetch external data and are keyed off a daily DatetimeIndex.
# Entry signals are pure OHLCV.

def entry_signal(name: str, candles: pd.DataFrame) -> pd.Series:
    if name == "vpt_z_200":
        return vpt_z_200(candles)
    if name == "rsi_14":
        return rsi_14(candles)
    if name == "bollinger_bandwidth_20":
        return bollinger_bandwidth_20(candles)
    if name == "chaikin_vol_10":
        return chaikin_vol_10(candles)
    raise KeyError(f"Unknown entry signal: {name!r}")


def bias_signal(name: str, daily_index: pd.DatetimeIndex) -> pd.Series:
    if name == "difficulty_chg_30":
        return difficulty_chg_30(daily_index)
    if name == "active_addr_z_90":
        return active_addr_z_90(daily_index)
    raise KeyError(f"Unknown bias signal: {name!r}")
