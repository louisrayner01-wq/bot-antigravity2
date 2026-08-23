"""
strategy_rules.py
─────────────────
Rule-based strategies — Conservative 1, Balanced 1, Aggressive 1.

All three share the same signal logic and only differ in:
  - max_concurrent  (1 / 2 / 3 positions allowed simultaneously)
  - asset_priority  (order to fill open slots when multiple signals fire)

Signal logic:
  LONG  = (4h low  <= EMA9) AND (4h close > EMA9) AND (4h EMA9 > EMA21)
          AND (BTC daily close > BTC daily SMA20)
  SHORT = (4h high >= EMA9) AND (4h close < EMA9) AND (4h EMA9 < EMA21)
          AND (BTC daily close < BTC daily SMA20)

Per-asset R:R (×ATR14 on 4h):
  BTC : TP = 3.0 × ATR  |  SL = 0.8 × ATR    (R:R 3.75)
  ETH : TP = 2.0 × ATR  |  SL = 0.8 × ATR    (R:R 2.50)
  SOL : TP = 2.0 × ATR  |  SL = 0.8 × ATR    (R:R 2.50)

Risk: configurable 0.5%–2% of CURRENT portfolio equity per trade.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LONG, FLAT, SHORT = 1, 0, -1

# Per-asset R:R multipliers (×ATR14). Lookup by the base symbol returned by
# normalise_symbol() below — never the exchange-specific suffix.
ASSET_RR: Dict[str, Dict[str, float]] = {
    "BTC": {"tp": 3.0, "sl": 0.8},
    "ETH": {"tp": 2.0, "sl": 0.8},
    "SOL": {"tp": 2.0, "sl": 0.8},
}


def normalise_symbol(symbol: str) -> str:
    """
    'BTCUSDT_UMCBL' → 'BTC'
    'ETHUSDT'       → 'ETH'
    'SOL'           → 'SOL'
    Returns the 3-letter base used in ASSET_RR.
    """
    base = (symbol.split("_")[0]
                  .replace("USDT", "")
                  .replace("USD",  "")
                  .upper())
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Strategy presets
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_PRESETS = {
    "conservative": {
        "max_concurrent": 1,
        "asset_priority": ["ETH", "SOL", "BTC"],
    },
    "balanced": {
        "max_concurrent": 2,
        "asset_priority": ["SOL", "ETH", "BTC"],
    },
    "aggressive": {
        # No cap → take every signal up to the number of tradable assets
        "max_concurrent": 3,
        "asset_priority": None,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Indicator computation (no external ML library — pure pandas)
# ─────────────────────────────────────────────────────────────────────────────

def compute_4h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the indicators needed for entry detection:
      ema9, ema21, atr14
    Expects df with columns [timestamp, open, high, low, close, volume] in
    ASCENDING time order (oldest first). Returns a copy.
    """
    if df is None or len(df) < 25:
        return df

    df = df.copy()
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_close = cl.shift()

    tr = pd.concat([
        (hi - lo),
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    df["ema9"]  = cl.ewm(span=9,  adjust=False).mean()
    df["ema21"] = cl.ewm(span=21, adjust=False).mean()
    return df


def compute_daily_btc_bias(btc_daily_df: pd.DataFrame) -> int:
    """
    Returns +1 (bullish), -1 (bearish), or 0 (unknown / insufficient data).
    Bull = BTC daily close > BTC daily SMA20.

    Uses the most recent CLOSED daily candle (the last row of btc_daily_df).
    If today's daily candle is still forming, callers should drop it first.
    """
    if btc_daily_df is None or len(btc_daily_df) < 21:
        return 0
    sma20 = btc_daily_df["close"].rolling(20).mean().iloc[-1]
    last_close = btc_daily_df["close"].iloc[-1]
    if pd.isna(sma20) or pd.isna(last_close):
        return 0
    return 1 if last_close > sma20 else -1


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """One candle's signal verdict for a single asset."""
    symbol:      str         # exchange-specific symbol (e.g. 'BTCUSDT_UMCBL')
    base:        str         # normalised base (e.g. 'BTC')
    side:        int         # LONG (1) / SHORT (-1) / FLAT (0)
    entry_price: float       # next-bar open we'd enter at (= last close as proxy)
    atr:         float       # ATR14 at signal candle
    sl_price:    float       # absolute SL price (computed from per-asset R:R)
    tp_price:    float       # absolute TP price
    rr:          float       # configured R:R (TP/SL multiplier)
    reason:      str = ""    # human-readable explanation

    def is_actionable(self) -> bool:
        return self.side != FLAT and self.atr > 0 and self.entry_price > 0


def evaluate_signal(symbol: str,
                    df_4h: pd.DataFrame,
                    btc_bias: int,
                    *,
                    use_last_closed: bool = True) -> Signal:
    """
    Evaluate the entry rule on the most recent 4h candle.

    use_last_closed: if True (default) we look at the second-to-last row
    (the most recent CLOSED candle). If False we look at the last row
    (useful during a backtest where rows are all closed candles).
    """
    base = normalise_symbol(symbol)
    rr = ASSET_RR.get(base)
    if rr is None:
        return Signal(symbol, base, FLAT, 0.0, 0.0, 0.0, 0.0, 0.0,
                      reason=f"no R:R config for {base}")

    df = compute_4h_indicators(df_4h)
    if df is None or len(df) < 22:
        return Signal(symbol, base, FLAT, 0.0, 0.0, 0.0, 0.0, 0.0,
                      reason="not enough candles")

    idx = -2 if use_last_closed and len(df) >= 2 else -1
    row = df.iloc[idx]

    cl, lo, hi = row["close"], row["low"], row["high"]
    e9, e21, atr = row["ema9"], row["ema21"], row["atr14"]

    if any(pd.isna(x) for x in (cl, lo, hi, e9, e21, atr)) or atr <= 0:
        return Signal(symbol, base, FLAT, 0.0, 0.0, 0.0, 0.0, 0.0,
                      reason="indicator NaN")

    long_setup  = (lo <= e9) and (cl > e9) and (e9 > e21) and (btc_bias ==  1)
    short_setup = (hi >= e9) and (cl < e9) and (e9 < e21) and (btc_bias == -1)

    if long_setup:
        sl = float(cl) - rr["sl"] * float(atr)
        tp = float(cl) + rr["tp"] * float(atr)
        return Signal(symbol, base, LONG, float(cl), float(atr),
                      sl_price=sl, tp_price=tp, rr=rr["tp"] / rr["sl"],
                      reason="4h pullback to EMA9 + BTC daily bull")

    if short_setup:
        sl = float(cl) + rr["sl"] * float(atr)
        tp = float(cl) - rr["tp"] * float(atr)
        return Signal(symbol, base, SHORT, float(cl), float(atr),
                      sl_price=sl, tp_price=tp, rr=rr["tp"] / rr["sl"],
                      reason="4h wick to EMA9 + BTC daily bear")

    return Signal(symbol, base, FLAT, 0.0, float(atr), 0.0, 0.0, 0.0,
                  reason=f"no setup (cl={cl:.2f} e9={e9:.2f} e21={e21:.2f} bias={btc_bias})")


def snapshot_signal(symbol: str,
                    df_4h: pd.DataFrame,
                    btc_bias: int,
                    btc_daily_close: float | None = None,
                    btc_daily_sma20: float | None = None,
                    *,
                    use_last_closed: bool = True) -> dict:
    """Full breakdown of the current 4h signal for the dashboard.

    Returns raw indicator values and each of the 4 firing conditions so the UI
    can render "how close is this to firing" bars. Never raises — on missing
    data returns an entry with `ok=False` and a `reason`.
    """
    base = normalise_symbol(symbol)
    rr = ASSET_RR.get(base)
    out: dict = {
        "base":         base,
        "symbol":       symbol,
        "side":         "FLAT",
        "rr_config":    rr,
        "btc_bias":     btc_bias,
        "btc_daily_close": btc_daily_close,
        "btc_daily_sma20": btc_daily_sma20,
    }
    if rr is None:
        out["reason"] = f"no R:R config for {base}"
        return out

    df = compute_4h_indicators(df_4h)
    if df is None or len(df) < 22:
        out["reason"] = "not enough candles"
        return out

    idx = -2 if use_last_closed and len(df) >= 2 else -1
    row = df.iloc[idx]
    cl, lo, hi = row["close"], row["low"], row["high"]
    e9, e21, atr14 = row["ema9"], row["ema21"], row["atr14"]
    if any(pd.isna(x) for x in (cl, lo, hi, e9, e21, atr14)) or atr14 <= 0:
        out["reason"] = "indicator NaN"
        return out

    out.update({
        "close": float(cl),
        "low":   float(lo),
        "high":  float(hi),
        "ema9":  float(e9),
        "ema21": float(e21),
        "atr14": float(atr14),
    })

    long_wick   = lo <= e9
    short_wick  = hi >= e9
    long_close  = cl > e9
    short_close = cl < e9
    ema_bull    = e9 > e21
    ema_bear    = e9 < e21
    bias_bull   = btc_bias == 1
    bias_bear   = btc_bias == -1

    out["conditions_long"] = {
        "wick_touch_ema9":   bool(long_wick),
        "close_above_ema9":  bool(long_close),
        "ema9_above_ema21":  bool(ema_bull),
        "btc_bias_bull":     bool(bias_bull),
    }
    out["conditions_short"] = {
        "wick_touch_ema9":   bool(short_wick),
        "close_below_ema9":  bool(short_close),
        "ema9_below_ema21":  bool(ema_bear),
        "btc_bias_bear":     bool(bias_bear),
    }

    if long_wick and long_close and ema_bull and bias_bull:
        side = "LONG"
        sl   = float(cl) - rr["sl"] * float(atr14)
        tp   = float(cl) + rr["tp"] * float(atr14)
    elif short_wick and short_close and ema_bear and bias_bear:
        side = "SHORT"
        sl   = float(cl) + rr["sl"] * float(atr14)
        tp   = float(cl) - rr["tp"] * float(atr14)
    else:
        side = "FLAT"
        sl = tp = None
    out["side"]     = side
    out["sl_price"] = sl
    out["tp_price"] = tp
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy controller — assembles candidate signals and applies concurrency
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyController:
    """
    A controller that:
      1. Builds candidate signals for each asset in the universe
      2. Filters them by the strategy's max_concurrent / priority rules
      3. Returns the trades that should be placed THIS tick

    The risk manager handles equity sizing, SL/TP enforcement, etc.
    """
    name:           str                  # "conservative" / "balanced" / "aggressive"
    max_concurrent: int
    asset_priority: Optional[List[str]]  # None = no preference (aggressive)

    @classmethod
    def from_preset(cls, name: str) -> "StrategyController":
        if name not in STRATEGY_PRESETS:
            raise ValueError(f"Unknown strategy preset '{name}'. "
                             f"Choose from {list(STRATEGY_PRESETS)}.")
        p = STRATEGY_PRESETS[name]
        return cls(
            name=name,
            max_concurrent=p["max_concurrent"],
            asset_priority=p["asset_priority"],
        )

    def select_entries(self,
                       candidates: List[Signal],
                       open_bases: List[str]) -> List[Signal]:
        """
        Given all evaluated candidate signals and the list of bases that
        currently have open positions, return the signals to actually act on.
        """
        n_open = len(open_bases)
        slots_left = max(0, self.max_concurrent - n_open)
        if slots_left == 0:
            return []

        # Drop signals where we already hold a position in that asset
        candidates = [c for c in candidates
                      if c.is_actionable() and c.base not in open_bases]
        if not candidates:
            return []

        if self.asset_priority:
            order = {a: i for i, a in enumerate(self.asset_priority)}
            candidates.sort(key=lambda c: order.get(c.base, 999))

        return candidates[:slots_left]
