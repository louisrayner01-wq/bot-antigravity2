"""Portfolio v1 configuration — the 5 strategies discovered in strategy-lab.

Every strategy runs concurrently on the same account, at 3× leverage, on
4h entry / 1d bias timeframes, bidirectional (long AND short). Shared
1% risk-per-trade sizing with 50% max gross exposure to bound the
combined margin footprint.
"""
from __future__ import annotations
from dataclasses import dataclass


# ── Global risk & execution ────────────────────────────────────────────────────
DEFAULT_STARTING_EQUITY   = 100.0

FEE_RATE_PER_SIDE         = 0.0003        # WEEX taker × WXT 50% discount
SLIPPAGE_PER_SIDE         = 0.0003
DEFAULT_RISK_PER_TRADE    = 0.01          # 1% of current equity per trade
MAX_GROSS_EXPOSURE        = 1.0           # cap on total margin as fraction of equity
CIRCUIT_BREAKER_DD_PCT    = 0.50          # halt trading at -50% account drawdown
MAINTENANCE_MARGIN_PCT    = 0.005

# ── Trade management ──────────────────────────────────────────────────────────
DEFAULT_LEVERAGE          = 3
SL_ATR_MULTIPLE           = 2.0           # SL = 2 × ATR(14, 4h)
TP_R_MULTIPLE             = 3.0           # TP = 3 × SL distance
MAX_HOLD_BARS_4H          = 12            # 12 × 4h = 48h max hold
ATR_PERIOD                = 14

# ── Timeframes ────────────────────────────────────────────────────────────────
BIAS_TF                   = "1d"
ENTRY_TF                  = "4h"

# ── Rolling windows for signal tercile classification ─────────────────────────
BIAS_TERCILE_WINDOW       = 90            # rolling 90d for bias state
ENTRY_TERCILE_WINDOW      = 60            # rolling 60-bar for entry signal

# ── Bar cadence ───────────────────────────────────────────────────────────────
# All 5 sub-strategies use 4h entry / 1d bias, so entry decisions can only
# change on 4h bar close. Between bars, only position management (SL/TP
# checks against the currently-forming 4h bar's high/low) benefits from
# faster ticks. 5 min matches Strat 1 and keeps SL response tight without
# adding meaningful load (≈48 candle fetches/hr across 4 assets).
DEFAULT_POLL_SECONDS      = 300           # 5 minutes


@dataclass(frozen=True)
class StrategyConfig:
    """One strategy in the portfolio."""
    key:          str
    bias_signal:  str
    entry_signal: str
    variant:      str      # "single" (XRP only) or "multi" (all 4)
    leverage:     int


# The 5-strategy portfolio (config.py in ~/strategy-lab/portfolio kept in sync).
STRATEGIES: tuple[StrategyConfig, ...] = (
    StrategyConfig("S1_diff_vpt_multi",      "difficulty_chg_30", "vpt_z_200",              "multi",  3),
    StrategyConfig("S2_addr_vpt_multi",      "active_addr_z_90",  "vpt_z_200",              "multi",  3),
    StrategyConfig("S3_addr_chaikin_multi",  "active_addr_z_90",  "chaikin_vol_10",         "multi",  3),
    StrategyConfig("S4_diff_rsi_xrp",        "difficulty_chg_30", "rsi_14",                 "single", 3),
    StrategyConfig("S5_diff_bb_xrp",         "difficulty_chg_30", "bollinger_bandwidth_20", "single", 3),
)


ASSETS_MULTI  = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ASSETS_SINGLE = ("XRPUSDT",)


def assets_for(variant: str) -> tuple[str, ...]:
    if variant == "single":
        return ASSETS_SINGLE
    if variant == "multi":
        return ASSETS_MULTI
    raise ValueError(f"Unknown variant: {variant!r}")
