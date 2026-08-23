"""Portfolio v1 live paper engine.

Runs the 5 confluence strategies concurrently against live 4h candles.
Each (strategy, asset) pair can have at most one open position at a time.
Shared equity + shared risk budget: every strategy sizes 1% of the same
account, up to the MAX_GROSS_EXPOSURE margin cap.

Persists state to disk between ticks so restarts don't lose open positions.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.config import (
    StrategyConfig, STRATEGIES, assets_for,
    ATR_PERIOD, SL_ATR_MULTIPLE, TP_R_MULTIPLE, MAX_HOLD_BARS_4H,
    BIAS_TERCILE_WINDOW, ENTRY_TERCILE_WINDOW,
    FEE_RATE_PER_SIDE, SLIPPAGE_PER_SIDE, MAX_GROSS_EXPOSURE,
    CIRCUIT_BREAKER_DD_PCT, DEFAULT_LEVERAGE, FUTURES_SUFFIX,
)
from portfolio.signals import entry_signal, bias_signal, atr


logger = logging.getLogger(__name__)


LONG  = 1
SHORT = -1


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Position:
    strategy_key: str
    asset:        str
    side:         int          # +1 long, -1 short
    entry_price:  float
    entry_ts:     str          # ISO-8601 UTC
    quantity:     float
    stop_loss:    float
    take_profit:  float
    leverage:     int
    entry_atr:    float        # for R-multiple bookkeeping
    bars_held:    int = 0

    def slot_key(self) -> str:
        return f"{self.strategy_key}::{self.asset}"


@dataclass
class ClosedTrade:
    strategy_key: str
    asset:        str
    side:         int
    entry_price:  float
    exit_price:   float
    quantity:     float
    leverage:     int
    entry_ts:     str
    exit_ts:      str
    pnl_usdt:     float
    pnl_pct:      float
    bars_held:    int
    exit_reason:  str


@dataclass
class EngineState:
    equity:          float
    hwm:             float
    initial_capital: float = 0.0        # locked at first init; used for PnL %
    halted:          bool  = False      # true once circuit breaker fires
    positions:       Dict[str, dict] = field(default_factory=dict)   # slot_key -> Position.__dict__

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "EngineState":
        d = json.loads(raw)
        equity = float(d["equity"])
        return cls(
            equity=equity,
            hwm=float(d["hwm"]),
            # Legacy state files (pre initial_capital) fall back to equity so
            # PnL % starts at 0% rather than blowing up on a missing baseline.
            initial_capital=float(d.get("initial_capital") or equity),
            halted=bool(d.get("halted", False)),
            positions=d.get("positions", {}),
        )


# ── Confluence + entry logic ──────────────────────────────────────────────────

def _tercile_bias_state(bias: pd.Series, window: int) -> pd.Series:
    """Bucket a bias series into {"long_bias", "neutral", "short_bias"} using
    a rolling `window`-bar tercile split. Returns an object Series aligned to bias.
    """
    q_lo = bias.rolling(window).quantile(1 / 3)
    q_hi = bias.rolling(window).quantile(2 / 3)
    state = pd.Series("neutral", index=bias.index, dtype="object")
    state[bias > q_hi] = "long_bias"
    state[bias < q_lo] = "short_bias"
    return state


def confluence_signal(cfg: StrategyConfig,
                      candles_4h: pd.DataFrame,
                      daily_index: pd.DatetimeIndex) -> pd.Series:
    """Return {-1, 0, +1} per bar for one (strategy, asset). Fires on
    transitions only (bar t-1 was 0 and bar t is not).
    """
    if len(candles_4h) < ENTRY_TERCILE_WINDOW + 5:
        return pd.Series(0.0, index=candles_4h.index)

    bias = bias_signal(cfg.bias_signal, daily_index)
    bias_state = _tercile_bias_state(bias, BIAS_TERCILE_WINDOW)
    # Align daily bias state onto the 4h entry index
    bias_aligned = bias_state.reindex(candles_4h.index, method="ffill")

    entry = entry_signal(cfg.entry_signal, candles_4h)
    e_lo = entry.rolling(ENTRY_TERCILE_WINDOW).quantile(1 / 3)
    e_hi = entry.rolling(ENTRY_TERCILE_WINDOW).quantile(2 / 3)

    sig = pd.Series(0.0, index=candles_4h.index)
    long_cond  = (bias_aligned == "long_bias")  & (entry > e_hi)
    short_cond = (bias_aligned == "short_bias") & (entry < e_lo)
    sig[long_cond]  = LONG
    sig[short_cond] = SHORT

    prev = sig.shift(1).fillna(0)
    return sig.where((sig != 0) & (prev == 0), 0.0)


# ── Position sizing ───────────────────────────────────────────────────────────

def size_position(equity: float, risk_pct: float, sl_distance: float, price: float) -> float:
    """Units = (equity × risk%) / SL distance in price. No leverage effect on
    sizing — leverage only affects margin usage."""
    if sl_distance <= 0 or price <= 0:
        return 0.0
    risk_usdt = equity * risk_pct
    return risk_usdt / sl_distance


def used_margin(positions: Dict[str, Position]) -> float:
    total = 0.0
    for p in positions.values():
        notional = abs(p.quantity) * p.entry_price
        total += notional / max(p.leverage, 1)
    return total


# ── Engine ────────────────────────────────────────────────────────────────────

class PortfolioEngine:
    """One instance per (user_id, family="portfolio"). Owns its own state file.

    Paper vs live is controlled by the `paper` flag on init. In live mode a
    WeexClient must be supplied — every entry places a real futures order +
    exchange-side TP/SL plan order, and every close sends a real close order.
    Local SL/TP checks stay on as belt-and-braces (exchange plan may already
    have triggered; duplicate closes are caught + logged).
    """

    def __init__(
        self,
        user_id: str,
        starting_equity: float,
        risk_per_trade: float,
        state_dir: Path,
        fetch_candles,        # callable(asset, tf) -> pd.DataFrame
        paper: bool = True,
        weex_client=None,     # WeexClient — required when paper=False
    ):
        self.user_id = user_id
        self.risk_per_trade = risk_per_trade
        self.fetch_candles  = fetch_candles
        self.paper          = paper
        self.weex           = weex_client
        if not paper and weex_client is None:
            # Refuse to run live without keys — safer to silently downgrade
            # to paper than to no-op live orders. Same posture as Strat 1.
            logger.warning("[%s] paper=False but no weex_client — forcing paper=True", user_id[:8])
            self.paper = True

        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = state_dir / f"portfolio_{user_id}.json"

        if self.state_path.exists():
            self.state = EngineState.from_json(self.state_path.read_text())
            # Backfill initial_capital for pre-existing state files that predate
            # this field. We can't recover the true starting equity from disk,
            # so we treat the current equity as the baseline and go from here.
            if not self.state.initial_capital:
                self.state.initial_capital = self.state.equity
            logger.info("[%s] Restored portfolio state: equity=%.2f positions=%d paper=%s",
                        user_id[:8], self.state.equity, len(self.state.positions), self.paper)
        else:
            self.state = EngineState(
                equity=starting_equity,
                hwm=starting_equity,
                initial_capital=starting_equity,
            )
            logger.info("[%s] New portfolio state: equity=%.2f paper=%s",
                        user_id[:8], starting_equity, self.paper)

        self._closed_trades_this_tick: List[ClosedTrade] = []

    # ── live execution helpers ─────────────────────────────────────────────

    def _futures_symbol(self, asset: str) -> str:
        """Portfolio config stores plain spot symbols (BTCUSDT). Live futures
        orders on Weex need the _UMCBL suffix. Candle fetches still use spot
        (higher rate limit, no auth required)."""
        return asset if asset.endswith(FUTURES_SUFFIX) else asset + FUTURES_SUFFIX

    def _place_live_entry(self, asset: str, side: int, qty: float,
                          sl: float, tp: float, leverage: int) -> None:
        """Set leverage, open a real futures market order, attach exchange-side
        TP/SL. Failures are logged, not raised — a failed exchange order does
        not roll back the local Position record (bot will still track it as
        open until SL/TP resolves it)."""
        if self.paper or self.weex is None:
            return
        sym = self._futures_symbol(asset)
        side_word  = "long" if side == LONG else "short"
        order_side = "open_long" if side == LONG else "open_short"
        try:
            self.weex.set_leverage(sym, leverage, side_word)
        except Exception as exc:
            logger.warning("[%s] set_leverage failed for %s: %s", self.user_id[:8], sym, exc)
        try:
            self.weex.futures_order(sym, order_side, qty)
        except Exception as exc:
            logger.error("[%s] futures_order OPEN failed for %s: %s", self.user_id[:8], sym, exc)
            return
        try:
            self.weex.place_tpsl(sym, side_word, sl, tp, size=qty)
        except Exception as exc:
            logger.warning("[%s] place_tpsl failed for %s: %s", self.user_id[:8], sym, exc)

    def _place_live_close(self, asset: str, side: int, qty: float) -> None:
        """Send a real close order. If the exchange-side TP/SL already fired,
        the position no longer exists and Weex will return an error — we
        catch it because the local state has already computed exit correctly."""
        if self.paper or self.weex is None:
            return
        sym = self._futures_symbol(asset)
        order_side = "close_long" if side == LONG else "close_short"
        try:
            self.weex.futures_order(sym, order_side, qty)
        except Exception as exc:
            # Common: position already flat because exchange plan triggered first.
            logger.info("[%s] futures_order CLOSE for %s returned error (may already be flat): %s",
                        self.user_id[:8], sym, exc)

    # ── persistence ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self.state_path.write_text(self.state.to_json())
        except Exception as exc:
            logger.warning("[%s] state save failed: %s", self.user_id[:8], exc)

    def _load_position(self, slot_key: str) -> Optional[Position]:
        raw = self.state.positions.get(slot_key)
        if raw is None:
            return None
        return Position(**raw)

    def _store_position(self, pos: Position) -> None:
        self.state.positions[pos.slot_key()] = asdict(pos)

    def _drop_position(self, slot_key: str) -> None:
        self.state.positions.pop(slot_key, None)

    # ── main tick ───────────────────────────────────────────────────────────

    def tick(self) -> List[ClosedTrade]:
        """Run one full portfolio evaluation. Returns any trades closed this tick."""
        self._closed_trades_this_tick = []

        if self.state.halted:
            return []

        # Cache candles per asset+tf so multiple strategies share fetches
        candle_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        def get_candles(asset: str, tf: str) -> pd.DataFrame:
            key = (asset, tf)
            if key in candle_cache:
                return candle_cache[key]
            df = self.fetch_candles(asset, tf)
            candle_cache[key] = df
            return df

        # 1) Manage open positions (SL/TP/time exit)
        self._manage_open_positions(get_candles)

        # 2) Check for new entries per strategy
        for cfg in STRATEGIES:
            self._check_entries(cfg, get_candles)

        # 3) Circuit breaker
        if self.state.equity <= self.state.hwm * (1 - CIRCUIT_BREAKER_DD_PCT):
            self.state.halted = True
            logger.warning("[%s] CIRCUIT BREAKER hit: equity=%.2f hwm=%.2f",
                           self.user_id[:8], self.state.equity, self.state.hwm)

        # 4) HWM
        if self.state.equity > self.state.hwm:
            self.state.hwm = self.state.equity

        self._save()
        return list(self._closed_trades_this_tick)

    # ── position management ────────────────────────────────────────────────

    def _manage_open_positions(self, get_candles) -> None:
        for slot_key in list(self.state.positions.keys()):
            pos = self._load_position(slot_key)
            if pos is None:
                continue
            candles = get_candles(pos.asset, "4h")
            if candles is None or candles.empty:
                continue
            latest = candles.iloc[-1]
            price = float(latest["close"])
            high  = float(latest["high"])
            low   = float(latest["low"])

            exit_reason: Optional[str] = None
            exit_price = price

            # Bump bars_held on new bar
            pos.bars_held += 1

            # SL/TP checks — use bar's extreme against direction
            if pos.side == LONG:
                if low <= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price  = pos.stop_loss
                elif high >= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price  = pos.take_profit
            else:  # SHORT
                if high >= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price  = pos.stop_loss
                elif low <= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price  = pos.take_profit

            # Time exit
            if exit_reason is None and pos.bars_held >= MAX_HOLD_BARS_4H:
                exit_reason = "time_exit"
                exit_price  = price

            if exit_reason:
                self._close_position(pos, exit_price, exit_reason)
            else:
                # Persist bars_held bump
                self._store_position(pos)

    def _close_position(self, pos: Position, exit_price: float, reason: str) -> None:
        # Live-mode: send the real close order first so the exchange settles
        # around the price we computed exit at. If exchange TP/SL already
        # fired this will error and get logged — local state still updates.
        self._place_live_close(pos.asset, pos.side, pos.quantity)

        # PnL: (exit − entry) × qty × side, minus round-trip fees + slippage
        raw_pnl = (exit_price - pos.entry_price) * pos.quantity * pos.side
        notional_in  = abs(pos.quantity) * pos.entry_price
        notional_out = abs(pos.quantity) * exit_price
        cost = (notional_in + notional_out) * (FEE_RATE_PER_SIDE + SLIPPAGE_PER_SIDE)
        pnl_usdt = raw_pnl - cost
        pnl_pct = pnl_usdt / self.state.equity if self.state.equity > 0 else 0.0

        self.state.equity += pnl_usdt

        trade = ClosedTrade(
            strategy_key=pos.strategy_key,
            asset=pos.asset,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            leverage=pos.leverage,
            entry_ts=pos.entry_ts,
            exit_ts=datetime.now(timezone.utc).isoformat(),
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            bars_held=pos.bars_held,
            exit_reason=reason,
        )
        self._closed_trades_this_tick.append(trade)
        self._drop_position(pos.slot_key())
        logger.info("[%s] CLOSE %s %s %s @ %.4f reason=%s pnl=%.2f equity=%.2f",
                    self.user_id[:8], pos.strategy_key, pos.asset,
                    "LONG" if pos.side == LONG else "SHORT",
                    exit_price, reason, pnl_usdt, self.state.equity)

    # ── entry logic ─────────────────────────────────────────────────────────

    def _check_entries(self, cfg: StrategyConfig, get_candles) -> None:
        for asset in assets_for(cfg.variant):
            slot_key = f"{cfg.key}::{asset}"
            if slot_key in self.state.positions:
                continue  # already open — one slot per (strategy, asset)

            candles_4h = get_candles(asset, "4h")
            if candles_4h is None or candles_4h.empty:
                continue

            # Daily index for bias — reuse 4h candles' end, sample 1 per day
            # (blockchain.info series is daily; we just need a DatetimeIndex to
            # reindex onto)
            daily_index = pd.date_range(
                end=candles_4h.index[-1],
                periods=max(BIAS_TERCILE_WINDOW * 4, 365),
                freq="D",
                tz="UTC",
            )
            sig = confluence_signal(cfg, candles_4h, daily_index)
            latest_sig = float(sig.iloc[-1]) if not sig.empty else 0.0
            if latest_sig == 0.0:
                continue

            side = LONG if latest_sig > 0 else SHORT
            price = float(candles_4h["close"].iloc[-1])
            atr_series = atr(candles_4h, period=ATR_PERIOD)
            atr_now = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            if atr_now <= 0 or np.isnan(atr_now):
                continue

            sl_distance = SL_ATR_MULTIPLE * atr_now
            stop_loss   = price - side * sl_distance
            take_profit = price + side * TP_R_MULTIPLE * sl_distance

            qty = size_position(self.state.equity, self.risk_per_trade, sl_distance, price)
            if qty <= 0:
                continue

            # Margin cap check
            tentative_margin = (qty * price) / max(cfg.leverage, 1)
            existing = used_margin({
                k: Position(**v) for k, v in self.state.positions.items()
            })
            if existing + tentative_margin > self.state.equity * MAX_GROSS_EXPOSURE:
                logger.info("[%s] SKIP %s %s — margin cap %.2f+%.2f > %.2f",
                            self.user_id[:8], cfg.key, asset,
                            existing, tentative_margin,
                            self.state.equity * MAX_GROSS_EXPOSURE)
                continue

            pos = Position(
                strategy_key=cfg.key,
                asset=asset,
                side=side,
                entry_price=price,
                entry_ts=datetime.now(timezone.utc).isoformat(),
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=cfg.leverage,
                entry_atr=atr_now,
            )
            self._store_position(pos)
            # Live-mode: fire the real WEEX order + exchange-side TP/SL. In
            # paper mode this is a no-op so the local Position is the sole
            # record of the trade.
            self._place_live_entry(asset, side, qty, stop_loss, take_profit, cfg.leverage)
            logger.info("[%s] OPEN  %s %s %s @ %.4f qty=%.6f SL=%.4f TP=%.4f paper=%s",
                        self.user_id[:8], cfg.key, asset,
                        "LONG" if side == LONG else "SHORT",
                        price, qty, stop_loss, take_profit, self.paper)
