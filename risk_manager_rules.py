"""
risk_manager_rules.py
─────────────────────
Simplified risk manager for the rule-based strategies (Conservative 1,
Balanced 1, Aggressive 1). Independent from the ML-bot's risk_manager.py
so the two systems can coexist without interfering.

What it does:
  - Tracks portfolio equity, HWM and per-day starting equity
  - Sizes each entry off CURRENT equity (not HWM) at a fixed risk_pct
  - Caps total concurrent positions at the strategy's max_concurrent
  - Enforces a daily loss circuit (default 5%) and an HWM drawdown circuit
    (default 25%) — both pause new entries, never force-close positions
  - Persists state to JSON so restarts don't lose equity/HWM
  - Reports closed trades via the same TradeLogger format as the ML bot
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RulePosition:
    """A position held by the rule-based bot. Simpler than the ML bot's Position
    because we don't use partial TPs, breakeven moves, MAE/MFE, etc."""
    base:         str          # 'BTC', 'ETH', 'SOL'
    symbol:       str          # exchange-specific symbol e.g. 'BTCUSDT_UMCBL'
    side:         str          # 'long' | 'short'
    entry_price:  float
    quantity:     float
    stop_loss:    float
    take_profit:  float
    leverage:     int
    entry_time:   str
    atr:          float        # ATR at entry (for analysis)
    order_id:     Optional[str] = None
    # Running wick extremes (paper mode) — same idea as ML bot
    seen_high:    float = 0.0
    seen_low:     float = 0.0

    @property
    def rr_ratio(self) -> float:
        if self.side == "long":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return (reward / risk) if risk > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Risk manager
# ─────────────────────────────────────────────────────────────────────────────

class RuleRiskManager:

    def __init__(self,
                 initial_capital: float,
                 risk_per_trade: float = 0.01,        # 1% default
                 max_concurrent: int    = 1,           # set by strategy preset
                 max_leverage:   int    = 10,
                 daily_loss_pct: float  = 0.05,        # 5% daily circuit
                 hwm_drawdown_pct: float = 0.25,       # 25% HWM circuit
                 data_dir:       str    = "/data"):
        self.initial_capital  = float(initial_capital)
        self.risk_per_trade   = float(risk_per_trade)
        self.max_concurrent   = int(max_concurrent)
        self.max_leverage     = int(max_leverage)
        self.daily_loss_pct   = float(daily_loss_pct)
        self.hwm_drawdown_pct = float(hwm_drawdown_pct)

        self.equity           = self.initial_capital
        self.hwm              = self.initial_capital
        self.day_start_equity = self.initial_capital
        self.today            = datetime.now(timezone.utc).date()

        # Open positions keyed by BASE symbol ('BTC') so one position per asset.
        self.open_positions: Dict[str, RulePosition] = {}

        self._state_path = os.path.join(data_dir, "risk_state_rules.json")

        # ── One-shot reset triggers (set in Railway Variables, then remove) ──
        # RESET_RULE_STATE=true     → wipe state file, fresh start at initial_capital
        # RESET_RULE_EQUITY=123.45  → keep open positions, set equity + HWM to this £
        if os.getenv("RESET_RULE_STATE", "").lower() == "true":
            try:
                if os.path.exists(self._state_path):
                    os.remove(self._state_path)
                logger.warning(
                    "⚠️  RESET_RULE_STATE applied — wiped %s. "
                    "Fresh start at £%.2f. REMOVE this env var from Railway now.",
                    self._state_path, self.initial_capital,
                )
            except Exception as exc:
                logger.error("RESET_RULE_STATE failed: %s", exc)
        else:
            self._load_state()

        eq_override = os.getenv("RESET_RULE_EQUITY")
        if eq_override:
            try:
                v = float(eq_override)
                self.equity = v
                self.hwm = v
                self.day_start_equity = v
                self._save_state()
                logger.warning(
                    "⚠️  RESET_RULE_EQUITY applied — equity/HWM set to £%.2f. "
                    "REMOVE this env var from Railway now.", v,
                )
            except ValueError:
                logger.error("RESET_RULE_EQUITY value '%s' invalid — ignored", eq_override)

    # ── State persistence ────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            state = {
                "equity":           self.equity,
                "hwm":              self.hwm,
                "day_start_equity": self.day_start_equity,
                "today":            self.today.isoformat(),
                "open_positions":   {k: asdict(v) for k, v in self.open_positions.items()},
            }
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save rule risk state: %s", exc)

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.equity           = state["equity"]
            self.hwm              = state["hwm"]
            self.day_start_equity = state["day_start_equity"]
            self.today            = datetime.fromisoformat(state["today"]).date()
            for base, pos_data in state.get("open_positions", {}).items():
                self.open_positions[base] = RulePosition(**pos_data)
            logger.info("✅ Rule risk state restored — equity=£%.2f HWM=£%.2f open=%d",
                        self.equity, self.hwm, len(self.open_positions))
        except Exception as exc:
            logger.warning("Could not restore rule risk state (%s) — starting fresh", exc)

    # ── Equity tracking ──────────────────────────────────────────────────────

    def update_equity(self, new_equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.today:
            self.day_start_equity = new_equity
            self.today = today
        self.equity = float(new_equity)
        if self.equity > self.hwm:
            self.hwm = self.equity
        self._save_state()

    def daily_loss_pct_now(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)

    def drawdown_from_hwm(self) -> float:
        if self.hwm <= 0:
            return 0.0
        return max(0.0, (self.hwm - self.equity) / self.hwm)

    # ── Circuit breakers ─────────────────────────────────────────────────────

    def trading_halted(self) -> Tuple[bool, str]:
        """Return (halted?, reason)."""
        if self.daily_loss_pct_now() >= self.daily_loss_pct:
            return True, (f"daily loss limit hit "
                          f"({self.daily_loss_pct_now()*100:.1f}% "
                          f">= {self.daily_loss_pct*100:.0f}% of day start)")
        if self.drawdown_from_hwm() >= self.hwm_drawdown_pct:
            return True, (f"HWM drawdown circuit "
                          f"({self.drawdown_from_hwm()*100:.1f}% "
                          f">= {self.hwm_drawdown_pct*100:.0f}% from HWM £{self.hwm:.2f})")
        return False, "ok"

    # ── Position sizing ──────────────────────────────────────────────────────

    def calc_position(self, entry: float, sl: float) -> Tuple[float, int]:
        """
        Returns (quantity, leverage) for a trade whose loss-at-SL equals
        risk_per_trade × current equity.

        qty = (risk_per_trade × equity) / |entry - sl|
        leverage = ceil(notional / margin), capped at max_leverage. Margin is
        set so that liquidation ≈ SL (matches the original bot's approach).
        """
        import math
        stop_distance = abs(entry - sl)
        if entry <= 0 or stop_distance <= 0 or self.equity <= 0:
            return 0.0, 1

        risk_amount = self.equity * self.risk_per_trade
        qty = risk_amount / stop_distance

        notional = qty * entry
        # leverage so that margin == risk_amount → liquidation ≈ SL
        raw_lev = notional / max(risk_amount, 1e-9)
        leverage = max(1, min(self.max_leverage, math.ceil(raw_lev)))

        return round(qty, 6), int(leverage)

    # ── Open / close ─────────────────────────────────────────────────────────

    def can_open(self, base: str) -> Tuple[bool, str]:
        halted, reason = self.trading_halted()
        if halted:
            return False, reason
        if base in self.open_positions:
            return False, f"{base} position already open"
        if len(self.open_positions) >= self.max_concurrent:
            return False, (f"max concurrent positions reached "
                           f"({len(self.open_positions)}/{self.max_concurrent})")
        return True, "ok"

    def open_position(self, pos: RulePosition) -> None:
        self.open_positions[pos.base] = pos
        logger.info(
            "📈 OPEN  %s %s @ £%.4f | SL=£%.4f | TP=£%.4f | R/R=%.2f | "
            "Qty=%.6f | Lev=%dx | risk=%.1f%%",
            pos.side.upper(), pos.symbol, pos.entry_price,
            pos.stop_loss, pos.take_profit, pos.rr_ratio,
            pos.quantity, pos.leverage, self.risk_per_trade * 100,
        )
        self._save_state()

    def should_exit(self, base: str, current_price: float,
                    candle_high: float = 0.0,
                    candle_low:  float = 0.0) -> Optional[str]:
        """Returns 'stop_loss' / 'take_profit' / None."""
        pos = self.open_positions.get(base)
        if not pos:
            return None

        # Track wick extremes so wicks between polls aren't missed
        if candle_high > 0:
            pos.seen_high = max(pos.seen_high, candle_high)
        if candle_low > 0 and candle_low < 1e15:
            pos.seen_low = (min(pos.seen_low, candle_low)
                            if pos.seen_low > 0 else candle_low)

        if pos.side == "long":
            hit_sl = (current_price <= pos.stop_loss
                      or (candle_low > 0 and candle_low <= pos.stop_loss)
                      or (pos.seen_low > 0 and pos.seen_low <= pos.stop_loss))
            hit_tp = (current_price >= pos.take_profit
                      or (candle_high > 0 and candle_high >= pos.take_profit)
                      or (pos.seen_high > 0 and pos.seen_high >= pos.take_profit))
        else:  # short
            hit_sl = (current_price >= pos.stop_loss
                      or (candle_high > 0 and candle_high >= pos.stop_loss)
                      or (pos.seen_high > 0 and pos.seen_high >= pos.stop_loss))
            hit_tp = (current_price <= pos.take_profit
                      or (candle_low > 0 and candle_low <= pos.take_profit)
                      or (pos.seen_low > 0 and pos.seen_low <= pos.take_profit))

        # SL takes priority when both fire on the same bar (conservative)
        if hit_sl:
            return "stop_loss"
        if hit_tp:
            return "take_profit"
        return None

    def close_position(self, base: str, exit_price: float, exit_reason: str) -> Optional[dict]:
        pos = self.open_positions.pop(base, None)
        if not pos:
            return None

        if pos.side == "long":
            price_move = exit_price - pos.entry_price
        else:
            price_move = pos.entry_price - exit_price

        # qty was sized so qty × |entry-sl| = risk_amount. PnL = qty × actual move.
        pnl_usdt = pos.quantity * price_move
        pnl_pct = price_move / pos.entry_price if pos.entry_price else 0.0

        self.equity = max(self.equity + pnl_usdt, 0.0)
        if self.equity > self.hwm:
            self.hwm = self.equity
        self._save_state()

        emoji = "🟢" if pnl_usdt >= 0 else "🔴"
        logger.info(
            "%s CLOSE %s %s @ £%.4f | PnL=%+.2f%% (%+.4f £) | "
            "Equity=£%.2f | Reason=%s",
            emoji, pos.side.upper(), pos.symbol, exit_price,
            pnl_pct * 100, pnl_usdt, self.equity, exit_reason,
        )

        return {
            "pair":          pos.symbol,
            "base":          pos.base,
            "side":          pos.side,
            "entry_price":   pos.entry_price,
            "exit_price":    exit_price,
            "stop_loss":     pos.stop_loss,
            "take_profit":   pos.take_profit,
            "quantity":      pos.quantity,
            "leverage":      pos.leverage,
            "atr":           pos.atr,
            "pnl_pct":       pnl_pct,
            "pnl_usdt":      pnl_usdt,
            "exit_reason":   exit_reason,
            "entry_time":    pos.entry_time,
            "exit_time":     datetime.now(timezone.utc).isoformat(),
        }

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def open_bases(self) -> List[str]:
        return list(self.open_positions.keys())

    def summary(self) -> str:
        return (f"equity=£{self.equity:.2f} HWM=£{self.hwm:.2f} "
                f"DD={self.drawdown_from_hwm()*100:.1f}% "
                f"day_dd={self.daily_loss_pct_now()*100:.1f}% "
                f"open={len(self.open_positions)}/{self.max_concurrent}")
