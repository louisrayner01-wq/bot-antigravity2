"""
risk_manager.py
Position sizing, stop-loss / take-profit, R/R validation,
and daily drawdown protection.

Key changes vs v1:
  - Absolute £5 risk per trade (not a percentage), so position size
    is always calculated to lose exactly £5 if stop-loss is hit
  - R/R gate: trade is blocked if actual R/R < min_rr_ratio
  - Kelly-aware sizing: position never exceeds what the Kelly formula
    would recommend given historical win rate
  - Daily loss halt uses absolute £ amount, not percentage
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class Position:
    pair:               str
    side:               str        # 'long' | 'short'
    entry_price:        float
    quantity:           float      # current remaining quantity (reduced after TP1)
    stop_loss:          float
    take_profit:        float
    leverage:           int
    entry_time:         str
    candles_held:       int   = 0
    order_id:           Optional[str] = None
    # ── Multi-TP fields ──────────────────────────────────────────────────────
    tp1_price:          float = 0.0    # TP1 price level (set on open)
    tp1_hit:            bool  = False  # True once 50% has been closed at TP1
    quantity_original:  float = 0.0   # Full qty at entry (before any partial close)
    # ── MAE / MFE excursion tracking ─────────────────────────────────────────
    mae_pct:            float = 0.0   # Max Adverse Excursion as fraction of entry (e.g. 0.012 = 1.2%)
    mfe_pct:            float = 0.0   # Max Favorable Excursion as fraction of entry
    entry_candle_low:   float = 0.0   # Low of the entry candle (wick level for longs)
    entry_candle_high:  float = 0.0   # High of the entry candle (wick level for shorts)

    @property
    def rr_ratio(self) -> float:
        if self.side == "long":
            risk    = self.entry_price - self.stop_loss
            reward  = self.take_profit - self.entry_price
        else:
            risk    = self.stop_loss - self.entry_price
            reward  = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0.0


class RiskManager:

    def __init__(self, cfg: dict):
        rc = cfg["risk"]
        sc = cfg["strategy"]

        self.initial_capital    = rc["initial_capital"]      # £100
        self.risk_per_trade_abs = rc["risk_per_trade_abs"]   # £5
        self.sl_atr_mult        = rc["stop_loss_atr_mult"]   # 1.5
        self.tp_atr_mult        = rc["take_profit_atr_mult"] # 3.0
        self.tp1_atr_mult       = rc.get("take_profit_1_atr_mult",  1.5)  # 1:1 R/R
        self.tp1_close_pct      = rc.get("take_profit_1_close_pct", 0.50) # 50%
        self.min_rr             = rc["min_rr_ratio"]         # 1.5
        self.max_open           = rc["max_open_positions"]   # 2
        self.max_daily_loss     = rc["max_daily_loss_abs"]   # £10
        self.min_holding        = sc.get("min_holding_candles", 2)
        self.max_leverage       = rc.get("max_leverage", 20) # hard cap

        self.equity             = float(self.initial_capital)
        self.day_start_equity   = float(self.initial_capital)
        self.today              = date.today()
        self.open_positions: Dict[str, Position] = {}

    # ── Equity ────────────────────────────────────────────────────────────────

    def update_equity(self, new_equity: float):
        today = date.today()
        if today != self.today:
            self.day_start_equity = new_equity
            self.today = today
        self.equity = new_equity

    def daily_loss(self) -> float:
        return self.day_start_equity - self.equity   # positive = loss

    def trading_halted(self) -> bool:
        halted = self.daily_loss() >= self.max_daily_loss
        if halted:
            logger.warning(
                "🛑 Daily loss limit hit (£%.2f / £%.2f). Trading halted for today.",
                self.daily_loss(), self.max_daily_loss
            )
        return halted

    # ── Stop / take-profit prices ─────────────────────────────────────────────

    def stop_loss_price(self, entry: float, atr: float, side: str) -> float:
        dist = atr * self.sl_atr_mult
        return entry - dist if side == "long" else entry + dist

    def take_profit_price(self, entry: float, atr: float, side: str) -> float:
        dist = atr * self.tp_atr_mult
        return entry + dist if side == "long" else entry - dist

    def tp1_price_for(self, entry: float, atr: float, side: str) -> float:
        """First partial take-profit at tp1_atr_mult × ATR (default 1:1 R/R)."""
        dist = atr * self.tp1_atr_mult
        return entry + dist if side == "long" else entry - dist

    # ── Partial close at TP1 ─────────────────────────────────────────────────

    def partial_close(self, pair: str, exit_price: float) -> Optional[dict]:
        """
        Close tp1_close_pct (50%) of an open position at TP1.
        Moves stop loss to entry price (breakeven) so the remaining half
        is now a risk-free runner.
        Returns a trade-record dict for logging, or None if not applicable.
        """
        pos = self.open_positions.get(pair)
        if not pos or pos.tp1_hit:
            return None

        close_qty = round(pos.quantity_original * self.tp1_close_pct, 6)
        remaining = round(pos.quantity - close_qty, 6)

        if pos.side == "long":
            price_move = exit_price - pos.entry_price
        else:
            price_move = pos.entry_price - exit_price

        # Same fix: qty already sized for 1× risk — do not multiply by leverage.
        pnl_usdt = close_qty * price_move
        pnl_pct  = price_move / pos.entry_price

        self.equity = max(self.equity + pnl_usdt, 0.0)

        # Update position in place — reduce qty, move SL to breakeven, flag TP1 done
        pos.quantity  = max(remaining, 0.0)
        pos.stop_loss = pos.entry_price    # breakeven
        pos.tp1_hit   = True

        logger.info(
            "🎯 TP1  %s  closed %.0f%%  qty=%.5f @ £%.4f | "
            "PnL=%+.2f£ | Remaining=%.5f | SL→breakeven £%.4f",
            pair, self.tp1_close_pct * 100, close_qty, exit_price,
            pnl_usdt, remaining, pos.entry_price,
        )

        return {
            "pair":         pair,
            "side":         pos.side,
            "entry_price":  pos.entry_price,
            "exit_price":   exit_price,
            "quantity":     close_qty,
            "leverage":     pos.leverage,
            "pnl_pct":      pnl_pct,
            "pnl_usdt":     pnl_usdt,
            "candles_held": pos.candles_held,
            "exit_type":    "tp1_partial",
            "mae_pct":      round(pos.mae_pct * 100, 4),
            "mfe_pct":      round(pos.mfe_pct * 100, 4),
            "wick_breach":  0,   # wick breach measured at full close only
        }

    # ── R/R validation ────────────────────────────────────────────────────────

    def rr_acceptable(self, entry: float, sl: float, tp: float, side: str) -> Tuple[bool, float]:
        """
        Returns (is_acceptable, actual_rr).
        Checks the real R/R from the ATR-derived stop and take-profit levels.
        """
        if side == "long":
            risk   = entry - sl
            reward = tp - entry
        else:
            risk   = sl - entry
            reward = entry - tp

        if risk <= 0:
            return False, 0.0

        rr = reward / risk
        ok = rr >= self.min_rr
        return ok, round(rr, 2)

    # ── Dynamic leverage + position sizing ───────────────────────────────────

    def calc_position(self, entry: float, atr: float,
                      win_rate: Optional[float] = None) -> Tuple[float, int]:
        """
        Calculates BOTH position size (qty) and leverage dynamically.

        The logic:
          1. We always risk exactly £risk_per_trade_abs (£5).
          2. Stop distance (in price) = ATR × sl_multiplier.
          3. qty = risk_amount / stop_distance  — the number of units needed
             so that hitting the stop loses exactly £5 (at 1x leverage).
          4. Notional value = qty × entry.  If notional > equity we need leverage.
          5. leverage = ceil(notional / equity), capped at max_leverage.

        This means:
          • Wide stop (high volatility) → small qty needed → low leverage
          • Tight stop (low volatility) → large qty needed → high leverage
          ...but the £ risk is ALWAYS exactly £5 regardless.
        """
        if entry <= 0 or atr <= 0:
            return 0.0, 1

        stop_distance = atr * self.sl_atr_mult   # price distance to stop
        risk_amount   = self.risk_per_trade_abs  # £5

        # Qty to lose exactly £5 if stop hit (no leverage)
        qty = risk_amount / stop_distance

        # How much notional do we need?
        notional = qty * entry

        # What leverage does that require given our equity?
        import math
        leverage_needed = math.ceil(notional / self.equity) if self.equity > 0 else 1
        leverage = max(1, min(self.max_leverage, leverage_needed))

        # Optional Kelly cap on qty (never overbet based on win rate history)
        if win_rate is not None and 0 < win_rate < 1:
            avg_rr   = self.tp_atr_mult / self.sl_atr_mult
            kelly_f  = max(0.0, win_rate - (1 - win_rate) / avg_rr) * 0.5  # half-Kelly
            max_qty  = (self.equity * kelly_f * leverage) / entry
            qty      = min(qty, max_qty)

        return round(max(qty, 0.0), 6), leverage

    # ── Gate checks ───────────────────────────────────────────────────────────

    def can_open(self, slot_key: str) -> Tuple[bool, str]:
        """
        slot_key is "SYMBOL_TF" (e.g. "BTCUSDT_UMCBL_4h") so two different
        timeframe strategies on the same symbol each get their own slot.
        """
        if self.trading_halted():
            return False, "daily loss limit reached"
        if slot_key in self.open_positions:
            return False, f"already have an open position for {slot_key}"
        if len(self.open_positions) >= self.max_open:
            return False, f"max open positions ({self.max_open}) reached"
        if self.equity < self.risk_per_trade_abs:
            return False, f"equity (£{self.equity:.2f}) below minimum trade risk (£{self.risk_per_trade_abs})"
        return True, "ok"

    def update_excursion(self, pair: str, current_price: float):
        """
        Called on every price check (every 5 min) to keep MAE and MFE current.

        MAE (Max Adverse Excursion)  — how far price moved *against* us as % of entry
        MFE (Max Favorable Excursion) — how far price moved *for* us as % of entry

        For a long  → adverse = below entry, favorable = above entry
        For a short → adverse = above entry, favorable = below entry
        """
        pos = self.open_positions.get(pair)
        if not pos:
            return
        if pos.side == "long":
            adverse   = max(0.0, (pos.entry_price - current_price) / pos.entry_price)
            favorable = max(0.0, (current_price   - pos.entry_price) / pos.entry_price)
        else:
            adverse   = max(0.0, (current_price   - pos.entry_price) / pos.entry_price)
            favorable = max(0.0, (pos.entry_price - current_price)   / pos.entry_price)
        pos.mae_pct = max(pos.mae_pct, adverse)
        pos.mfe_pct = max(pos.mfe_pct, favorable)

    def should_exit(self, pair: str, current_price: float) -> Optional[str]:
        pos = self.open_positions.get(pair)
        if not pos:
            return None
        pos.candles_held += 1
        if pos.candles_held < self.min_holding:
            return None
        if pos.side == "long":
            if current_price <= pos.stop_loss:
                return "stop_loss"
            if not pos.tp1_hit and pos.tp1_price > 0 and current_price >= pos.tp1_price:
                return "tp1"                    # partial close — trade stays open
            if current_price >= pos.take_profit:
                return "take_profit"
        else:
            if current_price >= pos.stop_loss:
                return "stop_loss"
            if not pos.tp1_hit and pos.tp1_price > 0 and current_price <= pos.tp1_price:
                return "tp1"
            if current_price <= pos.take_profit:
                return "take_profit"
        return None

    # ── Position registry ─────────────────────────────────────────────────────

    def open_position(self, pos: Position, slot_key: str = ""):
        key = slot_key if slot_key else pos.pair
        self.open_positions[key] = pos
        logger.info(
            "📈 OPEN  %s %s @ £%.4f | SL=£%.4f | TP=£%.4f | R/R=%.2f | Qty=%.5f | Lev=%dx",
            pos.side.upper(), pos.pair, pos.entry_price,
            pos.stop_loss, pos.take_profit, pos.rr_ratio, pos.quantity, pos.leverage,
        )

    def close_position(self, pair: str, exit_price: float) -> Optional[dict]:
        pos = self.open_positions.pop(pair, None)
        if not pos:
            return None

        # PnL in price terms
        if pos.side == "long":
            price_move = exit_price - pos.entry_price
        else:
            price_move = pos.entry_price - exit_price

        # Actual £ PnL: qty × price_move
        # NOTE: qty is already sized so that qty × stop_distance = £5 risk at 1×.
        # Leverage is only needed for the exchange margin calculation — do NOT
        # multiply here or the PnL will be overstated by the leverage factor.
        pnl_usdt = pos.quantity * price_move
        pnl_pct  = price_move / pos.entry_price

        self.equity = max(self.equity + pnl_usdt, 0.0)

        emoji = "🟢" if pnl_usdt >= 0 else "🔴"
        logger.info(
            "%s CLOSE %s @ £%.4f | PnL=%+.4f%% (%+.2f £) | Equity=£%.2f",
            emoji, pair, exit_price, pnl_pct * 100, pnl_usdt, self.equity,
        )

        # Did price breach the entry candle's wick before closing?
        # Long: wick breach = price went below the entry candle's low
        # Short: wick breach = price went above the entry candle's high
        worst_price = pos.entry_price * (1 - pos.mae_pct) if pos.side == "long" \
                 else pos.entry_price * (1 + pos.mae_pct)
        if pos.side == "long":
            wick_breach = 1 if (pos.entry_candle_low > 0 and worst_price < pos.entry_candle_low) else 0
        else:
            wick_breach = 1 if (pos.entry_candle_high > 0 and worst_price > pos.entry_candle_high) else 0

        return {
            "pair":              pair,
            "side":              pos.side,
            "entry_price":       pos.entry_price,
            "exit_price":        exit_price,
            "quantity":          pos.quantity,
            "leverage":          pos.leverage,
            "pnl_pct":           pnl_pct,
            "pnl_usdt":          pnl_usdt,
            "candles_held":      pos.candles_held,
            "mae_pct":           round(pos.mae_pct * 100, 4),   # % e.g. 1.25
            "mfe_pct":           round(pos.mfe_pct * 100, 4),
            "wick_breach":       wick_breach,
        }
