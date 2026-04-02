"""
trade_logger.py
Logs every trade to CSV + a live performance summary to console.
"""

import os
import csv
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persists all trade data and prints performance summaries."""

    FIELDS = [
        "timestamp", "pair", "side", "entry_price", "exit_price",
        "quantity", "leverage", "pnl_pct", "pnl_usdt", "candles_held",
        "exit_reason", "equity_after",
        # MAE / MFE — recorded per trade for stop-loss optimisation
        "mae_pct",      # Max Adverse Excursion as % of entry (e.g. 1.25 = 1.25%)
        "mfe_pct",      # Max Favorable Excursion as % of entry
        "wick_breach",  # 1 if price violated the entry candle's wick before close
    ]

    def __init__(self, trades_file: str):
        self.trades_file = trades_file
        os.makedirs(os.path.dirname(trades_file) or ".", exist_ok=True)
        self._write_header()
        self.records: List[dict] = []

    def _write_header(self):
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def log_trade(self, trade: dict, equity_after: float, exit_reason: str = "signal"):
        row = {
            "timestamp":    datetime.utcnow().isoformat(),
            "pair":         trade.get("pair", ""),
            "side":         trade.get("side", ""),
            "entry_price":  round(trade.get("entry_price", 0), 4),
            "exit_price":   round(trade.get("exit_price", 0), 4),
            "quantity":     round(trade.get("quantity", 0), 6),
            "leverage":     trade.get("leverage", 1),
            "pnl_pct":      round(trade.get("pnl_pct", 0) * 100, 3),
            "pnl_usdt":     round(trade.get("pnl_usdt", 0), 2),
            "candles_held": trade.get("candles_held", 0),
            "exit_reason":  exit_reason,
            "equity_after": round(equity_after, 2),
            "mae_pct":      round(trade.get("mae_pct", 0.0), 4),
            "mfe_pct":      round(trade.get("mfe_pct", 0.0), 4),
            "wick_breach":  trade.get("wick_breach", 0),
        }
        self.records.append(row)
        with open(self.trades_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)

        emoji = "🟢" if row["pnl_usdt"] >= 0 else "🔴"
        logger.info("%s  %s %s | PnL: %.2f%% (%.2f USDT) | Equity: %.2f",
                    emoji, row["side"].upper(), row["pair"],
                    row["pnl_pct"], row["pnl_usdt"], equity_after)

    def print_summary(self):
        if not self.records:
            logger.info("No completed trades yet.")
            return

        wins  = [r for r in self.records if r["pnl_usdt"] > 0]
        loses = [r for r in self.records if r["pnl_usdt"] <= 0]
        total_pnl = sum(r["pnl_usdt"] for r in self.records)
        win_rate  = len(wins) / len(self.records) * 100 if self.records else 0

        logger.info("=" * 55)
        logger.info("  📊 PERFORMANCE SUMMARY")
        logger.info("  Total trades : %d", len(self.records))
        logger.info("  Win rate     : %.1f%%", win_rate)
        logger.info("  Total PnL    : %.2f USDT", total_pnl)
        logger.info("  Avg win      : %.2f USDT",
                    sum(r["pnl_usdt"] for r in wins)  / max(len(wins), 1))
        logger.info("  Avg loss     : %.2f USDT",
                    sum(r["pnl_usdt"] for r in loses) / max(len(loses), 1))
        logger.info("=" * 55)

        # Per-pair breakdown
        pairs = {r["pair"] for r in self.records}
        for pair in sorted(pairs):
            pair_trades = [r for r in self.records if r["pair"] == pair]
            pair_pnl    = sum(r["pnl_usdt"] for r in pair_trades)
            pair_wins   = sum(1 for r in pair_trades if r["pnl_usdt"] > 0)
            logger.info("  %-18s  trades=%d  wins=%d  PnL=%.2f USDT",
                        pair, len(pair_trades), pair_wins, pair_pnl)
