"""
trade_logger.py
Logs every trade to CSV + a live performance summary to console.
"""

import os
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persists all trade data and prints performance summaries."""

    FIELDS = [
        "timestamp", "pair", "slot_key", "side", "entry_price", "exit_price",
        "quantity", "leverage", "confidence", "pnl_pct", "pnl_usdt", "candles_held",
        "exit_reason", "equity_after",
        # MAE / MFE — recorded per trade for stop-loss optimisation
        "mae_pct",      # Max Adverse Excursion as % of entry (e.g. 1.25 = 1.25%)
        "mfe_pct",      # Max Favorable Excursion as % of entry
        "wick_breach",  # 1 if price violated the entry candle's wick before close
    ]

    # Fields recorded when a valid signal is skipped due to position already open.
    # sl_price / tp_price allow the hypothetical outcome to be calculated later
    # by comparing against subsequent candle data.
    SKIPPED_FIELDS = [
        "timestamp",      # when the signal fired
        "slot_key",       # e.g. "BTCUSDT_UMCBL_4h"
        "symbol",         # e.g. "BTCUSDT_UMCBL"
        "timeframe",      # e.g. "4h"
        "signal",         # "BUY" or "SELL"
        "confidence",     # model probability 0–1
        "entry_price",    # price at signal time
        "sl_price",       # where stop-loss would have been
        "tp_price",       # where take-profit would have been
        "rr",             # actual R/R ratio
        "ev_pct",         # expected value % at time of skip (None if not enough data)
        "skip_reason",    # e.g. "symbol already has an open position (BTCUSDT_UMCBL)"
        "blocking_slot",  # which open position caused the skip
    ]

    def __init__(self, trades_file: str):
        self.trades_file   = trades_file
        self.skipped_file  = trades_file.replace(".csv", "_skipped.csv")
        self.stats_file    = trades_file.replace(".csv", "_strategy_stats.json")
        os.makedirs(os.path.dirname(trades_file) or ".", exist_ok=True)
        self._migrate_csv()
        self._write_header()
        self._write_skipped_header()
        self.records: List[dict] = self._load_existing_records()

    def _load_existing_records(self) -> List[dict]:
        """Load all rows from trades.csv into memory on startup so stats survive restarts."""
        if not os.path.exists(self.trades_file):
            return []
        rows = []
        try:
            with open(self.trades_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Cast numeric fields so stats math works without type errors
                    cast: dict = {}
                    for field in self.FIELDS:
                        val = row.get(field, "")
                        if field in ("pnl_usdt", "pnl_pct", "equity_after",
                                     "entry_price", "exit_price", "quantity",
                                     "confidence", "mae_pct", "mfe_pct"):
                            try:
                                cast[field] = float(val)
                            except (ValueError, TypeError):
                                cast[field] = 0.0
                        elif field in ("leverage", "candles_held", "wick_breach"):
                            try:
                                cast[field] = int(val)
                            except (ValueError, TypeError):
                                cast[field] = 0
                        else:
                            cast[field] = val
                    rows.append(cast)
        except Exception as exc:
            logger.warning("Could not pre-load trade records: %s", exc)
        return rows

    def _migrate_csv(self):
        """
        Fix column-shift bugs caused by adding fields (e.g. slot_key) to FIELDS
        after the CSV was already created.  Reads the existing file header, and
        if it doesn't match the current FIELDS list, rewrites the file with the
        correct header while remapping existing rows.
        """
        if not os.path.exists(self.trades_file):
            return

        with open(self.trades_file, newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            if list(old_fields) == self.FIELDS:
                return   # already up to date
            rows = list(reader)

        logger.info("🔧 trades.csv header mismatch — migrating from %d to %d columns",
                    len(old_fields), len(self.FIELDS))

        # Symbols whose names appear in the slot_key column.
        # Used to detect rows written with the new FIELDS order but read
        # against the old header (where slot_key data ends up in 'side').
        _pair_tokens = ("USDT_UMCBL", "USDT_SPBL", "USDT")

        # Build a positional lookup for the OLD fields list so we can
        # re-read new-format rows that were written in new FIELDS order.
        new_fields_pos = {name: i for i, name in enumerate(self.FIELDS)}

        fixed_rows = []
        for row in rows:
            side_val = row.get("side", "")
            # Detect a row that was written with the NEW FIELDS order (slot_key
            # at position 3) but read with the OLD header (slot_key value
            # appears under the 'side' key).
            if any(tok in str(side_val) for tok in _pair_tokens):
                # The raw CSV values for this row are in NEW-FIELDS order.
                # Re-read them positionally.
                raw_vals = list(row.values())
                # If extra columns were spilled into the DictReader's 'None' key
                rest = row.get(None) or []
                raw_vals = [v for v in raw_vals if v is not None] + list(rest)

                new_row: dict = {}
                for fname in self.FIELDS:
                    idx = new_fields_pos.get(fname)
                    new_row[fname] = raw_vals[idx] if idx is not None and idx < len(raw_vals) else ""
                fixed_rows.append(new_row)
            else:
                # Old-format row — columns are correctly named, just add
                # any new fields that weren't present with empty defaults.
                new_row = {f: row.get(f, "") for f in self.FIELDS}
                fixed_rows.append(new_row)

        # Rewrite file with correct header
        import shutil
        backup = self.trades_file.replace(".csv", "_pre_migration.csv")
        shutil.copy2(self.trades_file, backup)
        with open(self.trades_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(fixed_rows)
        logger.info("✅ trades.csv migrated — %d rows fixed (backup: %s)", len(fixed_rows), backup)

    def _write_header(self):
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def _write_skipped_header(self):
        if not os.path.exists(self.skipped_file):
            with open(self.skipped_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.SKIPPED_FIELDS)
                writer.writeheader()

    def log_skipped(self, slot_key: str, symbol: str, timeframe: str,
                    signal: str, confidence: float,
                    entry_price: float, sl_price: float, tp_price: float,
                    rr: float, ev_pct: Optional[float],
                    skip_reason: str, blocking_slot: str = ""):
        """
        Record a signal that was valid (passed EV + R/R gates) but skipped
        because a position was already open on this symbol.
        The sl_price and tp_price fields allow the hypothetical outcome to be
        reconstructed later by replaying subsequent candle data.
        """
        row = {
            "timestamp":    datetime.utcnow().isoformat(),
            "slot_key":     slot_key,
            "symbol":       symbol,
            "timeframe":    timeframe,
            "signal":       signal,
            "confidence":   round(confidence, 4),
            "entry_price":  round(entry_price, 4),
            "sl_price":     round(sl_price, 4),
            "tp_price":     round(tp_price, 4),
            "rr":           round(rr, 3),
            "ev_pct":       round(ev_pct, 4) if ev_pct is not None else "",
            "skip_reason":  skip_reason,
            "blocking_slot": blocking_slot,
        }
        with open(self.skipped_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SKIPPED_FIELDS)
            writer.writerow(row)
        logger.info("⏭️  SKIPPED  %s[%s]  %s  conf=%.2f  reason=%s",
                    symbol, timeframe, signal, confidence, skip_reason)

    def log_trade(self, trade: dict, equity_after: float, exit_reason: str = "signal"):
        entry  = trade.get("entry_price", 0)
        sl     = trade.get("stop_loss",   0)
        qty    = trade.get("quantity",    0)
        pnl    = trade.get("pnl_usdt",   0)
        risk   = abs(entry - sl) * qty if entry and sl and qty else 0
        rr     = round(pnl / risk, 3) if risk > 0 else 0.0

        row = {
            "timestamp":    datetime.utcnow().isoformat(),
            "pair":         trade.get("pair", ""),
            "slot_key":     trade.get("slot_key", ""),
            "side":         trade.get("side", ""),
            "entry_price":  round(entry, 4),
            "exit_price":   round(trade.get("exit_price", 0), 4),
            "quantity":     round(qty, 6),
            "leverage":     trade.get("leverage", 1),
            "confidence":   round(trade.get("confidence", 0.0), 4),
            "pnl_pct":      round(trade.get("pnl_pct", 0) * 100, 3),
            "pnl_usdt":     round(pnl, 2),
            "candles_held": trade.get("candles_held", 0),
            "exit_reason":  exit_reason,
            "equity_after": round(equity_after, 2),
            "mae_pct":      round(trade.get("mae_pct", 0.0), 4),
            "mfe_pct":      round(trade.get("mfe_pct", 0.0), 4),
            "wick_breach":  trade.get("wick_breach", 0),
            "rr":           rr,
        }
        self.records.append(row)
        with open(self.trades_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)

        emoji = "🟢" if row["pnl_usdt"] >= 0 else "🔴"
        logger.info("%s  %s %s | PnL: %.2f%% (%.2f USDT) | Equity: %.2f",
                    emoji, row["side"].upper(), row["pair"],
                    row["pnl_pct"], row["pnl_usdt"], equity_after)
        self._update_strategy_stats(row)

    def _update_strategy_stats(self, row: dict):
        """
        Rebuild and persist per-slot stats to trades_strategy_stats.json.
        File is private — never pushed to GitHub.
        Stats: trades, wins, win_rate, avg_confidence, total_pnl_usdt, avg_pnl_usdt.
        """
        import json

        # Re-read all completed trades from the in-memory list (fast, already loaded)
        # Fall back to reading the CSV if records list is empty (e.g. first trade after restart)
        all_rows = self.records if self.records else []

        by_slot: dict = {}
        for r in all_rows:
            slot = r.get("slot_key") or r.get("pair", "unknown")
            by_slot.setdefault(slot, []).append(r)

        stats = {}
        for slot, rows in by_slot.items():
            wins  = [r for r in rows if r.get("pnl_usdt", 0) > 0]
            confs = [r["confidence"] for r in rows
                     if isinstance(r.get("confidence"), (int, float)) and r["confidence"] > 0]
            stats[slot] = {
                "trades":          len(rows),
                "wins":            len(wins),
                "losses":          len(rows) - len(wins),
                "win_rate_pct":    round(len(wins) / len(rows) * 100, 1) if rows else 0,
                "avg_confidence":  round(sum(confs) / len(confs), 4) if confs else None,
                "total_pnl_usdt":  round(sum(r.get("pnl_usdt", 0) for r in rows), 2),
                "avg_pnl_usdt":    round(sum(r.get("pnl_usdt", 0) for r in rows) / len(rows), 2),
                "last_trade":      rows[-1].get("timestamp", ""),
            }

        try:
            with open(self.stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write strategy stats: %s", exc)

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
