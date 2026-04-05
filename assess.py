"""
assess.py
═════════
Standalone performance assessment tool.

Reads:
  logs/trades.csv         — completed live/paper trades
  logs/trades_skipped.csv — valid signals blocked by per-symbol cap
  /data/{SYMBOL}_{TF}.csv — OHLCV candles for replaying skipped trades

Outputs a full report covering:
  1. Live trade performance — per strategy, vs backtest expectation
  2. Skipped trade replay   — hypothetical outcome if cap had been relaxed
  3. Cap assessment         — would allowing stacked positions have helped?
  4. Strategy ranking       — which slots are over/under performing

Run locally:
  python assess.py

Run against Railway volume (if mounted locally):
  python assess.py --trades /data/logs/trades.csv --data-dir /data

Optional flags:
  --trades      path to trades.csv     (default: logs/trades.csv)
  --skipped     path to trades_skipped.csv (default: logs/trades_skipped.csv)
  --data-dir    path to OHLCV CSVs     (default: /data)
  --out         save JSON report to file
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# ── Backtest baseline (2-year walk-forward results for comparison) ─────────────
# Taken directly from the backtest output — used to measure live vs expected.
BACKTEST_BASELINE = {
    "BTCUSDT_UMCBL_5m":  {"ev_pct": 0.102, "win_rate": 0.487, "total_pnl_pct": 105.5,  "trades_per_year": 515},
    "BTCUSDT_UMCBL_15m": {"ev_pct": 0.126, "win_rate": 0.449, "total_pnl_pct": 47.8,   "trades_per_year": 191},
    "BTCUSDT_UMCBL_1h":  {"ev_pct": 0.174, "win_rate": 0.451, "total_pnl_pct": 25.1,   "trades_per_year": 72},
    "ETHUSDT_UMCBL_5m":  {"ev_pct": 0.110, "win_rate": 0.475, "total_pnl_pct": 145.1,  "trades_per_year": 662},
    "ETHUSDT_UMCBL_15m": {"ev_pct": 0.168, "win_rate": 0.463, "total_pnl_pct": 81.8,   "trades_per_year": 243},
    "ETHUSDT_UMCBL_1h":  {"ev_pct": 0.615, "win_rate": 0.504, "total_pnl_pct": 75.6,   "trades_per_year": 62},
    "SOLUSDT_UMCBL_5m":  {"ev_pct": 0.155, "win_rate": 0.478, "total_pnl_pct": 200.8,  "trades_per_year": 649},
    "SOLUSDT_UMCBL_15m": {"ev_pct": 0.216, "win_rate": 0.449, "total_pnl_pct": 84.1,   "trades_per_year": 195},
    "SOLUSDT_UMCBL_1h":  {"ev_pct": 0.464, "win_rate": 0.468, "total_pnl_pct": 71.4,   "trades_per_year": 77},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  [!] Not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [!] Empty: {path}")
        return None
    return df


def _max_drawdown(pnl_series: pd.Series) -> float:
    equity = (1 + pnl_series / 100).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak * 100
    return float(dd.min())


def _streak(outcomes: pd.Series) -> tuple:
    max_w = max_l = cur_w = cur_l = 0
    for o in outcomes:
        if o:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _replay_skipped(row: pd.Series, candles: pd.DataFrame) -> Optional[str]:
    """
    Given a skipped signal row and subsequent OHLCV candles, determine
    whether the trade would have been a WIN or LOSS.
    Returns 'WIN', 'LOSS', or None if outcome cannot be determined.
    """
    try:
        ts         = pd.to_datetime(row["timestamp"], utc=True)
        entry      = float(row["entry_price"])
        sl         = float(row["sl_price"])
        tp         = float(row["tp_price"])
        signal     = row["signal"]

        future = candles[candles["timestamp"] > ts].head(200)
        if future.empty:
            return None

        for _, c in future.iterrows():
            high = float(c["high"])
            low  = float(c["low"])

            if signal == "BUY":
                if low <= sl:
                    return "LOSS"
                if high >= tp:
                    return "WIN"
            else:  # SELL
                if high >= sl:
                    return "LOSS"
                if low <= tp:
                    return "WIN"
        return None  # neither hit within 200 candles
    except Exception:
        return None


# ── Section 1: Live trade performance ────────────────────────────────────────

def analyse_live_trades(trades: pd.DataFrame) -> dict:
    print("\n" + "═" * 72)
    print("  SECTION 1 — LIVE TRADE PERFORMANCE")
    print("═" * 72)

    # Normalise slot_key — use pair as fallback
    if "slot_key" not in trades.columns:
        trades["slot_key"] = trades["pair"]
    else:
        trades["slot_key"] = trades["slot_key"].fillna(trades["pair"])

    overall_wins  = (trades["pnl_usdt"] > 0).sum()
    overall_total = len(trades)
    overall_pnl   = trades["pnl_usdt"].sum()
    overall_wr    = overall_wins / overall_total * 100 if overall_total else 0

    print(f"\n  Total trades  : {overall_total}")
    print(f"  Overall WR    : {overall_wr:.1f}%")
    print(f"  Total PnL     : £{overall_pnl:.2f}")

    results = {}
    slots   = trades["slot_key"].unique()

    header = (f"\n  {'Slot':<28} {'Trades':>6} {'WR%':>6} {'AvgW':>7} "
              f"{'AvgL':>7} {'EV%':>7} {'PnL£':>7} {'MaxDD':>7} "
              f"{'vs BT EV':>9} {'vs BT WR':>9}")
    print(header)
    print("  " + "─" * 98)

    for slot in sorted(slots):
        s      = trades[trades["slot_key"] == slot]
        wins   = s[s["pnl_usdt"] > 0]
        losses = s[s["pnl_usdt"] <= 0]
        n      = len(s)
        wr     = len(wins) / n * 100 if n else 0
        avg_w  = wins["pnl_pct"].mean()   if len(wins)   else 0.0
        avg_l  = losses["pnl_pct"].mean() if len(losses) else 0.0
        ev     = (wr / 100) * avg_w + (1 - wr / 100) * avg_l
        pnl    = s["pnl_usdt"].sum()
        dd     = _max_drawdown(s["pnl_pct"]) if n > 1 else 0.0

        bl          = BACKTEST_BASELINE.get(slot, {})
        bt_ev       = bl.get("ev_pct",   None)
        bt_wr       = bl.get("win_rate", None)
        ev_vs_bt    = f"{ev - bt_ev:+.3f}%" if bt_ev is not None and n >= 5 else "N/A"
        wr_vs_bt    = f"{wr/100 - bt_wr:+.3f}" if bt_wr is not None and n >= 5 else "N/A"

        flag = "✅" if ev > 0 else "❌"
        print(f"  {slot:<28} {n:>6} {wr:>5.1f}% {avg_w:>+6.2f}% "
              f"{avg_l:>+6.2f}% {ev:>+6.3f}%{flag} {pnl:>+6.2f} "
              f"{dd:>+6.1f}% {ev_vs_bt:>9} {wr_vs_bt:>9}")

        results[slot] = {
            "trades": n, "win_rate": wr / 100,
            "avg_win_pct": avg_w, "avg_loss_pct": avg_l,
            "ev_pct": ev, "total_pnl_usdt": pnl, "max_dd": dd,
        }

    return results


# ── Section 2: Skipped trade replay ──────────────────────────────────────────

def analyse_skipped_trades(skipped: pd.DataFrame, data_dir: str) -> dict:
    print("\n" + "═" * 72)
    print("  SECTION 2 — SKIPPED TRADE REPLAY")
    print("═" * 72)
    print("  Replaying skipped signals against subsequent candles...\n")

    # Load candle CSVs once per symbol/TF combo
    _candle_cache = {}

    def _get_candles(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        key  = f"{symbol}_{timeframe}"
        base = symbol.replace("_UMCBL", "").replace("_SPBL", "")
        fp   = os.path.join(data_dir, f"{base}_{timeframe}.csv")
        if key not in _candle_cache:
            if os.path.exists(fp):
                df = pd.read_csv(fp, parse_dates=["timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                _candle_cache[key] = df
            else:
                _candle_cache[key] = None
        return _candle_cache[key]

    skipped["timestamp"] = pd.to_datetime(skipped["timestamp"], utc=True)

    results   = {}
    win_count = loss_count = unknown_count = 0
    hyp_pnl   = 0.0

    slot_results = {}
    for _, row in skipped.iterrows():
        slot      = row.get("slot_key", row.get("symbol", "unknown"))
        symbol    = str(row.get("symbol", ""))
        timeframe = str(row.get("timeframe", ""))
        candles   = _get_candles(symbol, timeframe)

        outcome = _replay_skipped(row, candles) if candles is not None else None

        if outcome == "WIN":
            win_count += 1
            rr     = float(row.get("rr", 1.5))
            ep     = float(row.get("entry_price", 1))
            sl     = float(row.get("sl_price", ep))
            risk   = abs(ep - sl) / ep * 100
            trade_pnl = risk * rr
            hyp_pnl  += trade_pnl
        elif outcome == "LOSS":
            loss_count += 1
            ep    = float(row.get("entry_price", 1))
            sl    = float(row.get("sl_price", ep))
            risk  = abs(ep - sl) / ep * 100
            trade_pnl = -risk
            hyp_pnl  += trade_pnl
        else:
            unknown_count += 1
            trade_pnl = 0.0

        if slot not in slot_results:
            slot_results[slot] = {"wins": 0, "losses": 0, "unknown": 0, "hyp_pnl_pct": 0.0}
        slot_results[slot][outcome.lower() if outcome else "unknown"] += 1
        slot_results[slot]["hyp_pnl_pct"] += trade_pnl

    total_known = win_count + loss_count
    hyp_wr      = win_count / total_known * 100 if total_known else 0

    print(f"  Total skipped signals : {len(skipped)}")
    print(f"  Outcome determined    : {total_known}  ({unknown_count} inconclusive)")
    print(f"  Hypothetical WR       : {hyp_wr:.1f}%")
    print(f"  Hypothetical PnL      : {hyp_pnl:+.2f}% (% of entry price, not £)")

    print(f"\n  {'Slot':<28} {'Skipped':>7} {'Wins':>5} {'Losses':>6} {'WR%':>6} {'HypPnL%':>9}")
    print("  " + "─" * 65)
    for slot, r in sorted(slot_results.items()):
        n  = r["wins"] + r["losses"] + r["unknown"]
        kn = r["wins"] + r["losses"]
        wr = r["wins"] / kn * 100 if kn else 0
        print(f"  {slot:<28} {n:>7} {r['wins']:>5} {r['losses']:>6} "
              f"{wr:>5.1f}% {r['hyp_pnl_pct']:>+8.2f}%")

    results = {
        "total_skipped":   len(skipped),
        "wins":            win_count,
        "losses":          loss_count,
        "unknown":         unknown_count,
        "hyp_win_rate":    hyp_wr / 100,
        "hyp_pnl_pct":     hyp_pnl,
        "by_slot":         slot_results,
    }
    return results


# ── Section 3: Cap assessment ────────────────────────────────────────────────

def cap_assessment(live: dict, skipped: dict):
    print("\n" + "═" * 72)
    print("  SECTION 3 — PER-SYMBOL CAP ASSESSMENT")
    print("═" * 72)

    skipped_wins   = skipped.get("wins", 0)
    skipped_losses = skipped.get("losses", 0)
    skipped_known  = skipped_wins + skipped_losses
    hyp_wr         = skipped.get("hyp_win_rate", 0)
    hyp_pnl        = skipped.get("hyp_pnl_pct", 0)

    print(f"\n  Skipped trades with known outcome : {skipped_known}")
    print(f"  Skipped win rate                 : {hyp_wr*100:.1f}%")
    print(f"  Hypothetical additional PnL      : {hyp_pnl:+.2f}%")

    if skipped_known < 10:
        print("\n  ⚠️  Not enough skipped trades to assess cap reliably yet.")
        print("     Revisit once 20+ skipped outcomes are known.")
        return

    if hyp_wr >= 0.48 and hyp_pnl > 0:
        print("\n  📊 VERDICT: Skipped trades would have been profitable.")
        print("     Consider raising max_open_positions or relaxing the per-symbol cap.")
        print("     Suggested: monitor for 50+ skipped outcomes before changing.")
    elif hyp_wr < 0.42 or hyp_pnl < 0:
        print("\n  📊 VERDICT: Per-symbol cap is working correctly.")
        print("     Skipped trades underperformed — the cap is protecting capital.")
    else:
        print("\n  📊 VERDICT: Mixed evidence — maintain current cap for now.")


# ── Section 4: Strategy ranking ──────────────────────────────────────────────

# ── Section 5: Temporal breakdown from saved backtest_results.json ────────────

DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _temporal_breakdown(trades: list) -> dict:
    by_dow   = {d: {"trades": 0, "wins": 0, "pnl": 0.0} for d in DOW_ORDER}
    by_month: dict = {}

    for t in trades:
        try:
            ts = pd.Timestamp(t["entry_ts"])
        except Exception:
            continue
        win = t.get("outcome") == "WIN"
        pnl = float(t.get("pnl_pct", 0))

        dow = DOW_ORDER[ts.dayofweek]
        by_dow[dow]["trades"] += 1
        by_dow[dow]["wins"]   += int(win)
        by_dow[dow]["pnl"]    += pnl

        month = ts.strftime("%Y-%m")
        if month not in by_month:
            by_month[month] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_month[month]["trades"] += 1
        by_month[month]["wins"]   += int(win)
        by_month[month]["pnl"]    += pnl

    for b in list(by_dow.values()) + list(by_month.values()):
        n = b["trades"]
        b["win_rate"] = round(b["wins"] / n, 4) if n else None
        b["pnl"]      = round(b["pnl"], 4)

    return {"by_dow": by_dow, "by_month": dict(sorted(by_month.items()))}


def analyse_backtest_temporal(data_dir: str) -> None:
    """
    Section 5 — reads the already-saved backtest_results.json and prints
    day-of-week and monthly win-rate tables for every strategy.
    No rerun needed — uses the trade-level data already on disk.
    """
    path = os.path.join(data_dir, "backtest_results.json")
    if not os.path.exists(path):
        print(f"\n  [!] backtest_results.json not found at {path}")
        print("      Run backtest.py on Railway first to generate it.")
        return

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("\n  [!] backtest_results.json contains no results.")
        return

    print("\n" + "═" * 72)
    print("  SECTION 5 — BACKTEST TEMPORAL BREAKDOWN")
    print("  (day-of-week and monthly win rates from saved backtest data)")
    print("═" * 72)

    for r in sorted(results, key=lambda x: x.get("ev_pct", 0), reverse=True):
        trades = r.get("trades", [])
        if not trades:
            continue

        sym    = r.get("symbol", "?")
        tf     = r.get("timeframe", "?")
        ftf    = r.get("filter_tf", "")
        label  = f"{sym}  {tf}+{ftf}" if ftf else f"{sym}  {tf}"
        total  = r.get("total_trades", len(trades))
        ov_wr  = r.get("win_rate", 0) * 100

        # Use pre-computed temporal if present (future runs), otherwise compute now
        temporal = r.get("temporal") or _temporal_breakdown(trades)

        print(f"\n  ── {label}  (trades: {total}  overall WR: {ov_wr:.1f}%) ──")

        # Day-of-week
        by_dow = temporal.get("by_dow", {})
        print(f"\n  {'Day':<6}  {'Trades':>6}  {'Wins':>5}  {'WR%':>6}  {'PnL%':>8}  Bar")
        print("  " + "─" * 55)
        for dow in DOW_ORDER:
            b  = by_dow.get(dow, {})
            n  = b.get("trades", 0)
            if n == 0:
                print(f"  {dow:<6}  {'—':>6}")
                continue
            wr   = b.get("win_rate") or 0
            pnl  = b.get("pnl", 0.0)
            bar  = "█" * int(wr * 100 / 10) + ("▌" if int(wr * 100) % 10 >= 5 else "")
            flag = "✅" if wr >= 0.50 else ("⚠️ " if wr >= 0.44 else "❌")
            print(f"  {dow:<6}  {n:>6}  {b['wins']:>5}  {wr*100:>5.1f}%  {pnl:>+7.2f}%  {flag} {bar}")

        # Monthly
        by_month = temporal.get("by_month", {})
        print(f"\n  {'Month':<9}  {'Trades':>6}  {'Wins':>5}  {'WR%':>6}  {'PnL%':>8}")
        print("  " + "─" * 46)
        for month, b in by_month.items():
            n  = b.get("trades", 0)
            if n == 0:
                continue
            wr  = b.get("win_rate") or 0
            pnl = b.get("pnl", 0.0)
            flag = "✅" if wr >= 0.50 else ("⚠️ " if wr >= 0.44 else "❌")
            print(f"  {month:<9}  {n:>6}  {b['wins']:>5}  {wr*100:>5.1f}%  {pnl:>+7.2f}%  {flag}")

def strategy_ranking(live: dict):
    print("\n" + "═" * 72)
    print("  SECTION 4 — STRATEGY RANKING (by live EV%)")
    print("═" * 72)

    ranked = sorted(live.items(), key=lambda x: x[1]["ev_pct"], reverse=True)

    print(f"\n  {'Rank':<5} {'Slot':<28} {'EV%':>7} {'Trades':>7} {'Status'}")
    print("  " + "─" * 60)
    for i, (slot, r) in enumerate(ranked, 1):
        n      = r["trades"]
        ev     = r["ev_pct"]
        bt     = BACKTEST_BASELINE.get(slot, {})
        bt_ev  = bt.get("ev_pct")
        if n < 5:
            status = "⏳ insufficient data"
        elif bt_ev and ev >= bt_ev * 0.8:
            status = "✅ on track"
        elif bt_ev and ev >= 0:
            status = "⚠️  below backtest but profitable"
        else:
            status = "❌ underperforming"
        print(f"  {i:<5} {slot:<28} {ev:>+6.3f}% {n:>7}  {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bot performance assessment")
    parser.add_argument("--trades",   default="logs/trades.csv")
    parser.add_argument("--skipped",  default="logs/trades_skipped.csv")
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--out",      default=None, help="Save JSON report to file")
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  BOT PERFORMANCE ASSESSMENT".center(70) + "║")
    print("║" + f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    trades  = _load_csv(args.trades)
    skipped = _load_csv(args.skipped)

    live_results    = {}
    skipped_results = {}

    if trades is not None:
        live_results = analyse_live_trades(trades)
    else:
        print("\n  No live trades yet — run the bot first.")

    if skipped is not None:
        skipped_results = analyse_skipped_trades(skipped, args.data_dir)
    else:
        print("\n  No skipped trades yet.")

    if live_results:
        cap_assessment(live_results, skipped_results)
        strategy_ranking(live_results)

    analyse_backtest_temporal(args.data_dir)

    if args.out:
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "live_trades":     live_results,
            "skipped_trades":  skipped_results,
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved → {args.out}")

    print("\n" + "═" * 72 + "\n")


if __name__ == "__main__":
    main()
