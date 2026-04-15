"""
fix_false_stops.py
Reverses the 4 false stop-outs on SOL_5m+4h caused by the pre-entry
wick bug on 15 Apr between 22:10 and 22:27 UTC.

Run once on Railway:  railway run python fix_false_stops.py
"""

import csv
import json
import os
from datetime import datetime, timezone

TRADES_FILE = "/data/trades.csv"
STATE_FILE  = "/data/risk_state.json"

# ── Identify the false trades ─────────────────────────────────────────────────
def is_false_trade(row: dict) -> bool:
    ts = row.get("timestamp", "")
    return (
        row.get("pair", "")        == "SOLUSDT_UMCBL" and
        row.get("exit_reason", "") == "stop_loss" and
        "2026-04-15" in ts and
        any(f"22:{m}" in ts for m in ["10", "16", "21", "27"])
    )

# ── Read trades, split out the false ones ─────────────────────────────────────
with open(TRADES_FILE, newline="") as f:
    reader    = csv.DictReader(f)
    fieldnames = reader.fieldnames
    all_trades = list(reader)

false_trades  = [r for r in all_trades if is_false_trade(r)]
clean_trades  = [r for r in all_trades if not is_false_trade(r)]

if not false_trades:
    print("No matching false trades found — nothing to do.")
    exit(0)

total_loss = sum(float(r["pnl_usdt"]) for r in false_trades)
print(f"Found {len(false_trades)} false trade(s) totalling £{total_loss:.4f}")
for r in false_trades:
    print(f"  {r['timestamp']}  pnl=£{r['pnl_usdt']}  equity_after=£{r['equity_after']}")

# ── Rewrite trades CSV without the false trades ───────────────────────────────
with open(TRADES_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean_trades)
print(f"Trades CSV updated — {len(false_trades)} row(s) removed.")

# ── Restore equity in risk_state.json ─────────────────────────────────────────
with open(STATE_FILE) as f:
    state = json.load(f)

old_equity = state["equity"]
state["equity"] += abs(total_loss)
# If the restored equity exceeds HWM, update HWM too
if state["equity"] > state["hwm"]:
    state["hwm"] = state["equity"]
# Restore day_start_equity too so daily loss calc is clean
state["day_start_equity"] = max(state["day_start_equity"], state["equity"])

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

print(f"Risk state updated — equity £{old_equity:.2f} → £{state['equity']:.2f}  HWM=£{state['hwm']:.2f}")
print("Done. Restart the bot to pick up the restored state.")
