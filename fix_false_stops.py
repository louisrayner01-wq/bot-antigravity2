"""
fix_false_stops.py
Reverses the 4 false SOL stop-outs caused by the pre-entry wick bug.

Run in two steps:
  1. Preview:  railway run python fix_false_stops.py
  2. Confirm:  railway run python fix_false_stops.py --apply

Step 1 prints the trades it would remove so you can verify before committing.
"""

import csv
import json
import sys

TRADES_FILE = "/data/trades.csv"
STATE_FILE  = "/data/risk_state.json"
APPLY       = "--apply" in sys.argv
N_TO_REMOVE = 4   # the 4 false stop-outs

# ── Read all trades ───────────────────────────────────────────────────────────
with open(TRADES_FILE, newline="") as f:
    reader     = csv.DictReader(f)
    fieldnames = reader.fieldnames
    all_trades = list(reader)

# ── Find the last N SOL stop losses on 15 Apr ─────────────────────────────────
sol_stops = [
    (i, r) for i, r in enumerate(all_trades)
    if r.get("pair") == "SOLUSDT_UMCBL"
    and r.get("exit_reason") == "stop_loss"
    and "2026-04-15" in r.get("timestamp", "")
]

print(f"Found {len(sol_stops)} SOL stop loss(es) on 15 Apr:")
for i, (idx, r) in enumerate(sol_stops):
    print(f"  [{i}] row={idx}  {r['timestamp']}  pnl=£{r['pnl_usdt']}  equity_after=£{r['equity_after']}")

if len(sol_stops) < N_TO_REMOVE:
    print(f"\nExpected {N_TO_REMOVE} trades, only found {len(sol_stops)} — aborting.")
    sys.exit(1)

# Take the last N_TO_REMOVE (most recent)
to_remove   = sol_stops[-N_TO_REMOVE:]
remove_idxs = {idx for idx, _ in to_remove}
total_loss  = sum(float(r["pnl_usdt"]) for _, r in to_remove)

print(f"\nWill remove rows: {[idx for idx, _ in to_remove]}")
print(f"Total loss to restore: £{abs(total_loss):.2f}")

if not APPLY:
    print("\nDry run — pass --apply to actually apply the fix.")
    sys.exit(0)

# ── Rewrite CSV ───────────────────────────────────────────────────────────────
clean_trades = [r for i, r in enumerate(all_trades) if i not in remove_idxs]
with open(TRADES_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean_trades)
print(f"Trades CSV updated — {N_TO_REMOVE} row(s) removed.")

# ── Restore equity in risk_state.json ─────────────────────────────────────────
with open(STATE_FILE) as f:
    state = json.load(f)

old_equity = state["equity"]
state["equity"] += abs(total_loss)
if state["equity"] > state["hwm"]:
    state["hwm"] = state["equity"]
state["day_start_equity"] = max(state["day_start_equity"], state["equity"])

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

print(f"Risk state: equity £{old_equity:.2f} → £{state['equity']:.2f}  HWM=£{state['hwm']:.2f}")
print("Done. Redeploy the bot to pick up the restored state.")
