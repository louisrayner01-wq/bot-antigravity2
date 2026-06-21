# Strategy Swap — Rule-Based Bot (Conservative / Balanced / Aggressive)

This document explains what was added, how to switch the running bot from the
old ML strategy to the new rule-based one, and how to roll back if anything
goes wrong.

## TL;DR

A second bot lives alongside the existing one. Nothing in the original bot
was touched. To switch:

```
# Procfile — change this line:
worker: python bot.py
# to:
worker: python bot_rules.py
```

Push to `main`, Railway redeploys, paper bot is now running Conservative 1
at 1% risk. Roll back by reverting the Procfile.

## What was added

| File | Purpose |
|---|---|
| `strategy_rules.py` | Pure signal logic — 4h EMA9 pullback + BTC daily SMA20 bias |
| `risk_manager_rules.py` | Simple equity-% risk sizing, per-asset R:R, concurrency cap, circuit breakers |
| `bot_rules.py` | Standalone bot main loop using the above |
| `config.yaml` | New `strategy_rules:` section appended (additive only) |

## What was NOT touched

- `bot.py`            — original ML bot, unchanged
- `strategy.py`       — ML strategy module, unchanged
- `risk_manager.py`   — old risk manager, unchanged
- `weex_client.py`, `indicators.py`, `trade_logger.py`, `fortuna_client.py`,
  `telegram_notifier.py` — all reused as-is by the new bot

## The three strategies

| Name | Max positions | Asset priority | Backtest (Oct 23 → Apr 26) | MaxDD |
|---|---|---|---|---|
| **Conservative** | 1 | ETH → SOL → BTC | +1,741% (CAGR 210%) | 18.6% |
| **Balanced** | 2 | SOL → ETH → BTC | +11,557% (CAGR 534%) | 32.1% |
| **Aggressive** | 3 | take-all | +33,110% (CAGR 852%) | 44.7% |

All three use the same entry rule (4h EMA9 pullback aligned with BTC daily
SMA20 trend) and the same per-asset R:R:

| Asset | TP × ATR14 | SL × ATR14 | R:R |
|---|---|---|---|
| BTC | 3.0 | 0.8 | 3.75 |
| ETH | 2.0 | 0.8 | 2.50 |
| SOL | 2.0 | 0.8 | 2.50 |

## Configuration

`config.yaml` → `strategy_rules:` section

```yaml
strategy_rules:
  strategy_mode:    "conservative"   # conservative | balanced | aggressive
  risk_per_trade:   0.01             # 0.005–0.02 (0.5%–2%); clamped if outside
  max_leverage:     10
  daily_loss_pct:   0.05             # circuit: pause if down 5% in a day
  hwm_drawdown_pct: 0.25             # circuit: pause if down 25% from HWM
  poll_seconds:     300
  assets:
    - {base: "BTC", symbol: "BTCUSDT_UMCBL"}
    - {base: "ETH", symbol: "ETHUSDT_UMCBL"}
    - {base: "SOL", symbol: "SOLUSDT_UMCBL"}
```

### Environment variable overrides (highest priority)

| Variable | Values | Purpose |
|---|---|---|
| `STRATEGY_MODE` | conservative / balanced / aggressive | overrides `strategy_mode` |
| `RISK_PER_TRADE` | 0.005 – 0.02 | overrides `risk_per_trade` |
| `PAPER_TRADING` | true / false | paper vs live (defaults to value in config) |
| `FORTUNA_API_URL` | https://… | when set, switches to multi-user mode |

## Multi-user mode (for the Fortuna dashboard)

When `FORTUNA_API_URL` is set, `bot_rules.py` polls Fortuna for active users
each tick, fetching each user's `strategy_mode` and `risk_per_trade` via the
existing `/api/bot/internal/user-config/{user_id}` endpoint and running a
separate per-user `RuleBot` instance.

**Fortuna API needs to return these new fields per user** (alongside the
existing `api_key` / `api_secret` / `capital`):

```json
{
  "user_id": "abc123",
  "capital": 100,
  "api_key": "…",
  "api_secret": "…",
  "passphrase": "…",
  "strategy_mode":  "conservative",
  "risk_per_trade": 0.01
}
```

The dashboard repo (not in this codebase) needs:
1. Strategy picker on the user settings page:
   `Conservative` / `Balanced` / `Aggressive` (drop the "1" suffix in the UI)
2. Risk slider: 0.5% – 2.0%
3. Persist both to the user record
4. Surface them in the `/api/bot/internal/user-config/{user_id}` response

## Deploy checklist

1. **Verify nothing is live-trading real money.** This deploys to a paper
   bot — but double-check `paper_trading: true` in `config.yaml` AND
   `PAPER_TRADING=true` (or unset) in Railway env vars.

2. **Test locally first (optional but recommended):**
   ```bash
   cd /Users/Louis/bot-antigravity2
   PAPER_TRADING=true STRATEGY_MODE=conservative RISK_PER_TRADE=0.01 \
     python3 bot_rules.py
   ```
   Watch the logs for one tick (~5 min). It should print the BTC bias and any
   signal evaluations.

3. **Commit and push the new files:**
   ```bash
   git add strategy_rules.py risk_manager_rules.py bot_rules.py \
           config.yaml STRATEGY_SWAP.md
   git commit -m "Add rule-based Conservative/Balanced/Aggressive strategies"
   git push
   ```
   This alone does NOT change running behaviour — the Procfile still points
   to `bot.py`.

4. **Activate the new bot:**
   Edit `Procfile`:
   ```
   worker: python bot_rules.py
   ```
   Commit and push. Railway redeploys; paper bot now runs Conservative 1.

5. **Watch the first hour of logs** for sane output:
   - "BTC daily bias = BULL/BEAR" line every tick
   - "Signal X LONG @ Y" lines on relevant 4h closes
   - "OPEN long X" when an entry actually fires
   - "CLOSE" lines with PnL on SL/TP hits

## Rollback

If anything looks wrong:

```bash
# Revert the Procfile change
git checkout HEAD~1 -- Procfile
git commit -m "Rollback to bot.py"
git push
```

Or just edit `Procfile` back to `worker: python bot.py` and push.

The new bot has its own state file (`risk_state_rules.json`) so the old
bot's `risk_state.json` is preserved untouched. Rolling back loses NO data
from the original system.

## State files

| File | Used by | Persists |
|---|---|---|
| `/data/risk_state.json` | bot.py (ML) | equity, HWM, open positions |
| `/data/risk_state_rules.json` | bot_rules.py (new) | equity, HWM, open positions |
| `/data/trades.csv` | bot.py | trade history |
| `/data/trades_rules.csv` | bot_rules.py | trade history |

The two sets are fully independent.

## Operational notes

- **Poll interval is 5 minutes** but entries are only evaluated when a NEW
  4h candle closes. Exits are checked every 5 min so SL/TP don't drift.
- **Circuit breakers** never force-close positions; they only pause new
  entries until the condition clears (intraday for daily, equity-bounce for
  HWM DD).
- **Risk clamping**: any `risk_per_trade` outside 0.005–0.02 is clamped to
  the nearest bound — protects against accidental fat fingers from the
  dashboard.
- **One position per asset** is enforced regardless of strategy mode (so
  Conservative can hold BTC OR ETH OR SOL, never two of the same).
- **The fortuna-web dashboard** is in a separate repository and is NOT
  modified by this work. See the "Multi-user mode" section for what the
  dashboard needs.

## Scale-up plan

Start at 1% risk for ~4 weeks, then:

| Week | risk_per_trade |
|---|---|
| 1–4   | 0.01 (1%) |
| 5–8   | 0.015 (1.5%) |
| 9+    | 0.02 (2%) — the backtested setting |

Adjust via the dashboard slider or by editing `risk_per_trade` in
`config.yaml` and redeploying.
