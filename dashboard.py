"""
dashboard.py
Mobile-friendly web dashboard for the trading bot.

Endpoints:
  GET /            — HTML dashboard (auto-refreshes every 30 s)
  GET /api/state   — current equity, HWM, open positions (JSON)
  GET /api/trades  — last 100 completed trades (JSON)

Start with: uvicorn dashboard:app --host 0.0.0.0 --port $PORT
"""

import csv
import json
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

STATE_FILE  = os.environ.get("STATE_FILE",  "/data/state.json")
TRADES_FILE = os.environ.get("TRADES_FILE", "logs/trades.csv")


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state():
    if not os.path.exists(STATE_FILE):
        return JSONResponse(
            {"error": "State file not found — bot may still be initialising."},
            status_code=503,
        )
    with open(STATE_FILE) as f:
        return json.load(f)


@app.get("/api/trades")
def get_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    trades = []
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return list(reversed(trades[-100:]))  # newest first


# ── HTML dashboard ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Bot</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0f1117;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      padding-bottom: 32px;
    }

    header {
      background: #1a1d2e;
      padding: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #2d3148;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    header h1 { font-size: 17px; font-weight: 700; letter-spacing: 0.3px; }

    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 99px;
      letter-spacing: 0.5px;
    }
    .badge-paper { background: #2d3748; color: #90cdf4; }
    .badge-live  { background: #22543d; color: #68d391; }

    .updated {
      font-size: 11px;
      color: #718096;
      margin-top: 2px;
    }

    .section { padding: 16px; }
    .section-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      color: #718096;
      margin-bottom: 10px;
    }

    /* Stat cards row */
    .stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }

    .stat-card {
      background: #1a1d2e;
      border: 1px solid #2d3148;
      border-radius: 10px;
      padding: 14px;
    }

    .stat-label {
      font-size: 11px;
      color: #718096;
      margin-bottom: 4px;
    }

    .stat-value {
      font-size: 22px;
      font-weight: 700;
    }

    .stat-sub {
      font-size: 11px;
      color: #718096;
      margin-top: 2px;
    }

    .pos  { color: #68d391; }
    .neg  { color: #fc8181; }
    .neu  { color: #e2e8f0; }

    /* Position cards */
    .pos-card {
      background: #1a1d2e;
      border: 1px solid #2d3148;
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 10px;
    }

    .pos-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }

    .pos-title { font-size: 15px; font-weight: 700; }

    .dir-long  { color: #68d391; font-size: 12px; font-weight: 700; }
    .dir-short { color: #fc8181; font-size: 12px; font-weight: 700; }

    .pos-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }

    .pos-field-label { font-size: 10px; color: #718096; }
    .pos-field-value { font-size: 13px; font-weight: 600; margin-top: 1px; }

    .tp1-pill {
      font-size: 10px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 99px;
    }
    .tp1-hit  { background: #22543d; color: #68d391; }
    .tp1-open { background: #2d3748; color: #90cdf4; }

    /* Trades table */
    .trade-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    .trade-table th {
      text-align: left;
      padding: 6px 8px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      color: #718096;
      border-bottom: 1px solid #2d3148;
    }

    .trade-table td {
      padding: 8px 8px;
      border-bottom: 1px solid #1e2235;
    }

    .empty {
      text-align: center;
      color: #718096;
      padding: 32px 0;
      font-size: 13px;
    }

    .spinner {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 2px solid #4a5568;
      border-top-color: #90cdf4;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 6px;
      vertical-align: middle;
    }

    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <div>
    <h1>&#x1F916; Trading Bot</h1>
    <div class="updated" id="updated">Loading…</div>
  </div>
  <div id="mode-badge" class="badge badge-paper">PAPER</div>
</header>

<!-- Stat cards -->
<div class="section">
  <div class="section-title">Account</div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Equity</div>
      <div class="stat-value neu" id="equity">—</div>
      <div class="stat-sub" id="equity-sub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total PnL</div>
      <div class="stat-value" id="pnl">—</div>
      <div class="stat-sub" id="pnl-sub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">High-Water Mark</div>
      <div class="stat-value neu" id="hwm">—</div>
      <div class="stat-sub">HWM ratchet</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Open Positions</div>
      <div class="stat-value neu" id="pos-count">—</div>
      <div class="stat-sub" id="pos-sub"></div>
    </div>
  </div>
</div>

<!-- Open positions -->
<div class="section">
  <div class="section-title">Open Positions</div>
  <div id="positions-container">
    <div class="empty"><span class="spinner"></span>Loading…</div>
  </div>
</div>

<!-- Trade history -->
<div class="section">
  <div class="section-title">Recent Trades</div>
  <div style="overflow-x:auto;">
    <table class="trade-table">
      <thead>
        <tr>
          <th>Pair</th>
          <th>Side</th>
          <th>PnL%</th>
          <th>PnL £</th>
          <th>Exit</th>
          <th>Date</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="6" class="empty"><span class="spinner"></span>Loading…</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
  function fmt(val, decimals=2) {
    return val !== null && val !== undefined ? parseFloat(val).toFixed(decimals) : "—";
  }

  function pnlClass(val) {
    const n = parseFloat(val);
    if (isNaN(n)) return "neu";
    return n > 0 ? "pos" : n < 0 ? "neg" : "neu";
  }

  function pnlSign(val) {
    const n = parseFloat(val);
    return n > 0 ? "+" : "";
  }

  function shortPair(pair) {
    // BTCUSDT_UMCBL → BTC
    return pair.replace("USDT_UMCBL","").replace("USDT","");
  }

  function tfFromSlot(slot) {
    // BTCUSDT_UMCBL_15m → 15m
    const parts = slot.split("_");
    return parts[parts.length - 1];
  }

  function formatDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString("en-GB", {day:"2-digit", month:"short"})
           + " " + d.toLocaleTimeString("en-GB", {hour:"2-digit", minute:"2-digit"});
  }

  async function loadState() {
    try {
      const r = await fetch("/api/state");
      if (!r.ok) {
        document.getElementById("updated").textContent = "Bot initialising…";
        return;
      }
      const s = await r.json();

      // Mode badge
      const badge = document.getElementById("mode-badge");
      if (s.paper) {
        badge.textContent = "PAPER";
        badge.className = "badge badge-paper";
      } else {
        badge.textContent = "LIVE";
        badge.className = "badge badge-live";
      }

      // Updated time
      document.getElementById("updated").textContent =
        "Updated " + formatDate(s.updated_at);

      // Equity
      document.getElementById("equity").textContent = "£" + fmt(s.equity);
      document.getElementById("equity-sub").textContent =
        "Start £" + fmt(s.initial_capital);

      // PnL
      const pnlPct = ((s.equity - s.initial_capital) / s.initial_capital * 100);
      const pnlUsdt = s.equity - s.initial_capital;
      const pEl = document.getElementById("pnl");
      pEl.textContent = pnlSign(pnlPct) + fmt(pnlPct) + "%";
      pEl.className = "stat-value " + pnlClass(pnlPct);
      document.getElementById("pnl-sub").textContent =
        pnlSign(pnlUsdt) + "£" + fmt(Math.abs(pnlUsdt));

      // HWM
      document.getElementById("hwm").textContent = "£" + fmt(s.hwm);

      // Positions count
      const pos = s.positions || {};
      const posKeys = Object.keys(pos);
      document.getElementById("pos-count").textContent = posKeys.length;
      document.getElementById("pos-sub").textContent =
        "max " + (s.max_open || 5);

      // Render position cards
      const container = document.getElementById("positions-container");
      if (posKeys.length === 0) {
        container.innerHTML = "<div class=\\"empty\\">No open positions</div>";
      } else {
        container.innerHTML = posKeys.map(slot => {
          const p = pos[slot];
          const sym = shortPair(p.symbol || slot);
          const tf  = p.tf || tfFromSlot(slot);
          const dir = p.side === "long" ? "LONG" : "SHORT";
          const dirClass = p.side === "long" ? "dir-long" : "dir-short";
          const tp1Label = p.tp1_hit
            ? "<span class=\\"tp1-pill tp1-hit\\">TP1 ✓</span>"
            : "<span class=\\"tp1-pill tp1-open\\">TP1 open</span>";

          return \`
          <div class="pos-card">
            <div class="pos-header">
              <div>
                <span class="pos-title">\${sym} <span style="font-size:12px;color:#718096">\${tf}</span></span>
              </div>
              <div style="display:flex;gap:6px;align-items:center">
                \${tp1Label}
                <span class="\${dirClass}">\${dir}</span>
              </div>
            </div>
            <div class="pos-grid">
              <div>
                <div class="pos-field-label">Entry</div>
                <div class="pos-field-value">\${fmt(p.entry_price, 4)}</div>
              </div>
              <div>
                <div class="pos-field-label">Stop Loss</div>
                <div class="pos-field-value neg">\${fmt(p.sl, 4)}</div>
              </div>
              <div>
                <div class="pos-field-label">Take Profit</div>
                <div class="pos-field-value pos">\${fmt(p.tp, 4)}</div>
              </div>
              <div>
                <div class="pos-field-label">Leverage</div>
                <div class="pos-field-value">\${p.leverage || "—"}×</div>
              </div>
              <div>
                <div class="pos-field-label">Qty</div>
                <div class="pos-field-value">\${fmt(p.qty, 4)}</div>
              </div>
              <div>
                <div class="pos-field-label">Opened</div>
                <div class="pos-field-value" style="font-size:11px">\${formatDate(p.entry_time)}</div>
              </div>
            </div>
          </div>\`;
        }).join("");
      }

    } catch(e) {
      document.getElementById("updated").textContent = "Connection error";
    }
  }

  async function loadTrades() {
    try {
      const r = await fetch("/api/trades");
      const trades = await r.json();
      const tbody = document.getElementById("trades-body");

      if (!trades || trades.length === 0) {
        tbody.innerHTML = "<tr><td colspan=\\"6\\" class=\\"empty\\">No completed trades yet</td></tr>";
        return;
      }

      tbody.innerHTML = trades.slice(0, 30).map(t => {
        const pnlPct = parseFloat(t.pnl_pct || 0);
        const cls = pnlClass(pnlPct);
        const sign = pnlSign(pnlPct);
        const pair = shortPair(t.pair || "");
        const dateStr = formatDate(t.timestamp);
        const exitShort = (t.exit_reason || "").replace("_", " ").slice(0, 8);

        return \`<tr>
          <td><strong>\${pair}</strong></td>
          <td style="color:\${t.side==='long'?'#68d391':'#fc8181'}">\${t.side||"—"}</td>
          <td class="\${cls}">\${sign}\${fmt(pnlPct)}%</td>
          <td class="\${cls}">\${sign}£\${fmt(Math.abs(parseFloat(t.pnl_usdt||0)))}</td>
          <td style="color:#718096;font-size:11px">\${exitShort}</td>
          <td style="color:#718096;font-size:11px">\${dateStr}</td>
        </tr>\`;
      }).join("");

    } catch(e) {
      document.getElementById("trades-body").innerHTML =
        "<tr><td colspan=\\"6\\" class=\\"empty\\">Error loading trades</td></tr>";
    }
  }

  async function refresh() {
    await Promise.all([loadState(), loadTrades()]);
  }

  refresh();
  setInterval(refresh, 30000);  // refresh every 30 seconds
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML
