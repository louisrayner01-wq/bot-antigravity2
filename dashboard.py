"""
dashboard.py
Mobile-friendly web dashboard for the trading bot.

Endpoints:
  GET /            — HTML dashboard (auto-refreshes every 30 s)
  GET /static/app.js — dashboard JavaScript (served separately to avoid escaping issues)
  GET /api/state   — current equity, HWM, open positions (JSON)
  GET /api/trades  — last 100 completed trades (JSON)

Start with: uvicorn dashboard:app --host 0.0.0.0 --port $PORT
"""

import csv
import io
import json
import os
import zipfile

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

app = FastAPI()

STATE_FILE  = os.environ.get("STATE_FILE",  "/data/state.json")
TRADES_FILE = os.environ.get("TRADES_FILE", "logs/trades.csv")


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/download-models")
def download_models():
    """Temporary endpoint — download all model files from /data/models/ as a zip."""
    models_dir = "/data/models"
    if not os.path.exists(models_dir):
        return JSONResponse({"error": "No models directory found"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(models_dir):
            fpath = os.path.join(models_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=models.zip"},
    )

@app.get("/api/calendar")
def get_calendar():
    """Return daily PnL summary grouped by date for the calendar view."""
    if not os.path.exists(TRADES_FILE):
        return {}
    days = {}
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")[:10]  # YYYY-MM-DD
            if not ts:
                continue
            pnl = float(row.get("pnl_usdt", 0) or 0)
            if ts not in days:
                days[ts] = {"pnl": 0.0, "trades": 0}
            days[ts]["pnl"]    += pnl
            days[ts]["trades"] += 1
    return days


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


# ── JavaScript (served separately so template literals work without escaping) ──

JS = r"""
var _rawState  = null;
var _rawTrades = [];
var _scaleFactor = 1.0;

function getScaleFactor() {
  var sel = document.getElementById("acct-select");
  if (!sel || !_rawState) return 1.0;
  var selectedCap = parseFloat(sel.value);
  var actualCap   = parseFloat(_rawState.initial_capital) || 100;
  return selectedCap / actualCap;
}

function fmt(val, decimals) {
  var d = (decimals === undefined) ? 2 : decimals;
  return (val !== null && val !== undefined && val !== "") ? parseFloat(val).toFixed(d) : "\u2014";
}

function pnlClass(val) {
  var n = parseFloat(val);
  if (isNaN(n)) return "neu";
  return n > 0 ? "pos" : n < 0 ? "neg" : "neu";
}

function pnlSign(val) {
  var n = parseFloat(val);
  return n > 0 ? "+" : "";
}

function shortPair(pair) {
  return (pair || "").replace("USDT_UMCBL","").replace("USDT","");
}

function tfFromSlot(slot) {
  var parts = (slot || "").split("_");
  return parts[parts.length - 1];
}

function formatDate(iso) {
  if (!iso) return "\u2014";
  var d = new Date(iso);
  return d.toLocaleDateString("en-GB", {day:"2-digit", month:"short"})
       + " " + d.toLocaleTimeString("en-GB", {hour:"2-digit", minute:"2-digit"});
}

function el(id) { return document.getElementById(id); }

function posCard(slot, p, sf) {
  var sym      = shortPair(p.symbol || slot);
  var tf       = p.tf || tfFromSlot(slot);
  var dir      = p.side === "long" ? "LONG" : "SHORT";
  var dirClass = p.side === "long" ? "dir-long" : "dir-short";
  var scaledQty = (parseFloat(p.qty || 0) * sf);

  return '<div class="pos-card">'
    + '<div class="pos-header">'
    +   '<div><span class="pos-title">' + sym + ' <span style="font-size:12px;color:#718096">' + tf + '</span></span></div>'
    +   '<div><span class="' + dirClass + '">' + dir + '</span></div>'
    + '</div>'
    + '<div class="pos-grid">'
    +   '<div><div class="pos-field-label">Entry</div><div class="pos-field-value">' + fmt(p.entry_price, 4) + '</div></div>'
    +   '<div><div class="pos-field-label">Stop Loss</div><div class="pos-field-value neg">' + fmt(p.sl, 4) + '</div></div>'
    +   '<div><div class="pos-field-label">Take Profit</div><div class="pos-field-value pos">' + fmt(p.tp, 4) + '</div></div>'
    +   '<div><div class="pos-field-label">Leverage</div><div class="pos-field-value">' + (p.leverage || "\u2014") + '\xd7</div></div>'
    +   '<div><div class="pos-field-label">Qty</div><div class="pos-field-value">' + scaledQty.toFixed(4) + '</div></div>'
    +   '<div><div class="pos-field-label">Opened</div><div class="pos-field-value" style="font-size:11px">' + formatDate(p.entry_time) + '</div></div>'
    + '</div>'
    + '</div>';
}

function tradeRow(t, sf) {
  var pnlPct  = parseFloat(t.pnl_pct || 0);
  var pnlAbs  = parseFloat(t.pnl_usdt || 0) * sf;
  var cls     = pnlClass(pnlPct);
  var sign    = pnlSign(pnlPct);
  var pair    = shortPair(t.pair || "");
  var dateStr = formatDate(t.timestamp);
  var exitTxt = (t.exit_reason || "").replace("_", " ").slice(0, 8);
  var sideCol = t.side === "long" ? "#68d391" : "#fc8181";

  return '<tr>'
    + '<td><strong>' + pair + '</strong></td>'
    + '<td style="color:' + sideCol + '">' + (t.side || "\u2014") + '</td>'
    + '<td class="' + cls + '">' + sign + fmt(pnlPct) + '%</td>'
    + '<td class="' + cls + '">' + sign + '\xa3' + fmt(Math.abs(pnlAbs)) + '</td>'
    + '<td style="color:#718096;font-size:11px">' + exitTxt + '</td>'
    + '<td style="color:#718096;font-size:11px">' + dateStr + '</td>'
    + '</tr>';
}

function renderState() {
  var s  = _rawState;
  var sf = _scaleFactor;
  if (!s) return;

  var badge = el("mode-badge");
  if (s.paper) {
    badge.textContent = "PAPER";
    badge.className = "badge badge-paper";
  } else {
    badge.textContent = "LIVE";
    badge.className = "badge badge-live";
  }

  el("updated").textContent = "Updated " + formatDate(s.updated_at);

  var selectedCap = parseFloat(el("acct-select").value);
  var scaledEquity = parseFloat(s.equity) * sf;
  var scaledStart  = selectedCap;
  var pnlUsdt = (parseFloat(s.equity) - parseFloat(s.initial_capital)) * sf;
  var pnlPct  = (parseFloat(s.equity) - parseFloat(s.initial_capital)) / parseFloat(s.initial_capital) * 100;
  var scaledHWM = parseFloat(s.hwm) * sf;

  el("equity").textContent = "\xa3" + fmt(scaledEquity);
  el("equity-sub").textContent = "Start \xa3" + fmt(scaledStart);

  var pEl = el("pnl");
  pEl.textContent = pnlSign(pnlPct) + fmt(pnlPct) + "%";
  pEl.className = "stat-value " + pnlClass(pnlPct);
  el("pnl-sub").textContent = pnlSign(pnlUsdt) + "\xa3" + fmt(Math.abs(pnlUsdt));

  el("hwm").textContent = "\xa3" + fmt(scaledHWM);

  var pos     = s.positions || {};
  var posKeys = Object.keys(pos);
  el("pos-count").textContent = posKeys.length;
  el("pos-sub").textContent = "max " + (s.max_open || 5);

  var container = el("positions-container");
  if (posKeys.length === 0) {
    container.innerHTML = '<div class="empty">No open positions</div>';
  } else {
    container.innerHTML = posKeys.map(function(slot) {
      return posCard(slot, pos[slot], sf);
    }).join("");
  }
}

function renderTrades() {
  var sf    = _scaleFactor;
  var tbody = el("trades-body");
  if (!_rawTrades || _rawTrades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No completed trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = _rawTrades.slice(0, 30).map(function(t) {
    return tradeRow(t, sf);
  }).join("");
}

function onAccountChange() {
  _scaleFactor = getScaleFactor();
  renderState();
  renderTrades();
}

async function loadState() {
  try {
    var r = await fetch("/api/state");
    if (!r.ok) {
      el("updated").textContent = "Bot initialising\u2026";
      return;
    }
    _rawState = await r.json();
    _scaleFactor = getScaleFactor();
    renderState();
  } catch(e) {
    el("updated").textContent = "Connection error";
  }
}

async function loadTrades() {
  try {
    var r      = await fetch("/api/trades");
    _rawTrades = await r.json();
    _scaleFactor = getScaleFactor();
    renderTrades();
  } catch(e) {
    el("trades-body").innerHTML = '<tr><td colspan="6" class="empty">Error loading trades</td></tr>';
  }
}

async function refresh() {
  await Promise.all([loadState(), loadTrades()]);
}

refresh();
setInterval(refresh, 30000);
"""


@app.get("/static/app.js")
def serve_js():
    return Response(content=JS, media_type="application/javascript")


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Bot</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; padding-bottom: 32px;
    }
    header {
      background: #1a1d2e; padding: 16px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148; position: sticky; top: 0; z-index: 10;
    }
    header h1 { font-size: 17px; font-weight: 700; letter-spacing: 0.3px; }
    .header-right { display: flex; align-items: center; gap: 10px; }
    .badge { font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 99px; letter-spacing: 0.5px; }
    .badge-paper { background: #2d3748; color: #90cdf4; }
    .badge-live  { background: #22543d; color: #68d391; }
    .acct-select {
      appearance: none; -webkit-appearance: none;
      background: #2d3148; color: #e2e8f0;
      border: 1px solid #3d4268; border-radius: 8px;
      padding: 5px 28px 5px 10px; font-size: 13px; font-weight: 600;
      cursor: pointer; outline: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2390cdf4' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 8px center;
    }
    .acct-select:focus { border-color: #90cdf4; }
    .acct-label { font-size: 11px; color: #718096; white-space: nowrap; }
    .updated { font-size: 11px; color: #718096; margin-top: 2px; }
    .section { padding: 16px; }
    .section-title { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #718096; margin-bottom: 10px; }
    .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .stat-card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 10px; padding: 14px; }
    .stat-label { font-size: 11px; color: #718096; margin-bottom: 4px; }
    .stat-value { font-size: 22px; font-weight: 700; }
    .stat-sub { font-size: 11px; color: #718096; margin-top: 2px; }
    .pos  { color: #68d391; }
    .neg  { color: #fc8181; }
    .neu  { color: #e2e8f0; }
    .acct-banner {
      margin: 0 16px 4px; background: #1e2235; border: 1px solid #3d4268;
      border-radius: 8px; padding: 8px 14px;
      font-size: 12px; color: #90cdf4; display: flex; align-items: center; gap: 6px;
    }
    .pos-card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
    .pos-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .pos-title { font-size: 15px; font-weight: 700; }
    .dir-long  { color: #68d391; font-size: 12px; font-weight: 700; }
    .dir-short { color: #fc8181; font-size: 12px; font-weight: 700; }
    .pos-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .pos-field-label { font-size: 10px; color: #718096; }
    .pos-field-value { font-size: 13px; font-weight: 600; margin-top: 1px; }
    .trade-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .trade-table th { text-align: left; padding: 6px 8px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: #718096; border-bottom: 1px solid #2d3148; }
    .trade-table td { padding: 8px 8px; border-bottom: 1px solid #1e2235; }
    .empty { text-align: center; color: #718096; padding: 32px 0; font-size: 13px; }
    .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #4a5568; border-top-color: #90cdf4; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 6px; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <div>
    <h1>&#x1F916; Trading Bot</h1>
    <div class="updated" id="updated">Loading&#x2026;</div>
  </div>
  <div class="header-right">
    <div>
      <div class="acct-label">Account size</div>
      <select class="acct-select" id="acct-select" onchange="onAccountChange()">
        <option value="100">&#xa3;100</option>
        <option value="500">&#xa3;500</option>
        <option value="1000">&#xa3;1,000</option>
        <option value="5000">&#xa3;5,000</option>
      </select>
    </div>
    <a href="/calendar" target="_blank" style="text-decoration:none;">
      <div class="badge" style="background:#2d3148;color:#90cdf4;cursor:pointer;">&#x1F4C5; Calendar</div>
    </a>
    <div id="mode-badge" class="badge badge-paper">PAPER</div>
  </div>
</header>

<div class="acct-banner" id="acct-banner">
  &#x1F4CA; Showing projected figures for a <strong id="acct-banner-size">&#xa3;100</strong> account &mdash; same trades, scaled P&amp;L.
</div>

<div class="section">
  <div class="section-title">Account</div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Equity</div>
      <div class="stat-value neu" id="equity">&#x2014;</div>
      <div class="stat-sub" id="equity-sub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total PnL</div>
      <div class="stat-value" id="pnl">&#x2014;</div>
      <div class="stat-sub" id="pnl-sub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">High-Water Mark</div>
      <div class="stat-value neu" id="hwm">&#x2014;</div>
      <div class="stat-sub">HWM ratchet</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Open Positions</div>
      <div class="stat-value neu" id="pos-count">&#x2014;</div>
      <div class="stat-sub" id="pos-sub"></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">Open Positions</div>
  <div id="positions-container">
    <div class="empty"><span class="spinner"></span>Loading&#x2026;</div>
  </div>
</div>

<div class="section">
  <div class="section-title">Recent Trades</div>
  <div style="overflow-x:auto;">
    <table class="trade-table">
      <thead>
        <tr>
          <th>Pair</th><th>Side</th><th>PnL%</th><th>PnL</th><th>Exit</th><th>Date</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="6" class="empty"><span class="spinner"></span>Loading&#x2026;</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script src="/static/app.js"></script>
<script>
  // Keep banner label in sync with dropdown
  document.getElementById("acct-select").addEventListener("change", function() {
    var val = parseFloat(this.value);
    var fmt = val >= 1000 ? "\xa3" + (val/1000).toFixed(0) + ",000" : "\xa3" + val;
    document.getElementById("acct-banner-size").textContent = fmt;
  });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


CALENDAR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PnL Calendar</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; padding-bottom: 32px;
    }
    header {
      background: #1a1d2e; padding: 16px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148; position: sticky; top: 0; z-index: 10;
    }
    header h1 { font-size: 17px; font-weight: 700; }
    .nav-btn {
      background: #2d3148; color: #e2e8f0; border: 1px solid #3d4268;
      border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 600;
      cursor: pointer; outline: none;
    }
    .nav-btn:hover { background: #3d4268; }
    .month-label { font-size: 16px; font-weight: 700; min-width: 160px; text-align: center; }
    .summary-bar {
      display: flex; gap: 16px; padding: 12px 16px;
      background: #1a1d2e; border-bottom: 1px solid #2d3148;
      font-size: 13px; flex-wrap: wrap;
    }
    .summary-item { display: flex; flex-direction: column; }
    .summary-label { font-size: 10px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .summary-value { font-size: 15px; font-weight: 700; margin-top: 2px; }
    .pos { color: #68d391; }
    .neg { color: #fc8181; }
    .neu { color: #e2e8f0; }
    .cal-grid {
      display: grid; grid-template-columns: repeat(7, 1fr);
      gap: 4px; padding: 12px 10px;
    }
    .day-header {
      text-align: center; font-size: 10px; font-weight: 700;
      color: #718096; text-transform: uppercase; letter-spacing: 0.5px;
      padding: 6px 0;
    }
    .day-cell {
      background: #1a1d2e; border: 1px solid #2d3148; border-radius: 8px;
      padding: 8px 6px; min-height: 72px;
      display: flex; flex-direction: column; justify-content: space-between;
    }
    .day-cell.green { background: #1a2e22; border-color: #276749; }
    .day-cell.red   { background: #2d1515; border-color: #9b2c2c; }
    .day-cell.empty { background: transparent; border-color: transparent; }
    .day-cell.today { border-color: #90cdf4; }
    .day-num { font-size: 11px; font-weight: 700; color: #718096; }
    .day-cell.green .day-num { color: #68d391; }
    .day-cell.red   .day-num { color: #fc8181; }
    .day-pnl { font-size: 13px; font-weight: 700; margin-top: 4px; }
    .day-trades { font-size: 10px; color: #718096; margin-top: 2px; }
    .day-cell.green .day-trades { color: #48bb78; }
    .day-cell.red   .day-trades { color: #f56565; }
  </style>
</head>
<body>
<header>
  <a href="/" style="text-decoration:none;color:#90cdf4;font-size:13px;">&#x2190; Dashboard</a>
  <h1>&#x1F4C5; PnL Calendar</h1>
  <div style="width:80px"></div>
</header>

<div class="summary-bar">
  <div class="summary-item">
    <span class="summary-label">Month PnL</span>
    <span class="summary-value" id="month-pnl">&#x2014;</span>
  </div>
  <div class="summary-item">
    <span class="summary-label">Trades</span>
    <span class="summary-value neu" id="month-trades">&#x2014;</span>
  </div>
  <div class="summary-item">
    <span class="summary-label">Green Days</span>
    <span class="summary-value pos" id="green-days">&#x2014;</span>
  </div>
  <div class="summary-item">
    <span class="summary-label">Red Days</span>
    <span class="summary-value neg" id="red-days">&#x2014;</span>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
    <button class="nav-btn" onclick="prevMonth()">&#x2190;</button>
    <span class="month-label" id="month-label"></span>
    <button class="nav-btn" onclick="nextMonth()">&#x2192;</button>
  </div>
</div>

<div class="cal-grid" id="cal-grid"></div>

<script>
var _data = {};
var _now  = new Date();
var _year = _now.getFullYear();
var _mon  = _now.getMonth(); // 0-based

var MONTHS = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"];
var DAYS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

function fmt(v, d) {
  return parseFloat(v).toFixed(d === undefined ? 2 : d);
}

function pad(n) { return n < 10 ? "0" + n : "" + n; }

function dateKey(y, m, d) {
  return y + "-" + pad(m + 1) + "-" + pad(d);
}

function render() {
  document.getElementById("month-label").textContent = MONTHS[_mon] + " " + _year;

  var firstDay = new Date(_year, _mon, 1).getDay(); // 0=Sun
  var daysInMonth = new Date(_year, _mon + 1, 0).getDate();
  // Convert Sunday=0 to Mon-based offset
  var offset = (firstDay + 6) % 7;

  var todayKey = dateKey(_now.getFullYear(), _now.getMonth(), _now.getDate());

  var monthPnl = 0, monthTrades = 0, greenDays = 0, redDays = 0;

  var html = DAYS.map(function(d) {
    return '<div class="day-header">' + d + '</div>';
  }).join("");

  // Empty cells before first day
  for (var i = 0; i < offset; i++) {
    html += '<div class="day-cell empty"></div>';
  }

  for (var d = 1; d <= daysInMonth; d++) {
    var key  = dateKey(_year, _mon, d);
    var info = _data[key];
    var cls  = "day-cell";
    var pnlHtml = "";
    var tradeHtml = "";

    if (key === todayKey) cls += " today";

    if (info) {
      monthPnl    += info.pnl;
      monthTrades += info.trades;
      if (info.pnl > 0) { cls += " green"; greenDays++; }
      else if (info.pnl < 0) { cls += " red"; redDays++; }
      var sign = info.pnl >= 0 ? "+" : "";
      pnlHtml   = '<div class="day-pnl">' + sign + "\xa3" + fmt(Math.abs(info.pnl)) + '</div>';
      tradeHtml = '<div class="day-trades">' + info.trades + ' trade' + (info.trades !== 1 ? "s" : "") + '</div>';
    }

    html += '<div class="' + cls + '">'
          + '<div class="day-num">' + d + '</div>'
          + pnlHtml
          + tradeHtml
          + '</div>';
  }

  document.getElementById("cal-grid").innerHTML = html;

  var mpEl = document.getElementById("month-pnl");
  var sign = monthPnl >= 0 ? "+" : "";
  mpEl.textContent = sign + "\xa3" + fmt(Math.abs(monthPnl));
  mpEl.className = "summary-value " + (monthPnl > 0 ? "pos" : monthPnl < 0 ? "neg" : "neu");
  document.getElementById("month-trades").textContent = monthTrades;
  document.getElementById("green-days").textContent   = greenDays;
  document.getElementById("red-days").textContent     = redDays;
}

function prevMonth() {
  _mon--;
  if (_mon < 0) { _mon = 11; _year--; }
  render();
}

function nextMonth() {
  _mon++;
  if (_mon > 11) { _mon = 0; _year++; }
  render();
}

fetch("/api/calendar").then(function(r) { return r.json(); }).then(function(d) {
  _data = d;
  render();
});
</script>
</body>
</html>"""


@app.get("/calendar", response_class=HTMLResponse)
def calendar_view():
    return CALENDAR_HTML
