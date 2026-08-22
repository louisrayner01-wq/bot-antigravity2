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
import glob
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image, ImageDraw

app = FastAPI()

STATE_FILE  = os.environ.get("STATE_FILE",  "/data/state.json")
TRADES_FILE = os.environ.get("TRADES_FILE", "/data/trades.csv")

# Portfolio-family (Portfolio v1) data — written by portfolio_bot.py.
# Each active user has their own state file; trades are appended to a
# single CSV keyed by user_id. Paths mirror what portfolio_bot writes.
PORTFOLIO_STATE_DIR    = os.environ.get(
    "PORTFOLIO_PAPER_STATE_DIR",
    "/data/portfolio_state" if os.path.isdir("/data") else "./portfolio_state",
)
PORTFOLIO_TRADES_FILE  = os.environ.get(
    "PORTFOLIO_TRADES_FILE",
    "/data/portfolio_trades.csv" if os.path.isdir("/data") else "./portfolio_trades.csv",
)

STRAT_1  = "strat_1"
PORTFOLIO = "portfolio"


def _aggregate_portfolio_state() -> dict:
    """Read every portfolio_<user>.json and roll them up into a single
    dashboard-compatible state dict. In single-user mode this is just one file;
    in multi-user mode we sum equity/HWM and merge positions across users.
    Returns a dict shaped like the Strat 1 state.json so the same UI renders.
    """
    files = sorted(glob.glob(os.path.join(PORTFOLIO_STATE_DIR, "portfolio_*.json")))
    if not files:
        return {}

    total_equity = 0.0
    total_hwm    = 0.0
    total_start  = 0.0
    positions: dict = {}
    latest_mtime = 0.0
    any_halted = False

    for path in files:
        try:
            with open(path) as f:
                s = json.load(f)
        except Exception:
            continue

        equity = float(s.get("equity") or 0.0)
        hwm    = float(s.get("hwm")    or 0.0)
        # initial_capital is persisted by portfolio_bot.py from v2 onward.
        # Legacy files fall back to equity so PnL renders as 0% instead of
        # blowing up on a missing baseline.
        start  = float(s.get("initial_capital") or equity)

        total_equity += equity
        total_hwm    += hwm
        total_start  += start
        any_halted = any_halted or bool(s.get("halted"))

        mtime = os.path.getmtime(path)
        if mtime > latest_mtime:
            latest_mtime = mtime

        # Merge positions — prefix slot key with user-short so multi-user
        # rollup doesn't collide on shared (strategy, asset) slots.
        user_short = os.path.basename(path).replace("portfolio_", "").replace(".json", "")[:8]
        for slot, pos in (s.get("positions") or {}).items():
            key = f"{user_short}::{slot}" if len(files) > 1 else slot
            positions[key] = {
                "symbol":      pos.get("asset"),
                "tf":          "4h",
                "side":        "long" if pos.get("side", 0) > 0 else "short",
                "entry_price": pos.get("entry_price"),
                "sl":          pos.get("stop_loss"),
                "tp":          pos.get("take_profit"),
                "qty":         pos.get("quantity"),
                "leverage":    pos.get("leverage"),
                "entry_time":  pos.get("entry_ts"),
            }

    return {
        "equity":          total_equity,
        "hwm":             total_hwm,
        "initial_capital": total_start if total_start > 0 else total_equity,
        "positions":       positions,
        "max_open":        len(positions) or 5,
        "paper":           True,
        "halted":          any_halted,
        "updated_at":      datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat() if latest_mtime else "",
    }


def _read_portfolio_trades() -> list:
    """Load the portfolio trades CSV as a list of dicts shaped for the dashboard."""
    if not os.path.exists(PORTFOLIO_TRADES_FILE):
        return []
    rows = []
    try:
        with open(PORTFOLIO_TRADES_FILE, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "pair":         row.get("pair"),
                    "side":         row.get("side"),
                    "pnl_pct":      float(row.get("pnl_pct") or 0) * 100.0,
                    "pnl_usdt":     row.get("pnl_usdt"),
                    "exit_reason":  row.get("exit_reason"),
                    "timestamp":    row.get("timestamp"),
                })
    except Exception:
        return []
    return list(reversed(rows[-100:]))


def _make_logo_png(size: int = 512) -> bytes:
    """Generate the Fortuna hexagon logo as a PNG using Pillow."""
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    gold  = (255, 215, 0, 255)
    black = (0, 0, 0, 255)
    sx, sy = size / 500, size / 520

    # Hexagon outline
    hex_pts = [
        (250*sx, 6*sy), (425*sx, 107*sy), (425*sx, 309*sy),
        (250*sx, 410*sy), (75*sx, 309*sy), (75*sx, 107*sy),
    ]
    stroke_w = max(1, int(11 * sx))
    for i in range(len(hex_pts)):
        draw.line([hex_pts[i], hex_pts[(i+1) % len(hex_pts)]], fill=gold, width=stroke_w)

    # Outer circle (black fill, gold stroke)
    cx, cy = 250*sx, 208*sy
    r  = 90 * min(sx, sy)
    ri = 22 * min(sx, sy)
    bw = max(1, int(8 * sx))
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=black, outline=gold, width=bw)

    # Inner filled circle
    draw.ellipse([cx-ri, cy-ri, cx+ri, cy+ri], fill=gold)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_LOGO_PNG: bytes = b""   # populated on first request


@app.get("/apple-touch-icon.png")
@app.get("/icon.png")
def serve_icon():
    global _LOGO_PNG
    if not _LOGO_PNG:
        _LOGO_PNG = _make_logo_png(512)
    return Response(content=_LOGO_PNG, media_type="image/png")


@app.get("/manifest.json")
def serve_manifest():
    manifest = {
        "name": "Fortuna",
        "short_name": "Fortuna",
        "description": "Automated trading bot",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f1117",
        "theme_color": "#0f1117",
        "icons": [
            {"src": "/icon.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }
    return JSONResponse(manifest)


# ── API endpoints ──────────────────────────────────────────────────────────────

def _trades_csv_for_family(family: str) -> str:
    """Route to the CSV that backs the given strategy family."""
    return PORTFOLIO_TRADES_FILE if family.lower() == PORTFOLIO else TRADES_FILE


@app.get("/api/pnl-chart")
def get_pnl_chart(family: str = Query(STRAT_1)):
    """Aggregate trades CSV into daily / weekly / monthly PnL and equity series."""
    from collections import defaultdict
    path = _trades_csv_for_family(family)
    if not os.path.exists(path):
        return {"daily": [], "weekly": [], "monthly": [], "equity_series": []}

    daily   = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    weekly  = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    equity_series = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts  = row.get("timestamp", "")[:10]
            if not ts:
                continue
            pnl = float(row.get("pnl_usdt", 0) or 0)
            try:
                from datetime import datetime
                dt       = datetime.strptime(ts, "%Y-%m-%d")
                week_key = dt.strftime("%Y-W%W")
                mon_key  = ts[:7]
            except ValueError:
                continue

            daily[ts]["pnl"]       += pnl
            daily[ts]["trades"]    += 1
            weekly[week_key]["pnl"]    += pnl
            weekly[week_key]["trades"] += 1
            monthly[mon_key]["pnl"]    += pnl
            monthly[mon_key]["trades"] += 1

            # equity_after only exists in the Strat 1 CSV; portfolio_bot writes
            # equity to state.json instead. Missing/blank is fine — that curve
            # just won't render for portfolio family.
            eq = float(row.get("equity_after", 0) or 0)
            if eq:
                equity_series.append({"date": ts, "equity": round(eq, 2)})

    def to_list(d, key):
        return [{"{}".format(key): k, "pnl": round(v["pnl"], 2), "trades": v["trades"]}
                for k, v in sorted(d.items())]

    return {
        "daily":         to_list(daily,   "date"),
        "weekly":        to_list(weekly,  "week"),
        "monthly":       to_list(monthly, "month"),
        "equity_series": equity_series,
    }


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
def get_calendar(family: str = Query(STRAT_1)):
    """Return daily PnL summary grouped by date for the calendar view."""
    path = _trades_csv_for_family(family)
    if not os.path.exists(path):
        return {}
    days = {}
    with open(path, newline="") as f:
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
def get_state(family: str = Query(STRAT_1)):
    fam = family.lower()
    if fam == PORTFOLIO:
        state = _aggregate_portfolio_state()
        if not state:
            return JSONResponse(
                {"error": "Portfolio state not found — portfolio bot may still be initialising."},
                status_code=503,
            )
        return state
    # Default: Strat 1 (original behavior)
    if not os.path.exists(STATE_FILE):
        return JSONResponse(
            {"error": "State file not found — bot may still be initialising."},
            status_code=503,
        )
    with open(STATE_FILE) as f:
        return json.load(f)


@app.get("/api/trades")
def get_trades(family: str = Query(STRAT_1)):
    fam = family.lower()
    if fam == PORTFOLIO:
        return _read_portfolio_trades()
    if not os.path.exists(TRADES_FILE):
        return []
    trades = []
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return list(reversed(trades[-100:]))  # newest first


_MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")


@app.get("/api/system")
def get_system():
    now = datetime.now(timezone.utc)
    checks = {}

    # ── Load state.json once ──────────────────────────────────────────────────
    state = None
    state_age_s = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            updated_at = state.get("updated_at", "")
            if updated_at:
                ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                state_age_s = int((now - ts).total_seconds())
        except Exception:
            pass

    # ── 1. Bot Cycle (is the bot ticking?) ───────────────────────────────────
    if state is None:
        checks["bot_cycle"] = {"ok": False, "label": "Bot Cycle", "detail": "state.json not found — bot may be starting up"}
    elif state_age_s is None:
        checks["bot_cycle"] = {"ok": False, "label": "Bot Cycle", "detail": "Could not read last tick time"}
    else:
        mins = state_age_s // 60
        secs = state_age_s % 60
        age_str = f"{mins}m {secs}s ago" if mins else f"{secs}s ago"
        # Tick interval is 15 min; allow up to 25 min before flagging
        ok = state_age_s < 1500
        checks["bot_cycle"] = {
            "ok": ok,
            "label": "Bot Cycle",
            "detail": f"Last tick {age_str}" if ok else f"No tick for {age_str} — bot may be down",
        }

    # ── 2. Trading Mode ───────────────────────────────────────────────────────
    is_paper = state.get("paper", True) if state else True
    checks["trading_mode"] = {
        "ok": not is_paper,
        "warn": is_paper,
        "label": "Trading Mode",
        "detail": "Paper trading — simulated orders" if is_paper else "LIVE — real orders on WEEX",
    }

    # ── 3. Equity Health ──────────────────────────────────────────────────────
    if state:
        equity = state.get("equity", 0)
        hwm    = state.get("hwm", equity) or equity
        dd_pct = round((1 - equity / hwm) * 100, 1) if hwm else 0
        eq_ok  = dd_pct < 10
        eq_warn = 10 <= dd_pct < 20
        checks["equity"] = {
            "ok": eq_ok and not eq_warn,
            "warn": eq_warn,
            "label": "Equity",
            "detail": f"${equity:.2f}  (HWM ${hwm:.2f}  DD {dd_pct:.1f}%)" if dd_pct > 0 else f"${equity:.2f}  at high-water mark",
        }
    else:
        checks["equity"] = {"ok": False, "label": "Equity", "detail": "No state data"}

    # ── 4. Open Positions ─────────────────────────────────────────────────────
    positions = state.get("positions", {}) if state else {}
    pos_details = []
    for slot, p in positions.items():
        side = p.get("side", "?").upper()
        entry = p.get("entry_price", 0)
        sym = slot.split("_")[0]
        pos_details.append(f"{sym} {side} @ {entry:.4f}")
    checks["open_positions"] = {
        "ok": True,
        "label": "Open Positions",
        "detail": f"{len(positions)} open" if positions else "None — flat",
        "positions": pos_details,
    }

    # ── 5. ML Models ─────────────────────────────────────────────────────────
    expected_models = ["BTCUSDT_5m.joblib", "ETHUSDT_5m.joblib", "SOLUSDT_5m.joblib", "XRPUSDT_5m.joblib"]
    model_results = {}
    for m in expected_models:
        path = os.path.join(_MODELS_DIR, m)
        exists = os.path.exists(path)
        size_mb = round(os.path.getsize(path) / 1e6, 1) if exists else 0
        model_results[m] = {"ok": exists, "size_mb": size_mb}
    all_models_ok = all(v["ok"] for v in model_results.values())
    checks["ml_models"] = {
        "ok": all_models_ok,
        "label": "ML Models",
        "detail": "All 4 models present" if all_models_ok else "One or more models missing",
        "models": model_results,
    }

    # ── 6. Recent Trades ─────────────────────────────────────────────────────
    last_trade_str = "No trades yet"
    trade_count    = 0
    today_pnl      = 0.0
    today_str      = now.strftime("%Y-%m-%d")
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, newline="") as f:
                rows = list(csv.DictReader(f))
            trade_count = len(rows)
            if rows:
                last_ts  = rows[-1].get("timestamp", "")
                last_pair = rows[-1].get("pair", "")
                last_pnl  = rows[-1].get("pnl_usdt", "")
                last_trade_str = f"{last_pair}  {last_ts[:16].replace('T',' ')}  {'+' if float(last_pnl or 0) >= 0 else ''}{float(last_pnl or 0):.2f} USDT"
            today_pnl = sum(
                float(r.get("pnl_usdt", 0) or 0)
                for r in rows if r.get("timestamp", "").startswith(today_str)
            )
        except Exception:
            pass
    checks["recent_trades"] = {
        "ok": True,
        "label": "Recent Trades",
        "detail": f"{trade_count} total  |  Today: {'+' if today_pnl >= 0 else ''}{today_pnl:.2f} USDT  |  Last: {last_trade_str}",
    }

    # ── 7. Telegram Alerts ───────────────────────────────────────────────────
    tg_token   = bool(os.environ.get("TELEGRAM_TOKEN"))
    tg_chat    = bool(os.environ.get("TELEGRAM_CHAT_ID"))
    tg_enabled = os.environ.get("TELEGRAM_ENABLED", "true").lower() == "true"
    tg_ok = tg_token and tg_chat and tg_enabled
    checks["telegram"] = {
        "ok": tg_ok,
        "label": "Telegram Alerts",
        "detail": "Configured and enabled" if tg_ok else (
            "Disabled (TELEGRAM_ENABLED=false)" if (tg_token and tg_chat and not tg_enabled)
            else "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"
        ),
    }

    overall_ok = all(
        c["ok"] for k, c in checks.items()
        if k not in ("trading_mode", "open_positions", "recent_trades")
    )
    return {"ok": overall_ok, "checked_at": now.isoformat(), "checks": checks}


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
  renderWeeklyPnl();
}

function onFamilyChange() {
  try { localStorage.setItem("dashboard_family", currentFamily()); } catch(e) {}
  // Clear all previous-family data before refetching so nothing lingers.
  _rawState = null;
  _rawTrades = [];
  // Weekly PnL lives in an inline script further down — call it if defined.
  if (typeof loadWeeklyPnl === "function") loadWeeklyPnl();
  refresh();
}

function currentFamily() {
  var s = document.getElementById("family-select");
  return (s && s.value) || "strat_1";
}

async function loadState() {
  try {
    var r = await fetch("/api/state?family=" + encodeURIComponent(currentFamily()));
    if (!r.ok) {
      el("updated").textContent = (currentFamily() === "portfolio")
        ? "Portfolio bot initialising\u2026"
        : "Bot initialising\u2026";
      _rawState = null;
      renderState();
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
    var r      = await fetch("/api/trades?family=" + encodeURIComponent(currentFamily()));
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

// Restore last-selected family from localStorage before the first fetch.
try {
  var savedFamily = localStorage.getItem("dashboard_family");
  var fs = document.getElementById("family-select");
  if (savedFamily && fs && (savedFamily === "strat_1" || savedFamily === "portfolio")) {
    fs.value = savedFamily;
  }
} catch(e) {}

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
  <title>Fortuna</title>
  <link rel="icon" href="/icon.png" type="image/png">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  <link rel="manifest" href="/manifest.json">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Fortuna">
  <meta name="theme-color" content="#0f1117">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; padding-bottom: 32px;
    }
    header {
      background: #1a1d2e; padding: 12px 16px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148; position: sticky; top: 0; z-index: 10;
    }
    .header-brand { display: flex; align-items: center; gap: 10px; }
    .header-brand svg { width: 36px; height: 38px; }
    .header-brand-text { display: flex; flex-direction: column; }
    .header-brand-name { font-size: 16px; font-weight: 700; letter-spacing: 2px; background: linear-gradient(180deg,#FFE566,#B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .header-brand-sub  { font-size: 9px; letter-spacing: 3px; color: #718096; text-transform: uppercase; }
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
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
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
  <div class="header-brand">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 420">
      <defs>
        <linearGradient id="hg" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#B8860B"/></linearGradient>
        <linearGradient id="hwl" x1="100%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <linearGradient id="hwr" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <filter id="hglow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <polygon points="250,6 425,107 425,309 250,410 75,309 75,107" fill="transparent" stroke="url(#hg)" stroke-width="11" stroke-linejoin="round" filter="url(#hglow)"/>
      <polygon points="250,32 404,122 404,294 250,384 96,294 96,122" fill="none" stroke="#FFD700" stroke-width="2.5" opacity="0.3"/>
      <polygon points="168,205 101,92 156,211" fill="url(#hwl)" opacity="1.00"/>
      <polygon points="167,203 75,135 158,213" fill="url(#hwl)" opacity="0.87"/>
      <polygon points="164,201 75,176 160,215" fill="url(#hwl)" opacity="0.74"/>
      <polygon points="162,201 75,208 162,215" fill="url(#hwl)" opacity="0.62"/>
      <polygon points="165,215 75,243 159,202" fill="url(#hwl)" opacity="0.50"/>
      <polygon points="167,213 75,286 157,203" fill="url(#hwl)" opacity="0.38"/>
      <polygon points="332,205 399,92 344,211" fill="url(#hwr)" opacity="1.00"/>
      <polygon points="333,203 425,135 342,213" fill="url(#hwr)" opacity="0.87"/>
      <polygon points="336,201 425,176 340,215" fill="url(#hwr)" opacity="0.74"/>
      <polygon points="338,201 425,208 338,215" fill="url(#hwr)" opacity="0.62"/>
      <polygon points="335,215 425,243 341,202" fill="url(#hwr)" opacity="0.50"/>
      <polygon points="333,213 425,286 343,203" fill="url(#hwr)" opacity="0.38"/>
      <circle cx="250" cy="208" r="90" fill="#1a1d2e" stroke="url(#hg)" stroke-width="8" filter="url(#hglow)"/>
      <circle cx="250" cy="208" r="59" fill="none" stroke="url(#hg)" stroke-width="3.5"/>
      <g stroke="#FFD700" stroke-width="4.5" stroke-linecap="round">
        <line x1="250" y1="118" x2="250" y2="298"/>
        <line x1="160" y1="208" x2="340" y2="208"/>
        <line x1="186" y1="144" x2="314" y2="272"/>
        <line x1="314" y1="144" x2="186" y2="272"/>
      </g>
      <circle cx="250" cy="208" r="22" fill="url(#hg)"/>
      <circle cx="250" cy="208" r="12" fill="#1a1d2e"/>
      <circle cx="250" cy="208" r="5"  fill="#FFD700"/>
      <polygon points="250,109 244,118 250,127 256,118" fill="#FFD700"/>
      <polygon points="250,289 244,298 250,307 256,298" fill="#FFD700"/>
      <polygon points="151,202 160,208 151,214 160,208" fill="#FFD700"/>
      <polygon points="340,202 349,208 340,214 349,208" fill="#FFD700"/>
    </svg>
    <div class="header-brand-text">
      <span class="header-brand-name">FORTUNA</span>
      <span class="header-brand-sub">Trading Bot</span>
      <div class="updated" id="updated">Loading&#x2026;</div>
    </div>
  </div>
  <div class="header-right">
    <div>
      <div class="acct-label">Strategy</div>
      <select class="acct-select" id="family-select" onchange="onFamilyChange()">
        <option value="strat_1">Strat 1</option>
        <option value="portfolio">Portfolio Strategy</option>
      </select>
    </div>
    <div>
      <div class="acct-label">Account size</div>
      <select class="acct-select" id="acct-select" onchange="onAccountChange()">
        <option value="100">&#xa3;100</option>
        <option value="500">&#xa3;500</option>
        <option value="1000">&#xa3;1,000</option>
        <option value="5000">&#xa3;5,000</option>
        <option value="10000">&#xa3;10,000</option>
        <option value="15000">&#xa3;15,000</option>
        <option value="20000">&#xa3;20,000</option>
      </select>
    </div>
    <a href="/pnl" style="text-decoration:none;">
      <div class="badge" style="background:#2d3148;color:#90cdf4;cursor:pointer;">&#x1F4C8; P&amp;L</div>
    </a>
    <a href="/calendar" style="text-decoration:none;">
      <div class="badge" style="background:#2d3148;color:#90cdf4;cursor:pointer;">&#x1F4C5; Calendar</div>
    </a>
    <a href="/system" style="text-decoration:none;">
      <div class="badge" style="background:#2d3148;color:#90cdf4;cursor:pointer;">&#x2699;&#xFE0F; System</div>
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
    <div class="stat-card">
      <div class="stat-label">This Week P&amp;L</div>
      <div class="stat-value" id="week-pnl">&#x2014;</div>
      <div class="stat-sub" id="week-trades"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Last Week P&amp;L</div>
      <div class="stat-value" id="lastweek-pnl">&#x2014;</div>
      <div class="stat-sub" id="lastweek-trades"></div>
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

  // Weekly PnL from calendar data
  var _rawWeekly = { tw: { pnl: 0, trades: 0 }, lw: { pnl: 0, trades: 0 } };

  function getISOWeek(date) {
    var d = new Date(date);
    var day = d.getDay() || 7;
    d.setHours(0,0,0,0);
    d.setDate(d.getDate() + 4 - day);
    var yearStart = new Date(d.getFullYear(), 0, 1);
    return d.getFullYear() + "-W" + Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
  }

  function renderWeeklyPnl() {
    var sf = _scaleFactor;
    function setPnl(idPnl, idTrades, data) {
      var scaled = data.pnl * sf;
      var el = document.getElementById(idPnl);
      var sign = scaled >= 0 ? "+" : "";
      el.textContent = sign + "\xa3" + Math.abs(scaled).toFixed(2);
      el.className = "stat-value " + (scaled > 0 ? "pos" : scaled < 0 ? "neg" : "neu");
      document.getElementById(idTrades).textContent = data.trades + " trade" + (data.trades !== 1 ? "s" : "");
    }
    setPnl("week-pnl",     "week-trades",     _rawWeekly.tw);
    setPnl("lastweek-pnl", "lastweek-trades", _rawWeekly.lw);
  }

  function loadWeeklyPnl() {
    // Reset instantly so the previous family's numbers don't linger while the
    // new fetch is in flight.
    _rawWeekly = { tw: { pnl: 0, trades: 0 }, lw: { pnl: 0, trades: 0 } };
    renderWeeklyPnl();

    var fam = (typeof currentFamily === "function") ? currentFamily() : "strat_1";
    fetch("/api/calendar?family=" + encodeURIComponent(fam))
      .then(function(r) { return r.json(); })
      .then(function(days) {
        var thisWeek = getISOWeek(new Date());
        var lastWeekDate = new Date(); lastWeekDate.setDate(lastWeekDate.getDate() - 7);
        var lastWeek = getISOWeek(lastWeekDate);

        var tw = { pnl: 0, trades: 0 };
        var lw = { pnl: 0, trades: 0 };

        Object.keys(days).forEach(function(date) {
          var w = getISOWeek(new Date(date));
          if (w === thisWeek) { tw.pnl += days[date].pnl; tw.trades += days[date].trades; }
          if (w === lastWeek) { lw.pnl += days[date].pnl; lw.trades += days[date].trades; }
        });

        _rawWeekly = { tw: tw, lw: lw };
        renderWeeklyPnl();
      });
  }

  loadWeeklyPnl();
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
    .header-brand { display: flex; align-items: center; gap: 8px; }
    .header-brand svg { width: 28px; height: 29px; }
    .header-brand-name { font-size: 15px; font-weight: 700; letter-spacing: 2px; background: linear-gradient(180deg,#FFE566,#B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .header-brand-sub  { font-size: 9px; letter-spacing: 2px; color: #718096; text-transform: uppercase; }
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
  <div class="header-brand">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 420">
      <defs>
        <linearGradient id="cg" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#B8860B"/></linearGradient>
        <linearGradient id="cwl" x1="100%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <linearGradient id="cwr" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <filter id="cglow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <polygon points="250,6 425,107 425,309 250,410 75,309 75,107" fill="transparent" stroke="url(#cg)" stroke-width="11" stroke-linejoin="round" filter="url(#cglow)"/>
      <polygon points="250,32 404,122 404,294 250,384 96,294 96,122" fill="none" stroke="#FFD700" stroke-width="2.5" opacity="0.3"/>
      <polygon points="168,205 101,92 156,211" fill="url(#cwl)" opacity="1.00"/>
      <polygon points="167,203 75,135 158,213" fill="url(#cwl)" opacity="0.87"/>
      <polygon points="164,201 75,176 160,215" fill="url(#cwl)" opacity="0.74"/>
      <polygon points="162,201 75,208 162,215" fill="url(#cwl)" opacity="0.62"/>
      <polygon points="165,215 75,243 159,202" fill="url(#cwl)" opacity="0.50"/>
      <polygon points="167,213 75,286 157,203" fill="url(#cwl)" opacity="0.38"/>
      <polygon points="332,205 399,92 344,211" fill="url(#cwr)" opacity="1.00"/>
      <polygon points="333,203 425,135 342,213" fill="url(#cwr)" opacity="0.87"/>
      <polygon points="336,201 425,176 340,215" fill="url(#cwr)" opacity="0.74"/>
      <polygon points="338,201 425,208 338,215" fill="url(#cwr)" opacity="0.62"/>
      <polygon points="335,215 425,243 341,202" fill="url(#cwr)" opacity="0.50"/>
      <polygon points="333,213 425,286 343,203" fill="url(#cwr)" opacity="0.38"/>
      <circle cx="250" cy="208" r="90" fill="#1a1d2e" stroke="url(#cg)" stroke-width="8" filter="url(#cglow)"/>
      <circle cx="250" cy="208" r="59" fill="none" stroke="url(#cg)" stroke-width="3.5"/>
      <g stroke="#FFD700" stroke-width="4.5" stroke-linecap="round">
        <line x1="250" y1="118" x2="250" y2="298"/>
        <line x1="160" y1="208" x2="340" y2="208"/>
        <line x1="186" y1="144" x2="314" y2="272"/>
        <line x1="314" y1="144" x2="186" y2="272"/>
      </g>
      <circle cx="250" cy="208" r="22" fill="url(#cg)"/>
      <circle cx="250" cy="208" r="12" fill="#1a1d2e"/>
      <circle cx="250" cy="208" r="5"  fill="#FFD700"/>
      <polygon points="250,109 244,118 250,127 256,118" fill="#FFD700"/>
      <polygon points="250,289 244,298 250,307 256,298" fill="#FFD700"/>
      <polygon points="151,202 160,208 151,214 160,208" fill="#FFD700"/>
      <polygon points="340,202 349,208 340,214 349,208" fill="#FFD700"/>
    </svg>
    <div>
      <div class="header-brand-name">FORTUNA</div>
      <div class="header-brand-sub">PnL Calendar</div>
    </div>
  </div>
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


# ── PnL Performance Page ───────────────────────────────────────────────────────

PNL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fortuna — Performance</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; padding-bottom: 40px;
    }
    header {
      background: #1a1d2e; padding: 12px 16px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148; position: sticky; top: 0; z-index: 10;
    }
    .header-brand { display: flex; align-items: center; gap: 10px; }
    .header-brand svg { width: 36px; height: 38px; }
    .header-brand-text { display: flex; flex-direction: column; }
    .header-brand-name { font-size: 16px; font-weight: 700; letter-spacing: 2px; background: linear-gradient(180deg,#FFE566,#B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .header-brand-sub  { font-size: 9px; letter-spacing: 3px; color: #718096; text-transform: uppercase; }
    .back-btn { font-size: 13px; color: #90cdf4; text-decoration: none; }
    .back-btn:hover { color: #e2e8f0; }
    .section { padding: 16px; }
    .section-title { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #718096; margin-bottom: 12px; }
    .card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 16px; }
    .stat-pill { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 10px; padding: 14px 12px; }
    .stat-pill .label { font-size: 10px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-pill .value { font-size: 18px; font-weight: 700; margin-top: 4px; }
    .pos { color: #68d391; } .neg { color: #fc8181; } .neu { color: #e2e8f0; }
    .chart-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
    .nav-btn { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4268; border-radius: 6px; padding: 5px 12px; font-size: 13px; font-weight: 600; cursor: pointer; }
    .nav-btn:hover { background: #3d4268; }
    .nav-btn:disabled { opacity: 0.3; cursor: default; }
    .month-label { font-size: 13px; font-weight: 700; min-width: 100px; text-align: center; }
    .month-meta { display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
    .month-meta-item .ml { font-size: 10px; color: #718096; }
    .month-meta-item .mv { font-size: 14px; font-weight: 700; margin-top: 2px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
    .grid-2 .card { margin-bottom: 0; }
    .chart-wrap { position: relative; height: 200px; width: 100%; }
    .chart-wrap-sm { position: relative; height: 160px; width: 100%; }
    canvas { position: absolute; top: 0; left: 0; }
  </style>
</head>
<body>

<header>
  <div class="header-brand">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 420">
      <defs>
        <linearGradient id="pg" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#B8860B"/></linearGradient>
        <linearGradient id="pwl" x1="100%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <linearGradient id="pwr" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#7A5500"/></linearGradient>
        <filter id="pglow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <polygon points="250,6 425,107 425,309 250,410 75,309 75,107" fill="transparent" stroke="url(#pg)" stroke-width="11" stroke-linejoin="round" filter="url(#pglow)"/>
      <polygon points="250,32 404,122 404,294 250,384 96,294 96,122" fill="none" stroke="#FFD700" stroke-width="2.5" opacity="0.3"/>
      <polygon points="168,205 101,92 156,211" fill="url(#pwl)" opacity="1.00"/>
      <polygon points="167,203 75,135 158,213" fill="url(#pwl)" opacity="0.87"/>
      <polygon points="164,201 75,176 160,215" fill="url(#pwl)" opacity="0.74"/>
      <polygon points="162,201 75,208 162,215" fill="url(#pwl)" opacity="0.62"/>
      <polygon points="165,215 75,243 159,202" fill="url(#pwl)" opacity="0.50"/>
      <polygon points="167,213 75,286 157,203" fill="url(#pwl)" opacity="0.38"/>
      <polygon points="332,205 399,92 344,211" fill="url(#pwr)" opacity="1.00"/>
      <polygon points="333,203 425,135 342,213" fill="url(#pwr)" opacity="0.87"/>
      <polygon points="336,201 425,176 340,215" fill="url(#pwr)" opacity="0.74"/>
      <polygon points="338,201 425,208 338,215" fill="url(#pwr)" opacity="0.62"/>
      <polygon points="335,215 425,243 341,202" fill="url(#pwr)" opacity="0.50"/>
      <polygon points="333,213 425,286 343,203" fill="url(#pwr)" opacity="0.38"/>
      <circle cx="250" cy="208" r="90" fill="#1a1d2e" stroke="url(#pg)" stroke-width="8" filter="url(#pglow)"/>
      <circle cx="250" cy="208" r="59" fill="none" stroke="url(#pg)" stroke-width="3.5"/>
      <g stroke="#FFD700" stroke-width="4.5" stroke-linecap="round">
        <line x1="250" y1="118" x2="250" y2="298"/>
        <line x1="160" y1="208" x2="340" y2="208"/>
        <line x1="186" y1="144" x2="314" y2="272"/>
        <line x1="314" y1="144" x2="186" y2="272"/>
      </g>
      <circle cx="250" cy="208" r="22" fill="url(#pg)"/>
      <circle cx="250" cy="208" r="12" fill="#1a1d2e"/>
      <circle cx="250" cy="208" r="5"  fill="#FFD700"/>
      <polygon points="250,109 244,118 250,127 256,118" fill="#FFD700"/>
      <polygon points="250,289 244,298 250,307 256,298" fill="#FFD700"/>
      <polygon points="151,202 160,208 151,214 160,208" fill="#FFD700"/>
      <polygon points="340,202 349,208 340,214 349,208" fill="#FFD700"/>
    </svg>
    <div class="header-brand-text">
      <span class="header-brand-name">FORTUNA</span>
      <span class="header-brand-sub">Performance</span>
    </div>
  </div>
  <a href="/" class="back-btn">&#x2190; Dashboard</a>
</header>

<div class="section">

  <!-- Summary pills -->
  <div class="stats-row">
    <div class="stat-pill"><div class="label">All-time P&L</div><div class="value" id="all-pnl">&mdash;</div></div>
    <div class="stat-pill"><div class="label">Total Trades</div><div class="value neu" id="all-trades">&mdash;</div></div>
    <div class="stat-pill"><div class="label">Win Days</div><div class="value pos" id="win-days">&mdash;</div></div>
  </div>

  <!-- Daily PnL -->
  <div class="card">
    <div class="chart-header">
      <div class="section-title" style="margin:0">Daily P&L</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <button class="nav-btn" id="prev-mon" onclick="shiftMonth(-1)">&#x2190;</button>
        <span class="month-label" id="mon-label"></span>
        <button class="nav-btn" id="next-mon" onclick="shiftMonth(1)">&#x2192;</button>
      </div>
    </div>
    <div class="month-meta">
      <div class="month-meta-item"><div class="ml">Month P&L</div><div class="mv" id="m-pnl">&mdash;</div></div>
      <div class="month-meta-item"><div class="ml">Trades</div><div class="mv neu" id="m-trades">&mdash;</div></div>
      <div class="month-meta-item"><div class="ml">Green days</div><div class="mv pos" id="m-green">&mdash;</div></div>
      <div class="month-meta-item"><div class="ml">Red days</div><div class="mv neg" id="m-red">&mdash;</div></div>
    </div>
    <div class="chart-wrap"><canvas id="daily-chart"></canvas></div>
  </div>

  <!-- Weekly + Monthly side by side -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">Weekly P&L</div>
      <div class="chart-wrap-sm"><canvas id="weekly-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-title">Monthly P&L</div>
      <div class="chart-wrap-sm"><canvas id="monthly-chart"></canvas></div>
    </div>
  </div>

  <!-- Equity curve -->
  <div class="card">
    <div class="section-title">Equity Curve</div>
    <div class="chart-wrap"><canvas id="equity-chart"></canvas></div>
  </div>

</div>

<script>
var _daily = [], _weekly = [], _monthly = [], _equity = [];
var _months = [], _monIdx = 0;
var _charts = {};

var CHART_DEFAULTS = {
  plugins: { legend: { display: false }, tooltip: {
    backgroundColor: "#1a1d2e", borderColor: "#2d3148", borderWidth: 1,
    titleColor: "#90cdf4", bodyColor: "#e2e8f0", padding: 10,
  }},
  scales: {
    x: { ticks: { color: "#718096", font: { size: 10 } }, grid: { color: "#1e2235" }, border: { display: false } },
    y: { ticks: { color: "#718096", font: { size: 10 } }, grid: { color: "#1e2235" }, border: { display: false } },
  },
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
};

function sign(v) { return v >= 0 ? "+" : ""; }
function fmt(v)  { return sign(v) + "$" + Math.abs(parseFloat(v)).toFixed(2); }
function pnlColor(v) { return parseFloat(v) >= 0 ? "#68d391" : "#fc8181"; }

function buildBarChart(id, labels, values, labelFn) {
  var ctx = document.getElementById(id).getContext("2d");
  if (_charts[id]) _charts[id].destroy();
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(function(v) { return pnlColor(v); }),
        borderRadius: 3,
      }]
    },
    options: Object.assign({}, CHART_DEFAULTS, {
      plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
        tooltip: Object.assign({}, CHART_DEFAULTS.plugins.tooltip, {
          callbacks: {
            title: function(items) { return labelFn ? labelFn(items[0].label) : items[0].label; },
            label: function(item) { return " P&L: " + fmt(item.raw); },
          }
        })
      })
    })
  });
}

function renderDaily() {
  var mon = _months[_months.length - 1 - _monIdx] || "";
  var days = _daily.filter(function(d) { return d.date.slice(0,7) === mon; });
  document.getElementById("mon-label").textContent = mon || "—";
  document.getElementById("prev-mon").disabled = (_monIdx >= _months.length - 1);
  document.getElementById("next-mon").disabled = (_monIdx === 0);

  var mPnl    = days.reduce(function(s,d) { return s + d.pnl; }, 0);
  var mTrades = days.reduce(function(s,d) { return s + d.trades; }, 0);
  var mGreen  = days.filter(function(d) { return d.pnl > 0; }).length;
  var mRed    = days.filter(function(d) { return d.pnl < 0; }).length;

  var pEl = document.getElementById("m-pnl");
  pEl.textContent = fmt(mPnl);
  pEl.className = "mv " + (mPnl >= 0 ? "pos" : "neg");
  document.getElementById("m-trades").textContent = mTrades;
  document.getElementById("m-green").textContent  = mGreen;
  document.getElementById("m-red").textContent    = mRed;

  buildBarChart("daily-chart",
    days.map(function(d) { return d.date.slice(8); }),
    days.map(function(d) { return d.pnl; }),
    function(l) { return "Day " + l; }
  );
}

function shiftMonth(dir) {
  _monIdx = Math.max(0, Math.min(_months.length - 1, _monIdx - dir));
  renderDaily();
}

function render() {
  // All-time summary
  var totalPnl    = _monthly.reduce(function(s,m) { return s + m.pnl; }, 0);
  var totalTrades = _monthly.reduce(function(s,m) { return s + m.trades; }, 0);
  var winDays     = _daily.filter(function(d) { return d.pnl > 0; }).length;
  var allDays     = _daily.filter(function(d) { return d.pnl !== 0; }).length;

  var pEl = document.getElementById("all-pnl");
  pEl.textContent = fmt(totalPnl);
  pEl.className = "value " + (totalPnl >= 0 ? "pos" : "neg");
  document.getElementById("all-trades").textContent = totalTrades;
  document.getElementById("win-days").textContent   = winDays + " / " + allDays;

  // Daily
  _months = Array.from(new Set(_daily.map(function(d) { return d.date.slice(0,7); }))).sort();
  renderDaily();

  // Weekly
  var wSlice = _weekly.slice(-12);
  buildBarChart("weekly-chart",
    wSlice.map(function(w) { return "W" + (w.week.split("-W")[1] || w.week); }),
    wSlice.map(function(w) { return w.pnl; }),
    function(l) { return "Week " + l.replace("W",""); }
  );

  // Monthly
  var mSlice = _monthly.slice(-12);
  var MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  buildBarChart("monthly-chart",
    mSlice.map(function(m) { var p = m.month.split("-"); return MONTHS[parseInt(p[1])-1] + " " + p[0]; }),
    mSlice.map(function(m) { return m.pnl; }),
    function(l) { return l; }
  );

  // Equity curve
  var ctx = document.getElementById("equity-chart").getContext("2d");
  if (_charts["equity-chart"]) _charts["equity-chart"].destroy();
  _charts["equity-chart"] = new Chart(ctx, {
    type: "line",
    data: {
      labels: _equity.map(function(e) { return e.date.slice(5); }),
      datasets: [{
        data: _equity.map(function(e) { return e.equity; }),
        borderColor: "#6366f1",
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: { target: "origin", above: "rgba(99,102,241,0.08)", below: "rgba(252,129,129,0.08)" },
        tension: 0.3,
      }]
    },
    options: Object.assign({}, CHART_DEFAULTS, {
      plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
        tooltip: Object.assign({}, CHART_DEFAULTS.plugins.tooltip, {
          callbacks: { label: function(item) { return " Equity: $" + parseFloat(item.raw).toFixed(2); } }
        })
      })
    })
  });
}

fetch("/api/pnl-chart")
  .then(function(r) { return r.json(); })
  .then(function(d) {
    _daily   = d.daily   || [];
    _weekly  = d.weekly  || [];
    _monthly = d.monthly || [];
    _equity  = d.equity_series || [];
    render();
  });
</script>
</body>
</html>"""


@app.get("/pnl", response_class=HTMLResponse)
def pnl_view():
    return PNL_HTML


# ── System health page ─────────────────────────────────────────────────────────

SYSTEM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fortuna — System</title>
  <link rel="icon" href="/icon.png" type="image/png">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; padding-bottom: 40px;
    }
    header {
      background: #1a1d2e; padding: 12px 16px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148; position: sticky; top: 0; z-index: 10;
    }
    .header-brand { display: flex; align-items: center; gap: 8px; }
    .header-brand svg { width: 28px; height: 29px; }
    .brand-text { display: flex; flex-direction: column; }
    .brand-name { font-size: 15px; font-weight: 700; letter-spacing: 2px; background: linear-gradient(180deg,#FFE566,#B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-sub  { font-size: 9px; letter-spacing: 2px; color: #718096; text-transform: uppercase; }
    .nav-btn { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4268; border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 600; cursor: pointer; text-decoration: none; }
    .nav-btn:hover { background: #3d4268; }

    .page-title { padding: 20px 16px 8px; font-size: 18px; font-weight: 700; }
    .refresh-note { padding: 0 16px 16px; font-size: 12px; color: #718096; }

    .overall-banner {
      margin: 0 16px 20px; padding: 14px 16px; border-radius: 10px;
      display: flex; align-items: center; gap: 12px; font-size: 15px; font-weight: 700;
    }
    .overall-banner.ok   { background: #1a2e22; border: 1px solid #276749; color: #68d391; }
    .overall-banner.fail { background: #2d1515; border: 1px solid #9b2c2c; color: #fc8181; }
    .overall-banner.loading { background: #1a1d2e; border: 1px solid #2d3148; color: #718096; }

    .section { margin: 0 16px 12px; }
    .check-card {
      background: #1a1d2e; border: 1px solid #2d3148; border-radius: 10px;
      margin-bottom: 8px; overflow: hidden;
    }
    .check-header {
      display: flex; align-items: center; gap: 12px;
      padding: 14px 16px; cursor: pointer; user-select: none;
    }
    .check-header:hover { background: #20243a; }
    .dot {
      width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
    }
    .dot.ok   { background: #48bb78; box-shadow: 0 0 6px #48bb78aa; }
    .dot.fail { background: #fc8181; box-shadow: 0 0 6px #fc8181aa; }
    .dot.warn { background: #f6ad55; box-shadow: 0 0 6px #f6ad55aa; }
    .dot.loading { background: #718096; }
    .check-label { font-weight: 600; font-size: 14px; flex: 1; }
    .check-detail { font-size: 12px; color: #a0aec0; }
    .chevron { font-size: 12px; color: #718096; transition: transform 0.2s; }
    .check-card.open .chevron { transform: rotate(180deg); }

    .check-body { display: none; border-top: 1px solid #2d3148; padding: 12px 16px; }
    .check-card.open .check-body { display: block; }

    .sub-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 0; border-bottom: 1px solid #2d3148;
      font-size: 13px;
    }
    .sub-row:last-child { border-bottom: none; }
    .sub-name { color: #a0aec0; }
    .sub-ok   { color: #68d391; font-weight: 600; }
    .sub-fail { color: #fc8181; font-weight: 600; }
    .sub-warn { color: #f6ad55; font-weight: 600; }
    .sub-detail { font-size: 11px; color: #718096; }

    .checked-at { text-align: center; font-size: 11px; color: #4a5568; padding-top: 24px; }
  </style>
</head>
<body>
<header>
  <div class="header-brand">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 420">
      <defs>
        <linearGradient id="sg" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#FFE566"/><stop offset="100%" stop-color="#B8860B"/></linearGradient>
        <filter id="sglow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <polygon points="250,6 425,107 425,309 250,410 75,309 75,107" fill="transparent" stroke="url(#sg)" stroke-width="11" stroke-linejoin="round" filter="url(#sglow)"/>
      <circle cx="250" cy="208" r="90" fill="#1a1d2e" stroke="url(#sg)" stroke-width="8" filter="url(#sglow)"/>
      <circle cx="250" cy="208" r="22" fill="#FFD700"/>
    </svg>
    <div class="brand-text">
      <span class="brand-name">FORTUNA</span>
      <span class="brand-sub">System Health</span>
    </div>
  </div>
  <a href="/" class="nav-btn">&#x2190; Dashboard</a>
</header>

<div class="page-title">System Status</div>
<div class="refresh-note" id="refresh-note">Checking...</div>

<div id="overall-banner" class="overall-banner loading">
  <span id="overall-icon">⏳</span>
  <span id="overall-text">Loading system status...</span>
</div>

<div class="section" id="checks-container"></div>

<div class="checked-at" id="checked-at"></div>

<script>
var EXPAND_ON_FAIL = true;

function dot(ok, warn) {
  if (ok === null) return '<span class="dot loading"></span>';
  if (warn)        return '<span class="dot warn"></span>';
  return ok ? '<span class="dot ok"></span>' : '<span class="dot fail"></span>';
}

function renderModels(models) {
  if (!models) return '';
  return Object.entries(models).map(function(kv) {
    var name = kv[0], v = kv[1];
    return '<div class="sub-row">'
      + '<span class="sub-name">' + name + '</span>'
      + (v.ok
        ? '<span class="sub-ok">&#10003; ' + v.size_mb + ' MB</span>'
        : '<span class="sub-fail">&#10007; Missing</span>')
      + '</div>';
  }).join('');
}

function renderPositions(positions) {
  if (!positions || positions.length === 0) return '<div class="sub-row"><span class="sub-detail">No open positions — flat</span></div>';
  return positions.map(function(p) {
    return '<div class="sub-row"><span class="sub-name">' + p + '</span><span class="sub-ok">Open</span></div>';
  }).join('');
}

function renderCard(key, check) {
  var isWarn = check.warn || false;
  var bodyHtml = '';

  if (key === 'ml_models' && check.models) bodyHtml = renderModels(check.models);
  else if (key === 'open_positions') bodyHtml = renderPositions(check.positions || []);
  else if (check.detail) bodyHtml = '<div class="sub-row"><span class="sub-detail">' + check.detail + '</span></div>';

  var hasBody = bodyHtml.length > 0;
  var autoOpen = EXPAND_ON_FAIL && !check.ok && hasBody;

  return '<div class="check-card' + (autoOpen ? ' open' : '') + '" id="card-' + key + '">'
    + '<div class="check-header" onclick="toggleCard(\'' + key + '\')">'
    + dot(check.ok, isWarn)
    + '<span class="check-label">' + check.label + '</span>'
    + '<span class="check-detail">' + (check.detail || '') + '</span>'
    + (hasBody ? '<span class="chevron">&#9660;</span>' : '')
    + '</div>'
    + (hasBody ? '<div class="check-body">' + bodyHtml + '</div>' : '')
    + '</div>';
}

function toggleCard(key) {
  var card = document.getElementById('card-' + key);
  if (card) card.classList.toggle('open');
}

function render(data) {
  var banner = document.getElementById('overall-banner');
  var icon   = document.getElementById('overall-icon');
  var text   = document.getElementById('overall-text');

  banner.className = 'overall-banner ' + (data.ok ? 'ok' : 'fail');
  icon.textContent = data.ok ? '✅' : '⚠️';
  text.textContent = data.ok ? 'All systems operational' : 'One or more systems need attention';

  var order = ['bot_cycle','trading_mode','equity','open_positions','ml_models','recent_trades','telegram'];
  var html = '';
  order.forEach(function(k) {
    if (data.checks[k]) html += renderCard(k, data.checks[k]);
  });
  document.getElementById('checks-container').innerHTML = html;

  var d = new Date(data.checked_at);
  document.getElementById('checked-at').textContent = 'Last checked: ' + d.toLocaleTimeString();
  document.getElementById('refresh-note').textContent = 'Auto-refreshes every 30s';
}

function load() {
  fetch('/api/system')
    .then(function(r) { return r.json(); })
    .then(render)
    .catch(function(e) {
      document.getElementById('overall-banner').className = 'overall-banner fail';
      document.getElementById('overall-text').textContent = 'Could not reach /api/system';
    });
}

load();
setInterval(load, 30000);
</script>
</body>
</html>"""


@app.get("/system", response_class=HTMLResponse)
def system_view():
    return SYSTEM_HTML
