"""
push_reports.py
───────────────
Reads backtest_results.json and analysis_results.json from the Railway /data
volume and pushes a human-readable markdown summary to the GitHub repo under
reports/backtest_summary.md and reports/temporal_breakdown.md.

Uses the GitHub API (no git required in the container) — just needs:
  GITHUB_TOKEN  — a GitHub personal access token with repo write access
  GITHUB_REPO   — owner/repo e.g. "louisrayner01-wq/bot-antigravity2"

Called automatically after RUN_BACKTEST completes.
Can also be run standalone: python push_reports.py
"""

import os
import json
import base64
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── GitHub API helpers ────────────────────────────────────────────────────────

def _gh_request(method: str, path: str, token: str, body: dict = None) -> Optional[dict]:
    url  = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept",        "application/vnd.github.v3+json")
    req.add_header("Content-Type",  "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        log.warning("GitHub API %s %s → %d: %s", method, path, e.code, e.read())
        return None


def _get_file_sha(repo: str, path: str, token: str) -> Optional[str]:
    """Return the blob SHA of an existing file (needed for updates)."""
    result = _gh_request("GET", f"/repos/{repo}/contents/{path}", token)
    return result.get("sha") if result else None


def _push_file(repo: str, path: str, content: str, message: str, token: str):
    """Create or update a file in the repo."""
    sha = _get_file_sha(repo, path, token)
    body = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        body["sha"] = sha
    result = _gh_request("PUT", f"/repos/{repo}/contents/{path}", token, body)
    if result:
        log.info("📤 Pushed → %s/%s", repo, path)
    else:
        log.warning("Failed to push %s", path)


# ── Report builders ───────────────────────────────────────────────────────────

def _flag(wr: float) -> str:
    if wr >= 0.50: return "✅"
    if wr >= 0.44: return "⚠️"
    return "❌"


def build_backtest_summary(results: list) -> str:
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Backtest Summary",
        f"_Generated: {now}_",
        "",
        "## Strategy Results",
        "",
        "| Symbol | Strategy | Trades | WR% | Avg Win | Avg Loss | R/R | EV% | Total PnL | Max DD |",
        "|--------|----------|--------|-----|---------|----------|-----|-----|-----------|--------|",
    ]

    for r in sorted(results, key=lambda x: x.get("ev_pct", 0), reverse=True):
        sym    = r.get("symbol", "?")
        tf     = r.get("timeframe", "?")
        ftf    = r.get("filter_tf", "")
        strat  = f"{tf}+{ftf}" if ftf else tf
        trades = r.get("total_trades", 0)
        wr     = r.get("win_rate", 0)
        avgw   = r.get("avg_win_pct", 0)
        avgl   = r.get("avg_loss_pct", 0)
        rr     = r.get("avg_rr", 0)
        ev     = r.get("ev_pct", 0)
        pnl    = r.get("total_pnl_pct", 0)
        dd     = r.get("max_drawdown_pct", 0)
        lines.append(
            f"| {sym} | {strat} | {trades} | {wr*100:.1f}% {_flag(wr)} | "
            f"{avgw:+.2f}% | {avgl:+.2f}% | {rr:.2f} | {ev:+.3f}% | "
            f"{pnl:+.2f}% | {dd:+.2f}% |"
        )

    return "\n".join(lines) + "\n"


def build_temporal_breakdown(results: list) -> str:
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Temporal Breakdown — Win Rate by Day & Month",
        f"_Generated: {now}_",
        "",
    ]

    for r in sorted(results, key=lambda x: x.get("ev_pct", 0), reverse=True):
        temporal = r.get("temporal")
        trades   = r.get("trades", [])
        if not temporal and trades:
            temporal = _compute_temporal(trades)
        if not temporal:
            continue

        sym   = r.get("symbol", "?")
        tf    = r.get("timeframe", "?")
        ftf   = r.get("filter_tf", "")
        label = f"{sym} {tf}+{ftf}" if ftf else f"{sym} {tf}"
        ov_wr = r.get("win_rate", 0)
        total = r.get("total_trades", 0)

        lines += [
            f"## {label}",
            f"Overall: **{total} trades**, **{ov_wr*100:.1f}% WR**",
            "",
            "### Day of Week",
            "| Day | Trades | Wins | WR% | PnL% |",
            "|-----|--------|------|-----|------|",
        ]

        by_dow = temporal.get("by_dow", {})
        for dow in DOW_ORDER:
            b  = by_dow.get(dow, {})
            n  = b.get("trades", 0)
            if n == 0:
                continue
            wr  = b.get("win_rate") or 0
            pnl = b.get("pnl", 0.0)
            lines.append(
                f"| {dow} | {n} | {b['wins']} | {wr*100:.1f}% {_flag(wr)} | {pnl:+.2f}% |"
            )

        lines += [
            "",
            "### Monthly",
            "| Month | Trades | Wins | WR% | PnL% |",
            "|-------|--------|------|-----|------|",
        ]

        by_month = temporal.get("by_month", {})
        for month, b in by_month.items():
            n  = b.get("trades", 0)
            if n == 0:
                continue
            wr  = b.get("win_rate") or 0
            pnl = b.get("pnl", 0.0)
            lines.append(
                f"| {month} | {n} | {b['wins']} | {wr*100:.1f}% {_flag(wr)} | {pnl:+.2f}% |"
            )

        lines.append("")

    return "\n".join(lines)


def _compute_temporal(trades: list) -> dict:
    """Compute temporal breakdown from raw trade list (fallback)."""
    try:
        import pandas as pd
    except ImportError:
        return {}

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


# ── Main ──────────────────────────────────────────────────────────────────────

def push_reports(data_dir: str = "/data"):
    token = os.getenv("GITHUB_TOKEN")
    repo  = os.getenv("GITHUB_REPO")

    if not token or not repo:
        log.warning("GITHUB_TOKEN or GITHUB_REPO not set — skipping report push.")
        return

    bt_path = os.path.join(data_dir, "backtest_results.json")
    if not os.path.exists(bt_path):
        log.warning("backtest_results.json not found — skipping report push.")
        return

    with open(bt_path) as f:
        data    = json.load(f)
    results = data.get("results", [])

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary_md  = build_backtest_summary(results)
    temporal_md = build_temporal_breakdown(results)

    _push_file(repo, "reports/backtest_summary.md",    summary_md,
               f"reports: update backtest summary ({now_str})", token)
    _push_file(repo, "reports/temporal_breakdown.md",  temporal_md,
               f"reports: update temporal breakdown ({now_str})", token)

    log.info("✅ Reports pushed to github.com/%s/reports/", repo)


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    cfg      = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
    data_dir = cfg.get("data", {}).get("data_dir", "/data")
    push_reports(data_dir)
