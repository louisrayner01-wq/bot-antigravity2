"""
news_calendar.py — Static economic news event calendar.

Defines high-impact USD macro events that can move crypto violently.
The bot uses this to:
  • Block new entries from 60 min before an event until 5 min after
  • Tighten open stops at T-5 min, restore them at T+5 min

Events are stored as UTC datetime strings: "YYYY-MM-DD HH:MM"
Update the EVENTS list monthly — takes ~2 minutes.

High-impact events to track:
  - FOMC rate decisions  (~8/year, always Wednesday ~18:00 UTC)
  - FOMC minutes         (~8/year, always Wednesday ~18:00 UTC, 3 weeks after decision)
  - US CPI               (~12/year, usually Tuesday/Wednesday ~12:30 UTC)
  - US NFP (jobs)        (~12/year, first Friday of month ~12:30 UTC)
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Event calendar ────────────────────────────────────────────────────────────
# Add events as you approach each month. Times are UTC.
# Format: ("LABEL", "YYYY-MM-DD HH:MM")

EVENTS = [
    # ── April 2026 ───────────────────────────────────────────────────────────
    ("FOMC Meeting Minutes",  "2026-04-08 18:00"),
    ("US CPI",                "2026-04-10 12:30"),
    ("US PPI",                "2026-04-14 12:30"),
    ("FOMC Rate Decision",    "2026-04-29 18:00"),

    # ── May 2026 ─────────────────────────────────────────────────────────────
    ("US NFP",                "2026-05-01 12:30"),
    ("FOMC Rate Decision",    "2026-05-07 18:00"),
    ("US CPI",                "2026-05-14 12:30"),

    # ── June 2026 ────────────────────────────────────────────────────────────
    ("US NFP",                "2026-06-05 12:30"),
    ("FOMC Rate Decision",    "2026-06-11 18:00"),
    ("US CPI",                "2026-06-11 12:30"),
]


# ── Blackout windows ──────────────────────────────────────────────────────────
ENTRY_BLOCK_MINUTES  = 60   # block new entries this many minutes before event
STOP_TIGHTEN_MINUTES = 5    # tighten stops this many minutes before event
STOP_RESTORE_MINUTES = 5    # restore stops this many minutes after event
NEWS_TIGHTEN_PCT     = 0.008  # tighten stop to 0.8% from current price


def _parse(event_time_str: str) -> datetime:
    return datetime.strptime(event_time_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def _events_as_dt() -> list:
    return [(label, _parse(ts)) for label, ts in EVENTS]


def entries_blocked(now: datetime = None) -> Optional[str]:
    """
    Returns the event label if new entries should be blocked right now,
    otherwise returns None.
    Entries are blocked from 60 min before until 5 min after the event.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    for label, event_dt in _events_as_dt():
        window_open  = event_dt - timedelta(minutes=ENTRY_BLOCK_MINUTES)
        window_close = event_dt + timedelta(minutes=STOP_RESTORE_MINUTES)
        if window_open <= now < window_close:
            return label
    return None


def stops_should_tighten(now: datetime = None) -> Optional[str]:
    """
    Returns the event label if open stops should be tightened right now,
    otherwise returns None.
    Stops are tightened from 5 min before until 5 min after the event.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    for label, event_dt in _events_as_dt():
        window_open  = event_dt - timedelta(minutes=STOP_TIGHTEN_MINUTES)
        window_close = event_dt + timedelta(minutes=STOP_RESTORE_MINUTES)
        if window_open <= now < window_close:
            return label
    return None


def next_event(now: datetime = None) -> Optional[tuple]:
    """
    Returns (label, minutes_until) for the nearest upcoming event, or None.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    upcoming = [
        (label, dt) for label, dt in _events_as_dt() if dt > now
    ]
    if not upcoming:
        return None
    label, dt = min(upcoming, key=lambda x: x[1])
    minutes_until = (dt - now).total_seconds() / 60
    return label, round(minutes_until, 1)
