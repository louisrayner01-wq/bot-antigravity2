"""
data_collector.py
═════════════════
Fetches and ACCUMULATES OHLCV data for multiple timeframes from Weex.
Saves to /data/{SYMBOL}_{TIMEFRAME}.csv  (Railway persistent volume).

Timeframes collected : 5m, 15m, 1h, 4h, 1d
Symbols              : BTCUSDT, ETHUSDT, SOLUSDT

Initial fetch (no CSV exists)
──────────────────────────────
On the very first run, the collector paginates backwards from the current
time — fetching 1 000 candles per request, stepping endTime back to the
oldest candle seen so far — until it has collected the full target window
defined in INITIAL_FETCH_DAYS.  This gives every model 2 years of training
data right from the first deployment.

Target windows:
  5m  → 730 days  (~210 240 candles — full training dataset, same as all TFs)
  15m → 730 days  (~70 080 candles)
  1h  → 730 days  (~17 520 candles)
  4h  → 730 days  (~4 380 candles  — BTC/SOL training timeframe)
  1d  → 730 days  (~730 candles)

Append strategy (CSV already exists)
──────────────────────────────────────
Loads the existing file, finds the latest saved timestamp, fetches the
newest batch, and appends only candles that are genuinely new.
The CSVs grow continuously — giving the ML model ever-increasing history.

Refresh rate
────────────
Each timeframe refreshes at its own natural cadence:
  5m  → skip if last candle < 5 min old
  15m → skip if last candle < 15 min old
  1h  → skip if last candle < 60 min old
  4h  → skip if last candle < 4 h old
  1d  → skip if last candle < 24 h old
"""

import os
import time
import logging
import requests
import pandas as pd
from typing import Optional, List

from weex_client import WeexClient

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TIMEFRAMES: List[str] = ["5", "15", "60", "240", "1440"]   # minutes (string keys)

TF_LABELS = {
    "5":    "5m",
    "15":   "15m",
    "60":   "1h",
    "240":  "4h",
    "1440": "1d",
}

# Use plain (spot) symbol format for OHLCV history collection.
# The spot API at /api/v3/market/klines supports endTime pagination so we can
# backfill 2 years of data.  Spot and futures prices for BTC/ETH/SOL are
# effectively identical for ML training purposes.
# _UMCBL symbols are only used for live order execution in weex_client.
SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

REQUEST_DELAY = 0.5   # seconds between requests (rate-limit safety)

# Minimum age of the most-recent candle before we bother fetching again.
TF_REFRESH_MINUTES = {
    "5":    5,
    "15":   15,
    "60":   60,
    "240":  240,
    "1440": 1440,
}

# How many days of history to fetch on the very first run (no CSV exists).
# All timeframes now fetch 2 years so the analysis can freely compare 5m
# against 15m/1h/4h without penalising shorter candles for having less data.
INITIAL_FETCH_DAYS = {
    "5":    730,   # ~210 240 candles in ~211 batches — full 2-year training set
    "15":   730,   # ~70 080 candles  in  ~71 batches
    "60":   730,   # ~17 520 candles  in  ~18 batches
    "240":  730,   # ~4 380 candles   in   ~5 batches
    "1440": 730,   # ~730 candles     in    1 batch
}

# 200 candles per request — conservative limit that works reliably with
# time-bounded queries (startTime / endTime) on the Weex spot API.
BATCH_SIZE = 200


# ── Core collector ────────────────────────────────────────────────────────────

class DataCollector:
    """Fetches and caches multi-timeframe historical OHLCV data from Weex."""

    def __init__(self, client: WeexClient, data_dir: str = "/data"):
        self.client   = client
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def collect_all(self, quiet: bool = False) -> None:
        """
        Collect (or refresh) historical data for every symbol × timeframe
        combination.  Skips any file that is still fresh.

        quiet=True suppresses the header/progress lines — used for the
        background accumulation calls that run every 5 minutes so they
        don't flood the logs.
        """
        total = len(SYMBOLS) * len(TIMEFRAMES)
        done  = 0

        if not quiet:
            logger.info("═══ Data collection starting — %d symbol/timeframe combos ═══", total)

        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    self.collect_one(symbol, tf, quiet=quiet)
                except Exception as exc:
                    # 5m is monitoring-only — degrade quietly if unavailable
                    level = logging.WARNING if tf == "5" else logging.ERROR
                    logger.log(level, "Failed collecting %s %s: %s",
                               symbol, TF_LABELS[tf], exc)
                done += 1
                if not quiet:
                    logger.info("Progress: %d / %d", done, total)

        if not quiet:
            logger.info("═══ Data collection complete ═══")

    def collect_one(self, symbol: str, timeframe_min: str,
                    quiet: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch and APPEND new candles for one symbol/timeframe.

        First run (no CSV): paginates backwards to collect INITIAL_FETCH_DAYS
        of history so the models have real training data immediately.

        Subsequent runs: appends only candles newer than the last saved row.
        """
        label       = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        filepath    = self._filepath(symbol, label)
        refresh_min = TF_REFRESH_MINUTES.get(timeframe_min, 60)

        # ── Load existing data ────────────────────────────────────────────────
        existing: Optional[pd.DataFrame] = None
        if os.path.exists(filepath):
            try:
                existing = pd.read_csv(filepath, parse_dates=["timestamp"])
                if existing.empty:
                    existing = None
            except Exception as exc:
                logger.warning("Could not load %s: %s — will re-fetch.", filepath, exc)
                existing = None

        days = INITIAL_FETCH_DAYS.get(timeframe_min, 730)

        # ── Thin-data check: backfill even when CSV exists but is too short ───
        # If the oldest candle in the CSV is more than 30 days newer than our
        # 2-year target, the data is too thin for good model training.
        # We backfill by paginating backwards and merging with the existing CSV.
        # Applies to ALL timeframes including 5m — 5m now has a full 2-year target.
        if existing is not None and not existing.empty:
            target_start = (pd.Timestamp.now('UTC').tz_localize(None)
                            - pd.Timedelta(days=days))
            oldest_candle = existing["timestamp"].min()
            gap_days = (oldest_candle - target_start).total_seconds() / 86400
            if gap_days > 30:
                logger.info(
                    "  THIN  %s %s — oldest candle %s, target %s (%d days short) — backfilling…",
                    symbol, label,
                    oldest_candle.strftime("%Y-%m-%d"),
                    target_start.strftime("%Y-%m-%d"),
                    int(gap_days),
                )
                raw = self._fetch_history_paginated(symbol, timeframe_min, days)
                if raw:
                    old_df   = self._to_dataframe(raw)
                    combined = pd.concat([old_df, existing], ignore_index=True)
                    combined = (combined
                                .drop_duplicates("timestamp")
                                .sort_values("timestamp")
                                .reset_index(drop=True))
                    combined.to_csv(filepath, index=False)
                    n_added = len(combined) - len(existing)
                    logger.info("  BACKFILL  %s %s  +%d candles → %d total  (%d days)",
                                symbol, label, n_added, len(combined), days)
                    return combined
                else:
                    logger.warning("  Backfill returned no data for %s %s — keeping existing %d candles",
                                   symbol, label, len(existing))

        # ── Freshness check: skip if last candle is recent enough ─────────────
        if existing is not None and not existing.empty:
            last_ts  = existing["timestamp"].max()
            now_utc  = pd.Timestamp.now('UTC').tz_localize(None)
            age_min  = (now_utc - last_ts).total_seconds() / 60
            if age_min < refresh_min:
                log = logger.debug if quiet else logger.info
                log("  FRESH   %s %s  — last candle %.0f min ago (refresh at %d min)",
                    symbol, label, age_min, refresh_min)
                return existing

        # ── First run: paginate backwards to collect full history ─────────────
        if existing is None or existing.empty:
            logger.info("  BACKFILL  %s %s  — fetching %d days of history…",
                        symbol, label, days)
            raw = self._fetch_history_paginated(symbol, timeframe_min, days)
            if not raw:
                logger.warning("  No data returned for %s %s", symbol, label)
                return None
            new_df = self._to_dataframe(raw)
            new_df.to_csv(filepath, index=False)
            logger.info("  SAVED   %s %s  → %d candles  (%d days backfill)",
                        symbol, label, len(new_df), days)
            return new_df

        # ── Subsequent runs: append latest candles ────────────────────────────
        logger.info("  FETCH   %s %s  …", symbol, label)
        raw = self._fetch_latest(symbol, timeframe_min)

        if not raw:
            logger.warning("  No data returned for %s %s", symbol, label)
            return existing

        new_df = self._to_dataframe(raw)

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (combined
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        n_new = len(combined) - len(existing)
        if n_new > 0:
            combined.to_csv(filepath, index=False)
            logger.info("  APPEND  %s %s  +%d new candles → %d total",
                        symbol, label, n_new, len(combined))
        else:
            logger.info("  UP-TO-DATE  %s %s  (%d candles, nothing new)",
                        symbol, label, len(existing))
        return combined

    def load(self, symbol: str, timeframe_min: str) -> Optional[pd.DataFrame]:
        """Load a saved CSV; returns None if it doesn't exist yet."""
        label    = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        filepath = self._filepath(symbol, label)
        if not os.path.exists(filepath):
            return None
        return pd.read_csv(filepath, parse_dates=["timestamp"])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filepath(self, symbol: str, label: str) -> str:
        # Strip exchange suffix so filenames stay clean: BTCUSDT_4h.csv
        clean = symbol.replace("_UMCBL", "").replace("_SPBL", "").replace("_DMCBL", "")
        return os.path.join(self.data_dir, f"{clean}_{label}.csv")

    # Binance public klines API — used for historical backfill only.
    # Weex spot API does not support startTime/endTime pagination, so we use
    # Binance (free, no auth) to get 2 years of OHLCV training data.
    # BTC/ETH/SOL spot prices are effectively identical across exchanges.
    _BINANCE_URL = "https://api.binance.com/api/v3/klines"
    _BINANCE_INTERVAL_MAP = {
        "5": "5m", "15": "15m", "60": "1h", "240": "4h", "1440": "1d",
    }

    def _fetch_history_paginated(self, symbol: str, timeframe_min: str,
                                  days: int) -> List[List]:
        """
        Fetch `days` worth of historical candles by paginating FORWARD from
        (now - days) using Binance's public klines API.

        Binance supports startTime/endTime in milliseconds, returns up to 1000
        candles per request, and provides full multi-year history.

        Returns all candles combined in chronological order (oldest first).
        """
        tf_min          = int(timeframe_min)
        tf_ms           = tf_min * 60 * 1000
        target_start_ms = int((time.time() - days * 86400) * 1000)
        start_time_ms   = target_start_ms
        interval        = self._BINANCE_INTERVAL_MAP.get(timeframe_min, "1h")
        label           = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        # Plain symbol for Binance (strip any exchange suffix)
        clean_symbol    = symbol.replace("_UMCBL","").replace("_SPBL","").replace("_DMCBL","")

        all_candles: List[List] = []
        batch_num = 0

        while True:
            batch_num += 1
            try:
                resp = requests.get(
                    self._BINANCE_URL,
                    params={
                        "symbol":    clean_symbol,
                        "interval":  interval,
                        "startTime": start_time_ms,
                        "limit":     1000,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                raw = resp.json()
            except Exception as exc:
                logger.warning("  Binance fetch error batch %d for %s %s: %s",
                               batch_num, symbol, label, exc)
                break

            if not raw:
                break  # no more data

            # Binance format: [openTime, open, high, low, close, volume, ...]
            candles = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in raw]
            all_candles.extend(candles)

            newest_ts = int(candles[-1][0])

            if batch_num == 1 or batch_num % 10 == 0:
                oldest_dt  = pd.Timestamp(int(candles[0][0]),  unit="ms")
                newest_dt  = pd.Timestamp(newest_ts, unit="ms")
                logger.info("  … batch %d  %s → %s  (%d collected so far)",
                            batch_num,
                            oldest_dt.strftime("%Y-%m-%d"),
                            newest_dt.strftime("%Y-%m-%d"),
                            len(all_candles))

            now_ms = int(time.time() * 1000)
            if newest_ts >= now_ms - tf_ms or len(raw) < 1000:
                break  # reached present or got a partial batch (end of history)

            start_time_ms = newest_ts + tf_ms
            time.sleep(REQUEST_DELAY)

        logger.info("  … Binance pagination complete: %d candles for %s %s",
                    len(all_candles), symbol, label)
        return all_candles

    def _fetch_latest(self, symbol: str, timeframe_min: str) -> List[List]:
        """
        Fetch the most recent batch of candles (no endTime — used for
        appending new candles to an existing CSV).

        Tries progressively smaller limits until one succeeds.
        """
        for limit in [BATCH_SIZE, 100, 50]:
            batch = self.client.get_candles(
                symbol      = symbol,
                granularity = timeframe_min,
                limit       = limit,
            )
            if batch:
                return batch
            time.sleep(REQUEST_DELAY)
        return []

    def _to_dataframe(self, raw: List[List]) -> pd.DataFrame:
        """Convert raw candle list to a clean, deduped, sorted DataFrame."""
        trimmed = [row[:6] for row in raw]
        df = pd.DataFrame(trimmed,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df.dropna()
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
        return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    exc = cfg["exchange"]
    client = WeexClient(
        api_key    = exc.get("api_key", ""),
        api_secret = exc.get("api_secret", ""),
        passphrase = exc.get("passphrase", ""),
        base_url   = exc.get("base_url", "https://api-spot.weex.com"),
    )

    data_dir = cfg.get("data", {}).get("data_dir", "/data")
    collector = DataCollector(client, data_dir=data_dir)
    collector.collect_all()
