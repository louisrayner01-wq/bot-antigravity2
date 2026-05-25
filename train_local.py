"""
train_local.py — Train 5m models for all 4 slots using local CSV data.

Usage:
    python train_local.py

Reads CSVs from BOT_BACKTEST_DATA_DIR, saves .joblib files to
./local_models/ ready to be uploaded to Railway's /data/models/.
"""

import os
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = "/Users/Louis/bot-backtest/data"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "local_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLOTS = [
    {"symbol": "BTCUSDT_UMCBL", "name": "BTC", "tf_label": "5m", "csv": "BTCUSDT_5m.csv"},
    {"symbol": "ETHUSDT_UMCBL", "name": "ETH", "tf_label": "5m", "csv": "ETHUSDT_5m.csv"},
    {"symbol": "SOLUSDT_UMCBL", "name": "SOL", "tf_label": "5m", "csv": "SOLUSDT_5m.csv"},
    {"symbol": "XRPUSDT_UMCBL", "name": "XRP", "tf_label": "5m", "csv": "XRPUSDT_5m.csv"},
]

# Minimal config so TradingStrategy initialises without Railway env
import yaml
cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
cfg["logging"]["models_dir"] = OUTPUT_DIR
cfg["logging"]["trades_file"] = "/tmp/trades_train_local.csv"

sys.path.insert(0, os.path.dirname(__file__))
from strategy import TradingStrategy

strategy = TradingStrategy(cfg)

for slot in SLOTS:
    csv_path = os.path.join(DATA_DIR, slot["csv"])
    log.info("Loading %s...", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    log.info("  %d rows — training %s [%s]", len(df), slot["name"], slot["tf_label"])
    strategy.train(df, symbol=slot["symbol"], timeframe_label=slot["tf_label"])
    log.info("  ✅ %s [%s] done", slot["name"], slot["tf_label"])

log.info("\nAll models saved to: %s", OUTPUT_DIR)
log.info("Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    log.info("  %s  (%.1f MB)", f, size / 1e6)
log.info("\nNext: upload these to Railway /data/models/")
