#!/bin/sh
# Start dashboard in background, then run the multi-user bot in foreground.
# Railway restarts the whole service if multi_runner.py exits.
uvicorn dashboard:app --host 0.0.0.0 --port "${PORT:-8080}" &
python multi_runner.py
