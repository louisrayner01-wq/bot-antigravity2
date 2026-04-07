#!/bin/sh
# Start dashboard in background, then run the bot in foreground.
# Railway restarts the whole service if bot.py exits.
uvicorn dashboard:app --host 0.0.0.0 --port "${PORT:-8080}" &
python bot.py
