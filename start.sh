#!/bin/sh
# Start dashboard in background, then run the rule-based bot in foreground.
# Railway restarts the whole service if the bot process exits.
#
# To roll back to the ML bot, change `python bot_rules.py` to
# `python multi_runner.py` below and redeploy.
uvicorn dashboard:app --host 0.0.0.0 --port "${PORT:-8080}" &
python bot_rules.py
