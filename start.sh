#!/bin/sh
# Start dashboard + Portfolio worker in the background, then run the rule-based
# Strat 1 bot in the foreground. Railway monitors the foreground process and
# restarts the whole service if it exits (restartPolicyType = "always").
#
# Layout:
#   uvicorn dashboard:app  → :$PORT   (bot state UI)
#   python  portfolio_bot.py         (Portfolio v1 family, background)
#   python  bot_rules.py             (Strat 1 family, foreground)
#
# To roll back to the ML bot, change `python bot_rules.py` at the bottom to
# `python multi_runner.py` and redeploy.
uvicorn dashboard:app --host 0.0.0.0 --port "${PORT:-8080}" &
python portfolio_bot.py &
python bot_rules.py
