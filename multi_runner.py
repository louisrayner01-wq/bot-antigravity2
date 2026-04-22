"""
multi_runner.py — Multi-user entry point for the Fortuna bot engine.

Set FORTUNA_API_URL and BOT_ENGINE_SECRET in Railway env vars.
If FORTUNA_API_URL is not set, falls back to single-user mode using
WEEX_API_KEY / WEEX_API_SECRET from env vars (original behaviour).

Each active user gets their own:
  - WeexClient (their API keys)
  - RiskManager (their capital + state in /data/{user_id}/)
  - Trade reports posted to the Fortuna API dashboard

Shared across all users (loaded once on startup):
  - Candle data / DataCollector
  - ML models / TradingStrategy
  - Analyzer
"""

import os
import time
import logging
import yaml

import fortuna_client
from bot import TradingBot, load_config, setup_logging

logger = logging.getLogger("MultiRunner")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")


def run_multi_user():
    """
    Main loop. Fetches active users every cycle and runs one bot tick per user.
    """
    # Load base config once for logging setup
    base_cfg = load_config(CONFIG_PATH)
    setup_logging(base_cfg)

    logger.info("🚀 Fortuna Multi-User Runner starting...")
    logger.info("   API: %s", os.environ.get("FORTUNA_API_URL", "NOT SET — single-user mode"))

    # Cache of live bot instances — avoids re-running startup/training on every cycle
    # user_id → TradingBot instance
    bot_instances: dict = {}

    loop_interval = base_cfg["trading"].get("loop_interval_s", 900)

    while True:
        cycle_start = time.time()

        # ── Fetch active users from Fortuna API ───────────────────────────────
        active_users = fortuna_client.get_active_users()

        if not active_users:
            # Fall back to single-user mode if API is unavailable or no active users
            logger.info("No active users from API — running in single-user mode")
            active_users = [{"user_id": "", "capital": None}]

        logger.info("🔄 Cycle start — %d active user(s)", len(active_users))

        # ── Remove bot instances for users who deactivated ───────────────────
        active_ids = {u["user_id"] for u in active_users}
        for uid in list(bot_instances.keys()):
            if uid not in active_ids:
                logger.info("User %s deactivated — removing bot instance", uid[:8])
                del bot_instances[uid]

        # ── Run one cycle per active user ─────────────────────────────────────
        for user in active_users:
            user_id = user["user_id"]

            try:
                # Fetch full config (with decrypted API keys) from Fortuna API
                if user_id:
                    user_config = fortuna_client.get_user_config(user_id)
                    if not user_config:
                        logger.warning("Could not fetch config for user %s — skipping", user_id[:8])
                        continue
                else:
                    user_config = None   # single-user mode — keys come from env vars

                # Create bot instance on first run for this user
                if user_id not in bot_instances:
                    logger.info("Creating bot instance for user %s", user_id[:8] if user_id else "single")
                    bot_instances[user_id] = TradingBot(
                        config_path   = CONFIG_PATH,
                        user_id       = user_id,
                        user_override = user_config,
                    )
                    bot_instances[user_id].startup()

                bot = bot_instances[user_id]

                # Update capital in case the user changed it in the dashboard
                if user_config and user_config.get("capital"):
                    bot.cfg["risk"]["initial_capital"] = float(user_config["capital"])

                # Run the trading tick for this user
                bot.tick()

                # Push equity update to dashboard after each tick
                if user_id:
                    fortuna_client.post_equity(user_id, bot.risk.equity, bot.risk.hwm)

            except Exception as exc:
                logger.error("Error running bot for user %s: %s", user_id[:8] if user_id else "single", exc, exc_info=True)
                # Remove broken instance so it gets recreated fresh next cycle
                bot_instances.pop(user_id, None)

        elapsed = time.time() - cycle_start
        sleep_for = max(0, loop_interval - elapsed)
        logger.info("✅ Cycle complete in %.1fs — sleeping %.0fs", elapsed, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    run_multi_user()
