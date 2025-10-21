import asyncio
import logging
import os
from modules.notifier import TelegramNotifier
from modules.config_manager import ConfigManager

# ---------------------------------------------------------------------
# ✅ Logging Setup (Rotating Log File)
# ---------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/run_notifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RunNotifier")

# ---------------------------------------------------------------------
# 🚀 MAIN BOT LOOP
# ---------------------------------------------------------------------
async def main():
    """
    Launches TelegramNotifier as a persistent service.
    Keeps the bot alive indefinitely with automatic reconnection.
    """
    logger.info("🚀 Starting CoPilot Telegram Notifier...")

    # Load configuration
    config = ConfigManager()

    # Retrieve Telegram credentials
    creds = config.get_credentials("telegram")
    bot_token = creds.get("bot_token")
    chat_id = creds.get("chat_id") or os.getenv("ADMIN_CHAT_ID")

    if not bot_token:
        logger.error("❌ Missing Telegram bot token in config or environment.")
        return

    # Initialize notifier
    notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)

    try:
        await notifier.start_bot()
    except KeyboardInterrupt:
        logger.warning("🛑 Bot manually stopped by user.")
    except Exception as e:
        logger.exception(f"❌ Fatal error in notifier: {e}")
    finally:
        logger.info("🔚 TelegramNotifier shutting down gracefully.")

# ---------------------------------------------------------------------
# 🏁 ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("✅ Notifier stopped cleanly.")
