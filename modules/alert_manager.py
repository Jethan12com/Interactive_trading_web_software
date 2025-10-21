import os
import json
import stat
import threading
import time
from datetime import datetime, timedelta
from cryptography.fernet import Fernet, InvalidToken
import pytz
from modules.logger_setup import setup_logger
from modules.config_manager import ConfigManager
from modules.notifier import TelegramNotifier


class AlertManager:
    """
    Manages alerts for trading signals.
    - Prevents duplicate alerts within a cooldown period.
    - Persists alert state securely (encrypted).
    - Supports Telegram notifications with tier-based messages.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = setup_logger("AlertManager", "logs/alert_manager.log", to_console=True)

        # Telegram channel
        try:
            telegram_creds = config_manager.get_credentials("telegram")
            bot_token = telegram_creds.get("bot_token")
            if not bot_token or bot_token.startswith("your-"):
                raise ValueError("Telegram bot token missing or invalid")
            self.telegram = TelegramNotifier(bot_token=bot_token, config_dir="config")
            self.logger.info("TelegramNotifier initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize TelegramNotifier: {e}")
            self.telegram = None

        # Persistent cache setup
        self.cache_dir = "logs"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path_plain = os.path.join(self.cache_dir, "alert_cache.json")
        self.cache_path_enc = os.path.join(self.cache_dir, "alert_cache.json.enc")
        self.backup_path_enc = os.path.join(self.cache_dir, "alert_cache_backup.json.enc")

        # Encryption key
        self.key_envvar = "ALERTS_CACHE_KEY"
        self.key_file = os.path.join("secrets", "alerts_key.key")
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)

        # Internal alert memory
        self.last_sent = {}
        self.lock = threading.Lock()
        self.min_interval = self.config.get("alerts.throttle_minutes", 60)

        # Initialize crypto
        self._initialize_crypto()

        # Load previous state
        self._load_last_sent()

        # Background cleaner thread (every 3 hours)
        threading.Thread(target=self._background_cleaner, daemon=True).start()

    # ======================================================================
    # === CORE ALERT METHODS ===
    # ======================================================================

    async def send_alert(self, pair: str, signal: dict):
        """Send alerts via Telegram to all users with tier/risk filtering and duplicate prevention."""
        try:
            action = signal.get("direction", "HOLD")
            strategy = signal.get("strategy", "unknown")
            timestamp = datetime.now(pytz.timezone('Africa/Lagos'))

            # Check if recently sent
            if not self._should_send_alert(pair, action):
                return False

            signal["timestamp"] = timestamp.isoformat()
            signal["pair"] = pair
            signal["action"] = action
            signal["strategy"] = strategy

            if not self.telegram:
                self.logger.warning("TelegramNotifier not initialized, skipping alert")
                return False

            users = self.telegram.user_manager.list_users()  # returns list of dicts
            for user in users:
                telegram_id = user.get("telegram_id")
                if not telegram_id:
                    self.logger.warning(f"User {user.get('user_id')} has no Telegram ID, skipping")
                    continue

                # Determine message format based on tier
                tier = user.get("tier", "Free")
                risk_profile = user.get("risk_profile", "Medium")
                user_signal = signal.copy()  # make a copy for user-specific customization

                user_signal["message"] = self._format_signal_message(
                    signal=user_signal, tier=tier, risk_profile=risk_profile
                )

                await self.telegram.send_signal(telegram_id, user_signal)
                self.logger.info(f"Sent alert to user {user.get('user_id')} ({telegram_id}) with tier {tier}")

            # Update alert memory
            with self.lock:
                self.last_sent[pair] = {"direction": action, "timestamp": timestamp}
            self._save_last_sent()

            self.logger.info(f"Alert sent for {pair}: {action}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send alert for {pair}: {e}")
            return False

    # ----------------------------------------------------------------------
    def _format_signal_message(self, signal: dict, tier: str = "Free", risk_profile: str = "Medium") -> str:
        """Format the signal message according to user tier."""
        pair = signal.get('pair', 'UNKNOWN')
        action = signal.get('action', 'HOLD')
        price = signal.get('price', 0)
        strategy = signal.get('strategy', 'unknown')
        timestamp = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
        tp = signal.get('tp', 0)
        sl = signal.get('sl', 0)
        conf = signal.get('confidence', 0)
        sent = signal.get('sentiment_score', 0)

        message = f"📊 Trading Signal: {pair} | {action} ({strategy.upper()})\n"
        if tier in ["Premium", "VIP"]:
            message += f"TP: {tp:.4f} | SL: {sl:.4f}\nConfidence: {conf:.2f}\n"
        if tier == "VIP":
            message += f"Sentiment Score: {sent:.2f}\n"

        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        message += f"---\n{self._get_risk_disclaimer(risk_profile)}"
        return message

    def _get_risk_disclaimer(self, risk_profile):
        disclaimers = {
            'Low': "⚠️ Low risk: Trade conservatively.",
            'Medium': "⚠️ Medium risk: Use stops and appropriate leverage.",
            'Aggressive': "⚠️ High risk: Volatile setup."
        }
        return disclaimers.get(risk_profile, "⚠️ Trade at your own risk.")

    # ----------------------------------------------------------------------
    def _should_send_alert(self, pair, action):
        """Check if we should send an alert based on last sent time."""
        try:
            with self.lock:
                info = self.last_sent.get(pair)
                if not info:
                    return True

                last_action = info.get("direction")
                last_time = info.get("timestamp")
                if isinstance(last_time, str):
                    last_time = datetime.fromisoformat(last_time)

                # Throttle duplicates
                if last_action == action:
                    delta = datetime.now(pytz.timezone('Africa/Lagos')) - last_time
                    if delta.total_seconds() < self.min_interval * 60:
                        self.logger.info(
                            f"Skipping duplicate {action} alert for {pair} "
                            f"({delta.total_seconds()/60:.1f} min ago)"
                        )
                        return False
            return True
        except Exception as e:
            self.logger.warning(f"Throttle check failed: {e}")
            return True

    # ======================================================================
    # === ENCRYPTION MANAGEMENT ===
    # ======================================================================

    def _initialize_crypto(self):
        key_b64 = os.environ.get(self.key_envvar)
        if key_b64:
            self._fernet = Fernet(key_b64.encode())
            self.logger.info("Using encryption key from environment variable")
            return

        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, "rb") as f:
                    key = f.read().strip()
                self._fernet = Fernet(key)
                self.logger.info("Loaded encryption key from key file")
                return
        except Exception as e:
            self.logger.warning(f"Failed loading key file: {e}")

        key = Fernet.generate_key()
        with open(self.key_file, "wb") as f:
            f.write(key)
        try:
            os.chmod(self.key_file, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            self.logger.warning("Failed to set key file permissions")
        self._fernet = Fernet(key)
        self.logger.info(f"Generated new encryption key at {self.key_file}")

    def _encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def _decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)

    # ======================================================================
    # === CACHE MANAGEMENT ===
    # ======================================================================

    def _load_last_sent(self):
        try:
            if os.path.exists(self.cache_path_enc):
                with open(self.cache_path_enc, "rb") as f:
                    encrypted = f.read()
                plain = self._decrypt(encrypted)
                data = json.loads(plain.decode("utf-8"))
                for pair, info in data.items():
                    info["timestamp"] = datetime.fromisoformat(info["timestamp"])
                self.last_sent = data
                self.logger.info(f"Loaded {len(data)} encrypted alert states")
                return

            if os.path.exists(self.cache_path_plain):
                with open(self.cache_path_plain, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for pair, info in data.items():
                    info["timestamp"] = datetime.fromisoformat(info["timestamp"])
                self.last_sent = data
                self.logger.info("Migrating plaintext cache to encrypted store")
                self._save_last_sent()
        except Exception as e:
            self.logger.warning(f"Failed to load alert cache: {e}")
            self.last_sent = {}

    def _rotate_cache_backup(self):
        try:
            if os.path.exists(self.cache_path_enc):
                os.replace(self.cache_path_enc, self.backup_path_enc)
        except Exception as e:
            self.logger.warning(f"Cache rotation failed: {e}")

    def _save_last_sent(self):
        try:
            self._rotate_cache_backup()
            data = {}
            for pair, info in self.last_sent.items():
                ts = info.get("timestamp")
                if isinstance(ts, datetime):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)
                data[pair] = {"direction": info["direction"], "timestamp": ts_str}

            plain = json.dumps(data, indent=2).encode("utf-8")
            enc = self._encrypt(plain)

            tmp_path = self.cache_path_enc + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(enc)
            os.replace(tmp_path, self.cache_path_enc)
            self._prune_old_entries()
        except Exception as e:
            self.logger.error(f"Failed to save encrypted alert cache: {e}")

    def _prune_old_entries(self):
        try:
            ttl_hours = self.config.get("alerts.cache_ttl_hours", 12)
            cutoff = datetime.now(pytz.timezone('Africa/Lagos')) - timedelta(hours=ttl_hours)
            removed = []

            with self.lock:
                for pair, info in list(self.last_sent.items()):
                    ts = info["timestamp"]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if ts < cutoff:
                        removed.append(pair)
                        del self.last_sent[pair]

            if removed:
                self.logger.info(f"Pruned {len(removed)} expired alerts: {removed}")
                self._save_last_sent()
        except Exception as e:
            self.logger.warning(f"Prune failed: {e}")

    # ======================================================================
    # === BACKGROUND CLEANUP ===
    # ======================================================================

    def _background_cleaner(self):
        while True:
            try:
                time.sleep(3 * 3600)
                self._prune_old_entries()
            except Exception:
                self.logger.warning("Background cleaner interrupted")
                pass