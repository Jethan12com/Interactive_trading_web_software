import asyncio
import aiohttp
import aiofiles
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger
from modules.vault_manager import VaultSecretsManager
from modules.rate_limiter import RateLimiter
from datetime import datetime
import pytz

# Prometheus metrics
telegram_send_success = Counter('telegram_send_success_total', 'Successful Telegram sends', ['user_id'])
telegram_send_failure = Counter('telegram_send_failure_total', 'Failed Telegram sends', ['user_id'])
telegram_send_duration = Gauge('telegram_send_duration_seconds', 'Duration of Telegram sends', ['user_id'])

class SimpleTelegramSender:
    def __init__(self, config_manager, vault_url=None, vault_token=None):
        self.config = config_manager
        self.vault = VaultSecretsManager(vault_url, vault_token)
        self.logger = setup_logger("TelegramSender", "copilot/modules/logs/telegram_sender.log")
        self.rate_limiter = RateLimiter()
        self.api_key = None
        self.lock = asyncio.Lock()
        asyncio.create_task(self._load_api_key())

    async def _load_api_key(self):
        try:
            secrets = await self.vault.get_secret("secret/data/apis")
            self.api_key = secrets.get("telegram", {}).get("api_key")
            telegram_send_success.labels(user_id='none').inc()
        except Exception as e:
            self.logger.error({"event": "load_api_key_error", "error": str(e)})
            telegram_send_failure.labels(user_id='none').inc()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def send_message(self, telegram_id: int, message: str, session: str):
        try:
            start_time = asyncio.get_event_loop().time()
            if not await self.rate_limiter.is_allowed(telegram_id, session):
                self.logger.warning({"event": "rate_limit_exceeded", "telegram_id": telegram_id, "session": session})
                return False
            async with self.lock:
                async with aiohttp.ClientSession() as client:
                    url = f"https://api.telegram.org/bot{self.api_key}/sendMessage"
                    payload = {"chat_id": telegram_id, "text": message}
                    async with client.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        telegram_send_success.labels(user_id=telegram_id).inc()
                        telegram_send_duration.labels(user_id=telegram_id).set(asyncio.get_event_loop().time() - start_time)
                        await self._export_send_metrics(telegram_id, session, "success")
                        return True
        except Exception as e:
            self.logger.error({"event": "send_message_error", "telegram_id": telegram_id, "session": session, "error": str(e)})
            telegram_send_failure.labels(user_id=telegram_id).inc()
            await self._export_send_metrics(telegram_id, session, "failed")
            return False

    async def _export_send_metrics(self, telegram_id: int, session: str, status: str):
        try:
            metrics = {"telegram_id": telegram_id, "session": session, "status": status, "timestamp": datetime.now(pytz.UTC).isoformat()}
            async with aiofiles.open(f"copilot/modules/logs/telegram_metrics_{telegram_id}.json", mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(metrics, indent=4))
            telegram_send_success.labels(user_id=telegram_id).inc()
        except Exception as e:
            self.logger.error({"event": "export_metrics_error", "telegram_id": telegram_id, "session": session, "error": str(e)})
            telegram_send_failure.labels(user_id=telegram_id).inc()