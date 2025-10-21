import asyncio
import aiohttp
import aiofiles
import json
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger
from modules.vault_manager import VaultSecretsManager
from modules.rate_limiter import RateLimiter
from datetime import datetime, timezone
import pytz


# Prometheus metrics
notify_ops_success = Counter('notify_ops_success_total', 'Successful notification operations', ['type'])
notify_ops_failure = Counter('notify_ops_failure_total', 'Failed notification operations', ['type'])
notify_ops_duration = Gauge('notify_ops_duration_seconds', 'Duration of notification operations', ['type'])

class Notifier:
    def __init__(self, config_manager, vault_url=None, vault_token=None):
        self.config = config_manager
        self.vault = VaultSecretsManager(vault_url, vault_token)
        self.logger = setup_logger("Notifier", "copilot/modules/logs/notifier.log")
        self.rate_limiter = RateLimiter()
        self.telegram_api_key = None
        self.lock = asyncio.Lock()
        asyncio.create_task(self._load_api_key())

    async def _load_api_key(self):
        try:
            secrets = await self.vault.get_secret("secret/data/apis")
            self.telegram_api_key = secrets.get("telegram", {}).get("api_key")
            notify_ops_success.labels(type='load_api_key').inc()
        except Exception as e:
            self.logger.error({"event": "load_api_key_error", "error": str(e)})
            notify_ops_failure.labels(type='load_api_key').inc()

    async def send_notification(self, telegram_id: int, message: str, session: str):
        try:
            start_time = asyncio.get_event_loop().time()
            if not await self.rate_limiter.is_allowed(telegram_id):
                self.logger.warning({"event": "rate_limit_exceeded", "telegram_id": telegram_id, "session": session})
                return False
            async with self.lock:
                async with aiohttp.ClientSession() as client:
                    url = f"https://api.telegram.org/bot{self.telegram_api_key}/sendMessage"
                    payload = {"chat_id": telegram_id, "text": message}
                    async with client.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        notify_ops_success.labels(type='send_notification').inc()
                        notify_ops_duration.labels(type='send_notification').set(asyncio.get_event_loop().time() - start_time)
                        return True
        except Exception as e:
            self.logger.error({"event": "send_notification_error", "telegram_id": telegram_id, "session": session, "error": str(e)})
            notify_ops_failure.labels(type='send_notification').inc()
            return False

    async def export_notification_metrics(self, telegram_id: int, session: str, status: str):
        try:
            start_time = asyncio.get_event_loop().time()
            metrics = {"telegram_id": telegram_id, "session": session, "status": status, "timestamp": datetime.now(pytz.UTC).isoformat()}
            async with aiofiles.open(f"copilot/modules/logs/notification_metrics_{telegram_id}.json", mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(metrics, indent=4))
            notify_ops_success.labels(type='export_metrics').inc()
            notify_ops_duration.labels(type='export_metrics').set(asyncio.get_event_loop().time() - start_time)
        except Exception as e:
            self.logger.error({"event": "export_metrics_error", "telegram_id": telegram_id, "session": session, "error": str(e)})
            notify_ops_failure.labels(type='export_metrics').inc()