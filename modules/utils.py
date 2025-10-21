import asyncio
import aiofiles
import json
import pandas as pd
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger

# Prometheus metrics
utils_ops_success = Counter('utils_ops_success_total', 'Successful utils operations', ['operation'])
utils_ops_failure = Counter('utils_ops_failure_total', 'Failed utils operations', ['operation'])
utils_ops_duration = Gauge('utils_ops_duration_seconds', 'Duration of utils operations', ['operation'])

class Utils:
    def __init__(self):
        self.logger = setup_logger("Utils", "copilot/modules/logs/utils.log")

    async def log_metrics(self, metrics: dict, prefix: str, session: str = None):
        try:
            start_time = asyncio.get_event_loop().time()
            async with aiofiles.open(f"copilot/modules/logs/{prefix}_metrics.json", mode='w', encoding='utf-8') as f:
                metrics.update({"session": session, "timestamp": datetime.now(pytz.UTC).isoformat()})
                await f.write(json.dumps(metrics, indent=4, default=str))
            utils_ops_success.labels(operation='log_metrics').inc()
            utils_ops_duration.labels(operation='log_metrics').set(asyncio.get_event_loop().time() - start_time)
        except Exception as e:
            self.logger.error({"event": "log_metrics_error", "prefix": prefix, "session": session, "error": str(e)})
            utils_ops_failure.labels(operation='log_metrics').inc()

    async def preprocess_finbert_input(self, text: str) -> str:
        try:
            start_time = asyncio.get_event_loop().time()
            processed = text.lower().strip()[:512]
            utils_ops_success.labels(operation='preprocess_finbert').inc()
            utils_ops_duration.labels(operation='preprocess_finbert').set(asyncio.get_event_loop().time() - start_time)
            return processed
        except Exception as e:
            self.logger.error({"event": "preprocess_finbert_error", "error": str(e)})
            utils_ops_failure.labels(operation='preprocess_finbert').inc()
            return text

    async def export_for_dashboard(self, data: dict, filename: str, session: str = None):
        try:
            start_time = asyncio.get_event_loop().time()
            async with aiofiles.open(f"copilot/modules/logs/{filename}", mode='w', encoding='utf-8') as f:
                data.update({"session": session, "timestamp": datetime.now(pytz.UTC).isoformat()})
                await f.write(json.dumps(data, indent=4, default=str))
            utils_ops_success.labels(operation='export_dashboard').inc()
            utils_ops_duration.labels(operation='export_dashboard').set(asyncio.get_event_loop().time() - start_time)
        except Exception as e:
            self.logger.error({"event": "export_dashboard_error", "filename": filename, "session": session, "error": str(e)})
            utils_ops_failure.labels(operation='export_dashboard').inc()