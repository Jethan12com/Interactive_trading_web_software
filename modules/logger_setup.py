import logging
import os
import json
from datetime import datetime
import pytz
import asyncio
from logging.handlers import RotatingFileHandler
from prometheus_client import Counter, Gauge

# Prometheus metrics
log_ops_success = Counter('log_ops_success_total', 'Successful log operations', ['logger_name'])
log_ops_failure = Counter('log_ops_failure_total', 'Failed log operations', ['logger_name'])
log_ops_duration = Gauge('log_ops_duration_seconds', 'Duration of log operations', ['logger_name'])

class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, pytz.timezone('Africa/Lagos')).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "session": getattr(record, 'session', None),
            "symbol": getattr(record, 'symbol', None)
        }
        return json.dumps(log_entry)

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    to_console: bool = True,
    max_bytes: int = 10*1024*1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a logger with JSON formatting, Africa/Lagos timezone, UTF-8 file logging, optional console logging,
    and Prometheus metrics tracking.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # Prevent duplicate handlers

        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional console logging
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Prometheus metrics
        log_ops_success.labels(logger_name=name).inc()
        log_ops_duration.labels(logger_name=name).set(asyncio.get_event_loop().time() - start_time)

        return logger

    except Exception as e:
        logging.error({"event": "setup_logger_error", "logger_name": name, "error": str(e)})
        log_ops_failure.labels(logger_name=name).inc()
        return logging.getLogger(name)