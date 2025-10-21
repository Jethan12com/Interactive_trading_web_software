import asyncio
import redis.asyncio as redis
from datetime import datetime, timedelta
import pytz
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger

# Prometheus metrics
rate_limit_success = Counter('rate_limit_success_total', 'Successful rate limit checks', ['user_id'])
rate_limit_failure = Counter('rate_limit_failure_total', 'Failed rate limit checks', ['user_id'])
rate_limit_duration = Gauge('rate_limit_duration_seconds', 'Duration of rate limit checks', ['user_id'])

class RateLimiter:
    def __init__(self, redis_url="redis://localhost:6379", max_requests=10, window_seconds=60):
        self.redis = redis.from_url(redis_url)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.logger = setup_logger("RateLimiter", "copilot/modules/logs/rate_limiter.log")
        self.lock = asyncio.Lock()

    async def is_allowed(self, user_id: int, session: str = None) -> bool:
        try:
            start_time = asyncio.get_event_loop().time()
            key = f"rate_limit:{user_id}:{session or 'default'}"
            async with self.lock:
                async with self.redis.pipeline() as pipe:
                    current = await pipe.get(key)
                    pipe.multi()
                    pipe.incr(key)
                    pipe.expire(key, self.window_seconds)
                    results = await pipe.execute()
                count = int(current or 0)
                if count >= self.max_requests:
                    self.logger.warning({"event": "rate_limit_exceeded", "user_id": user_id, "session": session})
                    return False
                rate_limit_success.labels(user_id=user_id).inc()
                rate_limit_duration.labels(user_id=user_id).set(asyncio.get_event_loop().time() - start_time)
                return True
        except Exception as e:
            self.logger.error({"event": "rate_limit_error", "user_id": user_id, "session": session, "error": str(e)})
            rate_limit_failure.labels(user_id=user_id).inc()
            return False