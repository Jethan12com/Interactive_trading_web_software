import asyncio
import aiohttp
import aiofiles
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger
from modules.vault_manager import VaultSecretsManager
from modules.data_provider import MultiProviderDataProvider


# Prometheus metrics
fetch_ops_success = Counter('fetch_ops_success_total', 'Successful data fetches', ['type', 'pair'])
fetch_ops_failure = Counter('fetch_ops_failure_total', 'Failed data fetches', ['type', 'pair'])
fetch_ops_duration = Gauge('fetch_ops_duration_seconds', 'Duration of data fetches', ['type', 'pair'])


class BackgroundFetcher:
    """
    Unified Background Fetcher with Prometheus metrics, Vault-based secrets,
    and integrated MultiProviderDataProvider for live data + economic events.
    """

    def __init__(self, data_provider: MultiProviderDataProvider, config_manager=None, vault_url=None, vault_token=None, interval: int = 60):
        self.data_provider = data_provider
        self.config = config_manager
        self.vault = VaultSecretsManager(vault_url, vault_token) if vault_url and vault_token else None
        self.logger = setup_logger("BackgroundFetcher", "logs/background_fetcher.log")
        self.data_dir = "data"
        self.interval = interval
        self.running = False
        self.task = None
        self.lock = asyncio.Lock()

    # ====================== FETCHING MARKET DATA ======================
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_market_data(self, pairs: list, session: str) -> pd.DataFrame:
        """Fetch market data asynchronously from API or data provider."""
        start_time = asyncio.get_event_loop().time()
        try:
            data = await self.data_provider.fetch_live(pairs)
            fetch_ops_duration.labels(type='market', pair='all').set(asyncio.get_event_loop().time() - start_time)
            for pair in pairs:
                fetch_ops_success.labels(type='market', pair=pair).inc()
            return data
        except Exception as e:
            self.logger.error({"event": "fetch_market_error", "pairs": pairs, "session": session, "error": str(e)})
            for pair in pairs:
                fetch_ops_failure.labels(type='market', pair=pair).inc()
            return pd.DataFrame()

    # ====================== FETCHING ECONOMIC CALENDAR ======================
    async def fetch_economic_calendar(self, pairs: list, session: str) -> pd.DataFrame:
        """Fetch global economic calendar events and cache them."""
        start_time = asyncio.get_event_loop().time()
        if not self.vault:
            self.logger.warning("Vault not configured — skipping economic calendar fetch.")
            return pd.DataFrame()

        try:
            secrets = await self.vault.get_secret("secret/data/apis")
            async with aiohttp.ClientSession() as client:
                url = f"https://finnhub.io/api/v1/calendar/economic?token={secrets.get('finnhub', {}).get('api_key')}"
                async with client.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            events = [
                {"pair": pair, "event": item["event"], "impact": item["impact"], "session": session, "timestamp": item["date"]}
                for pair in pairs for item in data.get("economicCalendar", [])
                if pair.split('/')[0] in item.get("symbol", "") or pair.split('/')[1] in item.get("symbol", "")
            ]
            df = pd.DataFrame(events)

            async with aiofiles.open(f"{self.data_dir}/economic_calendar_{session}.csv", mode='w', encoding='utf-8') as f:
                await f.write(df.to_csv(index=False))

            fetch_ops_success.labels(type='calendar', pair='all').inc()
            fetch_ops_duration.labels(type='calendar', pair='all').set(asyncio.get_event_loop().time() - start_time)
            return df

        except Exception as e:
            self.logger.error({"event": "fetch_calendar_error", "pairs": pairs, "session": session, "error": str(e)})
            fetch_ops_failure.labels(type='calendar', pair='all').inc()
            return pd.DataFrame()

    # ====================== BACKGROUND LOOP ======================
    async def fetch_loop(self, pairs: list, callback, session: str = None):
        """Continuously fetch data and trigger callback on updates."""
        self.running = True
        self.logger.info(f"Starting fetch loop for {pairs}")

        while self.running:
            try:
                async with self.lock:
                    market_data, calendar_data = await asyncio.gather(
                        self.fetch_market_data(pairs, session),
                        self.fetch_economic_calendar(pairs, session)
                    )

                    if not market_data.empty:
                        for pair in pairs:
                            symbol_data = market_data[market_data["pair"] == pair]
                            if not symbol_data.empty:
                                callback(pair, symbol_data)

                    if not calendar_data.empty:
                        self.logger.info(f"Fetched {len(calendar_data)} economic events for session: {session}")

                await asyncio.sleep(self.interval)

            except Exception as e:
                self.logger.error(f"Fetch loop error: {e}")
                await asyncio.sleep(self.interval)

    def start(self, pairs: list, callback, session: str = None):
        """Start background fetching in asyncio task."""
        if not self.running:
            self.task = asyncio.create_task(self.fetch_loop(pairs, callback, session))
            self.logger.info("BackgroundFetcher started")

    def stop(self):
        """Stop the background fetcher."""
        self.running = False
        if self.task:
            self.task.cancel()
        self.logger.info("BackgroundFetcher stopped")


# ====================== HELPER INDICATORS ======================

def calculate_moving_average(data: pd.DataFrame, window: int) -> pd.Series:
    return data['close'].rolling(window=window).mean()


def calculate_rsi(data: pd.DataFrame, periods: int = 14) -> pd.Series:
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))