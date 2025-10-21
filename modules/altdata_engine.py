import asyncio
import aiohttp
import random
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Gauge
from transformers import pipeline
from modules.logger_setup import setup_logger
from modules.vault_manager import VaultSecretsManager
from bs4 import BeautifulSoup
import contextlib

# --- Prometheus metrics ---
alt_data_success = Counter('alt_data_success_total', 'Successful alternative data fetches', ['type', 'pair'])
alt_data_failure = Counter('alt_data_failure_total', 'Failed alternative data fetches', ['type', 'pair'])
alt_data_duration = Gauge('alt_data_duration_seconds', 'Duration of alternative data fetches', ['type', 'pair'])


class AlternativeDataEngine:
    def __init__(self, data_provider, vault_url=None, vault_token=None, use_mock=False, cache_ttl=300, cleanup_interval=60):
        self.data_provider = data_provider
        self.logger = setup_logger("AltDataEngine", "logs/altdata_engine.log")
        self.vault = VaultSecretsManager(vault_url, vault_token) if not use_mock else None
        self.use_mock = use_mock
        self.alpha_vantage_api_key = None
        self.finnhub_api_key = None
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert") if not use_mock else None
        self.session = None
        self.cache_ttl = cache_ttl
        self.cleanup_interval = cleanup_interval
        self.news_cache = {}
        self.events_cache = {}
        self._cleanup_task = None

        if not use_mock:
            asyncio.create_task(self._load_api_keys())

    # --- Context Manager ---
    async def __aenter__(self):
        if not self.use_mock:
            self.session = aiohttp.ClientSession()
            self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        if self.session:
            await self.session.close()

    # --- Load API keys from Vault ---
    async def _load_api_keys(self):
        try:
            secrets = await self.vault.get_secret("secret/data/apis")
            self.alpha_vantage_api_key = secrets.get("alpha_vantage", {}).get("api_key")
            self.finnhub_api_key = secrets.get("finnhub", {}).get("api_key")
            self.logger.info({"event": "load_api_keys_success"})
        except Exception as e:
            self.logger.error({"event": "load_api_keys_error", "error": str(e)})
            raise

    # --- Cache cleanup loop ---
    async def _cache_cleanup_loop(self):
        while True:
            now = datetime.utcnow().timestamp()
            for cache_dict in [self.news_cache, self.events_cache]:
                expired_keys = [key for key, (ts, _) in cache_dict.items() if now - ts > self.cache_ttl]
                for key in expired_keys:
                    del cache_dict[key]
            await asyncio.sleep(self.cleanup_interval)

    # --- Public API: Get edge signals ---
    async def get_edge_signals(self, pairs: list, session_name: str = None) -> pd.DataFrame:
        if self.use_mock:
            return self._get_mock_edge_signals(pairs)

        try:
            start_time = asyncio.get_event_loop().time()
            news_data, event_data = await asyncio.gather(
                self.fetch_news_sentiment(pairs, session_name),
                self.fetch_economic_events(pairs, session_name)
            )
            rows = []
            for pair in pairs:
                news_sentiment = sum(item["sentiment"] for item in news_data.get(pair, [])) / max(len(news_data.get(pair, [])), 1)
                event_impact = sum(
                    1 if item["impact"] == "high" else 0.5 if item["impact"] == "medium" else 0.1
                    for item in event_data.get(pair, [])
                ) / max(len(event_data.get(pair, [])), 1)
                edge_score = (0.6 * news_sentiment + 0.4 * event_impact) * 100
                rows.append({
                    'pair': pair,
                    'news_sentiment': news_sentiment,
                    'event_impact': event_impact,
                    'edge_score': round(edge_score, 2),
                    'timestamp': datetime.now(pytz.UTC).isoformat(),
                    'session': session_name
                })
            result = pd.DataFrame(rows)
            alt_data_duration.labels(type='edge_signals', pair='all').set(asyncio.get_event_loop().time() - start_time)
            self.logger.info({"event": "get_edge_signals_success", "pairs": pairs, "session": session_name})
            return result
        except Exception as e:
            self.logger.error({"event": "get_edge_signals_error", "pairs": pairs, "session": session_name, "error": str(e)})
            return pd.DataFrame()

    # --- Mock signals ---
    def _get_mock_edge_signals(self, pairs):
        rows = []
        for pair in pairs:
            rows.append({
                'pair': pair,
                'news_sentiment': random.choice([-1, 0, 1]),
                'event_impact': round(random.uniform(0, 1), 2),
                'edge_score': round(random.uniform(0, 100), 2),
                'timestamp': datetime.utcnow().isoformat(),
                'session': 'mock'
            })
        return pd.DataFrame(rows)

    # --- News Sentiment with fallback and batching ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_news_sentiment(self, pairs: list, session_name: str, batch_size: int = 32) -> dict:
        news_context = {}
        current_time = datetime.utcnow().timestamp()
        pairs_to_fetch = []

        for pair in pairs:
            cached = self.news_cache.get(pair)
            if cached and current_time - cached[0] < self.cache_ttl:
                news_context[pair] = cached[1]
            else:
                pairs_to_fetch.append(pair)

        if not pairs_to_fetch:
            return news_context

        all_summaries = []
        pair_mapping = []

        # Primary: Alpha Vantage
        try:
            for pair in pairs_to_fetch:
                symbol = pair.split('/')[0]
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_api_key}"
                async with self.session.get(url) as resp:
                    data = await resp.json()
                    articles = data.get("feed", [])[:5]
                    for article in articles:
                        summary = article.get("summary", "No summary")[:512]
                        all_summaries.append(summary)
                        pair_mapping.append((pair, article))
        except Exception as e:
            self.logger.warning({"event": "alphavantage_failed", "error": str(e)})
            all_summaries = []
            pair_mapping = []

        # Fallback to cache if no summaries
        if not all_summaries:
            self.logger.info("All news sources failed, using cached news if available.")
            for pair in pairs_to_fetch:
                news_context[pair] = self.news_cache.get(pair, (current_time, []))[1]
            return news_context

        # Parallel FinBERT
        total = len(all_summaries)
        batches = [all_summaries[i:i+batch_size] for i in range(0, total, batch_size)]
        batch_results = await asyncio.gather(*[asyncio.to_thread(self.finbert, batch) for batch in batches])
        results = [res for batch in batch_results for res in batch]

        temp_context = {pair: [] for pair in pairs_to_fetch}
        for (pair, article), res in zip(pair_mapping, results):
            sentiment_score = res["score"] * (1 if res["label"] == "positive" else -1)
            temp_context[pair].append({
                "sentiment": sentiment_score,
                "summary": article.get("summary", "No summary")[:100],
                "session": session_name
            })

        for pair in pairs_to_fetch:
            news_context[pair] = temp_context.get(pair, [])
            self.news_cache[pair] = (current_time, news_context[pair])
            alt_data_success.labels(type='news', pair=pair).inc()

        return news_context

    # --- Economic Events with fallback ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_economic_events(self, pairs: list, session_name: str) -> dict:
        event_context = {}
        current_time = datetime.utcnow().timestamp()
        pairs_to_fetch = []

        for pair in pairs:
            cached = self.events_cache.get(pair)
            if cached and current_time - cached[0] < self.cache_ttl:
                event_context[pair] = cached[1]
            else:
                pairs_to_fetch.append(pair)

        if not pairs_to_fetch:
            return event_context

        events = []
        # 1️⃣ Finnhub
        try:
            events = await self._fetch_events_finnhub()
        except Exception as e:
            self.logger.warning({"event": "finnhub_failed", "error": str(e)})

        # 2️⃣ Investing.com (scraping)
        if not events:
            try:
                events = await self._fetch_events_investing()
            except Exception as e:
                self.logger.warning({"event": "investing_failed", "error": str(e)})

        # 3️⃣ ForexFactory (scraping)
        if not events:
            try:
                events = await self._fetch_events_forexfactory()
            except Exception as e:
                self.logger.warning({"event": "forexfactory_failed", "error": str(e)})

        # 4️⃣ Fallback cache
        if not events:
            self.logger.info("All sources failed, using cached events if available.")
            for pair in pairs_to_fetch:
                event_context[pair] = self.events_cache.get(pair, (current_time, []))[1]
            return event_context

        temp_context = {pair: [] for pair in pairs_to_fetch}
        for event in events:
            event_symbols = event.get("symbol", "")
            for pair in pairs_to_fetch:
                base, quote = pair.split('/')
                if base in event_symbols or quote in event_symbols:
                    temp_context[pair].append({
                        "event": event.get("event", "No event"),
                        "impact": event.get("impact", "low"),
                        "session": session_name
                    })
                    alt_data_success.labels(type='events', pair=pair).inc()

        for pair in pairs_to_fetch:
            event_context[pair] = temp_context.get(pair, [])
            self.events_cache[pair] = (current_time, event_context[pair])

        return event_context

    # --- Helper methods for events (production-ready scraping) ---
    async def _fetch_events_finnhub(self):
        to_date = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
        from_date = (datetime.now(pytz.UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/calendar/economic?from={from_date}&to={to_date}&token={self.finnhub_api_key}"
        async with self.session.get(url) as resp:
            data = await resp.json()
            return data.get("economicCalendar", [])[:50]

    async def _fetch_events_investing(self):
        url = "https://www.investing.com/economic-calendar/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with self.session.get(url, headers=headers) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, "lxml")
            events = []
            rows = soup.select("table#economicCalendarData tbody tr")
            for row in rows[:50]:
                try:
                    symbol = row.get("data-event-instrument", "")
                    event_name = row.select_one("td.event").get_text(strip=True)
                    impact_class = row.select_one("td.impact span")["title"].lower()
                    impact = "high" if "high" in impact_class else "medium" if "medium" in impact_class else "low"
                    events.append({"symbol": symbol, "event": event_name, "impact": impact})
                except Exception:
                    continue
            return events

    async def _fetch_events_forexfactory(self):
        url = "https://www.forexfactory.com/calendar"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with self.session.get(url, headers=headers) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, "lxml")
            events = []
            rows = soup.select("table#calendarTable tbody tr.calendar__row")
            for row in rows[:50]:
                try:
                    symbol = row.select_one("td.calendar__currency").get_text(strip=True)
                    event_name = row.select_one("td.calendar__event").get_text(strip=True)
                    impact_class = row.select_one("td.calendar__impact span")["title"].lower()
                    impact = "high" if "high" in impact_class else "medium" if "medium" in impact_class else "low"
                    events.append({"symbol": symbol, "event": event_name, "impact": impact})
                except Exception:
                    continue
            return events