import os
import asyncio
import aiohttp
import aiofiles
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Gauge
import atexit

from modules.logger_setup import setup_logger
from modules.vault_manager import VaultSecretsManager
from modules.altdata_engine import AlternativeDataEngine
from modules.config_manager import ConfigManager

# Prometheus metrics
sentiment_ops_success = Counter('sentiment_ops_success_total', 'Successful sentiment operations', ['operation', 'pair'])
sentiment_ops_failure = Counter('sentiment_ops_failure_total', 'Failed sentiment operations', ['operation', 'pair'])
sentiment_ops_duration = Gauge('sentiment_ops_duration_seconds', 'Duration of sentiment operations', ['operation', 'pair'])

MAX_ROWS_PER_PAIR = 5000  # Max rows per pair in memory & CSV
FLUSH_INTERVAL = 60  # seconds


class SentimentAnalyzer:
    """Production-ready sentiment analyzer with in-memory cache, async fetching, VADER/FinBERT, event impact, and robust CSV management."""

    def __init__(self, config_manager: ConfigManager, vault_url: str = None, vault_token: str = None):
        self.config = config_manager
        self.vault = VaultSecretsManager(vault_url, vault_token)
        self.alt_data_engine = AlternativeDataEngine(None, vault_url, vault_token)
        self.vader = SentimentIntensityAnalyzer()

        self.finbert_enabled = config_manager.get_config("news").get("finbert_enabled", True)
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert") if self.finbert_enabled else None

        # Logging setup
        os.makedirs("logs", exist_ok=True)
        self.logger = setup_logger("SentimentAnalyzer", "logs/sentiment_analyzer.log")

        # CSV storage and cache
        self.sentiment_file = "logs/sentiment.csv"
        self.cache = {}  # {pair: DataFrame of latest sentiment rows}

        # Initialize
        asyncio.create_task(self._init_sentiment_file())
        asyncio.create_task(self._load_cache_from_csv())
        asyncio.create_task(self._periodic_flush_task())

        # Auto-flush on exit
        atexit.register(lambda: asyncio.run(self._flush_cache_on_exit()))

    # ------------------------ INITIALIZATION ------------------------
    async def _init_sentiment_file(self):
        """Initialize CSV with headers if missing."""
        if not os.path.exists(self.sentiment_file):
            async with aiofiles.open(self.sentiment_file, mode='w', encoding='utf-8') as f:
                headers = ["pair", "timestamp", "vader_score", "finbert_score", "aggregate_score", "source_count", "session"]
                await f.write(','.join(headers) + '\n')
            self.logger.info({"event": "init_sentiment_file_success"})

    async def _load_cache_from_csv(self):
        """Load the last MAX_ROWS_PER_PAIR rows per pair into memory."""
        if os.path.exists(self.sentiment_file):
            df = pd.read_csv(self.sentiment_file)
            for pair, group in df.groupby("pair"):
                self.cache[pair] = group.sort_values("timestamp").tail(MAX_ROWS_PER_PAIR)
            self.logger.info({"event": "cache_loaded", "pairs": list(self.cache.keys())})

    async def _periodic_flush_task(self):
        """Flush all cached pairs to CSV periodically."""
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            try:
                await self._flush_all_to_csv()
            except Exception as e:
                self.logger.error({"event": "periodic_flush_error", "error": str(e)})

    async def _flush_cache_on_exit(self):
        """Flush all cached sentiment data to CSV on process exit."""
        try:
            await self._flush_all_to_csv()
            self.logger.info({"event": "auto_flush_on_exit_success", "pairs": list(self.cache.keys())})
        except Exception as e:
            self.logger.error({"event": "auto_flush_on_exit_error", "error": str(e)})

    async def _flush_all_to_csv(self):
        """Flush the entire cache to CSV asynchronously."""
        if not self.cache:
            return
        async with aiofiles.open(self.sentiment_file, mode='r+', encoding='utf-8') as f:
            f.seek(0)
            df_existing = pd.read_csv(f) if os.path.exists(self.sentiment_file) else pd.DataFrame()
            # Remove all cached pairs from existing
            df_existing = df_existing[~df_existing["pair"].isin(self.cache.keys())]
            # Combine existing + cache
            combined = pd.concat([df_existing] + list(self.cache.values()), ignore_index=True)
            combined = combined.sort_values(["pair", "timestamp"])
            f.seek(0)
            await f.write(combined.to_csv(index=False))
            await f.truncate()
        self.logger.info({"event": "flush_all_to_csv_success", "pairs": list(self.cache.keys())})

    async def _flush_pair_to_csv(self, pair: str):
        """Flush a single pair's cached sentiment data to the CSV."""
        if pair not in self.cache or self.cache[pair].empty:
            return
        async with aiofiles.open(self.sentiment_file, mode='r+', encoding='utf-8') as f:
            f.seek(0)
            df_existing = pd.read_csv(f) if os.path.exists(self.sentiment_file) else pd.DataFrame()
            df_existing = df_existing[df_existing["pair"] != pair]
            df_to_save = pd.concat([df_existing, self.cache[pair]], ignore_index=True)
            df_to_save = df_to_save.sort_values(["pair", "timestamp"])
            f.seek(0)
            await f.write(df_to_save.to_csv(index=False))
            await f.truncate()
        self.logger.info({"event": "flush_pair_to_csv_success", "pair": pair})

    # ------------------------ DATA FETCHERS ------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_social_media_data(self, pair: str, session: str, limit: int = 30) -> pd.DataFrame:
        try:
            async with aiohttp.ClientSession() as client:
                url = f"https://www.reddit.com/search.json?q={pair.split('/')[0]}&limit={limit}"
                async with client.get(url, headers={"User-Agent": "SentimentAnalyzer"}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    posts = [{
                        "text": p["data"]["title"],
                        "timestamp": datetime.utcfromtimestamp(p["data"]["created_utc"]),
                        "session": session
                    } for p in data.get("data", {}).get("children", [])]
                    df = pd.DataFrame(posts)
            return df
        except Exception as e:
            self.logger.error({"event": "fetch_social_error", "pair": pair, "session": session, "error": str(e)})
            return pd.DataFrame()

    def fetch_news_data_light(self, pair: str, days: int = 1) -> pd.DataFrame:
        """Fallback lightweight news fetch without transformers."""
        try:
            end_date = datetime.utcnow()
            df = pd.DataFrame([{"text": f"Simulated headline for {pair.split('/')[0]}", "timestamp": end_date}])
            return df
        except Exception as e:
            self.logger.warning(f"⚠️ Failed lightweight news fetch for {pair}: {e}")
            return pd.DataFrame()

    # ------------------------ SENTIMENT ANALYSIS ------------------------
    def analyze_vader(self, text: str) -> float:
        try:
            return float(self.vader.polarity_scores(str(text))["compound"])
        except Exception:
            return 0.0

    def analyze_finbert(self, text: str) -> float:
        if not self.finbert:
            return 0.0
        try:
            truncated = text[:512]
            result = self.finbert(truncated)[0]
            return result["score"] * (1 if result["label"].lower() == "positive" else -1)
        except Exception:
            return 0.0

    async def get_sentiment(self, pair: str, session: str) -> pd.DataFrame:
        """Fetch, analyze, update cache with event impact."""
        try:
            social_df, news_df_light, events_df = await asyncio.gather(
                self.fetch_social_media_data(pair, session),
                asyncio.to_thread(self.fetch_news_data_light, pair),
                self.alt_data_engine.fetch_economic_events([pair], session)
            )

            events_df = pd.DataFrame(events_df.get(pair, [])) if events_df else pd.DataFrame()
            if social_df.empty and news_df_light.empty and events_df.empty:
                return pd.DataFrame()

            df = pd.concat([social_df.assign(source="social"), news_df_light.assign(source="news")])
            df["vader_score"] = df["text"].apply(self.analyze_vader)
            df["finbert_score"] = df["text"].apply(self.analyze_finbert) if self.finbert else df["vader_score"]
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("H")
            df["session"] = session

            # Event impact
            event_impact = events_df["impact"].map({"high": 1.0, "medium": 0.5, "low": 0.1}).mean() if not events_df.empty else 0.0

            grouped = df.groupby(["timestamp"]).agg({
                "vader_score": "mean",
                "finbert_score": "mean",
                "text": "count",
                "session": "first"
            }).reset_index()

            grouped["pair"] = pair
            grouped["aggregate_score"] = grouped["finbert_score"] * 0.7 + grouped["vader_score"] * 0.3 + event_impact * 0.1
            grouped["source_count"] = grouped["text"]

            # Update cache
            if pair in self.cache:
                self.cache[pair] = pd.concat([self.cache[pair], grouped], ignore_index=True)
            else:
                self.cache[pair] = grouped

            self.cache[pair] = self.cache[pair].drop_duplicates(subset=["pair", "timestamp"], keep="last")
            self.cache[pair] = self.cache[pair].sort_values("timestamp").tail(MAX_ROWS_PER_PAIR)

            return grouped

        except Exception as e:
            self.logger.error({"event": "get_sentiment_error", "pair": pair, "session": session, "error": str(e)})
            return pd.DataFrame()

    async def get_latest_aggregate_sentiment(self, pair: str, session: str = None) -> float:
        """Return latest aggregate sentiment score from cache."""
        if pair not in self.cache or self.cache[pair].empty:
            return 0.0
        df = self.cache[pair]
        if session:
            df = df[df["session"] == session]
        if df.empty:
            return 0.0
        latest = df.sort_values("timestamp").iloc[-1]["aggregate_score"]
        return float(latest)

    # ------------------------ MANUAL FLUSH METHODS ------------------------
    async def flush_cache_to_csv(self):
        """Manually flush the entire cache to CSV."""
        await self._flush_all_to_csv()

    async def flush_pair_to_csv(self, pair: str):
        """Manually flush a single pair to CSV."""
        await self._flush_pair_to_csv(pair)