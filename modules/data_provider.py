import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Callable, Optional

import aiohttp
import aiofiles
import pandas as pd
import numpy as np
from cachetools import TTLCache

# Prometheus
from prometheus_client import Counter, Histogram
from binance import AsyncClient, BinanceSocketManager
import ccxt.async_support as ccxt
import MetaTrader5 as mt5
import yfinance as yf
import nasdaqdatalink
from alpha_vantage.foreignexchange import ForeignExchange
import finnhub
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream
from polygon import RESTClient as PolygonRESTClient, WebSocketClient as PolygonWS
from pycoingecko import CoinGeckoAPI

# -----------------------------------------------------------------------------
# Prometheus metrics (with provider label)
# -----------------------------------------------------------------------------
# Tuned histogram buckets (seconds): very small to larger latencies
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

data_fetch_success = Counter(
    'data_fetch_success_total', 'Successful data fetches', ['type', 'pair', 'provider']
)
data_fetch_failure = Counter(
    'data_fetch_failure_total', 'Failed data fetches', ['type', 'pair', 'provider']
)
data_fetch_duration = Histogram(
    'data_fetch_duration_seconds', 'Duration of data fetches', ['type', 'pair', 'provider'], buckets=_BUCKETS
)

# -----------------------------------------------------------------------------
# Helper: small logger setup (replace with your setup_logger)
# -----------------------------------------------------------------------------
def setup_logger(name: str, logfile: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

import logging

# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class MultiProviderDataProvider:
    """
    Multi-provider data fetcher with Prometheus instrumentation.
    Providers included (core): MT5, OANDA, Binance, Kraken, CCXT (selected), Polygon,
    TwelveData, Finnhub, Alpha Vantage, Yahoo Finance (yfinance), CoinGecko, MetalPrice.

    Notes:
    - External provider libs must be installed for actual use (see commented imports).
    - Start a Prometheus server externally in your main program:
        from prometheus_client import start_http_server
        start_http_server(9090)
    """

    def __init__(self, config_manager, credentials: Optional[dict] = None, rate_limit: int = 10):
        self.config_manager = config_manager
        self.credentials = credentials or getattr(config_manager, "get_all_credentials", lambda: {})()
        # vault usage optional in your infra
        try:
            vault_url = config_manager.get_config().get("vault_url", os.getenv("VAULT_URL", "http://localhost:8200"))
            vault_token = config_manager.get_config().get("vault_token", os.getenv("VAULT_TOKEN", None))
            self.vault = VaultSecretsManager(vault_url, vault_token)
        except Exception:
            self.vault = None

        self.logger = setup_logger("DataProvider", "logs/data_provider.log")

        # caches
        self.historical_cache = TTLCache(maxsize=200, ttl=24 * 3600)
        self.live_cache = TTLCache(maxsize=100, ttl=300)

        # concurrency primitives
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.lock = asyncio.Lock()

        # HTTP session (shared)
        self._http_session: Optional[aiohttp.ClientSession] = None

        # provider credentials and clients (only core ones here)
        self.binance_api_key = self.credentials.get("binance", {}).get("api_key")
        self.binance_api_secret = self.credentials.get("binance", {}).get("api_secret")
        self.binance = None
        self.binance_socket_manager = None

        self.kraken_api_key = self.credentials.get("kraken", {}).get("api_key")
        self.kraken_api_secret = self.credentials.get("kraken", {}).get("api_secret")
        self.kraken = None

        self.mt5_credentials = self.credentials.get("mt5", {})
        self.mt5_initialized = False

        self.oanda_api_key = self.credentials.get("oanda", {}).get("api_key")
        self.oanda_account_id = self.credentials.get("oanda", {}).get("account_id")
        self.oanda = None
        if self.oanda_api_key and self.oanda_account_id:
            try:
                self.oanda = oandapyV20.API(access_token=self.oanda_api_key, environment="practice")
            except Exception:
                self.oanda = None

        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY") or (self.vault.get_secret("secret/data/apis").get("alpha_vantage", {}).get("api_key") if self.vault else None)
        self.alpha_vantage_forex = None
        if self.alpha_vantage_api_key:
            try:
                self.alpha_vantage_forex = ForeignExchange(key=self.alpha_vantage_api_key)
            except Exception:
                self.alpha_vantage_forex = None

        self.polygon_api_key = self.credentials.get("polygon", {}).get("api_key")
        self.polygon_rest = None
        self.polygon_ws = None
        if self.polygon_api_key:
            try:
                self.polygon_rest = PolygonRESTClient(api_key=self.polygon_api_key)
                self.polygon_ws = PolygonWS(api_key=self.polygon_api_key, feed="stocks")
            except Exception:
                self.polygon_rest = None
                self.polygon_ws = None

        self.twelvedata_api_key = self.credentials.get("twelvedata", {}).get("api_key")
        self.twelvedata = None
        if self.twelvedata_api_key:
            try:
                self.twelvedata = TDClient(apikey=self.twelvedata_api_key)
            except Exception:
                self.twelvedata = None

        self.finnhub_api_key = self.credentials.get("finnhub", {}).get("api_key")
        self.finnhub = None
        if self.finnhub_api_key:
            try:
                self.finnhub = finnhub.Client(api_key=self.finnhub_api_key)
            except Exception:
                self.finnhub = None

        self.metalprice_api_key = self.credentials.get("metalprice", {}).get("api_key")

        self.coingecko = CoinGeckoAPI()

        # CCXT selected exchanges
        self.ccxt_exchanges = {}
        self.ccxt_credentials = self.credentials.get("ccxt", {})
        for exchange_id in ['coinbase', 'bitfinex', 'kucoin']:
            creds = self.ccxt_credentials.get(exchange_id, {})
            if creds.get("api_key") and creds.get("api_secret"):
                try:
                    self.ccxt_exchanges[exchange_id] = getattr(ccxt, exchange_id)({
                        'apiKey': creds["api_key"],
                        'secret': creds["api_secret"],
                        'enableRateLimit': True
                    })
                except Exception:
                    pass

        # yfinance / nasdaqdatalink setup omitted - assume libs available

        # data directory
        self.data_dir = "copilot/modules/data"
        os.makedirs(self.data_dir, exist_ok=True)

        self.logger.info("MultiProviderDataProvider initialized")

    # -----------------------------
    # HTTP session helpers
    # -----------------------------
    async def _ensure_http_session(self):
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

    async def _close_http_session(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # -----------------------------
    # Metrics helpers
    # -----------------------------
    def _record_success(self, t: str, pair: str, provider: str, elapsed: float):
        try:
            data_fetch_success.labels(type=t, pair=pair, provider=provider).inc()
            data_fetch_duration.labels(type=t, pair=pair, provider=provider).observe(elapsed)
        except Exception:
            self.logger.debug("Metrics record_success failed", exc_info=True)

    def _record_failure(self, t: str, pair: str, provider: str, elapsed: Optional[float] = None):
        try:
            data_fetch_failure.labels(type=t, pair=pair, provider=provider).inc()
            if elapsed is not None:
                data_fetch_duration.labels(type=t, pair=pair, provider=provider).observe(elapsed)
        except Exception:
            self.logger.debug("Metrics record_failure failed", exc_info=True)

    # -----------------------------
    # Provider initializers
    # -----------------------------
    async def initialize_binance(self):
        if self.binance is None and self.binance_api_key and self.binance_api_secret:
            try:
                self.binance = await AsyncClient.create(self.binance_api_key, self.binance_api_secret)
                self.binance_socket_manager = BinanceSocketManager(self.binance)
                self.logger.info("Binance client initialized")
            except Exception as e:
                self.logger.error(f"Binance init failed: {e}")
                self.binance = None
                self.binance_socket_manager = None

    def initialize_kraken(self):
        if self.kraken is None and self.kraken_api_key and self.kraken_api_secret:
            try:
                self.kraken = KrakenAPI(key=self.kraken_api_key, secret=self.kraken_api_secret)
                self.logger.info("Kraken client initialized")
            except Exception as e:
                self.logger.error(f"Kraken init failed: {e}")
                self.kraken = None

    def initialize_mt5(self):
        if not self.mt5_initialized and self.mt5_credentials:
            try:
                mt5.initialize(
                    login=self.mt5_credentials.get("login"),
                    password=self.mt5_credentials.get("password"),
                    server=self.mt5_credentials.get("server")
                )
                self.mt5_initialized = True
                self.logger.info("MT5 initialized")
            except Exception as e:
                self.logger.error(f"MT5 init failed: {e}")
                self.mt5_initialized = False

    # -----------------------------
    # Type detection
    # -----------------------------
    def _is_forex(self, pair: str) -> bool:
        if '/' in pair:
            base = pair.split('/')[0]
            return base.isalpha() and len(base) == 3
        return False

    def _is_commodity(self, pair: str) -> bool:
        commodity_prefixes = {'XAU', 'XAG', 'CL', 'GC', 'SI'}
        if pair.endswith('=F'):
            return True
        if '/' in pair:
            return pair.split('/')[0] in commodity_prefixes
        return pair in commodity_prefixes

    def _is_crypto(self, pair: str) -> bool:
        if pair.endswith("USDT") or pair.endswith("USD"):
            return True
        return '/' in pair and pair.split('/')[0].upper() in {'BTC', 'ETH', 'BNB', 'ADA', 'XRP'}

    def _is_stock(self, pair: str) -> bool:
        return not self._is_forex(pair) and not self._is_commodity(pair) and not self._is_crypto(pair)

    # -----------------------------
    # Fetch historical (instrumented)
    # -----------------------------
    async def fetch_historical(self, pairs: List[str], start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical data across available providers, instrumented with Prometheus.
        Returns concatenated DataFrame with columns: timestamp, pair, open, high, low, close, volume
        """
        cache_key = f"historical_{'-'.join(pairs)}_{start_date}_{end_date}_{interval}"
        if cache_key in self.historical_cache:
            self.logger.info(f"Cache hit: {cache_key}")
            return self.historical_cache[cache_key]

        await self._ensure_http_session()
        data_frames = []

        async def append_and_metric(df: pd.DataFrame, pair: str, provider: str, start_t: float):
            elapsed = time.monotonic() - start_t
            if isinstance(df, pd.DataFrame) and not df.empty:
                data_frames.append(df)
                self._record_success('historical', pair, provider, elapsed)
            else:
                self._record_failure('historical', pair, provider, elapsed)

        try:
            # PROVIDER: MT5 (forex, commodities)
            if self.mt5_credentials:
                self.initialize_mt5()
                provider = "mt5"
                if self.mt5_initialized:
                    for pair in pairs:
                        if (self._is_forex(pair) or self._is_commodity(pair)) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                            start_t = time.monotonic()
                            try:
                                symbol = pair.replace("/", "")
                                rates = mt5.copy_rates_range(
                                    symbol,
                                    mt5.TIMEFRAME_H1 if interval == "1h" else mt5.TIMEFRAME_M1,
                                    datetime.strptime(start_date, '%Y-%m-%d'),
                                    datetime.strptime(end_date, '%Y-%m-%d')
                                )
                                if rates is None:
                                    df = pd.DataFrame()
                                else:
                                    df = pd.DataFrame(rates)
                                    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                                    df["pair"] = pair
                                    df = df.rename(columns={"tick_volume": "volume"})
                                    df = df[["timestamp", "pair", "open", "high", "low", "close", "volume"]]
                                await append_and_metric(df, pair, provider, start_t)
                            except Exception as e:
                                self.logger.warning(f"MT5 fetch failed for {pair}: {e}")
                                self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: OANDA (forex)
            if self.oanda:
                provider = "oanda"
                for pair in pairs:
                    if self._is_forex(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            instrument = pair.replace("/", "_")
                            params = {
                                "from": start_date,
                                "to": end_date,
                                "granularity": "H1" if interval == "1h" else "M1"
                            }
                            req = InstrumentsCandles(instrument=instrument, params=params)
                            res = self.oanda.request(req)
                            candles = res.get("candles", [])
                            if not candles:
                                df = pd.DataFrame()
                            else:
                                df = pd.DataFrame([{
                                    "timestamp": c["time"],
                                    "pair": pair,
                                    "open": float(c["mid"]["o"]),
                                    "high": float(c["mid"]["h"]),
                                    "low": float(c["mid"]["l"]),
                                    "close": float(c["mid"]["c"]),
                                    "volume": float(c.get("volume", 0))
                                } for c in candles])
                                df["timestamp"] = pd.to_datetime(df["timestamp"])
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"OANDA fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: Polygon (stocks)
            if self.polygon_rest:
                provider = "polygon"
                for pair in pairs:
                    if self._is_stock(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            bars = self.polygon_rest.get_aggs(
                                ticker=pair,
                                multiplier=1,
                                timespan="hour" if interval == "1h" else "minute",
                                from_=start_date,
                                to=end_date
                            )
                            if not bars:
                                df = pd.DataFrame()
                            else:
                                df = pd.DataFrame([{
                                    "timestamp": pd.to_datetime(bar.timestamp, unit="ms"),
                                    "pair": pair,
                                    "open": bar.open,
                                    "high": bar.high,
                                    "low": bar.low,
                                    "close": bar.close,
                                    "volume": bar.volume
                                } for bar in bars])
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"Polygon fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: TwelveData (stocks, forex)
            if self.twelvedata:
                provider = "twelvedata"
                for pair in pairs:
                    if (self._is_stock(pair) or self._is_forex(pair)) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            ts = self.twelvedata.time_series(
                                symbol=pair.replace("/", ""),
                                interval=interval,
                                start_date=start_date,
                                end_date=end_date
                            )
                            df = ts.as_pandas()
                            if df is None or df.empty:
                                df = pd.DataFrame()
                            else:
                                df = df.reset_index()
                                df["pair"] = pair
                                df = df.rename(columns={"datetime": "timestamp"})
                                df = df[["timestamp", "pair", "open", "high", "low", "close", "volume"]]
                                df["timestamp"] = pd.to_datetime(df["timestamp"])
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"TwelveData fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: Finnhub (stocks)
            if self.finnhub:
                provider = "finnhub"
                for pair in pairs:
                    if self._is_stock(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
                            data = self.finnhub.stock_candles(pair, '60' if interval == "1h" else '1', start_ts, end_ts)
                            if data.get('s') != 'ok':
                                df = pd.DataFrame()
                            else:
                                df = pd.DataFrame({
                                    "timestamp": pd.to_datetime(data['t'], unit="s"),
                                    "pair": pair,
                                    "open": data['o'],
                                    "high": data['h'],
                                    "low": data['l'],
                                    "close": data['c'],
                                    "volume": data['v']
                                })
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"Finnhub fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: MetalPrice (commodities)
            if self.metalprice_api_key:
                provider = "metalprice"
                for pair in pairs:
                    if self._is_commodity(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            symbol = pair.split('/')[0] if '/' in pair else pair
                            async with self.semaphore:
                                async with self._http_session.get(
                                    "https://api.metalpriceapi.com/v1/timeframe",
                                    params={
                                        "api_key": self.metalprice_api_key,
                                        "base": symbol,
                                        "currencies": "USD",
                                        "start_date": start_date,
                                        "end_date": end_date
                                    }
                                ) as resp:
                                    if resp.status != 200:
                                        df = pd.DataFrame()
                                    else:
                                        payload = await resp.json()
                                        if not payload.get("success"):
                                            df = pd.DataFrame()
                                        else:
                                            rates = payload.get("rates", {})
                                            df = pd.DataFrame([{
                                                "timestamp": pd.to_datetime(date),
                                                "pair": pair,
                                                "open": rate[symbol]["open"],
                                                "high": rate[symbol]["high"],
                                                "low": rate[symbol]["low"],
                                                "close": rate[symbol]["close"],
                                                "volume": 0
                                            } for date, rate in rates.items()])
                                            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"MetalPrice fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: Alpha Vantage (forex fallback)
            if self.alpha_vantage_forex:
                provider = "alpha_vantage"
                for pair in pairs:
                    if self._is_forex(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            from_currency, to_currency = pair.split('/')
                            df_av, _ = self.alpha_vantage_forex.get_currency_exchange_intraday(
                                from_symbol=from_currency, to_symbol=to_currency, interval=interval, outputsize='full'
                            )
                            if df_av is None or df_av.empty:
                                df = pd.DataFrame()
                            else:
                                df = df_av.reset_index().rename(columns={'date': 'timestamp'})
                                df["pair"] = pair
                                df["volume"] = 0
                                # map column names
                                if '1. open' in df.columns:
                                    df = df[["timestamp", "pair", "1. open", "2. high", "3. low", "4. close", "volume"]]
                                    df.columns = ["timestamp", "pair", "open", "high", "low", "close", "volume"]
                                df["timestamp"] = pd.to_datetime(df["timestamp"])
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"Alpha Vantage fetch failed for {pair}: {e}")
                            self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: Yahoo Finance fallback (yfinance)
            provider = "yfinance"
            for pair in pairs:
                if not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                    start_t = time.monotonic()
                    try:
                        ticker = pair.replace("/", "-") if '/' in pair else pair
                        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                        if df is None or df.empty:
                            df = pd.DataFrame()
                        else:
                            df = df.reset_index()
                            df["pair"] = pair
                            df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                            df = df[["timestamp", "pair", "open", "high", "low", "close", "volume"]]
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                        await append_and_metric(df, pair, provider, start_t)
                    except Exception as e:
                        self.logger.warning(f"Yahoo fetch failed for {pair}: {e}")
                        self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # PROVIDER: CoinGecko (crypto fallback)
            provider = "coingecko"
            for pair in pairs:
                if self._is_crypto(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                    start_t = time.monotonic()
                    try:
                        coin = pair.split('/')[0].lower()
                        raw = self.coingecko.get_coin_ohlc_by_id(id=coin, vs_currency='usd', days=30)
                        if not raw:
                            df = pd.DataFrame()
                        else:
                            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
                            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                            df["pair"] = pair
                            df["volume"] = 0
                        await append_and_metric(df, pair, provider, start_t)
                    except Exception as e:
                        self.logger.warning(f"CoinGecko fetch failed for {pair}: {e}")
                        self._record_failure('historical', pair, provider, time.monotonic() - start_t)

            # Combine
            if not data_frames:
                self.logger.error("No historical data fetched")
                return pd.DataFrame()

            combined = pd.concat(data_frames, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values(["pair", "timestamp"]).reset_index(drop=True)

            # persist cache + file async
            self.historical_cache[cache_key] = combined
            try:
                async with self.lock:
                    path = os.path.join(self.data_dir, f"historical_{cache_key}.csv")
                    async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
                        await f.write(combined.to_csv(index=False))
            except Exception:
                self.logger.debug("Failed to write historical csv", exc_info=True)

            self.logger.info(f"Historical fetched & cached: {cache_key}")
            return combined

        except Exception as e:
            self.logger.error(f"fetch_historical failed: {e}", exc_info=True)
            return pd.DataFrame()

    # -----------------------------
    # Fetch live (instrumented)
    # -----------------------------
    async def fetch_live(self, pairs: List[str], interval: str = "1m") -> pd.DataFrame:
        """
        Fetch latest live ticks/last bar across providers. Returns DataFrame.
        """
        cache_key = f"live_{'-'.join(pairs)}_{interval}"
        if cache_key in self.live_cache:
            self.logger.info(f"Cache hit (live): {cache_key}")
            return self.live_cache[cache_key]

        await self._ensure_http_session()
        data_frames = []

        async def append_and_metric(df: pd.DataFrame, pair: str, provider: str, start_t: float):
            elapsed = time.monotonic() - start_t
            if isinstance(df, pd.DataFrame) and not df.empty:
                data_frames.append(df)
                self._record_success('live', pair, provider, elapsed)
            else:
                self._record_failure('live', pair, provider, elapsed)

        try:
            # Example: Binance, CCXT, MetalPrice, TwelveData, AlphaVantage, Yahoo, CoinGecko, Finnhub, MT5, OANDA, Polygon
            # (Implementation mirrors historical but requests latest / tail(1) rows.)
            # For brevity, reuse earlier patterns: implement representative set

            # BINANCE live
            if self.binance_api_key and self.binance_api_secret:
                await self.initialize_binance()
                provider = "binance"
                if self.binance:
                    for pair in pairs:
                        if self._is_crypto(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                            start_t = time.monotonic()
                            try:
                                klines = await self.binance.get_klines(symbol=pair.replace("/", ""), interval=interval, limit=1)
                                if not klines:
                                    df = pd.DataFrame()
                                else:
                                    df = pd.DataFrame(klines, columns=[
                                        "timestamp", "open", "high", "low", "close", "volume",
                                        "close_time", "quote_volume", "trades", "taker_buy_base",
                                        "taker_buy_quote", "ignore"
                                    ])
                                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                                    df["pair"] = pair
                                    df = df[["timestamp", "pair", "open", "high", "low", "close", "volume"]]
                                await append_and_metric(df, pair, provider, start_t)
                            except Exception as e:
                                self.logger.warning(f"Binance live failed for {pair}: {e}")
                                self._record_failure('live', pair, provider, time.monotonic() - start_t)

            # METALPRICE live
            if self.metalprice_api_key:
                provider = "metalprice"
                for pair in pairs:
                    if self._is_commodity(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                        start_t = time.monotonic()
                        try:
                            symbol = pair.split('/')[0] if '/' in pair else pair
                            async with self.semaphore:
                                async with self._http_session.get(
                                    "https://api.metalpriceapi.com/v1/latest",
                                    params={"api_key": self.metalprice_api_key, "base": symbol, "currencies": "USD"}
                                ) as resp:
                                    if resp.status != 200:
                                        df = pd.DataFrame()
                                    else:
                                        payload = await resp.json()
                                        if not payload.get("success"):
                                            df = pd.DataFrame()
                                        else:
                                            df = pd.DataFrame([{
                                                "timestamp": datetime.utcnow(),
                                                "pair": pair,
                                                "open": payload["rates"][symbol]["price"],
                                                "high": payload["rates"][symbol]["price"],
                                                "low": payload["rates"][symbol]["price"],
                                                "close": payload["rates"][symbol]["price"],
                                                "volume": 0
                                            }])
                            await append_and_metric(df, pair, provider, start_t)
                        except Exception as e:
                            self.logger.warning(f"MetalPrice live failed for {pair}: {e}")
                            self._record_failure('live', pair, provider, time.monotonic() - start_t)

            # YAHOO live fallback
            provider = "yfinance"
            for pair in pairs:
                if (self._is_stock(pair) or self._is_commodity(pair)) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                    start_t = time.monotonic()
                    try:
                        ticker = pair.replace("/", "-") if '/' in pair else pair
                        df = yf.download(ticker, period="1d", interval=interval)
                        if df is None or df.empty:
                            df = pd.DataFrame()
                        else:
                            df = df.reset_index()
                            df["pair"] = pair
                            df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                            df = df[["timestamp", "pair", "open", "high", "low", "close", "volume"]].tail(1)
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                        await append_and_metric(df, pair, provider, start_t)
                    except Exception as e:
                        self.logger.warning(f"Yahoo live failed for {pair}: {e}")
                        self._record_failure('live', pair, provider, time.monotonic() - start_t)

            # COINGECKO live
            provider = "coingecko"
            for pair in pairs:
                if self._is_crypto(pair) and not any(not df.empty and df["pair"].iloc[0] == pair for df in data_frames):
                    start_t = time.monotonic()
                    try:
                        coin = pair.split('/')[0].lower()
                        data = self.coingecko.get_price(ids=coin, vs_currencies='usd', include_market_data=True)
                        if data and coin in data:
                            df = pd.DataFrame([{
                                "timestamp": datetime.utcnow(),
                                "pair": pair,
                                "open": data[coin]['usd'],
                                "high": data[coin]['usd'],
                                "low": data[coin]['usd'],
                                "close": data[coin]['usd'],
                                "volume": data[coin].get('usd_volume', 0)
                            }])
                        else:
                            df = pd.DataFrame()
                        await append_and_metric(df, pair, provider, start_t)
                    except Exception as e:
                        self.logger.warning(f"CoinGecko live failed for {pair}: {e}")
                        self._record_failure('live', pair, provider, time.monotonic() - start_t)

            # Additional live providers (MT5, OANDA, Polygon, TwelveData, Finnhub, AlphaVantage) can be added analogously.

            if not data_frames:
                self.logger.error("No live data fetched")
                return pd.DataFrame()

            combined = pd.concat(data_frames, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values(["pair", "timestamp"]).reset_index(drop=True)

            # cache + write file
            self.live_cache[cache_key] = combined
            try:
                async with self.lock:
                    path = os.path.join(self.data_dir, f"live_{cache_key}.csv")
                    async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
                        await f.write(combined.to_csv(index=False))
            except Exception:
                self.logger.debug("Failed to write live csv", exc_info=True)

            self.logger.info(f"Live fetched & cached: {cache_key}")
            return combined

        except Exception as e:
            self.logger.error(f"fetch_live failed: {e}", exc_info=True)
            return pd.DataFrame()

    # -----------------------------
    # Streaming realtime (instrumented)
    # -----------------------------
    async def stream_realtime(self, pairs: List[str], callback: Callable[[str, dict], None]):
        """
        Stream real-time data via WebSocket for supported providers (Binance, Polygon, OANDA).
        The callback receives (pair, point_dict) where point_dict contains timestamp, pair, open, high, low, close, volume.
        Metrics are recorded under type='stream'.
        """
        tasks = []

        # BINANCE WebSocket
        if self.binance_api_key and self.binance_api_secret:
            await self.initialize_binance()
            if self.binance and self.binance_socket_manager:
                for pair in pairs:
                    if self._is_crypto(pair):
                        async def _binance_ws(symbol, pair_label):
                            provider = "binance"
                            try:
                                async with self.binance_socket_manager.trade_socket(symbol) as stream:
                                    while True:
                                        start_t = time.monotonic()
                                        msg = await stream.recv()
                                        if not msg:
                                            self._record_failure('stream', pair_label, provider, time.monotonic() - start_t)
                                            continue
                                        # message parsing depends on client; adapt as necessary
                                        try:
                                            data = msg.get('data') or msg
                                            df = pd.DataFrame([{
                                                "timestamp": pd.to_datetime(data.get('T') or data.get('E'), unit="ms"),
                                                "pair": pair_label,
                                                "open": float(data.get('p')),
                                                "high": float(data.get('p')),
                                                "low": float(data.get('p')),
                                                "close": float(data.get('p')),
                                                "volume": float(data.get('q') or data.get('v') or 0)
                                            }])
                                            callback(pair_label, df.to_dict(orient="records")[0])
                                            elapsed = time.monotonic() - start_t
                                            self._record_success('stream', pair_label, provider, elapsed)
                                        except Exception as ex:
                                            self.logger.debug("Binance stream parse error", exc_info=True)
                                            self._record_failure('stream', pair_label, provider, time.monotonic() - start_t)
                            except Exception as e:
                                self.logger.error(f"Binance WebSocket failed for {pair_label}: {e}")
                        tasks.append(asyncio.create_task(_binance_ws(pair.replace("/", ""), pair)))

        # POLYGON WebSocket
        if self.polygon_ws and self.polygon_api_key:
            for pair in pairs:
                if self._is_stock(pair):
                    async def _polygon_ws(symbol, pair_label):
                        provider = "polygon"
                        try:
                            async def handle_msg(msgs):
                                for msg in msgs:
                                    if msg.get("ev") == "T":  # trade event
                                        start_t = time.monotonic()
                                        try:
                                            df = pd.DataFrame([{
                                                "timestamp": pd.to_datetime(msg["t"], unit="ms"),
                                                "pair": pair_label,
                                                "open": float(msg["p"]),
                                                "high": float(msg["p"]),
                                                "low": float(msg["p"]),
                                                "close": float(msg["p"]),
                                                "volume": float(msg.get("s", 0))
                                            }])
                                            callback(pair_label, df.to_dict(orient="records")[0])
                                            elapsed = time.monotonic() - start_t
                                            self._record_success('stream', pair_label, provider, elapsed)
                                        except Exception:
                                            self._record_failure('stream', pair_label, provider, time.monotonic() - start_t)
                            self.polygon_ws.subscribe([f"T.{symbol}"])
                            await self.polygon_ws.run(handle_msg)
                        except Exception as e:
                            self.logger.error(f"Polygon WebSocket failed for {pair_label}: {e}")
                    tasks.append(asyncio.create_task(_polygon_ws(pair, pair)))

        # OANDA streaming (note: many OANDA libs are sync; adapt if needed)
        if self.oanda:
            for pair in pairs:
                if self._is_forex(pair):
                    async def _oanda_stream(symbol, pair_label):
                        provider = "oanda"
                        try:
                            params = {"instruments": symbol.replace("/", "_")}
                            req = PricingStream(accountID=self.oanda_account_id, params=params)
                            # some OANDA clients yield iterables - adapt to your client
                            for tick in self.oanda.request(req):
                                start_t = time.monotonic()
                                if tick.get("type") == "PRICE":
                                    try:
                                        df = pd.DataFrame([{
                                            "timestamp": pd.to_datetime(tick["time"]),
                                            "pair": pair_label,
                                            "open": float(tick["closeoutBid"]),
                                            "high": float(tick["closeoutBid"]),
                                            "low": float(tick["closeoutBid"]),
                                            "close": float(tick["closeoutAsk"]),
                                            "volume": 0
                                        }])
                                        callback(pair_label, df.to_dict(orient="records")[0])
                                        elapsed = time.monotonic() - start_t
                                        self._record_success('stream', pair_label, provider, elapsed)
                                    except Exception:
                                        self._record_failure('stream', pair_label, provider, time.monotonic() - start_t)
                        except Exception as e:
                            self.logger.error(f"OANDA stream failed for {pair_label}: {e}")
                    tasks.append(asyncio.create_task(_oanda_stream(pair, pair)))

        if not tasks:
            self.logger.error("No WebSocket streams initialized")
            return

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            self.logger.info("Streaming cancelled")
        except Exception as e:
            self.logger.error(f"Streaming error: {e}", exc_info=True)

    # -----------------------------
    # Close / cleanup
    # -----------------------------
    async def close(self):
        try:
            # close clients
            if self.binance:
                try:
                    await self.binance.close_connection()
                except Exception:
                    pass
            # ccxt exchanges
            for exchange in self.ccxt_exchanges.values():
                try:
                    await exchange.close()
                except Exception:
                    pass
            # mt5 shutdown
            if self.mt5_initialized:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            # polygon ws close
            if self.polygon_ws:
                try:
                    self.polygon_ws.close()
                except Exception:
                    pass
            # alpaca/ib or others not included in trimmed version
            await self._close_http_session()
        except Exception as e:
            self.logger.error(f"Error during close: {e}", exc_info=True)