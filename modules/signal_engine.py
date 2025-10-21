import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import talib
import pytz
from arch import arch_model
import aiofiles
from prometheus_client import Counter, Gauge

from modules.logger_setup import setup_logger
from modules.altdata_engine import AlternativeDataEngine
from modules.reinforcement_trader import RLTrader

# ----------------------- Prometheus Metrics -----------------------
signal_gen_success = Counter('signal_gen_success_total', 'Successful signal generations', ['pair', 'session'])
signal_gen_failure = Counter('signal_gen_failure_total', 'Failed signal generations', ['pair', 'session'])
signal_gen_duration = Gauge('signal_gen_duration_seconds', 'Duration of signal generations', ['pair', 'session'])

# ----------------------- Logging -----------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = setup_logger("HybridSignalEngine", os.path.join(log_dir, "signal_engine.log"))

# Directory for storing signals
signals_dir = "signals"
os.makedirs(signals_dir, exist_ok=True)

HIGH_IMPACT_EVENTS = [
    "non-farm payroll", "nfp", "interest rate decision", "fomc", "ecb rate", "boj rate",
    "gdp", "cpi", "unemployment rate", "retail sales", "trade balance", "opec meeting",
    "fed minutes", "earnings report", "halving", "etf approval"
]

class HybridSignalEngine:
    def __init__(self, config_manager, data_provider, signal_logger, utils=None, pattern_discovery=None, volatility_model=None):
        self.config_manager = config_manager
        self.data_provider = data_provider
        self.signal_logger = signal_logger
        self.utils = utils
        self.pattern_discovery = pattern_discovery
        self.volatility_model = volatility_model
        self.alt_data_engine = AlternativeDataEngine(data_provider)
        self.rl_trader = RLTrader(data_provider, config_manager)
        self.trading_config = config_manager.get_config("trading")
        self.news_config = config_manager.get_config("news")
        self.signals_dir = signals_dir
        logger.info("HybridSignalEngine initialized")

    # ----------------------- Sentiment -----------------------
    async def fetch_sentiment(self, pair: str) -> float:
        try:
            keywords = self.news_config.get("keywords", ["forex", "market", "economy", "bitcoin", "gold"])
            to_date = datetime.now(pytz.timezone('Africa/Lagos')).strftime("%Y-%m-%d")
            from_date = (datetime.now(pytz.timezone('Africa/Lagos')) - timedelta(days=7)).strftime("%Y-%m-%d")
            articles = await self.config_manager.fetch_news(query=pair.split("/")[0].lower(),
                                                            from_date=from_date, to_date=to_date)
            if not articles:
                return 0.0
            scores = []
            for article in articles:
                text = (article.get("title", "") + " " + article.get("description", "")).lower()
                impact_score = 0.5 if any(kw in text for kw in HIGH_IMPACT_EVENTS) else 0.1
                scores.append(impact_score)
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.error(f"Sentiment fetch failed for {pair}: {e}")
            return 0.0

    # ----------------------- Technical Indicators -----------------------
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            if df.empty or len(df) < self.trading_config.get("slow_ma_period", 50):
                return {"rsi": 0.0, "macd": 0.0, "ma_signal": 0.0}
            close = df["close"].values
            rsi = talib.RSI(close, timeperiod=self.trading_config.get("rsi_period", 14))[-1]
            macd, signal, _ = talib.MACD(
                close,
                fastperiod=self.trading_config.get("fast_ma_period", 10),
                slowperiod=self.trading_config.get("slow_ma_period", 50),
                signalperiod=9
            )
            macd_diff = (macd[-1] - signal[-1]) if not np.isnan(macd[-1]) and not np.isnan(signal[-1]) else 0.0
            fast_ma = talib.SMA(close, timeperiod=self.trading_config.get("fast_ma_period", 10))[-1]
            slow_ma = talib.SMA(close, timeperiod=self.trading_config.get("slow_ma_period", 50))[-1]
            ma_signal = 1 if fast_ma > slow_ma else -1 if fast_ma < slow_ma else 0
            return {"rsi": float(rsi), "macd": float(macd_diff), "ma_signal": float(ma_signal)}
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return {"rsi": 0.0, "macd": 0.0, "ma_signal": 0.0}

    # ----------------------- GARCH Volatility -----------------------
    def calculate_garch_volatility(self, df: pd.DataFrame) -> float:
        try:
            if df.empty or len(df) < 50:
                return 0.0
            returns = df["close"].pct_change().dropna()
            model = arch_model(returns, vol="Garch", p=1, q=1, dist="Normal")
            res = model.fit(disp="off")
            return float(res.conditional_volatility[-1])
        except Exception as e:
            logger.error(f"GARCH volatility failed: {e}")
            return 0.0

    # ----------------------- Async Retry Helper -----------------------
    async def retry_async(self, func, *args, retries=3, delay=5, alert_func=None, **kwargs):
        for attempt in range(1, retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                if attempt == retries and alert_func:
                    await alert_func(f"Function {func.__name__} failed after {retries} attempts: {e}")
                await asyncio.sleep(delay * attempt)
        return None

    # ----------------------- Signal Generation -----------------------
    async def generate_signals(self, pairs: List[str], active_session: str = None, alert_func=None) -> List[Dict[str, Any]]:
        signals = []
        start_time = asyncio.get_event_loop().time()
        try:
            historical_df = await self.data_provider.fetch_historical(
                pairs,
                (datetime.now(pytz.timezone('Africa/Lagos')) - timedelta(days=60)).strftime("%Y-%m-%d"),
                datetime.now(pytz.timezone('Africa/Lagos')).strftime("%Y-%m-%d")
            )
            if historical_df is None or historical_df.empty:
                logger.warning("No historical data for signal generation.")
                return signals

            edge_signals = await self.alt_data_engine.get_edge_signals(pairs, active_session)
            for pair in pairs:
                try:
                    df_pair = historical_df[historical_df["pair"] == pair]
                    if df_pair.empty:
                        continue
                    indicators = self.calculate_technical_indicators(df_pair)
                    volatility = self.calculate_garch_volatility(df_pair)
                    sentiment_score = await self.fetch_sentiment(pair)
                    pattern_signal = self.pattern_discovery.detect_patterns(df_pair) if self.pattern_discovery else {"score": 0.0}
                    vol_signal = self.volatility_model.forecast_volatility(df_pair) if self.volatility_model else {"score": 0.0}

                    # RLTrader action
                    news_sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
                    event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0
                    rsi_series = df_pair['close'].pct_change().rolling(14).mean() * 100
                    state = np.array([df_pair['close'].iloc[-1], rsi_series.iloc[-1], news_sentiment, event_impact])
                    action, strategy = await self.rl_trader.predict(state, df_pair, pair)

                    # Hybrid direction
                    rsi, macd, ma_signal = indicators["rsi"], indicators["macd"], indicators["ma_signal"]
                    rsi_lower, rsi_upper = self.trading_config.get("rsi_lower", 30), self.trading_config.get("rsi_upper", 70)
                    direction = "Hold"
                    if rsi < rsi_lower and macd > 0 and ma_signal > 0:
                        direction = "Buy"
                    elif rsi > rsi_upper and macd < 0 and ma_signal < 0:
                        direction = "Sell"
                    # Override with RL action if present
                    if action == 1:
                        direction = "Buy"
                    elif action == -1:
                        direction = "Sell"

                    hybrid_score = (
                        0.3 * (1 if direction == "Buy" else -1 if direction == "Sell" else 0) +
                        0.25 * (macd / abs(macd) if abs(macd) > 0 else 0) +
                        0.15 * ma_signal +
                        0.15 * sentiment_score +
                        0.15 * ((pattern_signal["score"] + vol_signal["score"]) / 2)
                    )
                    hybrid_score = float(np.clip(hybrid_score, -1.0, 1.0))
                    confidence = float(min(max(0.5 + abs(hybrid_score) * 0.5, 0.5), 1.0))

                    last_close = float(df_pair["close"].iloc[-1])
                    risk_multiplier = {'Low': 0.5, 'Medium': 1.0, 'Aggressive': 1.5}.get(self.config_manager.get('risk_profile', 'Medium'), 1.0)
                    tp = last_close * (1 + 0.04 * risk_multiplier) if direction == "Buy" else last_close * (1 - 0.04 * risk_multiplier)
                    sl = last_close * (1 - 0.02 * risk_multiplier) if direction == "Buy" else last_close * (1 + 0.02 * risk_multiplier)
                    returns = df_pair["close"].pct_change().iloc[-1] if len(df_pair) > 1 else 0.0

                    signal = {
                        "pair": pair,
                        "direction": direction,
                        "hybrid_score": hybrid_score,
                        "confidence": confidence,
                        "sentiment_score": sentiment_score,
                        "volatility": volatility,
                        "rsi": rsi,
                        "macd": macd,
                        "tp": tp,
                        "sl": sl,
                        "close": last_close,
                        "returns": float(returns),
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).isoformat(),
                        "strategy": strategy
                    }

                    if direction != "Hold":
                        signals.append(signal)
                        await self.signal_logger.log_signal(pair, strategy, direction, last_close, None, tp=tp, sl=sl, session=active_session, volatility=volatility)
                        if alert_func:
                            await alert_func(f"New signal for {pair}: {direction} at {last_close:.2f}")
                    signal_gen_success.labels(pair=pair, session=active_session or 'all').inc()

                except Exception as e:
                    logger.error(f"Signal generation failed for {pair}: {e}")
                    signal_gen_failure.labels(pair=pair, session=active_session or 'all').inc()

            signal_gen_duration.labels(pair='all', session=active_session or 'all').set(asyncio.get_event_loop().time() - start_time)
            return signals

        except Exception as e:
            logger.error(f"Overall signal generation error: {e}")
            return []

    # ----------------------- Save Signals -----------------------
    async def save_signals(self, signals: List[Dict[str, Any]], session: str):
        if not signals:
            return
        filename = os.path.join(self.signals_dir, f"{session.lower()}_{datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')}.json")
        try:
            async with aiofiles.open(filename, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(signals, indent=4))
            logger.info(f"Saved {len(signals)} signals to {filename}")
        except Exception as e:
            logger.error(f"Failed to save signals to {filename}: {e}")

    # ----------------------- Merge & Update -----------------------
    def merge_signals_to_master(self, master_filename: str = "signals_master.csv"):
        all_files = [os.path.join(self.signals_dir, f) for f in os.listdir(self.signals_dir) if f.endswith(".csv")]
        if not all_files:
            return
        df_list = []
        for file in all_files:
            try:
                df_list.append(pd.read_csv(file))
            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
        if not df_list:
            return
        master_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["pair", "timestamp"])
        master_path = os.path.join(self.signals_dir, master_filename)
        master_df.to_csv(master_path, index=False)
        logger.info(f"Merged {len(master_df)} signals into master file: {master_path}")

    def update_latest_signals(self, signals: List[Dict[str, Any]], filename: str = "latest_signals.json"):
        if not signals:
            return
        latest_path = os.path.join(self.signals_dir, filename)
        try:
            if os.path.exists(latest_path):
                with open(latest_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = {}
        except Exception as e:
            logger.error(f"Failed to load existing latest signals: {e}")
            existing = {}
        for sig in signals:
            pair = sig["pair"]
            if pair not in existing or sig["timestamp"] > existing[pair]["timestamp"]:
                existing[pair] = sig
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4)
            logger.info(f"Updated latest signals JSON: {latest_path}")
        except Exception as e:
            logger.error(f"Failed to save latest signals JSON: {e}")

    # ----------------------- Scheduler -----------------------
    async def run_scheduler(self, pairs: List[str], interval_minutes: int = 30, delivery_func=None, alert_func=None):
        logger.info(f"Starting HybridSignalEngine scheduler (interval: {interval_minutes} min)")
        while True:
            try:
                now = datetime.now(pytz.timezone('Africa/Lagos'))
                hour = now.hour
                # Determine active session
                if 7 <= hour < 15:
                    session = "London"
                elif 13 <= hour < 21:
                    session = "New_York"
                else:
                    session = "Tokyo"
                logger.info(f"Active session: {session}")

                # Generate signals with retry
                signals = await self.retry_async(
                    self.generate_signals,
                    pairs,
                    active_session=session,
                    retries=3,
                    delay=5,
                    alert_func=alert_func
                )

                if signals:
                    # Save signals JSON
                    await self.save_signals(signals, session)

                    # Save signals CSV for master merge
                    df_signals = pd.DataFrame(signals)
                    csv_filename = os.path.join(self.signals_dir, f"signals_{session}_{now.strftime('%Y%m%d_%H%M%S')}.csv")
                    df_signals.to_csv(csv_filename, index=False)

                    # Merge to master CSV
                    self.merge_signals_to_master()

                    # Update latest signals JSON
                    self.update_latest_signals(signals)

                    # Deliver signals with retry
                    if delivery_func:
                        for sig in signals:
                            await self.retry_async(delivery_func, sig, retries=3, delay=3, alert_func=alert_func)

                    logger.info(f"{len(signals)} signals processed for {session} session.")
                else:
                    logger.info(f"No valid signals generated for {session} session.")

            except Exception as e:
                logger.error(f"Scheduler unexpected error: {e}")
                if alert_func:
                    await alert_func(f"Scheduler unexpected error: {e}")

            logger.info(f"Sleeping for {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)