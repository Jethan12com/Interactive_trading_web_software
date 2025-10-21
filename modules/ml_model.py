import asyncio
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from stable_baselines3.common.envs import SimpleMultiObsEnv
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger
from modules.altdata_engine import AlternativeDataEngine
from datetime import datetime
import pytz
import json
import aiofiles

# Prometheus metrics
ml_ops_success = Counter('ml_ops_success_total', 'Successful ML operations', ['operation', 'pair'])
ml_ops_failure = Counter('ml_ops_failure_total', 'Failed ML operations', ['operation', 'pair'])
ml_ops_duration = Gauge('ml_ops_duration_seconds', 'Duration of ML operations', ['operation', 'pair'])

class TradingEnv(SimpleMultiObsEnv):
    def __init__(self, df: pd.DataFrame, data_provider, config_manager, session: str = None):
        self.df = df
        self.data_provider = data_provider
        self.config = config_manager
        self.alt_data_engine = AlternativeDataEngine(data_provider)
        self.logger = setup_logger("TradingEnv", "copilot/modules/logs/ml_model.log")
        self.current_step = 0
        self.session = session
        self.observation_space = {'price': 1, 'sentiment': 1, 'event_impact': 1}
        self.action_space = 3  # Buy, Sell, Hold

    async def reset(self):
        self.current_step = 0
        state = await self._get_state()
        return state

    async def _get_state(self):
        try:
            start_time = asyncio.get_event_loop().time()
            pair = self.df['pair'].iloc[0]
            edge_signals = await self.alt_data_engine.get_edge_signals([pair], self.session)
            sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
            event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0
            state = np.array([self.df['close'].iloc[self.current_step], sentiment, event_impact])
            ml_ops_success.labels(operation='get_state', pair=pair).inc()
            ml_ops_duration.labels(operation='get_state', pair=pair).set(asyncio.get_event_loop().time() - start_time)
            return state
        except Exception as e:
            self.logger.error({"event": "get_state_error", "pair": self.df['pair'].iloc[0], "session": self.session, "error": str(e)})
            ml_ops_failure.labels(operation='get_state', pair=self.df['pair'].iloc[0]).inc()
            return np.zeros(3)

class MLModel:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = setup_logger("MLModel", "copilot/modules/logs/ml_model.log")
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0 if torch.cuda.is_available() else -1)
        self.finbert_cache = {}
        self.lock = asyncio.Lock()

    async def predict_sentiment(self, texts: list, pair: str, session: str) -> list:
        try:
            start_time = asyncio.get_event_loop().time()
            async with self.lock:
                results = []
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    cache_key = tuple(batch)
                    if cache_key in self.finbert_cache:
                        results.extend(self.finbert_cache[cache_key])
                    else:
                        outputs = await asyncio.to_thread(self.finbert, batch)
                        scores = [output["score"] * (1 if output["label"] == "positive" else -1) for output in outputs]
                        self.finbert_cache[cache_key] = scores
                        results.extend(scores)
                ml_ops_success.labels(operation='predict_sentiment', pair=pair).inc()
                ml_ops_duration.labels(operation='predict_sentiment', pair=pair).set(asyncio.get_event_loop().time() - start_time)
                return results
        except Exception as e:
            self.logger.error({"event": "predict_sentiment_error", "pair": pair, "session": session, "error": str(e)})
            ml_ops_failure.labels(operation='predict_sentiment', pair=pair).inc()
            return [0.0] * len(texts)

    async def export_metrics(self, pair: str, session: str, metrics: dict):
        try:
            start_time = asyncio.get_event_loop().time()
            async with aiofiles.open(f"copilot/modules/logs/ml_metrics_{pair}_{session}.json", mode='w', encoding='utf-8') as f:
                metrics.update({"pair": pair, "session": session, "timestamp": datetime.now(pytz.UTC).isoformat()})
                await f.write(json.dumps(metrics, indent=4, default=str))
            ml_ops_success.labels(operation='export_metrics', pair=pair).inc()
            ml_ops_duration.labels(operation='export_metrics', pair=pair).set(asyncio.get_event_loop().time() - start_time)
        except Exception as e:
            self.logger.error({"event": "export_metrics_error", "pair": pair, "session": session, "error": str(e)})
            ml_ops_failure.labels(operation='export_metrics', pair=pair).inc()