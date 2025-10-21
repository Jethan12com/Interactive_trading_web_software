import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import asyncio
from modules.pattern_discovery import PatternDiscovery
from modules.sentiment_analyzer import SentimentAnalyzer
import logging

logger = logging.getLogger("TradingEnv")

class TradingEnv(Env):
    def __init__(self, df, data_provider, config_manager, sentiment_weight=0.1, pattern_weight=0.2):
        super().__init__()
        self.df = df.copy()
        self.data_provider = data_provider
        self.config_manager = config_manager
        self.pattern_discovery = PatternDiscovery(data_provider)
        self.sentiment_analyzer = SentimentAnalyzer(config_manager)
        self.sentiment_weight = sentiment_weight
        self.pattern_weight = pattern_weight

        self.action_space = Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.current_step = 0
        self.position = 0
        self.cash = 10000
        self.holdings = 0
        self.total_trades = 0
        self.total_reward = 0
        self.equity = [self.cash]

        self._prepare_indicators()
        self.sentiment_data = asyncio.run(self._preload_sentiments())

    def _prepare_indicators(self):
        try:
            import pandas_ta as ta
            self.df['rsi'] = ta.rsi(self.df['close'], length=14)
            macd = ta.macd(self.df['close'])
            self.df['macd'] = macd['MACD_12_26_9']
            self.df['returns'] = self.df['close'].pct_change().fillna(0)
        except Exception as e:
            logger.error(f"Indicator prep failed: {e}")

    async def _preload_sentiments(self):
        try:
            pair = self.df.get('pair', pd.Series(['unknown'] * len(self.df))).iloc[0]
            sentiment_df = await asyncio.to_thread(
                self.sentiment_analyzer.analyze_sentiment, pair
            )
            return sentiment_df
        except Exception as e:
            logger.error(f"Sentiment preload failed: {e}")
            return pd.DataFrame()

    def _get_sentiment(self, timestamp):
        if self.sentiment_data.empty:
            return 0.0, 0.0
        row = self.sentiment_data[self.sentiment_data['timestamp'] == timestamp]
        if row.empty:
            return 0.0, 0.0
        return row['aggregate_score'].iloc[-1], row['news_count'].iloc[-1]

    def get_all_observations(self):
        """
        Vectorized observation array for all steps: shape (n_steps, obs_dim)
        """
        n_steps = len(self.df)
        obs_array = np.zeros((n_steps, self.observation_space.shape[0]), dtype=np.float32)
        equity_series = np.array([self.cash]*n_steps)
        returns = self.df['returns'].fillna(0).to_numpy()
        sharpe_series = np.zeros(n_steps, dtype=np.float32)
        drawdown_series = np.zeros(n_steps, dtype=np.float32)
        peak = equity_series[0]

        for i in range(n_steps):
            if i > 0:
                equity_series[i] = equity_series[i-1] * (1 + returns[i])
            if i >= 2:
                r = equity_series[:i+1].pct_change()[1:]
                sharpe_series[i] = (r.mean()/r.std()*np.sqrt(252)) if r.std() > 0 else 0
            peak = max(peak, equity_series[i])
            drawdown_series[i] = (peak - equity_series[i])/peak if peak > 0 else 0

            timestamp = self.df.index[i]
            social_score, news_score = self._get_sentiment(timestamp)
            df_slice = self.df.iloc[:i+1]
            candlestick_signal, anomaly_score, divergence_signal = self._get_pattern_signals(df_slice, self.df.get('pair', pd.Series(['unknown'])).iloc[0])

            row = self.df.iloc[i]
            obs_array[i] = np.array([
                row['close'], row['rsi'], row['macd'], social_score,
                row['returns'], sharpe_series[i], candlestick_signal,
                anomaly_score, divergence_signal, news_score
            ], dtype=np.float32)

        return obs_array

    def _get_pattern_signals(self, df_slice, pair):
        try:
            patterns = self.pattern_discovery.detect_candlestick_patterns(df_slice, pair)
            anomalies = self.pattern_discovery.detect_anomalies(df_slice, pair, features=['close','volume'])
            divergences = self.pattern_discovery.detect_divergences(df_slice, pair, indicator='rsi')

            def latest_signal(df, positive='buy'):
                if df.empty: return 0
                latest = df[df['timestamp']==df_slice.index[-1]]
                if latest.empty: return 0
                return latest['confidence'].iloc[0] if latest['signal'].iloc[0]==positive else -latest['confidence'].iloc[0]

            return latest_signal(patterns), latest_signal(anomalies,'anomaly'), latest_signal(divergences)
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return 0,0,0
