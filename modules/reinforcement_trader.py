import asyncio
import numpy as np
import pandas as pd
import logging
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from prometheus_client import Counter, Gauge
from arch import arch_model

from modules.ml_model import TradingEnv
from modules.utils import Utils
from modules.pattern_discovery import PatternDiscovery
from modules.altdata_engine import AlternativeDataEngine
from modules.logger_setup import setup_logger

# Prometheus metrics
rl_train_success = Counter('rl_train_success_total', 'Successful RL trainings', ['model_type', 'pair'])
rl_train_failure = Counter('rl_train_failure_total', 'Failed RL trainings', ['model_type', 'pair'])
rl_predict_duration = Gauge('rl_predict_duration_seconds', 'Duration of RL predictions', ['model_type', 'pair'])


class RLTrader:
    def __init__(self, data_provider, config_manager, sentiment_weight=0.1, pattern_weight=0.2):
        self.config_manager = config_manager
        self.data_provider = data_provider
        self.alt_data_engine = AlternativeDataEngine(data_provider)
        self.sentiment_weight = sentiment_weight
        self.pattern_weight = pattern_weight
        self.utils = Utils()
        self.pattern_discovery = PatternDiscovery(data_provider)
        self.logger = setup_logger("RLTrader", "logs/rl_trader.log")
        self.env = None
        self.ppo_model = None
        self.dqn_model = None
        self.volatility_cache = {}
        self.model_performance = {}
        self.selected_model = {}
        self.last_evaluation = {}

    # ---------------------
    # Volatility Forecast
    # ---------------------
    async def _forecast_volatility_async(self, df: pd.DataFrame, pair: str) -> float:
        try:
            cache = self.volatility_cache.get(pair, {})
            if cache.get('timestamp') and (pd.Timestamp.now() - cache['timestamp']) < pd.Timedelta(minutes=5):
                return cache['volatility']

            if len(df) < 20:
                volatility = 0.01
            else:
                try:
                    returns = df['close'].pct_change().dropna() * 100
                    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
                    fitted_model = await asyncio.to_thread(model.fit, disp='off', options={'maxiter': 50})
                    volatility = np.sqrt(fitted_model.forecast(horizon=1).variance.values[-1, 0]) / 100
                except Exception:
                    volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1] or 0.01

            self.volatility_cache[pair] = {'volatility': volatility, 'timestamp': pd.Timestamp.now()}
            return volatility
        except Exception as e:
            self.logger.error(f"Volatility forecast failed for {pair}: {e}")
            return 0.01

    def forecast_volatility(self, df: pd.DataFrame, pair: str) -> float:
        """Sync wrapper for volatility forecast."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._forecast_volatility_async(df, pair))
        except RuntimeError:
            return asyncio.run(self._forecast_volatility_async(df, pair))

    # ---------------------
    # Environment
    # ---------------------
    def create_env(self, df: pd.DataFrame, pair: str):
        try:
            df = self.pattern_discovery.get_sentiment_feature(df, pair)
            pattern_signals = self.pattern_discovery.detect_candlestick_patterns(df, pair)
            anomaly_signals = self.pattern_discovery.detect_anomalies(df, pair)
            divergence_signals = self.pattern_discovery.detect_divergences(df, pair)

            df['candlestick_signal'] = 0.0
            df['anomaly_score'] = 0.0
            df['divergence_signal'] = 0.0
            df['combined_sentiment'] = df.get('combined_sentiment', 0.0)

            for _, row in pattern_signals.iterrows():
                df.loc[row['timestamp'], 'candlestick_signal'] = (
                    row['confidence'] if row['signal'] == 'buy' else -row['confidence'] if row['signal'] == 'sell' else 0.0
                )
            for _, row in anomaly_signals.iterrows():
                df.loc[row['timestamp'], 'anomaly_score'] = row['confidence']
            for _, row in divergence_signals.iterrows():
                df.loc[row['timestamp'], 'divergence_signal'] = (
                    row['confidence'] if row['signal'] == 'buy' else -row['confidence'] if row['signal'] == 'sell' else 0.0
                )

            self.env = DummyVecEnv([
                lambda: TradingEnv(df, self.data_provider, self.config_manager, self.sentiment_weight, self.pattern_weight)
            ])
            return self.env
        except Exception as e:
            self.logger.error(f"Failed to create environment for {pair}: {e}", exc_info=True)
            return None

    # ---------------------
    # Training
    # ---------------------
    async def _train_async(self, df: pd.DataFrame, pair: str, total_timesteps=10000):
        try:
            env = self.create_env(df, pair)
            if env is None:
                rl_train_failure.labels(model_type='hybrid', pair=pair).inc()
                return

            volatility = await self._forecast_volatility_async(df, pair)
            ppo_params = self.config_manager.get_config('model').get('ppo', {})
            dqn_params = self.config_manager.get_config('model').get('dqn', {})

            scale_factor = max(0.5, 1 / (1 + volatility * 10))
            ppo_params['learning_rate'] = ppo_params.get('learning_rate', 0.0003) * scale_factor
            dqn_params['learning_rate'] = dqn_params.get('learning_rate', 0.0003) * scale_factor

            self.ppo_model = PPO('MlpPolicy', env, verbose=0, **ppo_params)
            self.dqn_model = DQN('MlpPolicy', env, verbose=0, **dqn_params)

            await asyncio.gather(
                asyncio.to_thread(self.ppo_model.learn, total_timesteps=total_timesteps // 2),
                asyncio.to_thread(self.dqn_model.learn, total_timesteps=total_timesteps // 2)
            )

            self.model_performance[pair] = {'PPO': {'win_rate': 0.0, 'sharpe_ratio': 0.0, 'drawdown': 0.0},
                                            'DQN': {'win_rate': 0.0, 'sharpe_ratio': 0.0, 'drawdown': 0.0}}
            self.selected_model[pair] = 'PPO'
            self.last_evaluation[pair] = pd.Timestamp.min
            rl_train_success.labels(model_type='hybrid', pair=pair).inc()
            self.logger.info(f"Trained hybrid RL models for {pair} with {total_timesteps} timesteps, volatility={volatility:.4f}")
        except Exception as e:
            rl_train_failure.labels(model_type='hybrid', pair=pair).inc()
            self.logger.error(f"Failed to train RL models for {pair}: {e}", exc_info=True)

    def train(self, df: pd.DataFrame, pair: str, total_timesteps=10000):
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._train_async(df, pair, total_timesteps))
        except RuntimeError:
            return asyncio.run(self._train_async(df, pair, total_timesteps))

    # ---------------------
    # Model Evaluation & Selection
    # ---------------------
    def evaluate_model_performance(self, pair: str, trades_df: pd.DataFrame):
        try:
            recent_trades = trades_df[(trades_df['pair'] == pair) & (trades_df['status'] == 'CLOSED')].tail(50)
            if len(recent_trades) < 10:
                self.logger.info(f"Insufficient trades ({len(recent_trades)}) for {pair} to evaluate models")
                return

            metrics = {}
            for model_name, strategy in [('PPO', 'RL_Hybrid_PPO'), ('DQN', 'RL_Hybrid_DQN')]:
                model_trades = recent_trades[recent_trades['strategy'] == strategy]
                if model_trades.empty:
                    metrics[model_name] = {'win_rate': 0.0, 'sharpe_ratio': 0.0, 'drawdown': 1.0}
                    continue

                win_rate = (model_trades['pnl'] > 0).mean()
                returns = model_trades['pnl'] / model_trades['entry_price']
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
                cumulative = (1 + returns).cumprod()
                drawdown = (cumulative.cummax() - cumulative).max() if not cumulative.empty else 1.0

                metrics[model_name] = {
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe_ratio,
                    'drawdown': drawdown
                }

            # Weighted scoring
            scores = {}
            for model_name, vals in metrics.items():
                scores[model_name] = vals['win_rate'] * 0.4 + vals['sharpe_ratio'] * 0.5 - vals['drawdown'] * 0.1

            self.selected_model[pair] = max(scores, key=scores.get)
            self.model_performance[pair] = metrics
            self.last_evaluation[pair] = pd.Timestamp.now()

            self.logger.info(
                f"Model selection for {pair}: {self.selected_model[pair]}, Metrics: {metrics}, Scores: {scores}"
            )
        except Exception as e:
            self.logger.error(f"Failed to evaluate model performance for {pair}: {e}", exc_info=True)
            self.selected_model[pair] = 'PPO'

    # ---------------------
    # Prediction
    # ---------------------
    async def _predict_async(self, state: np.ndarray, df: pd.DataFrame, pair: str, trades_df=None, user_profile=None):
        try:
            start_time = asyncio.get_event_loop().time()
            edge_signals = await self.alt_data_engine.get_edge_signals([pair], user_profile.get('session', None)) if user_profile else pd.DataFrame()
            news_sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
            event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0
            state = np.append(state, [news_sentiment, event_impact])

            if trades_df is not None and (pair not in self.last_evaluation or (pd.Timestamp.now() - self.last_evaluation[pair]) > pd.Timedelta(hours=6)):
                self.evaluate_model_performance(pair, trades_df)

            volatility = await self._forecast_volatility_async(df, pair)
            risk_multiplier = {'Low': 0.5, 'Medium': 1.0, 'Aggressive': 1.5}.get(user_profile.get('risk_profile', 'Medium'), 1.0) if user_profile else 1.0

            strategy = 'RL_Hybrid_PPO'
            action = 0
            selected_model = self.selected_model.get(pair, 'PPO')
            if selected_model == 'PPO' and self.ppo_model:
                action, _ = self.ppo_model.predict(state, deterministic=True)
            elif self.dqn_model:
                action, _ = self.dqn_model.predict(state, deterministic=True)
                strategy = 'RL_Hybrid_DQN'

            if volatility > 0.05:
                action = 0
            action = action * risk_multiplier

            rl_predict_duration.labels(model_type=strategy, pair=pair).set(asyncio.get_event_loop().time() - start_time)
            return action, strategy
        except Exception as e:
            self.logger.error(f"Predict failed for {pair}: {e}", exc_info=True)
            return 0, 'RL_Hybrid_PPO'

    def predict(self, state: np.ndarray, df: pd.DataFrame, pair: str, trades_df=None, user_profile=None):
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._predict_async(state, df, pair, trades_df, user_profile))
        except RuntimeError:
            return asyncio.run(self._predict_async(state, df, pair, trades_df, user_profile))