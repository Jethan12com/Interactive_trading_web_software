import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import vectorbt as vbt
from tqdm import tqdm
try:
    from tqdm.asyncio import tqdm_asyncio
except Exception:
    tqdm_asyncio = None  # fallback: will use asyncio.gather

from .logger_setup import setup_logger  # your project logger helper
from .indicators import get_rsi, get_macd, get_atr  # keep these if implemented
from .journal_evaluator import JournalEvaluator
from .ml_model import MLModel

# ---------------------------------------------------------------------
# Minimal placeholder helpers (replace with your real implementations)
# ---------------------------------------------------------------------
class DataCache:
    """Simple DataCache placeholder. Replace with a real cache + fetch logic."""
    def __init__(self, data_provider):
        self.data_provider = data_provider

    async def get_historical_data(self, pairs: List[str], start_date, end_date) -> pd.DataFrame:
        """
        Expected return format:
        DataFrame with columns ['pair','timestamp','open','high','low','close','volume', ...]
        """
        # naive implementation: fetch each pair sequentially and concat
        frames = []
        for pair in pairs:
            df = await self.data_provider.fetch_historical([pair], start_date, end_date)
            if df is None:
                continue
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

class StrategyExecutor:
    """
    Wrapper holding ML model and any strategy helper methods.
    Replace .ml_model with your real ML model object that implements the interface used below.
    """
    def __init__(self, config_manager, data_provider):
        self.config = config_manager
        self.data_provider = data_provider
        self.ml_model = MLModel(config_manager)  # ensure MLModel.__init__ matches

class MetricsCalculator:
    """Placeholder. Replace with robust metrics calculation code (Sharpe, PnL, drawdown etc.)."""
    def calculate_metrics(self, portfolio: Optional[vbt.Portfolio], rewards=None) -> Dict[str, Any]:
        if portfolio is None:
            return {'error': 'No portfolio'}
        try:
            stats = portfolio.stats()
            # map to a simpler dict
            return {
                'Total PnL': float(stats.get('Total Return [%]', np.nan)) if 'Total Return [%]' in stats else float(stats.get('Total PnL', np.nan)),
                'Sharpe Ratio': float(stats.get('Sharpe Ratio', np.nan)),
                'Max Drawdown [%]': float(stats.get('Max Drawdown [%]', np.nan)),
                'Win Rate [%]': float(stats.get('Win Rate [%]', np.nan)) if 'Win Rate [%]' in stats else None,
                'raw_stats': stats
            }
        except Exception:
            return {'error': 'Failed to calculate metrics'}

    def calculate_monte_carlo_metrics(self, df: pd.DataFrame, num_sims: int, horizon: int, distribution: str = 'bootstrap') -> Tuple[Dict[str, Any], Any]:
        # TODO: implement Monte Carlo simulation for returns
        # For now return empty dict and placeholder
        return ({'Monte Carlo': 'not_implemented'}, None)

# ---------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class Backtester:
    def __init__(self, config_manager, data_provider):
        """
        Unified Backtester combining sync/async strategies, Monte Carlo scaffolding and logging.
        - config_manager: object with .get(key, default) and optionally attributes like pattern_discovery
        - data_provider: object with async fetch_historical(pairs, start, end) -> pd.DataFrame
        """
        self.config_manager = config_manager
        self.data_provider = data_provider

        # Debug: Log the type of config_manager to diagnose issues
        print(f"DEBUG: Type of config_manager: {type(config_manager)}")
        
        # internal helpers
        self.data_cache = DataCache(data_provider)
        self.strategy_executor = StrategyExecutor(config_manager, data_provider)
        self.metrics_calculator = MetricsCalculator()
        self.journal = JournalEvaluator()

        # logging & files
        # Check if config_manager has a get method; fallback to default if not
        log_path = 'logs/backtester.log'  # Default log path
        if hasattr(config_manager, 'get'):
            log_path = config_manager.get('backtester_log', log_path)
        else:
            print(f"WARNING: config_manager does not have a 'get' method, using default log path: {log_path}")
        
        self.logger = setup_logger("Backtester", log_path)
        self.logger.info("Backtester initialized")
        
        # Initialize backtest log file
        backtest_log_path = 'logs/backtests.csv'  # Default backtest log path
        if hasattr(config_manager, 'get'):
            backtest_log_path = config_manager.get('backtest_log_file', backtest_log_path)
        else:
            self.logger.warning(f"config_manager lacks 'get' method, using default backtest log path: {backtest_log_path}")
        
        self.backtest_log_file = backtest_log_path
        os.makedirs('logs', exist_ok=True)
        if not os.path.exists(self.backtest_log_file):
            pd.DataFrame(columns=[
                'backtest_id', 'strategy', 'pair', 'start_date', 'end_date', 'total_return',
                'sharpe_ratio', 'max_drawdown', 'execution_time', 'average_reward', 'timestamp'
            ]).to_csv(self.backtest_log_file, index=False)

    # ------------------------
    # SYNC STRATEGY EXECUTOR
    # ------------------------
    def execute_sync_strategy(self, df: pd.DataFrame, strategy: str, sentiment_weight: float = 0.0, model_weights: Optional[Dict[str, float]] = None):
        """
        Execute synchronous strategies (RL, MA crossover, RSI mean reversion).
        Returns (portfolio, processed_df, rewards)
        """
        if df is None or df.empty:
            self.logger.warning("execute_sync_strategy called with empty df")
            return None, None, None

        df = df.copy()
        try:
            if 'sentiment' in df.columns and sentiment_weight and sentiment_weight == 0.0:
                df['close'] = df['close'] * (1 + sentiment_weight * df['sentiment'])

            rewards = None

            if strategy == "RL":
                states_array = np.array(self.strategy_executor.ml_model.prepare_state(df))
                actions = self.strategy_executor.ml_model.predict_batch(states_array)
                entries = pd.Series(actions == 1, index=df.index)
                exits = pd.Series(actions == 2, index=df.index)
                rewards = self.strategy_executor.ml_model.calculate_batch_rewards(states_array, actions)

            elif strategy == "MA Crossover":
                fast_period = int(self.config_manager.get('fast_ma_period', 10))
                slow_period = int(self.config_manager.get('slow_ma_period', 50))
                fast_ma = df['close'].rolling(fast_period).mean()
                slow_ma = df['close'].rolling(slow_period).mean()
                entries = fast_ma > slow_ma
                exits = fast_ma < slow_ma

            elif strategy == "RSI Mean Reversion":
                rsi_period = int(self.config_manager.get('rsi_period', 14))
                rsi_lower = float(self.config_manager.get('rsi_lower', 30))
                rsi_upper = float(self.config_manager.get('rsi_upper', 70))
                # use indicator helper or vectorbt pandas_ta wrapper
                try:
                    rsi = vbt.IndicatorFactory.from_pandas_ta('rsi').run(df['close'], length=rsi_period).rsi
                except Exception:
                    rsi = get_rsi(df['close'], rsi_period)  # fallback if you have get_rsi
                entries = rsi < rsi_lower
                exits = rsi > rsi_upper

            else:
                self.logger.warning(f"Unsupported sync strategy: {strategy}")
                return None, None, None

            portfolio = vbt.Portfolio.from_signals(
                close=df['close'],
                entries=entries,
                exits=exits,
                init_cash=float(self.config_manager.get('init_cash', 10000)),
                freq=self.config_manager.get('timeframe', '1H'),
                fees=float(self.config_manager.get('fees', 0.001))
            )
            return portfolio, df, rewards

        except Exception as e:
            self.logger.error(f"Failed sync strategy {strategy}: {e}", exc_info=True)
            return None, None, None

    # ------------------------
    # ASYNC STRATEGY WRAPPERS
    # ------------------------
    async def backtest_tp_sl(self, pair: str, start_date, end_date, sl_pct: float = 0.02, tp_pct: float = 0.04) -> Dict[str, Any]:
        """Backtest TP/SL strategy."""
        return await self.execute_async_strategy(pair, "TP/SL", start_date, end_date, sl_pct=sl_pct, tp_pct=tp_pct)

    async def backtest_trailing_stops(self, pair: str, start_date, end_date, trail_pct: float = 0.02, use_atr: bool = False, atr_mult: float = 2.0) -> Dict[str, Any]:
        """Backtest Trailing Stop strategy."""
        return await self.execute_async_strategy(pair, "Trailing Stop", start_date, end_date, trail_pct=trail_pct, use_atr=use_atr, atr_mult=atr_mult)

    async def backtest_advanced_trailing(self, pair: str, start_date, end_date, base_mult: float = 3.0) -> Dict[str, Any]:
        """Backtest Advanced Trailing strategy."""
        return await self.execute_async_strategy(pair, "Advanced Trailing", start_date, end_date, base_mult=base_mult)

    # ------------------------
    # ASYNC STRATEGY EXECUTOR
    # ------------------------
    async def execute_async_strategy(self, pair: str, strategy: str, start_date, end_date,
                                     sl_pct: float = 0.02, tp_pct: float = 0.04, trail_pct: float = 0.02,
                                     use_atr: bool = False, atr_mult: float = 2.0, base_mult: float = 3.0,
                                     df_cached: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Async executor for TP/SL, trailing stop, advanced trailing, MACD divergence etc.
        Returns metrics dict (or {'error': ...}).
        """
        try:
            df = df_cached
            if df is None:
                # data_provider.fetch_historical expected to be async and return a dataframe for the single pair
                df = await self.data_provider.fetch_historical([pair], start_date, end_date)
                if df is None or df.empty:
                    self.logger.warning(f"No data for {pair} in {strategy}")
                    return {'error': 'No data'}

            # ensure df indexing is timestamp sorted
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').set_index('timestamp', drop=False)

            prices = df['close']
            if prices.empty or prices.isna().all():
                return {'error': 'Invalid price data'}

            pf = None
            if strategy == "TP/SL":
                fast = prices.rolling(int(self.config_manager.get('fast_ma_period', 10))).mean()
                slow = prices.rolling(int(self.config_manager.get('slow_ma_period', 50))).mean()
                entries = fast > slow
                pf = vbt.Portfolio.from_signals(prices, entries, sl_stop=sl_pct, tp_stop=tp_pct,
                                                slippage=float(self.config_manager.get('slippage', 0.001)))

            elif strategy == "Trailing Stop":
                entries = prices.rolling(int(self.config_manager.get('fast_ma_period', 10))).mean() > prices.rolling(int(self.config_manager.get('slow_ma_period', 50))).mean()
                if use_atr:
                    tsl_stop = (get_atr(df) / prices) * atr_mult
                else:
                    tsl_stop = trail_pct
                pf = vbt.Portfolio.from_signals(prices, entries, tsl_stop=tsl_stop, slippage=float(self.config_manager.get('slippage', 0.001)))

            elif strategy == "Advanced Trailing":
                atr = get_atr(df)
                rolling_high = df['high'].rolling(int(self.config_manager.get('rolling_high_period', 20))).max()
                tsl_stop = (rolling_high - atr * base_mult) / prices
                entries = prices.rolling(int(self.config_manager.get('fast_ma_period', 10))).mean() > prices.rolling(int(self.config_manager.get('slow_ma_period', 50))).mean()
                pf = vbt.Portfolio.from_signals(prices, entries, tsl_stop=tsl_stop, slippage=float(self.config_manager.get('slippage', 0.001)))

            elif strategy == "MACD Divergence":
                # pattern discovery hook expected on config_manager
                if hasattr(self.config_manager, 'pattern_discovery') and self.config_manager.pattern_discovery:
                    divs = self.config_manager.pattern_discovery.mine_macd_divergence(df, pair)
                    entries = divs['div_type'] == 'bullish'
                    pf = vbt.Portfolio.from_signals(prices, entries, slippage=float(self.config_manager.get('slippage', 0.001)))
                else:
                    return {'error': 'pattern_discovery not available in config_manager'}

            else:
                self.logger.warning(f"Unsupported async strategy: {strategy}")
                return {'error': f"Unsupported strategy: {strategy}"}

            # derive metrics
            metrics = {}
            try:
                metrics = pf.stats(['Win Rate [%]', 'Total PnL', 'Sharpe Ratio', 'Max Drawdown [%]'])
            except Exception:
                # fallback to generic stats
                try:
                    metrics = self.metrics_calculator.calculate_metrics(pf)
                except Exception as e:
                    self.logger.error("Failed to extract metrics from PF", exc_info=True)
                    return {'error': 'metrics extraction failed'}

            return metrics

        except Exception as e:
            self.logger.error(f"Async strategy {strategy} failed for {pair}: {e}", exc_info=True)
            return {'error': str(e)}

    # ------------------------
    # RUN ALL STRATEGIES (dashboard-style)
    # ------------------------
    async def run_all_strategies(self, pairs: List[str], start_date, end_date,
                                 sentiment_weight: float = 0.1, model_weights: Optional[Dict[str, float]] = None,
                                 sl_pct: float = 0.02, tp_pct: float = 0.04, trail_pct: float = 0.02,
                                 use_atr: bool = False, atr_mult: float = 2.0, base_mult: float = 3.0,
                                 num_sims: int = 2000, horizon: int = 30, mc_distribution: str = 'bootstrap',
                                 batch_size: int = 10, mc_display_limit: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Run a battery of sync + async strategies for provided pairs and return metrics.
        """
        all_results: Dict[str, Dict[str, Any]] = {}
        start_time = time.time()
        sync_strategies = ["RL", "MA Crossover", "RSI Mean Reversion"]
        async_strategies = ["TP/SL", "Trailing Stop", "Advanced Trailing", "MACD Divergence"]

        # Fetch & normalize historical data
        historical_df = await self.data_cache.get_historical_data(pairs, start_date, end_date)
        if historical_df is None or historical_df.empty:
            self.logger.warning("No historical data available for backtesting")
            return all_results

        # group / resample
        freq = self.config_manager.get('timeframe', '1H')
        try:
            historical_df = historical_df.groupby(['pair', pd.Grouper(key='timestamp', freq=freq)]).last().reset_index()
        except Exception:
            # fallback: assume timestamp already properly formatted
            pass

        pair_dfs = {}
        for pair in pairs:
            tmp = historical_df[historical_df['pair'] == pair]
            if not tmp.empty:
                pair_dfs[pair] = tmp.set_index('timestamp').sort_index()

        total_tasks = len(pairs) * (len(sync_strategies) + len(async_strategies) + 1)
        overall_progress = tqdm(total=total_tasks, desc="Overall Backtesting", unit="task", dynamic_ncols=True)
        category_times = {"sync": 0.0, "async": 0.0, "mc": {}}
        total_batches = (len(pairs) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_pairs = pairs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # SYNC STRATEGIES
            for pair in batch_pairs:
                all_results.setdefault(pair, {})
                df = pair_dfs.get(pair)
                for strategy in sync_strategies:
                    t_start = time.time()
                    if df is None or df.empty:
                        all_results[pair][strategy] = {'error': 'No data'}
                    else:
                        portfolio, df_processed, rewards = self.execute_sync_strategy(df, strategy, sentiment_weight, model_weights)
                        metrics = self.metrics_calculator.calculate_metrics(portfolio, rewards)
                        all_results[pair][strategy] = metrics
                        self._log_backtest(pair, strategy, start_date, end_date, metrics, time.time() - t_start)
                    category_times["sync"] += time.time() - t_start
                    overall_progress.update(1)

            # ASYNC STRATEGIES (gather)
            async_tasks = []
            task_info = []
            for strategy in async_strategies:
                for pair in batch_pairs:
                    async_tasks.append(
                        self.execute_async_strategy(pair, strategy, start_date, end_date,
                                                    sl_pct, tp_pct, trail_pct, use_atr, atr_mult, base_mult,
                                                    df_cached=pair_dfs.get(pair))
                    )
                    task_info.append((pair, strategy))

            t_start = time.time()
            if tqdm_asyncio is not None:
                async_results = await tqdm_asyncio.gather(*async_tasks, desc=f"Batch {batch_idx + 1} Async Strategies", leave=False)
            else:
                async_results = await asyncio.gather(*async_tasks, return_exceptions=False)
            category_times["async"] += time.time() - t_start

            for (pair, strategy), result in zip(task_info, async_results):
                all_results.setdefault(pair, {})[strategy] = result
                self._log_backtest(pair, strategy, start_date, end_date, result, time.time() - t_start)
                overall_progress.update(1)

            # MONTE CARLO (async)
            async def mc_task(pair, df_pair):
                if df_pair is None or df_pair.empty:
                    return pair, {}, 0.0
                t0 = time.time()
                metrics, _ = self.metrics_calculator.calculate_monte_carlo_metrics(df_pair, num_sims, horizon, mc_distribution)
                elapsed = time.time() - t0
                return pair, metrics, elapsed

            mc_tasks = [mc_task(pair, pair_dfs.get(pair)) for pair in batch_pairs]
            if tqdm_asyncio is not None:
                mc_results_list = await tqdm_asyncio.gather(*mc_tasks, desc=f"Batch {batch_idx + 1} Monte Carlo", leave=False)
            else:
                mc_results_list = await asyncio.gather(*mc_tasks, return_exceptions=False)

            for pair, mc_metrics, elapsed_task in mc_results_list:
                all_results.setdefault(pair, {})['Monte Carlo'] = mc_metrics
                self._log_backtest(pair, 'Monte Carlo', start_date, end_date, mc_metrics, elapsed_task)
                category_times["mc"][pair] = elapsed_task
                overall_progress.update(1)

                last_pairs = list(category_times["mc"].keys())[-mc_display_limit:]
                mc_eta_str = " | ".join([f"{p}: {format_seconds(category_times['mc'][p])}" for p in last_pairs])
                overall_progress.set_postfix_str(f"MC ETA (last {mc_display_limit}): {mc_eta_str}")

        overall_progress.close()
        self.logger.info(f"Completed all strategies and Monte Carlo for {len(pairs)} pairs in {format_seconds(time.time() - start_time)}")
        return all_results

    # ------------------------
    # LOGGING HELPER
    # ------------------------
    def _log_backtest(self, pair: str, strategy: str, start_date, end_date, metrics: Dict[str, Any], execution_time: float):
        try:
            backtest_id = str(uuid.uuid4())
            timestamp = pd.Timestamp.now().isoformat()
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            average_reward = 0.0

            if metrics and isinstance(metrics, dict) and 'error' not in metrics:
                # metrics may use different keys depending on source
                if 'Total PnL' in metrics and isinstance(metrics['Total PnL'], (int, float, np.number)):
                    total_return = float(metrics['Total PnL']) / float(self.config_manager.get('init_cash', 10000)) if hasattr(self.config_manager, 'get') else 0.0
                elif 'Total Return [%]' in metrics:
                    try:
                        total_return = float(metrics.get('Total Return [%]', 0)) / 100.0
                    except Exception:
                        total_return = 0.0
                sharpe_ratio = float(metrics.get('Sharpe Ratio', 0) or 0)
                max_drawdown = float(metrics.get('Max Drawdown [%]', 0) or 0)
                average_reward = float(metrics.get('average_reward', 0) or 0)

            log_entry = pd.DataFrame([{
                'backtest_id': backtest_id,
                'strategy': strategy,
                'pair': pair,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'execution_time': execution_time,
                'average_reward': average_reward,
                'timestamp': timestamp
            }])
            log_entry.to_csv(self.backtest_log_file, mode='a', header=not os.path.exists(self.backtest_log_file), index=False)
        except Exception as e:
            self.logger.error(f"Failed to log backtest for {pair}/{strategy}: {e}", exc_info=True)

# ---------------------------------------------------------------------
# Example runner (for quick local testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    async def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--pairs', nargs='+', default=['EURUSD'])
        parser.add_argument('--start', default='2023-01-01')
        parser.add_argument('--end', default='2023-12-31')
        args = parser.parse_args()

        # TODO: wire a real config_manager and data_provider here
        class DummyConfig:
            def get(self, k, d=None): return d
            pattern_discovery = None
        class DummyProvider:
            async def fetch_historical(self, pairs, start, end):
                # produce a tiny fake dataframe for local smoke test
                idx = pd.date_range(start='2023-01-01', periods=200, freq='1H')
                df = pd.DataFrame({
                    'timestamp': idx,
                    'pair': pairs[0],
                    'open': np.random.rand(len(idx)) * 1.2,
                    'high': np.random.rand(len(idx)) * 1.2 + 1,
                    'low': np.random.rand(len(idx)) * 1.2,
                    'close': np.cumsum(np.random.randn(len(idx))) + 100,
                    'volume': np.random.randint(1, 1000, size=len(idx))
                })
                return df

        cfg = DummyConfig()
        prov = DummyProvider()
        bt = Backtester(cfg, prov)
        results = await bt.run_all_strategies(args.pairs, args.start, args.end, batch_size=1)
        print("Results:", results)

    asyncio.run(main())