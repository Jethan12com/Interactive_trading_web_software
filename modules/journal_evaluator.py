import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from prometheus_client import Counter, Gauge
from modules.reinforcement_trader import RLTrader
from modules.data_provider import MultiProviderDataProvider
from modules.config_manager import ConfigManager
from modules.logger_setup import setup_logger

# Prometheus metrics
trade_log_success = Counter('trade_log_success_total', 'Successful trade logs', ['pair'])
trade_log_failure = Counter('trade_log_failure_total', 'Failed trade logs', ['pair'])
trade_monitor_duration = Gauge('trade_monitor_duration_seconds', 'Duration of trade monitoring', ['pair'])

class JournalEvaluator:
    def __init__(self, auto_flush_interval: int = 5):
        self.trades_file = 'copilot/modules/logs/trades.csv'
        os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
        self.logger = setup_logger("JournalEvaluator", "copilot/modules/logs/journal_evaluator.log")
        self.config_manager = ConfigManager()
        self.data_provider = MultiProviderDataProvider(self.config_manager, self.config_manager.get_all_credentials())
        self.rl_trader = RLTrader(self.data_provider, self.config_manager)
        self.last_retrain = {}  # Last retrain timestamp per pair
        self.lock = asyncio.Lock()
        self.auto_flush_interval = auto_flush_interval

        # Load trades or initialize empty DataFrame
        if os.path.exists(self.trades_file) and os.path.getsize(self.trades_file) > 0:
            self.trades_df = pd.read_csv(self.trades_file)
        else:
            self.trades_df = pd.DataFrame(columns=[
                'signal_id','pair','strategy','action','entry_price','exit_price','pnl',
                'asset_type','ab_group','status','timestamp','tp','sl','tp_hit','sl_hit','session'
            ])
            self._flush_to_csv_sync()

        # Periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())

    # ----------------- CSV Flush ----------------- #
    async def _periodic_flush(self):
        while True:
            await asyncio.sleep(self.auto_flush_interval)
            await self.flush_to_csv()

    async def flush_to_csv(self):
        async with self.lock:
            try:
                self.trades_df.to_csv(self.trades_file, index=False)
                self.logger.info("Flushed trades to CSV")
            except Exception as e:
                self.logger.error(f"Failed to flush trades to CSV: {e}")

    def _flush_to_csv_sync(self):
        try:
            self.trades_df.to_csv(self.trades_file, index=False)
        except Exception as e:
            self.logger.error(f"Failed to flush trades to CSV: {e}")

    # ----------------- Trade Logging ----------------- #
    async def log_trade(self, **trade_data):
        async with self.lock:
            try:
                trade_data['timestamp'] = datetime.now(pytz.UTC).isoformat()
                self.trades_df = pd.concat([self.trades_df, pd.DataFrame([trade_data])], ignore_index=True)
                trade_log_success.labels(pair=trade_data['pair']).inc()
                self.logger.info(f"Logged trade {trade_data['signal_id']} for {trade_data['pair']}")
            except Exception as e:
                trade_log_failure.labels(pair=trade_data.get('pair','unknown')).inc()
                self.logger.error(f"Failed to log trade: {e}")

    async def update_trade(self, signal_id, **update_data):
        async with self.lock:
            try:
                idx = self.trades_df[self.trades_df['signal_id']==signal_id].index
                if idx.empty:
                    self.logger.warning(f"Trade {signal_id} not found for update")
                    return
                for k,v in update_data.items():
                    self.trades_df.loc[idx, k] = v
                self.trades_df.loc[idx,'timestamp'] = datetime.now(pytz.UTC).isoformat()
                self.logger.info(f"Updated trade {signal_id}")
                if 'status' in update_data and update_data['status']=='CLOSED':
                    if self.trades_df.loc[idx,'ab_group'].iloc[0]=='A':
                        await self._trigger_rl_feedback()
            except Exception as e:
                self.logger.error(f"Failed to update trade {signal_id}: {e}")

    # ----------------- RL Feedback ----------------- #
    async def _trigger_rl_feedback(self):
        async with self.lock:
            closed_trades = self.trades_df[self.trades_df['status']=='CLOSED']
            if closed_trades.empty: return
            pairs = closed_trades['pair'].unique()
            historical_data = self.data_provider.fetch_historical_sync(
                pairs.tolist(),
                start_date='2024-01-01',
                end_date=datetime.now().isoformat()
            )
            if historical_data.empty: return
            for pair in pairs:
                pair_trades = closed_trades[closed_trades['pair']==pair]
                if not self._should_retrain(pair_trades): continue
                pair_data = historical_data[historical_data['pair']==pair]
                reward = pair_trades['pnl'].mean()
                if pair_trades['tp_hit'].any(): reward *= 1.5
                elif pair_trades['sl_hit'].any(): reward *= 0.5
                self.rl_trader.train(pair_data, total_timesteps=5000, pair=pair)
                self.last_retrain[pair] = pd.Timestamp.now()
                self.logger.info(f"RL retraining triggered for {pair}, reward={reward}")

    def _should_retrain(self, pair_trades):
        config = self.config_manager.get_config('model')
        if len(pair_trades) < config.get('min_trades_for_retrain',10): return False
        recent = pair_trades.tail(20)
        win_rate = len(recent[recent['pnl']>0])/len(recent)
        returns = recent['pnl']/recent['entry_price']
        sharpe = returns.mean()/returns.std()*np.sqrt(252) if returns.std()!=0 else 0
        last = self.last_retrain.get(pair_trades['pair'].iloc[0], pd.Timestamp.min)
        time_check = pd.Timestamp.now() - last > pd.Timedelta(hours=config.get('retrain_interval_hours',24))
        retrain = (win_rate<config.get('win_rate_threshold',0.5) or sharpe<config.get('sharpe_threshold',1.0)) and time_check
        return retrain

    # ----------------- Trade Monitoring ----------------- #
    async def monitor_trades(self, max_hold_time_hours=24):
        async with self.lock:
            open_trades = self.trades_df[self.trades_df['status']=='OPEN']
            if open_trades.empty: return
            config = self.config_manager.get_config('trading')
            for _, trade in open_trades.iterrows():
                signal_id, pair, entry_price, action, entry_time, tp, sl, session = (
                    trade['signal_id'], trade['pair'], trade['entry_price'], trade['action'],
                    pd.to_datetime(trade['timestamp']), trade['tp'], trade['sl'], trade['session']
                )
                latest_data = await self.data_provider.fetch_live([pair], session)
                if latest_data.empty: continue
                latest_price = latest_data['close'].iloc[-1]

                # Max hold enforcement
                if datetime.now(pytz.UTC)-entry_time > timedelta(hours=max_hold_time_hours):
                    pnl = (latest_price-entry_price) if action=='BUY' else (entry_price-latest_price)
                    await self.update_trade(signal_id, exit_price=latest_price, pnl=pnl, status='CLOSED')
                    continue

                # TP/SL adjustment with volatility
                volatility = await self.rl_trader.forecast_volatility(latest_data, pair)
                risk = {'Low':0.5,'Medium':1.0,'Aggressive':1.5}.get(config.get('risk_profile','Medium'),1.0)
                tp = entry_price*(1+config.get('take_profit_pct',0.04)*risk*(1+volatility)) if action=='BUY' else entry_price*(1-config.get('take_profit_pct',0.04)*risk*(1+volatility))
                sl = entry_price*(1-config.get('stop_loss_pct',0.02)*risk*(1+volatility)) if action=='BUY' else entry_price*(1+config.get('stop_loss_pct',0.02)*risk*(1+volatility))
                pnl = (latest_price-entry_price) if action=='BUY' else (entry_price-latest_price)
                tp_hit, sl_hit = (latest_price>=tp, latest_price<=sl) if action=='BUY' else (latest_price<=tp, latest_price>=sl)
                if tp_hit or sl_hit:
                    await self.update_trade(signal_id, exit_price=latest_price, pnl=pnl, status='CLOSED', tp_hit=tp_hit, sl_hit=sl_hit)

    # ----------------- Performance Metrics ----------------- #
    async def get_performance_metrics(self):
        async with self.lock:
            df = self.trades_df[self.trades_df['status']=='CLOSED']
            if df.empty: return {}
            total = len(df)
            return {
                'total_trades': total,
                'win_rate': len(df[df['pnl']>0])/total,
                'total_pnl': df['pnl'].sum(),
                'sharpe_ratio': (df['pnl']/df['entry_price']).mean()/(df['pnl']/df['entry_price']).std()*np.sqrt(252) if (df['pnl']/df['entry_price']).std()!=0 else 0,
                'expectancy': df['pnl'].mean(),
                'tp_hit_rate': df['tp_hit'].mean(),
                'sl_hit_rate': df['sl_hit'].mean()
            }

    # ----------------- A/B Test Evaluation ----------------- #
    async def evaluate_ab_test(self):
        async with self.lock:
            results = {}
            for group in ['A','B']:
                group_trades = self.trades_df[(self.trades_df['ab_group']==group) & (self.trades_df['status']=='CLOSED')]
                if group_trades.empty:
                    results[group] = {}
                    continue
                total = len(group_trades)
                strategy_metrics = {}
                for strat in group_trades['strategy'].unique():
                    st_trades = group_trades[group_trades['strategy']==strat]
                    strategy_metrics[strat] = {
                        'win_rate': len(st_trades[st_trades['pnl']>0])/len(st_trades),
                        'total_pnl': st_trades['pnl'].sum(),
                        'tp_hit_rate': st_trades['tp_hit'].mean(),
                        'sl_hit_rate': st_trades['sl_hit'].mean()
                    }
                returns = group_trades['pnl']/group_trades['entry_price']
                results[group] = {
                    'total_trades': total,
                    'win_rate': len(group_trades[group_trades['pnl']>0])/total,
                    'total_pnl': group_trades['pnl'].sum(),
                    'sharpe_ratio': returns.mean()/returns.std()*np.sqrt(252) if returns.std()!=0 else 0,
                    'expectancy': group_trades['pnl'].mean(),
                    'tp_hit_rate': group_trades['tp_hit'].mean(),
                    'sl_hit_rate': group_trades['sl_hit'].mean(),
                    'strategy_metrics': strategy_metrics
                }
            return results

    # ----------------- Shutdown ----------------- #
    async def shutdown(self):
        await self.flush_to_csv()
        self._flush_task.cancel()