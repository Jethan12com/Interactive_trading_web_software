import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import uuid
from modules.utils import Utils

class SignalLogger:
    def __init__(self):
        self.signals_file = 'logs/signals.csv'
        self.trades_file = 'logs/trades.csv'
        self.utils = Utils()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='logs/signal_logger.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        os.makedirs('logs', exist_ok=True)
        if not os.path.exists(self.signals_file):
            pd.DataFrame(columns=[
                'signal_id', 'pair', 'strategy', 'action', 'price', 'timestamp', 'user_id',
                'delivery_status', 'candlestick_signal', 'anomaly_score', 'divergence_signal', 'ab_group'
            ]).to_csv(self.signals_file, index=False)

    def log_signal(self, pair, strategy, action, price, user_id, candlestick_signal=0, anomaly_score=0, divergence_signal=0, ab_group='A'):
        """Log a trading signal to the signals CSV file."""
        try:
            signal_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            signal_data = {
                'signal_id': signal_id,
                'pair': pair,
                'strategy': strategy,  # e.g., PPO, DQN, MA Crossover, RSI Mean Reversion
                'action': action,  # buy, sell, hold
                'price': price,
                'timestamp': timestamp,
                'user_id': user_id,
                'delivery_status': 'pending',  # Updated by notifier
                'candlestick_signal': candlestick_signal,  # From pattern_discovery
                'anomaly_score': anomaly_score,  # From pattern_discovery
                'divergence_signal': divergence_signal,  # From pattern_discovery
                'ab_group': ab_group  # 'A' for RL (PPO/DQN), 'B' for rule-based
            }
            signals_df = pd.read_csv(self.signals_file)
            signals_df = pd.concat([signals_df, pd.DataFrame([signal_data])], ignore_index=True)
            signals_df.to_csv(self.signals_file, index=False)
            self.logger.info(f"Logged signal {signal_id} for {pair} with strategy {strategy}, user {user_id}")
            self.utils.log_metrics({
                'signals_logged': 1,
                'signal_action': action,
                'candlestick_signal': candlestick_signal,
                'anomaly_score': anomaly_score,
                'divergence_signal': divergence_signal
            }, prefix='signal_logger')
            return signal_id
        except Exception as e:
            self.logger.error(f"Failed to log signal for {pair}, user {user_id}: {e}")
            return None

    def update_signal_delivery(self, signal_id, delivery_status):
        """Update the delivery status of a signal."""
        try:
            signals_df = pd.read_csv(self.signals_file)
            signal_idx = signals_df[signals_df['signal_id'] == signal_id].index
            if not signal_idx.empty:
                signals_df.loc[signal_idx, 'delivery_status'] = delivery_status
                signals_df.loc[signal_idx, 'timestamp'] = datetime.now().isoformat()
                signals_df.to_csv(self.signals_file, index=False)
                self.logger.info(f"Updated signal {signal_id} delivery status to {delivery_status}")
                self.utils.log_metrics({'signals_delivered': 1 if delivery_status == 'delivered' else 0}, prefix='signal_logger')
            else:
                self.logger.warning(f"Signal {signal_id} not found for delivery update")
        except Exception as e:
            self.logger.error(f"Failed to update signal {signal_id} delivery status: {e}")

    def get_signal_history(self, pair=None, user_id=None, start_date=None, end_date=None):
        """Retrieve signal history filtered by pair, user, or date range."""
        try:
            signals_df = pd.read_csv(self.signals_file)
            if pair:
                signals_df = signals_df[signals_df['pair'] == pair]
            if user_id:
                signals_df = signals_df[signals_df['user_id'] == user_id]
            if start_date:
                signals_df = signals_df[pd.to_datetime(signals_df['timestamp']) >= pd.to_datetime(start_date)]
            if end_date:
                signals_df = signals_df[pd.to_datetime(signals_df['timestamp']) <= pd.to_datetime(end_date)]
            
            if signals_df.empty:
                self.logger.warning(f"No signals found for filters: pair={pair}, user_id={user_id}, start_date={start_date}, end_date={end_date}")
                return pd.DataFrame()
            
            self.logger.info(f"Retrieved {len(signals_df)} signals for filters: pair={pair}, user_id={user_id}, start_date={start_date}, end_date={end_date}")
            return signals_df
        except Exception as e:
            self.logger.error(f"Failed to retrieve signal history: {e}")
            return pd.DataFrame()

    def calculate_delivery_metrics(self):
        """Calculate signal delivery metrics for dashboard display."""
        try:
            signals_df = pd.read_csv(self.signals_file)
            if signals_df.empty:
                self.logger.warning("No signals available for delivery metrics")
                return {}

            total_signals = len(signals_df)
            delivered_signals = len(signals_df[signals_df['delivery_status'] == 'delivered'])
            delivery_rate = delivered_signals / total_signals if total_signals > 0 else 0
            strategy_counts = signals_df['strategy'].value_counts().to_dict()
            ab_group_counts = signals_df['ab_group'].value_counts().to_dict()

            metrics = {
                'total_signals': total_signals,
                'delivered_signals': delivered_signals,
                'delivery_rate': delivery_rate,
                'strategy_counts': strategy_counts,
                'ab_group_counts': ab_group_counts
            }
            self.logger.info(f"Calculated delivery metrics: {metrics}")
            self.utils.log_metrics(metrics, prefix='signal_logger')
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to calculate delivery metrics: {e}")
            return {}

    def archive_signals(self, days_old=30):
        """Archive signals older than a specified number of days."""
        try:
            signals_df = pd.read_csv(self.signals_file)
            if signals_df.empty:
                self.logger.warning("No signals to archive")
                return

            cutoff_date = pd.to_datetime(datetime.now()) - pd.Timedelta(days=days_old)
            old_signals = signals_df[pd.to_datetime(signals_df['timestamp']) < cutoff_date]
            if old_signals.empty:
                self.logger.info("No signals old enough to archive")
                return

            archive_file = f'logs/signals_archive_{datetime.now().strftime("%Y%m%d")}.csv'
            old_signals.to_csv(archive_file, index=False)
            signals_df = signals_df[pd.to_datetime(signals_df['timestamp']) >= cutoff_date]
            signals_df.to_csv(self.signals_file, index=False)
            self.logger.info(f"Archived {len(old_signals)} signals to {archive_file}")
            self.utils.log_metrics({'signals_archived': len(old_signals)}, prefix='signal_logger')
        except Exception as e:
            self.logger.error(f"Failed to archive signals: {e}")

    def analyze_signal_performance(self, pair=None, user_id=None, strategy=None, ab_group=None, start_date=None, end_date=None):
        """Analyze signal performance by correlating with trade outcomes."""
        try:
            # Load signals and trades
            signals_df = pd.read_csv(self.signals_file)
            trades_df = pd.read_csv(self.trades_file)

            # Apply filters
            if pair:
                signals_df = signals_df[signals_df['pair'] == pair]
                trades_df = trades_df[trades_df['pair'] == pair]
            if user_id:
                signals_df = signals_df[signals_df['user_id'] == user_id]
                trades_df = trades_df[trades_df['user_id'] == user_id]
            if strategy:
                signals_df = signals_df[signals_df['strategy'] == strategy]
                trades_df = trades_df[trades_df['strategy'] == strategy]
            if ab_group:
                signals_df = signals_df[signals_df['ab_group'] == ab_group]
                trades_df = trades_df[trades_df['ab_group'] == ab_group]
            if start_date:
                signals_df = signals_df[pd.to_datetime(signals_df['timestamp']) >= pd.to_datetime(start_date)]
                trades_df = trades_df[pd.to_datetime(trades_df['timestamp']) >= pd.to_datetime(start_date)]
            if end_date:
                signals_df = signals_df[pd.to_datetime(signals_df['timestamp']) <= pd.to_datetime(end_date)]
                trades_df = trades_df[pd.to_datetime(trades_df['timestamp']) <= pd.to_datetime(end_date)]

            # Filter for delivered signals and closed trades
            signals_df = signals_df[signals_df['delivery_status'] == 'delivered']
            trades_df = trades_df[trades_df['status'] == 'CLOSED']

            if signals_df.empty or trades_df.empty:
                self.logger.warning(f"No delivered signals or closed trades for analysis: pair={pair}, user_id={user_id}, strategy={strategy}, ab_group={ab_group}")
                return {}

            # Merge signals with trades on signal_id
            merged_df = signals_df.merge(trades_df, on=['signal_id', 'pair', 'strategy', 'ab_group'], how='inner')
            if merged_df.empty:
                self.logger.warning(f"No matching signals and trades for analysis")
                return {}

            # Calculate performance metrics
            total_signals = len(merged_df)
            win_trades = len(merged_df[merged_df['pnl'] > 0])
            win_rate = win_trades / total_signals if total_signals > 0 else 0
            average_pnl = merged_df['pnl'].mean() if total_signals > 0 else 0
            expectancy = average_pnl * win_rate - merged_df[merged_df['pnl'] <= 0]['pnl'].mean() * (1 - win_rate) if total_signals > 0 else 0

            # Assume TP/SL hit if exit_price differs significantly from signal price
            merged_df['tp_sl_hit'] = ((merged_df['action'] == 'buy') & (merged_df['exit_price'] > merged_df['price'])) | \
                                     ((merged_df['action'] == 'sell') & (merged_df['exit_price'] < merged_df['price']))
            tp_sl_hit_rate = merged_df['tp_sl_hit'].mean() if total_signals > 0 else 0

            # Pattern impact analysis
            pattern_metrics = {}
            for pattern in ['candlestick_signal', 'anomaly_score', 'divergence_signal']:
                if pattern in merged_df.columns:
                    high_pattern = merged_df[merged_df[pattern].abs() > 0.5]  # High-confidence patterns
                    if not high_pattern.empty:
                        pattern_win_rate = len(high_pattern[high_pattern['pnl'] > 0]) / len(high_pattern) if len(high_pattern) > 0 else 0
                        pattern_avg_pnl = high_pattern['pnl'].mean() if len(high_pattern) > 0 else 0
                        pattern_metrics[pattern] = {
                            'win_rate': pattern_win_rate,
                            'average_pnl': pattern_avg_pnl,
                            'count': len(high_pattern)
                        }

            metrics = {
                'total_signals': total_signals,
                'win_rate': win_rate,
                'average_pnl': average_pnl,
                'expectancy': expectancy,
                'tp_sl_hit_rate': tp_sl_hit_rate,
                'pattern_metrics': pattern_metrics,
                'strategy_breakdown': merged_df.groupby('strategy')['pnl'].agg(['count', 'mean']).to_dict(),
                'ab_group_breakdown': merged_df.groupby('ab_group')['pnl'].agg(['count', 'mean']).to_dict()
            }

            self.logger.info(f"Signal performance metrics: {metrics}")
            self.utils.log_metrics({
                'signal_total': metrics['total_signals'],
                'signal_win_rate': metrics['win_rate'],
                'signal_average_pnl': metrics['average_pnl'],
                'signal_expectancy': metrics['expectancy'],
                'signal_tp_sl_hit_rate': metrics['tp_sl_hit_rate']
            }, prefix='signal_performance')
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to analyze signal performance: {e}")
            return {}