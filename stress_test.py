import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
from modules.signal_engine import SignalEngine
from modules.data_provider import MultiProviderDataProvider
from modules.config_manager import ConfigManager
from modules.user_management import UserManager
import tracemalloc

async def stress_test():
    """Simulate high-frequency trading and measure system performance."""
    logging.basicConfig(filename='logs/stress_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        config_manager = ConfigManager()
        data_provider = MultiProviderDataProvider(config_manager, config_manager.get_credentials('apis'))
        user_manager = UserManager()
        signal_engine = SignalEngine(config_manager, data_provider, signal_logger=None)
        
        # Optimize cache and batch settings
        signal_engine.cache_expiry = pd.Timedelta(seconds=30)  # Reduced for high-frequency
        config_manager.get_config('data')['batch_size'] = 100  # Increased batch size
        
        pairs = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'ADA/USD', 
                 'SOL/USD', 'DOT/USD', 'BNB/USD', 'DOGE/USD', 'LINK/USD']
        users = user_manager.get_users()[:5]  # Limit to 5 users for testing
        signals_per_second = 100
        duration_seconds = 60
        total_signals = signals_per_second * duration_seconds

        logger.info(f"Starting stress test: {total_signals} signals over {duration_seconds}s across {len(pairs)} pairs and {len(users)} users")
        
        # Track memory and performance
        tracemalloc.start()
        start_time = time.time()
        signal_count = 0
        errors = 0
        
        # Mock historical data
        historical_data = {}
        for pair in pairs:
            dates = pd.date_range(start='2025-10-08', end='2025-10-09', freq='1min')
            historical_data[pair] = pd.DataFrame({
                'close': np.random.normal(100, 10, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates)),
                'pair': pair
            }, index=dates)
            signal_engine.data_cache[pair] = {'data': historical_data[pair], 'timestamp': pd.Timestamp.now()}

        async def generate_signal_batch():
            nonlocal signal_count, errors
            for _ in range(signals_per_second):
                for pair in pairs:
                    for user in users:
                        user_id = user['user_id']
                        user_profile = user_manager.get_user_profile(user_id)
                        signal = signal_engine.generate_signals(user_id, pair, user_profile)
                        if signal:
                            signal_count += 1
                            # Simulate user feedback (50% chance of acting)
                            signal_engine.update_user_feedback(signal['signal_id'], user_id, acted=np.random.choice([True, False]))
                        else:
                            errors += 1

        for _ in range(duration_seconds):
            await generate_signal_batch()
            await asyncio.sleep(1)

        end_time = time.time()
        memory_snapshot = tracemalloc.take_snapshot()
        top_stats = memory_snapshot.statistics('lineno')
        memory_usage = sum(stat.size for stat in top_stats) / 1024 / 1024  # MB
        
        success_rate = (signal_count / (signal_count + errors)) * 100 if (signal_count + errors) > 0 else 0
        throughput = signal_count / (end_time - start_time)
        
        logger.info(f"Stress test completed: {signal_count} signals generated, {errors} errors")
        logger.info(f"Success Rate: {success_rate:.2f}%, Throughput: {throughput:.2f} signals/s, Memory Usage: {memory_usage:.2f} MB")

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_signals': signal_count,
            'errors': errors,
            'success_rate': success_rate,
            'throughput': throughput,
            'memory_usage_mb': memory_usage,
            'optimizations': [
                'Reduced cache expiry to 30 seconds for high-frequency trading',
                'Increased batch size to 100 for data fetches',
                'Optimized RL model selection to reduce computation',
                'Cached historical data to minimize API calls'
            ]
        }
        pd.DataFrame([report]).to_csv('logs/stress_test_report.csv', index=False)
        logger.info("Stress test report saved to logs/stress_test_report.csv")
        
        tracemalloc.stop()
        return report

    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(stress_test())