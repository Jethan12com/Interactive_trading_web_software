import asyncio
import os
import pandas as pd
import numpy as np
import pytz
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from prometheus_client import Gauge, start_http_server
import requests

from modules.config_manager import ConfigManager
from modules.data_provider import MultiProviderDataProvider  # ⬅️ Metrics live here
from modules.signal_engine import HybridSignalEngine
from modules.notifier import TelegramNotifier
from modules.user_management import UserManagement
from modules.signal_logger import SignalLogger
from modules.ml_model import MLModel
from modules.user_filter import UserFilter
from modules.backtester import Backtester
from modules.journal_evaluator import JournalEvaluator
from modules.vault_manager import VaultSecretsManager
from modules.security.encryption_helper import EncryptionHelper
from modules.rate_limiter import RateLimiter

# -------------------- Logging Setup --------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S %Z")
formatter.converter = lambda *args: datetime.now(pytz.timezone('Africa/Lagos')).timetuple()

file_handler = logging.FileHandler(os.path.join(log_dir, "run_live.log"), encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- News Provider --------------------
class NewsProvider:
    def __init__(self, vault: VaultSecretsManager):
        self.logger = logging.getLogger("NewsProvider")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, "news_provider.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.handlers = [file_handler, console_handler]
        self.vault = vault
        self.api_key = self._get_api_key()

    def _get_api_key(self):
        try:
            creds = self.vault.get_api_keys("news_api")
            if not creds or not creds[0].get("api_key"):
                raise ValueError("Invalid News API credentials in Vault")
            return creds[0]["api_key"]
        except Exception as e:
            self.logger.error(f"Failed to load News API key: {e}")
            raise

    def fetch_news(self, pairs: list, session: str) -> dict:
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={','.join(pairs)}&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            news_data = response.json().get("feed", [])
            news_context = {}
            for pair in pairs:
                news_context[pair] = []
                for article in news_data[:5]:
                    if pair in article.get("tickers", []):
                        sentiment = article.get("overall_sentiment_score", 0.0)
                        summary = article.get("summary", "No summary")
                        news_context[pair].append({
                            "sentiment": sentiment,
                            "summary": summary[:100]
                        })
            self.logger.info(f"Fetched news for {pairs} during {session}")
            return news_context
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return {pair: [] for pair in pairs}

# -------------------- Economic Calendar Provider --------------------
class EconomicCalendarProvider:
    def __init__(self, vault: VaultSecretsManager):
        self.logger = logging.getLogger("EconomicCalendarProvider")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, "economic_calendar.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.handlers = [file_handler, console_handler]
        self.vault = vault
        self.api_key = self._get_api_key()

    def _get_api_key(self):
        try:
            creds = self.vault.get_api_keys("finnhub_api")
            if not creds or not creds[0].get("api_key"):
                raise ValueError("Invalid Finnhub API credentials in Vault")
            return creds[0]["api_key"]
        except Exception as e:
            self.logger.error(f"Failed to load Finnhub API key: {e}")
            raise

    def fetch_economic_events(self, pairs: list, from_date: str, to_date: str) -> dict:
        try:
            url = f"https://finnhub.io/api/v1/calendar/economic?from={from_date}&to={to_date}&token={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            events = response.json().get("economicCalendar", [])
            event_context = {}
            for pair in pairs:
                event_context[pair] = []
                for event in events:
                    if pair.split("/")[0] in event.get("country", "") or pair in event.get("currency", ""):
                        impact = event.get("impact", 1)
                        event_name = event.get("event", "No event")
                        event_context[pair].append({
                            "event_name": event_name,
                            "impact": "High" if impact == 3 else "Medium" if impact == 2 else "Low",
                            "actual": event.get("actual", "N/A"),
                            "previous": event.get("previous", "N/A")
                        })
            self.logger.info(f"Fetched economic events for {pairs} from {from_date} to {to_date}")
            return event_context
        except Exception as e:
            self.logger.error(f"Failed to fetch economic events: {e}")
            return {pair: [] for pair in pairs}

# -------------------- Pipeline --------------------
class Pipeline:
    def __init__(self, config_manager: ConfigManager, data_provider: MultiProviderDataProvider):
        self.config_manager = config_manager
        self.data_provider = data_provider
        vault_url = os.getenv("VAULT_URL", "http://127.0.0.1:8201")
        vault_token = os.getenv("VAULT_TOKEN")
        self.vault = VaultSecretsManager(vault_url, vault_token)
        self.user_manager = UserManagement(notifier=TelegramNotifier())
        self.signal_logger = SignalLogger()
        self.signal_engine = HybridSignalEngine(config_manager, data_provider, self.signal_logger)
        self.user_filter = UserFilter()
        self.notifier = TelegramNotifier()
        self.ml_model = MLModel(config_manager, data_provider)
        self.backtester = Backtester(config_manager, data_provider)
        self.journal = JournalEvaluator()
        self.news_provider = NewsProvider(self.vault)
        self.economic_provider = EconomicCalendarProvider(self.vault)
        self.encryption_helper = EncryptionHelper()
        self.rate_limiter = RateLimiter()
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone("Africa/Lagos"))
        self.last_model_reload = None
        self.last_retrain_time = None
        self.cov_matrix = None

        # ---- Prometheus Gauges (Pipeline-level only) ----
        self.cycle_duration = Gauge("copilot_cycle_duration", "Signal cycle duration")
        self.signal_count = Gauge("copilot_signal_count", "Signals per pair/user", ["pair", "user"])
        self.error_count = Gauge("copilot_error_count", "Errors per task", ["task"])
        self.model_confidence = Gauge("copilot_model_confidence", "RL Model Confidence", ["pair"])
        self.backup_success = Gauge("copilot_backup_success", "Database backup success", ["db_type"])

        try:
            start_http_server(8000)
            logger.info("✅ Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            self.error_count.labels(task="prometheus_init").inc()

        self.notifier.initialize_from_vault()

    def update_user_sessions(self):
        now_utc = datetime.now(pytz.UTC)
        users = self.user_manager.get_users()
        for user_id, user in users.items():
            try:
                user_tz = pytz.timezone(user.get("time_zone", "Africa/Lagos"))
                user_local = now_utc.astimezone(user_tz)
                hour = user_local.hour
                if 7 <= hour < 16:
                    session = "London"
                elif 13 <= hour < 22:
                    session = "NewYork"
                elif 22 <= hour or hour < 7:
                    session = "Tokyo"
                else:
                    session = "Off"
                user["active_session"] = session
            except Exception as e:
                logger.warning(f"Failed to update session for user {user_id}: {e}")
                user["active_session"] = "London"
        return users

    async def update_covariance_matrix(self):
        try:
            pairs = self.config_manager.get_pairs() + ["SP500", "TLT"]
            start_date = (datetime.now(pytz.timezone('Africa/Lagos')) - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = datetime.now(pytz.timezone('Africa/Lagos')).strftime("%Y-%m-%d")
            df = await self.data_provider.fetch_historical(pairs, start_date, end_date, interval="1h")
            if df is None or df.empty:
                logger.warning("No data for covariance matrix update")
                return
            returns_df = df.pivot(columns="pair", values="close").pct_change().dropna()
            if returns_df.empty:
                logger.warning("Empty returns data for covariance matrix")
                return
            self.cov_matrix = returns_df.cov().values
            logger.info("Covariance matrix updated successfully")
        except Exception as e:
            self.error_count.labels(task="covariance").inc()
            logger.error(f"Covariance update failed: {e}", exc_info=True)

    async def reload_model_if_needed(self):
        try:
            now = datetime.now(pytz.timezone("Africa/Lagos"))
            if not self.last_model_reload or (now - self.last_model_reload).total_seconds() > 12 * 3600:
                for pair in self.config_manager.get_pairs():
                    model_path = f"models/{pair}_ppo.zip"
                    if os.path.exists(model_path):
                        from stable_baselines3 import PPO
                        self.ml_model.rl_model = PPO.load(model_path)
                        logger.info(f"Reloaded RL model for {pair}")
                self.last_model_reload = now
        except Exception as e:
            self.error_count.labels(task="model_reload").inc()
            logger.error(f"Model reload failed: {e}", exc_info=True)

    async def retrain_models(self):
        try:
            now = datetime.now(pytz.timezone("Africa/Lagos"))
            if not self.last_retrain_time or (now - self.last_retrain_time).total_seconds() > 24 * 3600:
                pairs = self.config_manager.get_pairs()
                start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
                end_date = now.strftime("%Y-%m-%d")
                df_all = await self.data_provider.fetch_historical(pairs, start_date, end_date, interval="1h")
                if df_all is None or df_all.empty:
                    logger.warning("No data for model retraining")
                    return
                for pair in pairs:
                    df_pair = df_all[df_all["pair"] == pair]
                    if df_pair.empty:
                        logger.warning(f"No data for pair {pair}")
                        continue
                    await asyncio.to_thread(self.ml_model.train_rl_model, df_pair, pair)
                    if getattr(self.ml_model, "rl_model", None):
                        self.ml_model.rl_model.save(f"models/{pair}_ppo.zip")
                        logger.info(f"Saved retrained RL model for {pair}")
                self.last_retrain_time = now
        except Exception as e:
            self.error_count.labels(task="retrain").inc()
            logger.error(f"Model retraining failed: {e}", exc_info=True)

    async def run_periodic_backtest(self):
        try:
            pairs = self.config_manager.get_pairs()
            start_date = (datetime.now(pytz.timezone("Africa/Lagos")) - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = datetime.now(pytz.timezone("Africa/Lagos")).strftime("%Y-%m-%d")
            results = {}
            for pair in pairs:
                fixed_stats = await self.backtester.backtest_tp_sl(pair, start_date, end_date, sl_pct=0.02, tp_pct=0.04)
                trail_stats = await self.backtester.backtest_trailing_stops(pair, start_date, end_date, trail_pct=0.02)
                atr_stats = await self.backtester.backtest_advanced_trailing(pair, start_date, end_date, base_mult=3.0)
                results[pair] = {'fixed': fixed_stats, 'trailing': trail_stats, 'atr_trailing': atr_stats}
            pd.DataFrame(results).to_json("logs/backtest_results.json", indent=4)
            logger.info("Periodic backtest completed and results saved")
        except Exception as e:
            self.error_count.labels(task="backtest").inc()
            logger.error(f"Backtesting failed: {e}", exc_info=True)

    async def run_periodic_backup(self):
        try:
            backup_file = self.user_manager.backup_database()
            db_type = "postgresql" if self.user_manager.use_postgres else "sqlite"
            self.backup_success.labels(db_type=db_type).set(1)
            logger.info(f"Periodic backup completed: {backup_file}")
        except Exception as e:
            self.error_count.labels(task="backup").inc()
            self.backup_success.labels(db_type="postgresql" if self.user_manager.use_postgres else "sqlite").set(0)
            logger.error(f"Periodic backup failed: {e}", exc_info=True)

    def schedule_background_tasks(self):
        interval = 6 if os.getenv('ENV') == 'staging' else 24
        self.scheduler.add_job(self.update_covariance_matrix, 'interval', days=7)
        self.scheduler.add_job(self.reload_model_if_needed, 'interval', hours=12)
        self.scheduler.add_job(self.retrain_models, 'interval', hours=24)
        self.scheduler.add_job(self.run_periodic_backtest, 'interval', hours=interval)
        self.scheduler.add_job(self.run_periodic_backup, 'interval', hours=24)
        self.scheduler.add_job(lambda: self.signal_engine.prune_old_signals(max_age_days=30), 'cron', hour=3, minute=0)
        self.scheduler.add_job(lambda: self.encryption_helper.rotate_key(f"key_{datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y%m%d')}"), 'interval', days=90)
        logger.info("Background tasks scheduled")

    def compute_user_priority(self, signal: dict, user: dict) -> float:
        try:
            hybrid_score = abs(float(signal.get("hybrid_score", 0.0)))
            volatility = float(signal.get("volatility", 0.0))
            sentiment = abs(float(signal.get("sentiment_score", 0.0)))
            rl_conf = float(signal.get("rl_confidence", 0.0))
            news_sentiment = float(signal.get("news_sentiment", 0.0))
            news_impact = float(signal.get("news_impact", 0.0))
            priority = 0.3 * hybrid_score + 0.2 * volatility + 0.15 * sentiment + 0.1 * rl_conf + 0.15 * news_sentiment + 0.1 * news_impact
            priority *= user.get("priority_multiplier", 1.0)
            if signal.get("pair") in user.get("preferred_pairs", []):
                priority *= 1.1
            return priority
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid data in compute_user_priority for user {user.get('telegram_id')}: {e}")
            return 0.0

    async def run(self, cycle_interval: float = 5.0):
        try:
            self.notifier.send_message("🚀 Signal pipeline started")
        except Exception as e:
            logger.error(f"Failed to send Telegram startup message: {e}")
            self.error_count.labels(task="notifier_start").inc()
        self.user_manager.start_scheduler()
        try:
            self.scheduler.start()
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.error_count.labels(task="scheduler_start").inc()

        while True:
            start_time = datetime.now(pytz.timezone("Africa/Lagos"))
            try:
                users = self.update_user_sessions()
                if not users:
                    logger.warning("No active users found")
                    await asyncio.sleep(cycle_interval)
                    continue

                pairs = self.config_manager.get_pairs()
                from_date = (start_time - timedelta(days=1)).strftime("%Y-%m-%d")
                to_date = start_time.strftime("%Y-%m-%d")

                # Fetch live data
                data = await self.data_provider.fetch_live(pairs, interval="1m")
                if data.empty:
                    logger.error("No live data fetched")
                    self.notifier.send_message("🚨 No live data fetched")
                    await asyncio.sleep(cycle_interval)
                    continue

                # Fetch news and economic events
                news_context = {}
                economic_context = {}
                for user_id, user in users.items():
                    session = user.get("active_session", "all")
                    if session != "Off":
                        news_context.update(self.news_provider.fetch_news(pairs, session))
                        economic_context.update(self.economic_provider.fetch_economic_events(pairs, from_date, to_date))

                # Generate signals
                user_signal_tasks = [
                    self.signal_engine.generate_signals(pairs, active_session=user.get("active_session", "all"))
                    for user in users.values() if user.get("active_session") != "Off" and user.get("is_active", False)
                ]
                results = await asyncio.gather(*user_signal_tasks, return_exceptions=True)

                all_signals = []
                for res in results:
                    if isinstance(res, Exception):
                        self.error_count.labels(task="signal_generation").inc()
                        logger.error(f"Signal generation error: {res}", exc_info=True)
                    elif res:
                        for sig in res:
                            pair = sig.get("pair")
                            sig["news_sentiment"] = max([n.get("sentiment", 0.0) for n in news_context.get(pair, [])], default=0.0)
                            sig["news_context"] = "; ".join([n.get("summary", "") for n in news_context.get(pair, [])]) or "No recent news"
                            sig["economic_event"] = "; ".join([e.get("event_name", "") for e in economic_context.get(pair, [])]) or "No events"
                            sig["news_impact"] = max([1 if e.get("impact", "Low") == "Low" else 2 if e.get("impact", "Medium") == "Medium" else 3 for e in economic_context.get(pair, [])], default=0.0)
                            all_signals.append(sig)

                if not all_signals:
                    await asyncio.sleep(cycle_interval)
                    continue

                # Log signals
                try:
                    self.signal_engine.save_signals(all_signals)
                    self.signal_engine.merge_signals_to_master()
                    self.signal_engine.prune_old_signals(max_age_days=30)
                    signals_df = pd.DataFrame(all_signals)
                    signals_df["user_id"], signals_df["user_key_id"] = zip(*signals_df["user_id"].apply(self.encryption_helper.encrypt_text))
                    signals_df.to_json("logs/signals.json", orient="records", lines=True, mode="a")
                except Exception as e:
                    logger.error(f"Signal save/merge/prune failed: {e}")
                    self.error_count.labels(task="signal_processing").inc()

                # Dispatch signals
                admin_telegram_id = self.config_manager.get_config("telegram").get("admin_telegram_id")
                for user_id, user in users.items():
                    if not user.get("is_active") or user.get("active_session") == "Off":
                        continue

                    filtered_df = self.user_filter.apply(user, pd.DataFrame(all_signals))
                    if filtered_df.empty:
                        continue

                    # Vectorized RL predictions
                    states_array = filtered_df.apply(
                        lambda sig: np.array([
                            float(sig.get("close", 0.0)),
                            float(sig.get("rsi", 0.0)),
                            float(sig.get("macd", 0.0)),
                            float(sig.get("returns", 0.0)),
                            float(sig.get("volatility", 0.0)),
                            float(sig.get("hybrid_score", 0.0)),
                            float(sig.get("pattern1", 0.0)),
                            float(sig.get("pattern2", 0.0)),
                            float(sig.get("pattern3", 0.0)),
                            float(sig.get("sentiment_score", 0.0)),
                            float(sig.get("news_sentiment", 0.0)),
                            float(sig.get("news_impact", 0.0))
                        ], dtype=np.float32), axis=1
                    ).to_list()
                    states_array = np.stack(states_array, axis=0)

                    try:
                        rl_actions = self.ml_model.predict_batch(states_array)
                        rl_confidences = self.ml_model.calculate_batch_rewards(states_array, rl_actions)
                    except Exception as e:
                        logger.error(f"RL prediction failed for user {user_id}: {e}")
                        self.error_count.labels(task="rl_prediction").inc()
                        continue

                    filtered_df["rl_action"] = rl_actions
                    filtered_df["rl_action_text"] = [{0: "Hold", 1: "Buy", 2: "Sell"}.get(a, "Hold") for a in rl_actions]
                    filtered_df["rl_confidence"] = rl_confidences
                    filtered_df["model_confidence"] = np.clip(0.5 + np.abs(filtered_df["hybrid_score"].fillna(0.0)) * 0.5, 0.5, 1.0)
                    filtered_df["priority"] = -filtered_df.apply(lambda sig: self.compute_user_priority(sig, user), axis=1)

                    for idx, sig in filtered_df.iterrows():
                        self.model_confidence.labels(pair=sig.get("pair", "N/A")).set(sig.get("rl_confidence", 0.0))

                    sorted_signals = filtered_df.sort_values("priority", ascending=True).to_dict(orient="records")

                    async def dispatch_signals(signals, user_id, is_admin=False):
                        for sig in signals:
                            try:
                                telegram_id = user.get("telegram_id") if not is_admin else admin_telegram_id
                                signal_id = f"{sig.get('pair')}_{int(datetime.now(pytz.timezone('Africa/Lagos')).timestamp())}"
                                if telegram_id:
                                    encrypted_user_id, user_key_id = self.encryption_helper.encrypt_text(user_id) if not is_admin else self.encryption_helper.encrypt_text(admin_telegram_id)
                                    payload = {
                                        "pair": sig.get("pair"),
                                        "direction": sig.get("direction"),
                                        "confidence": sig.get("model_confidence", 0.0),
                                        "price": sig.get("close", 0.0),
                                        "timestamp": datetime.now(pytz.timezone("Africa/Lagos")).isoformat(),
                                        "rl_action": sig.get("rl_action_text"),
                                        "rl_confidence": sig.get("rl_confidence", 0.0),
                                        "tp": sig.get("tp", "N/A"),
                                        "sl": sig.get("sl", "N/A"),
                                        "news_context": sig.get("news_context", "No recent news"),
                                        "economic_event": sig.get("economic_event", "No events"),
                                        "user_id": encrypted_user_id,
                                        "user_key_id": user_key_id,
                                        "user_tier": user.get("tier", "Free")
                                    }
                                    message = (
                                        f"📈 *{payload['pair']} {payload['rl_action']} Signal*\n"
                                        f"Price: {payload['price']}\n"
                                        f"Confidence: {payload['confidence']:.2%}\n"
                                        f"TP: {payload['tp']} | SL: {payload['sl']}\n"
                                        f"News: {payload['news_context']}\n"
                                        f"Events: {payload['economic_event']}"
                                    )
                                    self.notifier.send_message(message)
                                    self.signal_count.labels(pair=sig.get("pair", "N/A"), user=user_id).inc()
                            except Exception as e:
                                self.error_count.labels(task=f"user_dispatch_{user_id}").inc()
                                logger.error(f"Error dispatching signal for {user_id}: {e}", exc_info=True)

                    await dispatch_signals(sorted_signals, user_id)
                    if user_id == admin_telegram_id:
                        await dispatch_signals(sorted_signals, user_id, is_admin=True)

            except Exception as e:
                self.error_count.labels(task="main_loop").inc()
                logger.error(f"Pipeline run error: {e}", exc_info=True)
            finally:
                duration = (datetime.now(pytz.timezone("Africa/Lagos")) - start_time).total_seconds()
                self.cycle_duration.set(duration)
                await asyncio.sleep(cycle_interval)

    async def close(self):
        await self.data_provider.close()
        self.scheduler.shutdown()
        logger.info("Pipeline closed")

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        data_provider = MultiProviderDataProvider(config_manager)
        pipeline = Pipeline(config_manager, data_provider)
        pipeline.schedule_background_tasks()
        asyncio.run(pipeline.run(cycle_interval=5.0))
    except Exception as e:
        logger.error(f"Pipeline startup failed: {e}", exc_info=True)
        raise