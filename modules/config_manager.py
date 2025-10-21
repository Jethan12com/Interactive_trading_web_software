import os
import json
import asyncio
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import aiofiles
import pandas as pd
from datetime import datetime
from prometheus_client import Counter, Gauge
from modules.logger_setup import setup_logger  # Make sure this exists

# ------------------------------
# Prometheus metrics
# ------------------------------
config_ops_success = Counter('config_ops_success_total', 'Successful config operations', ['operation'])
config_ops_failure = Counter('config_ops_failure_total', 'Failed config operations', ['operation'])
config_ops_duration = Gauge('config_ops_duration_seconds', 'Duration of config operations', ['operation'])

users_loaded_total = Gauge('users_loaded_total', 'Number of users loaded successfully')
users_load_failure_total = Counter('users_load_failure_total', 'Number of user load failures')
user_invalid_data_total = Counter('user_invalid_data_total', 'Number of users with invalid or missing data')

# Per-user metrics
users_by_tier = Gauge('users_by_tier', 'Number of users per subscription tier', ['tier'])
users_by_risk_profile = Gauge('users_by_risk_profile', 'Number of users per risk profile', ['risk_profile'])

# ------------------------------
# ConfigManager Class
# ------------------------------
class ConfigManager:
    def __init__(self,
                 config_path: str = "copilot/modules/config/config.json",
                 credentials_path: str = "copilot/modules/config/credentials.json",
                 users_path: str = "copilot/modules/data/users.csv",
                 async_init: bool = True):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.makedirs(os.path.dirname(users_path), exist_ok=True)
        os.makedirs("copilot/modules/logs", exist_ok=True)

        self.logger = setup_logger("ConfigManager", "copilot/modules/logs/config_manager.log")

        self.config_path = config_path
        self.credentials_path = credentials_path
        self.users_path = users_path
        self.config: Dict[str, Any] = {}
        self.credentials: Dict[str, Any] = {}
        self.users: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()

        # For debounced auto-save
        self._save_users_task: Optional[asyncio.Task] = None
        self._save_users_debounce_delay = 2  # seconds

        load_dotenv()

        if async_init:
            asyncio.create_task(self._initialize_async())
        else:
            self._initialize_sync()

    # ------------------------------
    # Initialization
    # ------------------------------
    async def _initialize_async(self):
        try:
            await asyncio.gather(
                self._load_config(),
                self._load_credentials(),
                self._load_users()
            )
            self.logger.info({"event": "init_success"})
            config_ops_success.labels(operation='initialize').inc()
        except Exception as e:
            self.logger.error({"event": "init_error", "error": str(e)})
            config_ops_failure.labels(operation='initialize').inc()

    def _initialize_sync(self):
        try:
            self._load_config_sync()
            self._load_credentials_sync()
            self._load_users_sync()
            self.logger.info({"event": "init_success_sync"})
            config_ops_success.labels(operation='initialize_sync').inc()
        except Exception as e:
            self.logger.error({"event": "init_error_sync", "error": str(e)})
            config_ops_failure.labels(operation='initialize_sync').inc()

    # ------------------------------
    # CONFIG
    # ------------------------------
    async def _load_config(self):
        start_time = asyncio.get_event_loop().time()
        try:
            async with self.lock:
                if os.path.exists(self.config_path):
                    async with aiofiles.open(self.config_path, 'r', encoding='utf-8') as f:
                        self.config = json.loads(await f.read())
                    self.logger.info(f"Loaded config (async) from {self.config_path}")
                else:
                    self.config = self._default_config()
                    async with aiofiles.open(self.config_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(self.config, indent=4))
                    self.logger.warning(f"{self.config_path} not found — created default config")
            config_ops_success.labels(operation='load_config').inc()
        except Exception as e:
            self.logger.error({"event": "load_config_error", "error": str(e)})
            config_ops_failure.labels(operation='load_config').inc()
        finally:
            config_ops_duration.labels(operation='load_config').set(asyncio.get_event_loop().time() - start_time)

    def _load_config_sync(self):
        start_time = datetime.now().timestamp()
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded config (sync) from {self.config_path}")
            else:
                self.config = self._default_config()
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
                self.logger.warning(f"{self.config_path} not found — created default config")
            config_ops_success.labels(operation='load_config_sync').inc()
        except Exception as e:
            self.logger.error({"event": "load_config_error_sync", "error": str(e)})
            config_ops_failure.labels(operation='load_config_sync').inc()
        finally:
            config_ops_duration.labels(operation='load_config_sync').set(datetime.now().timestamp() - start_time)

    def _default_config(self) -> Dict[str, Any]:
        return {
            "trading": {
                "pairs": ["BTC/USDT", "ETH/USDT", "SP500", "TLT"],
                "timeframe": "1h",
                "capital": 10000,
                "fees": 0.001,
                "slippage": 0.001,
                "fast_ma_period": 10,
                "slow_ma_period": 50,
                "rsi_period": 14,
                "rsi_lower": 30,
                "rsi_upper": 70
            },
            "sessions": {
                "London": "08:00-16:00",
                "New York": "13:00-21:00",
                "Asia": "00:00-08:00"
            },
            "model": {"vol_model": "GARCH", "window": 50},
            "notifications": {"enable_telegram": True, "enable_email": False},
            "news": {"keywords": ["bitcoin", "ethereum", "forex", "market"], "sources": ["reuters", "bloomberg"], "language": "en", "max_articles": 100}
        }

    # ------------------------------
    # CREDENTIALS
    # ------------------------------
    async def _load_credentials(self):
        start_time = asyncio.get_event_loop().time()
        try:
            async with self.lock:
                if os.path.exists(self.credentials_path):
                    async with aiofiles.open(self.credentials_path, 'r', encoding='utf-8') as f:
                        self.credentials = json.loads(await f.read())
                    self.logger.info(f"Loaded credentials (async) from {self.credentials_path}")
                else:
                    self.credentials = self._default_credentials()
                    async with aiofiles.open(self.credentials_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(self.credentials, indent=4))
                    self.logger.warning(f"{self.credentials_path} not found — created default credentials")
            config_ops_success.labels(operation='load_credentials').inc()
        except Exception as e:
            self.logger.error({"event": "load_credentials_error", "error": str(e)})
            config_ops_failure.labels(operation='load_credentials').inc()
        finally:
            config_ops_duration.labels(operation='load_credentials').set(asyncio.get_event_loop().time() - start_time)

    def _load_credentials_sync(self):
        start_time = datetime.now().timestamp()
        try:
            if os.path.exists(self.credentials_path):
                with open(self.credentials_path, 'r', encoding='utf-8') as f:
                    self.credentials = json.load(f)
                self.logger.info(f"Loaded credentials (sync) from {self.credentials_path}")
            else:
                self.credentials = self._default_credentials()
                with open(self.credentials_path, 'w', encoding='utf-8') as f:
                    json.dump(self.credentials, f, indent=4)
                self.logger.warning(f"{self.credentials_path} not found — created default credentials")
            config_ops_success.labels(operation='load_credentials_sync').inc()
        except Exception as e:
            self.logger.error({"event": "load_credentials_error_sync", "error": str(e)})
            config_ops_failure.labels(operation='load_credentials_sync').inc()
        finally:
            config_ops_duration.labels(operation='load_credentials_sync').set(datetime.now().timestamp() - start_time)

    def _default_credentials(self) -> Dict[str, Any]:
        return {
            "telegram": {"bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "")},
            "newsapi": {"api_key": os.getenv("NEWSAPI_KEY", "")},
            "binance": {"api_key": os.getenv("BINANCE_API_KEY", ""), "api_secret": os.getenv("BINANCE_API_SECRET", "")},
            "alpha_vantage": {"api_key": os.getenv("ALPHAVANTAGE_API_KEY", "")},
            "finnhub": {"api_key": os.getenv("FINNHUB_API_KEY", "")},
            "oanda": {"api_key": os.getenv("OANDA_API_KEY", ""), "account_id": os.getenv("OANDA_ACCOUNT_ID", "")},
            "mt5": {"login": os.getenv("MT5_LOGIN", ""), "password": os.getenv("MT5_PASSWORD", ""), "server": os.getenv("MT5_SERVER", "FBS-Demo")}
        }

# ------------------------------
# Part 2: Users, Metrics, Debounced Auto-Save, Accessors, Reload, Summary
# ------------------------------
# ------------------------------
# USERS (with validation & per-user metrics)
# ------------------------------
    async def _load_users(self):
        start_time = asyncio.get_event_loop().time()
        try:
            async with self.lock:
                if os.path.exists(self.users_path):
                    df = pd.read_csv(self.users_path)
                    self.users = {}
                    for _, row in df.iterrows():
                        telegram_id = row.get('telegram_id')
                        capital = row.get('capital', 10000)
                        tier = row.get('tier', 'Free')
                        if not telegram_id or not isinstance(capital, (int, float)) or tier not in ['Free', 'Pro', 'Premium']:
                            self.logger.warning(f"Invalid user data skipped: {row.to_dict()}")
                            user_invalid_data_total.inc()
                            continue
                        self.users[telegram_id] = {
                            'user_id': row.get('user_id', ''),
                            'capital': float(capital),
                            'name': row.get('name', ''),
                            'tier': tier,
                            'risk_profile': row.get('risk_profile', 'Medium'),
                            'time_zone': row.get('time_zone', 'Africa/Lagos')
                        }
                    self.logger.info(f"Loaded {len(self.users)} valid users (async)")
                else:
                    self.users = {}
                    pd.DataFrame(columns=['telegram_id','user_id','name','capital','tier','risk_profile','time_zone']).to_csv(self.users_path, index=False)
                    self.logger.warning(f"{self.users_path} not found — created empty users CSV")
            users_loaded_total.set(len(self.users))
            self._update_user_metrics()
            config_ops_success.labels(operation='load_users').inc()
        except Exception as e:
            self.logger.error({"event": "load_users_error", "error": str(e)})
            users_load_failure_total.inc()
            config_ops_failure.labels(operation='load_users').inc()
        finally:
            config_ops_duration.labels(operation='load_users').set(asyncio.get_event_loop().time() - start_time)

    def _load_users_sync(self):
        start_time = datetime.now().timestamp()
        try:
            if os.path.exists(self.users_path):
                df = pd.read_csv(self.users_path)
                self.users = {}
                for _, row in df.iterrows():
                    telegram_id = row.get('telegram_id')
                    capital = row.get('capital', 10000)
                    tier = row.get('tier', 'Free')
                    if not telegram_id or not isinstance(capital, (int, float)) or tier not in ['Free', 'Pro', 'Premium']:
                        self.logger.warning(f"Invalid user data skipped: {row.to_dict()}")
                        user_invalid_data_total.inc()
                        continue
                    self.users[telegram_id] = {
                        'user_id': row.get('user_id', ''),
                        'capital': float(capital),
                        'name': row.get('name', ''),
                        'tier': tier,
                        'risk_profile': row.get('risk_profile', 'Medium'),
                        'time_zone': row.get('time_zone', 'Africa/Lagos')
                    }
                self.logger.info(f"Loaded {len(self.users)} valid users (sync)")
            else:
                self.users = {}
                pd.DataFrame(columns=['telegram_id','user_id','name','capital','tier','risk_profile','time_zone']).to_csv(self.users_path, index=False)
                self.logger.warning(f"{self.users_path} not found — created empty users CSV")
            users_loaded_total.set(len(self.users))
            self._update_user_metrics()
            config_ops_success.labels(operation='load_users_sync').inc()
        except Exception as e:
            self.logger.error({"event": "load_users_error_sync", "error": str(e)})
            users_load_failure_total.inc()
            config_ops_failure.labels(operation='load_users_sync').inc()
        finally:
            config_ops_duration.labels(operation='load_users_sync').set(datetime.now().timestamp() - start_time)

# ------------------------------
# Debounced Auto-Save
# ------------------------------
    def _schedule_save_users(self):
        """Debounced async save for frequent updates."""
        if self._save_users_task and not self._save_users_task.done():
            self._save_users_task.cancel()
        loop = asyncio.get_event_loop()
        self._save_users_task = loop.create_task(self._debounced_save())

    async def _debounced_save(self):
        try:
            await asyncio.sleep(self._save_users_debounce_delay)
            self._save_users_sync()
        except asyncio.CancelledError:
            pass  # canceled due to new update

    def _save_users_sync(self):
        try:
            df = pd.DataFrame.from_dict(self.users, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'telegram_id'}, inplace=True)
            df.to_csv(self.users_path, index=False)
            self.logger.info(f"Users saved to {self.users_path}")
        except Exception as e:
            self.logger.error(f"Failed to save users: {e}")

# ------------------------------
# Update per-user metrics
# ------------------------------
    def _update_user_metrics(self):
        tier_counts = {}
        risk_counts = {}
        for user in self.users.values():
            tier = user.get('tier', 'Free')
            risk = user.get('risk_profile', 'Medium')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        for tier, count in tier_counts.items():
            users_by_tier.labels(tier=tier).set(count)
        for risk, count in risk_counts.items():
            users_by_risk_profile.labels(risk_profile=risk).set(count)

# ------------------------------
# User management
# ------------------------------
    def add_user(self, user_data: Dict[str, Any]):
        telegram_id = user_data.get('telegram_id')
        capital = user_data.get('capital', 10000)
        tier = user_data.get('tier', 'Free')
        risk = user_data.get('risk_profile', 'Medium')

        if not telegram_id or not isinstance(capital, (int, float)) or tier not in ['Free', 'Pro', 'Premium']:
            self.logger.warning(f"Invalid user data not added: {user_data}")
            user_invalid_data_total.inc()
            return False

        self.users[telegram_id] = {
            'user_id': user_data.get('user_id', ''),
            'capital': float(capital),
            'name': user_data.get('name', ''),
            'tier': tier,
            'risk_profile': risk,
            'time_zone': user_data.get('time_zone', 'Africa/Lagos')
        }

        self._update_user_metrics()
        users_loaded_total.set(len(self.users))
        self._schedule_save_users()
        self.logger.info(f"User {telegram_id} added successfully")
        return True

    def update_user(self, telegram_id: str, updates: Dict[str, Any]):
        if telegram_id not in self.users:
            self.logger.warning(f"Cannot update non-existent user {telegram_id}")
            return False

        user = self.users[telegram_id]
        for key, value in updates.items():
            if key in user:
                user[key] = value

        if not user.get('tier') or user.get('tier') not in ['Free', 'Pro', 'Premium'] or not isinstance(user.get('capital', 0), (int, float)):
            self.logger.warning(f"User {telegram_id} has invalid updated data, skipping metrics update")
            return False

        self._update_user_metrics()
        self._schedule_save_users()
        self.logger.info(f"User {telegram_id} updated successfully")
        return True

    def remove_user(self, telegram_id: str):
        if telegram_id in self.users:
            del self.users[telegram_id]
            self._update_user_metrics()
            users_loaded_total.set(len(self.users))
            self._schedule_save_users()
            self.logger.info(f"User {telegram_id} removed successfully")
            return True
        self.logger.warning(f"Cannot remove non-existent user {telegram_id}")
        return False

# ------------------------------
# Accessors
# ------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get("trading", {}).get(key, default)

    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        return self.config.get(section, {}) if section else self.config

    def get_credentials(self, service: str) -> Dict[str, Any]:
        return self.credentials.get(service, {})

    def get_api_key(self, provider_name: str) -> Optional[str]:
        section = self.credentials.get(provider_name, {})
        for k, v in section.items():
            if "key" in k.lower():
                return v
        return None

    def get_users(self) -> Dict[str, Dict]:
        return self.users

# ------------------------------
# Reload
# ------------------------------
    async def reload_async(self):
        await asyncio.gather(
            self._load_config(),
            self._load_credentials(),
            self._load_users()
        )

    def reload_sync(self):
        self._load_config_sync()
        self._load_credentials_sync()
        self._load_users_sync()

# ------------------------------
# Summary
# ------------------------------
    def summary(self):
        print("CONFIG SUMMARY")
        print(f"Trading pairs: {self.get_config('trading').get('pairs')}")
        print(f"Model config: {self.get_config('model')}")
        print(f"APIs available: {list(self.credentials.keys())}")
        print(f"Users loaded: {len(self.users)}")
        print("-" * 60)