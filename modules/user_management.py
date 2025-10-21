import os
import json
import asyncio
import sqlite3
import psycopg2
import psycopg2.extras
import psycopg2.pool
import uuid
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_fixed
from cachetools import TTLCache
from prometheus_client import Counter, Gauge
import aiofiles

from modules.logger_setup import setup_logger
from modules.security.encryption_helper import EncryptionHelper
from modules.vault_manager import VaultSecretsManager


# Prometheus metrics
user_ops_success = Counter('user_ops_success_total', 'Successful user operations', ['operation'])
user_ops_failure = Counter('user_ops_failure_total', 'Failed user operations', ['operation'])
user_ops_duration = Gauge('user_ops_duration_seconds', 'Duration of user operations', ['operation'])


class UserManagement:
    """
    Unified User Management System
    - Async operations
    - Supports PostgreSQL or SQLite
    - Uses encryption via VaultSecretsManager + EncryptionHelper
    - Tracks metrics via Prometheus
    - Manages sessions, tiers, expiry, and Telegram linking
    """

    def __init__(self, notifier=None, db_file: str = "data/copilot.db", vault_url=None, vault_token=None):
        self.logger = setup_logger("UserManagement", "logs/user_management.log")
        self.notifier = notifier
        self.vault = VaultSecretsManager(vault_url, vault_token)
        self.encryption = EncryptionHelper(vault_url, vault_token)

        self.db_file = db_file
        self.use_postgres = all(os.getenv(k) for k in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"])
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "copilot"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password")
        }

        # Metrics
        self.lock = asyncio.Lock()
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone("Africa/Lagos"))
        self.session_windows = {
            "London": {"start": "08:00", "end": "17:00", "timezone": "UTC"},
            "NewYork": {"start": "13:00", "end": "22:00", "timezone": "UTC"},
            "Tokyo": {"start": "00:00", "end": "09:00", "timezone": "UTC"}
        }
        self.tier_configs = {
            "Free": {"max_signals": 5, "session_limit": 1, "priority_multiplier": 1.0},
            "Premium": {"max_signals": 10, "session_limit": 2, "priority_multiplier": 1.2},
            "Pro": {"max_signals": 20, "session_limit": 3, "priority_multiplier": 1.5}
        }

        # Database setup
        if self.use_postgres:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **self.db_config)
        self._init_db()
        self.start_scheduler()

    # --- DATABASE INIT ------------------------------------------------------

    def _init_db(self):
        """Initialize DB tables for user management"""
        try:
            if self.use_postgres:
                conn = self.db_pool.getconn()
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        telegram_id TEXT UNIQUE,
                        telegram_key_id TEXT,
                        username TEXT,
                        tier TEXT,
                        account_type TEXT,
                        risk_profile TEXT,
                        time_zone TEXT,
                        status TEXT,
                        activation_code TEXT,
                        activation_code_key_id TEXT,
                        activation_code_expiry TEXT,
                        expiry_date TEXT,
                        created_on TEXT,
                        role TEXT,
                        is_active BOOLEAN,
                        max_signals INTEGER,
                        session_limit INTEGER
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        user_id TEXT REFERENCES users(user_id),
                        session_name TEXT,
                        PRIMARY KEY (user_id, session_name)
                    );
                """)
                conn.commit()
                self.db_pool.putconn(conn)
                self.logger.info("Initialized PostgreSQL user tables")
            else:
                os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            user_id TEXT PRIMARY KEY,
                            telegram_id TEXT UNIQUE,
                            telegram_key_id TEXT,
                            username TEXT,
                            tier TEXT,
                            account_type TEXT,
                            risk_profile TEXT,
                            time_zone TEXT,
                            status TEXT,
                            activation_code TEXT,
                            activation_code_key_id TEXT,
                            activation_code_expiry TEXT,
                            expiry_date TEXT,
                            created_on TEXT,
                            role TEXT,
                            is_active BOOLEAN,
                            max_signals INTEGER,
                            session_limit INTEGER
                        );
                    """)
                    conn.commit()
                self.logger.info(f"Initialized SQLite database at {self.db_file}")
            user_ops_success.labels(operation='init_db').inc()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            user_ops_failure.labels(operation='init_db').inc()

    # --- USER MANAGEMENT ----------------------------------------------------

    async def add_user(self, telegram_id: str, username: str, tier="Free", account_type="Standard", risk_profile="Medium", time_zone="Africa/Lagos", session="London"):
        """Add a user to the system"""
        start_time = asyncio.get_event_loop().time()
        try:
            user_id = str(uuid.uuid4())
            now = datetime.now(pytz.UTC)
            created_on = now.isoformat()
            tier_conf = self.tier_configs.get(tier, self.tier_configs["Free"])
            encrypted_tg_id, tg_key_id = await self.encryption.encrypt_text(telegram_id)

            user_record = {
                "user_id": user_id,
                "telegram_id": encrypted_tg_id,
                "telegram_key_id": tg_key_id,
                "username": username,
                "tier": tier,
                "account_type": account_type,
                "risk_profile": risk_profile,
                "time_zone": time_zone,
                "status": "Active",
                "created_on": created_on,
                "role": "user",
                "is_active": True,
                "max_signals": tier_conf["max_signals"],
                "session_limit": tier_conf["session_limit"]
            }

            if self.use_postgres:
                conn = self.db_pool.getconn()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (user_id, telegram_id, telegram_key_id, username, tier, account_type, risk_profile, time_zone, status, created_on, role, is_active, max_signals, session_limit)
                    VALUES (%(user_id)s, %(telegram_id)s, %(telegram_key_id)s, %(username)s, %(tier)s, %(account_type)s, %(risk_profile)s, %(time_zone)s, %(status)s, %(created_on)s, %(role)s, %(is_active)s, %(max_signals)s, %(session_limit)s)
                """, user_record)
                cursor.execute("INSERT INTO user_sessions (user_id, session_name) VALUES (%s, %s)", (user_id, session))
                conn.commit()
                self.db_pool.putconn(conn)
            else:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO users (user_id, telegram_id, telegram_key_id, username, tier, account_type, risk_profile, time_zone, status, created_on, role, is_active, max_signals, session_limit)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, tuple(user_record.values()))
                    cursor.execute("INSERT INTO user_sessions (user_id, session_name) VALUES (?, ?)", (user_id, session))
                    conn.commit()

            user_ops_success.labels(operation='add_user').inc()
            user_ops_duration.labels(operation='add_user').set(asyncio.get_event_loop().time() - start_time)

            await self._export_user_metrics(telegram_id, tier, session)
            self.logger.info(f"✅ Added user {username} ({telegram_id}) with tier {tier}")
        except Exception as e:
            user_ops_failure.labels(operation='add_user').inc()
            self.logger.error(f"Failed to add user {telegram_id}: {e}")

    async def _export_user_metrics(self, telegram_id: str, tier: str, session: str):
        """Export metrics to JSON log asynchronously"""
        try:
            metrics = {
                "telegram_id": telegram_id,
                "tier": tier,
                "session": session,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            async with aiofiles.open(f"logs/user_metrics_{telegram_id}.json", mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(metrics, indent=4))
            user_ops_success.labels(operation='export_user_metrics').inc()
        except Exception as e:
            user_ops_failure.labels(operation='export_user_metrics').inc()
            self.logger.error(f"Failed to export metrics for {telegram_id}: {e}")

    async def generate_access_code(self, telegram_id: str, expires_in_days: int = 7) -> Optional[str]:
        """Generate encrypted access code for activation"""
        try:
            code = f"{uuid.uuid4()}_{datetime.now(pytz.UTC).timestamp()}"
            encrypted_code, key_id = await self.encryption.encrypt_text(code)
            expiry = datetime.now(pytz.UTC) + timedelta(days=expires_in_days)
            user_ops_success.labels(operation='generate_access_code').inc()
            return json.dumps({
                "code": encrypted_code,
                "key_id": key_id,
                "expires_at": expiry.isoformat()
            })
        except Exception as e:
            user_ops_failure.labels(operation='generate_access_code').inc()
            self.logger.error(f"Failed to generate access code for {telegram_id}: {e}")
            return None

    # --- SCHEDULER ----------------------------------------------------------

    def start_scheduler(self):
        try:
            self.scheduler.add_job(self._check_user_expiry, 'interval', hours=24)
            self.scheduler.start()
            self.logger.info("User expiry scheduler started")
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")

    def _check_user_expiry(self):
        """Check and deactivate expired users"""
        try:
            now = datetime.now(pytz.UTC)
            if self.use_postgres:
                conn = self.db_pool.getconn()
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET is_active = FALSE, status = 'Expired' WHERE expiry_date < %s", (now.isoformat(),))
                conn.commit()
                self.db_pool.putconn(conn)
            else:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE users SET is_active = 0, status = 'Expired' WHERE expiry_date < ?", (now.isoformat(),))
                    conn.commit()
            self.logger.info("Checked and deactivated expired users")
            user_ops_success.labels(operation='check_expiry').inc()
        except Exception as e:
            self.logger.error(f"Failed to check expiry: {e}")
            user_ops_failure.labels(operation='check_expiry').inc()