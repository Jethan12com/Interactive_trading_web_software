import pandas as pd
import logging
import os
import json
from datetime import datetime, timedelta
from modules.utils import Utils


class UserFilter:
    """
    Persistent smart filter that:
    ✅ Applies filters based on user risk, tier, session, pairs, etc.
    ✅ Throttles signal frequency (per user & tier)
    ✅ Persists last signal timestamps across restarts (JSON cache)
    """

    def __init__(self, cache_path="data/user_signal_cache.json"):
        self.utils = Utils()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename='logs/user_filter.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.last_signal_time = self._load_cache()

        # Frequency limits per subscription tier (in minutes)
        self.signal_intervals = {
            'free': 60,       # 1 per hour
            'standard': 15,   # every 15 min
            'premium': 1      # near real-time
        }

    # ------------------------------------------------------------------
    # 🔁 Persistence helpers
    # ------------------------------------------------------------------
    def _load_cache(self):
        """Load persistent JSON cache of user timestamps."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    data = json.load(f)
                # Convert timestamps back to datetime
                for k, v in data.items():
                    data[k] = datetime.fromisoformat(v)
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save last signal times persistently."""
        try:
            cache_copy = {k: v.isoformat() for k, v in self.last_signal_time.items()}
            with open(self.cache_path, "w") as f:
                json.dump(cache_copy, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # 🧠 Main filter logic
    # ------------------------------------------------------------------
    def apply(self, user, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Filter and throttle signals for one user."""
        try:
            if signals_df.empty:
                return pd.DataFrame()

            filtered_df = signals_df.copy()
            user_id = str(user.get('user_id', 'unknown'))
            tier = str(user.get('subscription', 'Free')).lower()
            now = datetime.utcnow()

            # 1️⃣ Frequency Throttling
            last_time = self.last_signal_time.get(user_id)
            wait_mins = self.signal_intervals.get(tier, 60)
            if last_time and (now - last_time) < timedelta(minutes=wait_mins):
                self.logger.info(f"⏱️ Skipping user {user_id} — waiting {wait_mins}m interval.")
                return pd.DataFrame()

            # 2️⃣ Preferred Pairs
            if 'preferred_pairs' in user and isinstance(user['preferred_pairs'], str):
                pairs = [p.strip().upper() for p in user['preferred_pairs'].split(',') if p.strip()]
                filtered_df = filtered_df[filtered_df['pair'].isin(pairs)]

            # 3️⃣ Tier-Based Quantity
            if tier == 'free':
                filtered_df = filtered_df.nlargest(1, 'confidence')
            elif tier == 'standard':
                filtered_df = filtered_df.nlargest(3, 'confidence')

            # 4️⃣ Account Type
            account_type = str(user.get('account_type', 'demo')).lower()
            if account_type == 'demo':
                filtered_df = filtered_df[filtered_df['volatility'] < 0.03]
            elif account_type == 'real':
                filtered_df = filtered_df[filtered_df['volatility'] < 0.05]

            # 5️⃣ Risk Profile
            risk = str(user.get('risk_profile', 'medium')).lower()
            if risk == 'low':
                filtered_df = filtered_df[filtered_df['volatility'] < 0.015]
            elif risk == 'aggressive':
                filtered_df = filtered_df[filtered_df['volatility'] < 0.05]

            # 6️⃣ Market Session Filter
            current_session = self.utils.get_market_session()
            allowed_sessions = [s.strip() for s in str(user.get('active_sessions', 'London,NewYork')).split(',')]
            if current_session not in allowed_sessions:
                self.logger.info(f"User {user_id} outside session ({current_session})")
                return pd.DataFrame()

            # 7️⃣ Confidence Cutoff
            filtered_df = filtered_df[filtered_df['confidence'] >= 0.5]
            filtered_df = filtered_df.sort_values('confidence', ascending=False).reset_index(drop=True)

            # ✅ Update cache
            if not filtered_df.empty:
                self.last_signal_time[user_id] = now
                self._save_cache()
                self.logger.info(f"✅ Filtered {len(filtered_df)} signal(s) for {user_id} [{tier}]")

            return filtered_df

        except Exception as e:
            self.logger.error(f"Filter failed for {user.get('user_id', 'unknown')}: {e}")
            return pd.DataFrame()
