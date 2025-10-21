import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.utils import Utils


class MetricsAggregator:
    """
    Collects and analyzes metric JSON logs created by Utils.log_metrics().
    Generates performance summaries for dashboard analytics.
    """

    def __init__(self, log_dir="logs/metrics"):
        self.log_dir = log_dir
        self.utils = Utils(log_dir=log_dir)

    # ==========================================================
    # LOAD ALL METRIC LOGS
    # ==========================================================
    def load_logs(self, prefix_filter=None, days=None):
        """
        Load and combine all metric JSON logs.
        Args:
            prefix_filter (str): Only load logs from this module (e.g., 'rl_trader', 'signal_engine').
            days (int): Only include logs within the last X days.
        Returns:
            pd.DataFrame of combined metrics.
        """
        logs = []
        cutoff = None
        if days:
            cutoff = datetime.utcnow() - timedelta(days=days)

        for file in os.listdir(self.log_dir):
            if not file.endswith(".json"):
                continue
            if prefix_filter and not file.startswith(prefix_filter):
                continue

            path = os.path.join(self.log_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if "timestamp" in record:
                            ts = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
                            if cutoff and ts < cutoff:
                                continue
                            record["timestamp"] = ts
                        logs.append(record)
                    except json.JSONDecodeError:
                        continue

        if not logs:
            return pd.DataFrame()

        df = pd.DataFrame(logs)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ==========================================================
    # ANALYTICS
    # ==========================================================
    def compute_summary(self, df):
        """Compute aggregated trading performance metrics."""
        if df.empty:
            return {}

        summary = {}

        # Basic metrics
        summary["total_records"] = len(df)
        summary["start_date"] = df["timestamp"].min().strftime("%Y-%m-%d")
        summary["end_date"] = df["timestamp"].max().strftime("%Y-%m-%d")

        # Profit/Loss related
        if "pnl" in df.columns:
            pnl_series = df["pnl"].dropna().astype(float)
            summary["total_pnl"] = pnl_series.sum()
            summary["avg_pnl"] = pnl_series.mean()
            summary["sharpe_ratio"] = (
                pnl_series.mean() / pnl_series.std() * np.sqrt(252)
                if pnl_series.std() > 0 else 0
            )

        # Win rate & expectancy
        if "win" in df.columns:
            wins = df["win"].astype(int).sum()
            summary["win_rate"] = wins / len(df)
        else:
            summary["win_rate"] = None

        if "reward" in df.columns and "risk" in df.columns:
            expectancy = (df["reward"].mean() * summary.get("win_rate", 0)) - (
                abs(df["risk"].mean()) * (1 - summary.get("win_rate", 0))
            )
            summary["expectancy"] = expectancy

        # Volume or trade count
        if "volume" in df.columns:
            summary["avg_volume"] = df["volume"].mean()

        return summary

    # ==========================================================
    # COMBINE LOAD + ANALYSIS
    # ==========================================================
    def summarize_logs(self, prefix_filter=None, days=None):
        """
        Load logs, compute and return summary stats.
        Args:
            prefix_filter (str): Limit to logs for this module (e.g., 'rl_trader').
            days (int): Only include recent logs.
        Returns:
            dict: summary statistics.
        """
        df = self.load_logs(prefix_filter=prefix_filter, days=days)
        summary = self.compute_summary(df)
        return summary, df