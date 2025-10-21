"""
backtest_toolkit.py

Contains:
 - DatabaseManager (SQLite + Parquet)
 - RollingBacktester (with auto retrain, slippage & commission)
 - MonteCarloTester (drawdown & returns distribution)
 - WalkForwardCV (walk-forward cross validation)
 - Utilities for docker/service deployment info at the bottom
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from sqlalchemy import create_engine
import logging
from typing import Optional, Tuple, Dict, Any, List

# Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'backtest_toolkit.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------
# DatabaseManager (SQLite + Parquet)
# -------------------------
class DatabaseManager:
    def __init__(self, sqlite_path='models/model_runs.db', parquet_dir='models/parquet_runs'):
        self.sqlite_path = sqlite_path
        self.parquet_dir = parquet_dir
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)
        self.engine = create_engine(f'sqlite:///{self.sqlite_path}')

    def save_run(self, run_meta: dict, trades_df: pd.DataFrame, metrics: dict):
        run_id = run_meta.get('run_id', f"run_{int(time.time())}")
        run_meta = run_meta.copy()
        run_meta['run_id'] = run_id
        run_meta['saved_at'] = pd.Timestamp.utcnow()

        runs_df = pd.DataFrame([{
            **run_meta,
            **{f'metric_{k}': v for k, v in metrics.items()}
        }])
        runs_df.to_sql('runs', con=self.engine, if_exists='append', index=False)

        trades = trades_df.copy()
        trades['run_id'] = run_id
        trades.to_sql('trades', con=self.engine, if_exists='append', index=False)

        parquet_path = os.path.join(self.parquet_dir, f"{run_id}.parquet")
        trades.to_parquet(parquet_path, index=False)

        logging.info(f"Saved run {run_id} to sqlite and parquet")
        return run_id

    def list_runs(self):
        try:
            return pd.read_sql('SELECT * FROM runs ORDER BY saved_at DESC', con=self.engine)
        except Exception:
            return pd.DataFrame()

    def load_trades(self, run_id: str):
        try:
            df = pd.read_sql(f"SELECT * FROM trades WHERE run_id = '{run_id}'", con=self.engine)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame()

    def save_outcomes_history(self, outcomes_df: pd.DataFrame, table_name='outcomes_history'):
        """Append outcomes history used for retraining to a table."""
        outcomes_df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
        logging.info(f"Appended {len(outcomes_df)} rows to {table_name}")

    def load_outcomes_window(self, end_date: pd.Timestamp, window_days: int, pair: Optional[str]=None,
                             table_name='outcomes_history') -> pd.DataFrame:
        """
        Load outcomes from DB with timestamp >= end_date - window_days and < = end_date
        """
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=window_days)).isoformat()
        end_date_s = pd.to_datetime(end_date).isoformat()
        query = f"SELECT * FROM {table_name} WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date_s}'"
        if pair:
            query += f" AND pair = '{pair}'"
        try:
            df = pd.read_sql(query, con=self.engine)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame()

# -------------------------
# Utility functions
# -------------------------
def apply_slippage_and_commission(pnl: float, entry_price: float, slippage: float, commission: float, qty: float=1.0):
    """
    Apply slippage (fraction of entry_price per side) and fixed commission (in same units as pnl)
    slippage: e.g. 0.0005 -> 0.05% slippage per round trip or per side depending how used.
    commission: absolute value per trade (or per unit qty)
    qty: units traded (for scaling)
    Returns adjusted pnl.
    """
    # Treat slippage as cost relative to entry price * qty
    slippage_cost = slippage * entry_price * qty
    total_cost = slippage_cost + commission
    return pnl - total_cost

def compute_drawdown_from_equity(equity: pd.Series):
    eq = equity.fillna(method='ffill').fillna(0)
    running_max = eq.cummax()
    drawdown = eq - running_max
    drawdown_pct = drawdown / running_max.replace(0, np.nan)
    dd_df = pd.DataFrame({'equity': eq, 'running_max': running_max, 'drawdown': drawdown, 'drawdown_pct': drawdown_pct})
    return dd_df

# -------------------------
# RollingBacktester with automatic retrain
# -------------------------
class RollingBacktester:
    def __init__(self, model, dbmgr: Optional[DatabaseManager]=None, timeout_hours=4,
                 default_slippage=0.0, default_commission=0.0, qty=1.0):
        """
        model: MLModel-like object with .predict(signals_df) and .train(signals_df, outcomes_df)
        dbmgr: DatabaseManager to read/write historical outcomes for retraining
        default_slippage: fraction of price (e.g., 0.0005 = 0.05%) applied per trade (round-trip cost)
        default_commission: absolute commission applied per trade
        qty: units per trade (used to scale slippage cost)
        """
        self.model = model
        self.dbmgr = dbmgr
        self.timeout = timedelta(hours=timeout_hours)
        self.default_slippage = default_slippage
        self.default_commission = default_commission
        self.qty = qty

    @staticmethod
    def _simulate_single(signal_row, market_slice, slippage, commission, qty):
        entry_price = float(signal_row.get('entry_price', signal_row.get('close', np.nan)))
        tp = float(signal_row['tp'])
        sl = float(signal_row['sl'])
        direction = signal_row['signal']
        start_time = pd.to_datetime(signal_row['timestamp'])

        for _, r in market_slice.iterrows():
            t = pd.to_datetime(r['timestamp'])
            if t < start_time:
                continue
            price_high = float(r.get('high', r.get('close', np.nan)))
            price_low = float(r.get('low', r.get('close', np.nan)))

            if direction == 'BUY':
                if price_high >= tp:
                    pnl = tp - entry_price
                    pnl_adj = apply_slippage_and_commission(pnl, entry_price, slippage, commission, qty)
                    return {'outcome': 'TP', 'pnl': pnl_adj, 'exit_price': tp, 'exit_time': t}
                if price_low <= sl:
                    pnl = sl - entry_price
                    pnl_adj = apply_slippage_and_commission(pnl, entry_price, slippage, commission, qty)
                    return {'outcome': 'SL', 'pnl': pnl_adj, 'exit_price': sl, 'exit_time': t}
            else:  # SELL
                if price_low <= tp:
                    pnl = entry_price - tp
                    pnl_adj = apply_slippage_and_commission(pnl, entry_price, slippage, commission, qty)
                    return {'outcome': 'TP', 'pnl': pnl_adj, 'exit_price': tp, 'exit_time': t}
                if price_high >= sl:
                    pnl = entry_price - sl
                    pnl_adj = apply_slippage_and_commission(pnl, entry_price, slippage, commission, qty)
                    return {'outcome': 'SL', 'pnl': pnl_adj, 'exit_price': sl, 'exit_time': t}

        return {'outcome': 'OPEN', 'pnl': 0.0, 'exit_price': None, 'exit_time': None}

    def simulate(self,
                 signals_df: pd.DataFrame,
                 market_df: pd.DataFrame,
                 use_prediction_class: bool=True,
                 prediction_threshold: float=0.0,
                 retrain: bool=False,
                 retrain_window_days: int=30,
                 retrain_freq_days: int=7,
                 outcomes_table_name='outcomes_history',
                 slippage_col: Optional[str]=None,
                 commission_col: Optional[str]=None):
        """
        Main simulation loop with optional automatic retraining.

        retrain=True:
          - For each retrain date (every retrain_freq_days starting from first signal + retrain_window_days),
            load outcomes from DB for previous retrain_window_days and call model.train() on them if available.
          - This requires outcomes_history table to have columns compatible with model.train (pair, timestamp, pnl, outcome, plus features).
        """

        signals = signals_df.copy().sort_values('timestamp').reset_index(drop=True)
        market = market_df.copy()
        market['timestamp'] = pd.to_datetime(market['timestamp'])

        # predict once initially (later we may predict again after retrain)
        # We'll run simulation chronologically; if retrain==True, we'll retrain at specified dates and re-predict subsequent signals.
        start_ts = pd.to_datetime(signals['timestamp'].min())
        end_ts = pd.to_datetime(signals['timestamp'].max())

        # compute retrain schedule (dates at which to retrain BEFORE processing signals occurring after that date)
        retrain_dates = []
        if retrain:
            first_retrain = start_ts + pd.Timedelta(days=retrain_window_days)
            current = pd.to_datetime(first_retrain)
            while current <= end_ts + pd.Timedelta(days=1):
                retrain_dates.append(current)
                current += pd.Timedelta(days=retrain_freq_days)

        executed = []
        # We'll iterate signals in chronological order and retrain when passing retrain_date
        retrain_idx = 0

        # helper to perform retrain if DB has enough data
        def try_retrain(as_of_date):
            if self.dbmgr is None:
                logging.warning("Retrain requested but no DB manager provided. Skipping retrain.")
                return False
            window_df = self.dbmgr.load_outcomes_window(end_date=as_of_date, window_days=retrain_window_days)
            if window_df.empty:
                logging.info(f"No outcomes in DB for window ending {as_of_date}. Skipping retrain.")
                return False
            try:
                # Note: model.train expects signals_df and outcomes_df; we assume outcomes_df contains necessary columns
                # If your model.train expects signals+outcomes merged, adapt accordingly.
                # We'll assume window_df contains signals-like rows (pair,timestamp,features) AND pnl/outcome.
                logging.info(f"Retraining model with {len(window_df)} rows ending {as_of_date}")
                # Shuffle window_df into signals/outcomes parts for train call
                # We attempt to reconstruct signals_df subset (features) from window_df; adjust if needed.
                # Use same columns as preprocess_data expects in your MLModel
                signals_cols = ['pair', 'timestamp', 'adjusted_signal_score', 'volatility_score', 'session', 'pattern_id']
                if all(c in window_df.columns for c in signals_cols):
                    signals_for_train = window_df[signals_cols].copy()
                else:
                    # If DB stores only outcomes, we pass entire window_df; your model's preprocess_data should accept the minimal fields
                    signals_for_train = window_df.copy()

                outcomes_for_train = window_df[['pair', 'timestamp', 'outcome', 'pnl']].copy()
                self.model.train(signals_for_train, outcomes_for_train)
                logging.info("Retrain completed successfully.")
                return True
            except Exception as e:
                logging.error(f"Retrain failed at {as_of_date}: {e}")
                return False

        # iterate signals
        for _, sig in signals.iterrows():
            sig_ts = pd.to_datetime(sig['timestamp'])

            # if retrain scheduled and current signal time >= next retrain date, retrain first
            if retrain and retrain_idx < len(retrain_dates) and sig_ts >= retrain_dates[retrain_idx]:
                try_retrain(retrain_dates[retrain_idx])
                retrain_idx += 1
                # after retrain we may want to update predictions for all future signals (naive strategy: re-call predict for the remaining slice)
                # For simplicity, we'll continue predicting on-the-fly below (call predict for the single signal when needed).

            # Get prediction for this signal (model.predict expects DataFrame)
            try:
                pred = float(self.model.predict(pd.DataFrame([sig]))[0])
            except Exception as e:
                logging.error(f"Prediction failed for signal at {sig_ts}: {e}")
                pred = 0.0

            # decide whether to take trade
            take = pred > prediction_threshold if not use_prediction_class else (pred > prediction_threshold)

            # support per-trade slippage/commission columns
            trade_slippage = float(sig.get(slippage_col)) if slippage_col and pd.notnull(sig.get(slippage_col)) else self.default_slippage
            trade_commission = float(sig.get(commission_col)) if commission_col and pd.notnull(sig.get(commission_col)) else self.default_commission

            if not take:
                executed.append({**sig.to_dict(), 'predicted_pnl': pred, 'outcome': 'SKIPPED', 'pnl': 0.0, 'exit_time': pd.NaT, 'exit_price': None})
                continue

            # get market slice from timestamp to timestamp + timeout
            start_time = sig_ts
            end_time = start_time + self.timeout
            mask = (market['pair'] == sig['pair']) & (market['timestamp'] >= start_time) & (market['timestamp'] <= end_time)
            market_slice = market.loc[mask].sort_values('timestamp')

            if market_slice.empty:
                sres = {'outcome': 'OPEN', 'pnl': 0.0, 'exit_price': None, 'exit_time': None}
            else:
                sres = self._simulate_single(sig, market_slice, trade_slippage, trade_commission, self.qty)

            executed.append({**sig.to_dict(), 'predicted_pnl': pred, **sres})

            # Optionally append this trade result to DB outcomes so future retrain windows include it
            if self.dbmgr is not None:
                try:
                    # create a small outcomes row matching required columns and append
                    outcomes_row = pd.DataFrame([{
                        'pair': sig.get('pair'),
                        'timestamp': sig.get('timestamp'),
                        'outcome': sres['outcome'],
                        'pnl': sres['pnl'],
                        # include model features if available - this helps retrain later
                        'adjusted_signal_score': sig.get('adjusted_signal_score'),
                        'volatility_score': sig.get('volatility_score'),
                        'session': sig.get('session'),
                        'pattern_id': sig.get('pattern_id')
                    }])
                    self.dbmgr.save_outcomes_history(outcomes_row)
                except Exception as e:
                    logging.error(f"Failed to save outcome to DB: {e}")

        trades_df = pd.DataFrame(executed)
        # create ord_time for equity curve ordering
        trades_df['ord_time'] = pd.to_datetime(trades_df.get('exit_time')).fillna(pd.to_datetime(trades_df['timestamp']))
        trades_df = trades_df.sort_values('ord_time').reset_index(drop=True)
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()

        return trades_df

# -------------------------
# Monte Carlo Tester
# -------------------------
class MonteCarloTester:
    def __init__(self, trades_df: pd.DataFrame):
        """
        trades_df: DataFrame of executed trades with 'pnl' column
        """
        self.trades = trades_df.copy()

    def simulate(self, n_iterations=1000, replace=True, min_trades=None, seed: Optional[int]=42):
        """
        Resample trades (with replacement by default) to build MC distribution of equity curves and drawdowns.
        Returns a dict with distributions of max_drawdown, total_return, annualized_return_est (naive)
        """
        rng = np.random.default_rng(seed)
        pnl = self.trades['pnl'].values
        n = len(pnl)
        if min_trades is None:
            min_trades = n

        max_drawdowns = []
        total_returns = []
        final_equities = []

        for i in range(n_iterations):
            idx = rng.integers(0, n, size=n) if replace else rng.choice(np.arange(n), size=n, replace=False)
            sampled = pnl[idx]
            equity = np.cumsum(sampled)
            running_max = np.maximum.accumulate(equity)
            drawdown = equity - running_max
            max_dd = drawdown.min()
            max_drawdowns.append(max_dd)
            total_returns.append(equity[-1])
            final_equities.append(equity[-1])

        return {
            'max_drawdowns': np.array(max_drawdowns),
            'total_returns': np.array(total_returns),
            'final_equities': np.array(final_equities)
        }

    def summarize(self, mc_result: dict):
        """
        Return summary stats for MC result dictionary
        """
        stats = {
            'median_max_drawdown': np.median(mc_result['max_drawdowns']),
            '95pct_max_drawdown': np.percentile(mc_result['max_drawdowns'], 95),
            'median_return': np.median(mc_result['total_returns']),
            '95pct_return': np.percentile(mc_result['total_returns'], 95),
        }
        return stats

# -------------------------
# Walk-Forward Cross-Validation
# -------------------------
class WalkForwardCV:
    def __init__(self, model, initial_train_days: int=60, test_days: int=7, expand_window: bool=True):
        """
        model: MLModel-like object with .train() and .predict()
        initial_train_days: initial training window size (days)
        test_days: length of each test fold (days)
        expand_window: if True, training window expands by adding previous test fold; else fixed-length rolling window
        """
        self.model = model
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.expand_window = expand_window

    def run(self, signals_df: pd.DataFrame, outcomes_df: pd.DataFrame, max_folds: Optional[int]=None):
        """
        signals_df: all signals (with features)
        outcomes_df: ground-truth outcomes (pair,timestamp,outcome,pnl)
        Returns list of fold metrics and aggregated metrics.
        """
        signals = signals_df.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        outcomes = outcomes_df.copy()
        outcomes['timestamp'] = pd.to_datetime(outcomes['timestamp'])

        start_time = signals['timestamp'].min()
        train_start = start_time
        train_end = train_start + pd.Timedelta(days=self.initial_train_days)

        folds = []
        fold_idx = 0
        while train_end < signals['timestamp'].max():
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_days)

            # build train/test sets
            train_mask = (signals['timestamp'] >= train_start) & (signals['timestamp'] < train_end)
            test_mask = (signals['timestamp'] >= test_start) & (signals['timestamp'] < test_end)
            train_signals = signals.loc[train_mask]
            test_signals = signals.loc[test_mask]

            # outcomes for training
            train_outcomes_mask = (outcomes['timestamp'] >= train_start) & (outcomes['timestamp'] < train_end)
            train_outcomes = outcomes.loc[train_outcomes_mask]

            if train_signals.empty or test_signals.empty:
                # advance window
                train_end = test_end if self.expand_window else train_end + pd.Timedelta(days=self.test_days)
                if self.expand_window:
                    # train_start unchanged (expanding)
                    pass
                fold_idx += 1
                if max_folds and fold_idx >= max_folds:
                    break
                continue

            # (re)train model on train_signals/train_outcomes
            try:
                self.model.train(train_signals, train_outcomes)
            except Exception as e:
                logging.error(f"Train failed during walk-forward fold {fold_idx}: {e}")
                break

            # predict on test_signals
            y_pred = np.array(self.model.predict(test_signals)).flatten()
            test_outcomes = outcomes.loc[test_mask]
            # align predictions and actuals by pair+timestamp (best-effort)
            # Create DF for metric calc
            preds_df = test_signals.copy().reset_index(drop=True)
            preds_df['predicted_pnl'] = y_pred
            # merge actual pnl where available
            merged = preds_df.merge(test_outcomes[['pair','timestamp','pnl','outcome']], on=['pair','timestamp'], how='left')
            merged['pnl'] = merged['pnl'].fillna(0.0)
            merged['is_win'] = merged['outcome'] == 'TP'

            # compute fold metrics
            pred_class = (merged['predicted_pnl'] > 0).astype(int)
            from sklearn.metrics import f1_score
            try:
                f1 = f1_score(merged['is_win'].astype(int), pred_class)
            except Exception:
                f1 = np.nan
            win_rate = (merged['pnl'] > 0).mean()
            total_pnl = merged['pnl'].sum()

            fold_metrics = {
                'fold': fold_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'n_train': len(train_signals),
                'n_test': len(test_signals),
                'f1': f1,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            }
            folds.append(fold_metrics)

            # advance windows
            if self.expand_window:
                train_end = test_end  # expand training to include test period
            else:
                train_start = train_start + pd.Timedelta(days=self.test_days)
                train_end = train_end + pd.Timedelta(days=self.test_days)

            fold_idx += 1
            if max_folds and fold_idx >= max_folds:
                break

        return pd.DataFrame(folds)

# -------------------------
# Deployment artifacts
# -------------------------
DOCKERFILE = r"""
# Dockerfile for Dash app hosting the model dashboard
FROM python:3.10-slim

WORKDIR /app

# copy requirements (you can customize)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code
COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8050

# run app using gunicorn for production (dash app named 'app' in file app.py)
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server", "--workers", "2", "--timeout", "120"]
"""

SYSTEMD_SERVICE = r"""
# systemd unit file: /etc/systemd/system/dash-model.service
[Unit]
Description=Dash Model Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/srv/dash_app
Environment=PATH=/srv/dash_app/venv/bin
ExecStart=/srv/dash_app/venv/bin/gunicorn --bind 0.0.0.0:8050 app:server --workers 2 --timeout 120
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""

# -------------------------
# Example usage helper
# -------------------------
def example_workflow(ml_model, signals_df, market_df, db_path='models/model_runs.db'):
    """
    Quick demonstration of a full workflow:
      - create DB manager
      - run rolling backtester with retrain using DB
      - perform Monte Carlo simulation
      - perform walk-forward CV
    """
    dbmgr = DatabaseManager(sqlite_path=db_path)
    backtester = RollingBacktester(model=ml_model, dbmgr=dbmgr, default_slippage=0.0005, default_commission=0.5, qty=1.0)

    # simulate with retrain every 7 days using 30-day history
    trades = backtester.simulate(signals_df, market_df, use_prediction_class=True, prediction_threshold=0.0,
                                 retrain=True, retrain_window_days=30, retrain_freq_days=7,
                                 outcomes_table_name='outcomes_history')

    # compute equity and save
    trades['cum_pnl'] = trades['pnl'].cumsum()
    metrics = {
        'total_trades': len(trades),
        'win_rate': float((trades['pnl'] > 0).mean()),
        'total_pnl': float(trades['pnl'].sum())
    }
    run_meta = {'model_type': getattr(ml_model, 'model_type', 'mlp'), 'notes': 'auto retrain example'}
    run_id = dbmgr.save_run(run_meta, trades, metrics)

    # Monte Carlo
    mc = MonteCarloTester(trades)
    mc_res = mc.simulate(n_iterations=2000)
    mc_summary = mc.summarize(mc_res)

    # Walk-forward CV
    # requires outcomes history in DB; for quick example we'll fetch what's in outcomes_history and run
    outcomes_history = dbmgr.load_outcomes_window(end_date=pd.Timestamp.utcnow(), window_days=365)
    wfcv = WalkForwardCV(ml_model, initial_train_days=60, test_days=7, expand_window=True)
    if not outcomes_history.empty:
        wfcv_results = wfcv.run(signals_df, outcomes_history, max_folds=20)
    else:
        wfcv_results = pd.DataFrame()

    return {'run_id': run_id, 'trades': trades, 'mc_summary': mc_summary, 'wfcv': wfcv_results}

# End of backtest_toolkit.py
