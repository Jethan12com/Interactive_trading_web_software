import asyncio
import numpy as np
import pandas as pd
from arch import arch_model
from datetime import datetime, timedelta
import pytz
import aiofiles
import json

from modules.logger_setup import setup_logger
from modules.altdata_engine import AlternativeDataEngine
from modules.data_provider import MultiProviderDataProvider


class VolatilityForecaster:
    """
    Asynchronous volatility forecasting engine.

    Supports:
    - GARCH(1,1)
    - RealGARCH (with bipower variation, sentiment, and implied volatility)
    """

    def __init__(self, data_provider: MultiProviderDataProvider):
        self.data_provider = data_provider
        self.alt_data_engine = AlternativeDataEngine(data_provider)
        self.logger = setup_logger("VolatilityForecaster", "copilot/modules/logs/volatility_forecaster.log")
        self.volatility_cache = {}
        self.lock = asyncio.Lock()

    # -----------------------------
    # Utility methods
    # -----------------------------
    async def _compute_bipower_variation(self, df: pd.DataFrame) -> float:
        """Compute bipower variation to mitigate microstructure noise."""
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            bv = np.sum(np.abs(returns[:-1]) * np.abs(returns[1:])) * (np.pi / 2)
            return np.sqrt(bv * (252 * 288)) / 100  # Annualized (288 5-min intervals/day)
        except Exception as e:
            self.logger.error({"event": "bipower_variation_error", "error": str(e)})
            return 0.0

    async def _fetch_implied_volatility(self, pair: str, session: str) -> float:
        """Fetch implied volatility from options data via data provider."""
        try:
            df_options = await self.data_provider.fetch_options_data([pair], session)
            if df_options.empty or 'implied_volatility' not in df_options.columns:
                self.logger.warning({"event": "no_iv_data", "pair": pair, "session": session})
                return 0.0
            return df_options['implied_volatility'].mean()
        except Exception as e:
            self.logger.error({"event": "fetch_iv_error", "pair": pair, "session": session, "error": str(e)})
            return 0.0

    # -----------------------------
    # Main Forecasting Interface
    # -----------------------------
    async def forecast(self, df: pd.DataFrame, pair: str, session: str, model: str = "garch11", interval: str = '5min') -> float:
        """
        Forecast volatility using either:
        - GARCH(1,1)
        - RealGARCH (with alternative/exogenous data)
        """
        try:
            start_time = asyncio.get_event_loop().time()
            async with self.lock:
                cache_key = f"{pair}:{session}:{interval}:{model}"
                cache = self.volatility_cache.get(cache_key, {})
                if cache.get('timestamp') and (datetime.now(pytz.UTC) - cache['timestamp']) < timedelta(minutes=5):
                    self.logger.info({"event": "volatility_cache_hit", "pair": pair, "model": model})
                    return cache['volatility']

                # Refresh data if missing or stale
                if df is None or df.empty or 'timestamp' not in df.columns or df['timestamp'].max() < datetime.now(pytz.UTC) - timedelta(minutes=30):
                    df = await self.data_provider.fetch_live([pair], session, interval=interval)

                if df.empty or 'close' not in df.columns:
                    self.logger.warning({"event": "no_data_for_forecast", "pair": pair, "session": session})
                    return 0.01

                # Compute returns (%)
                df['returns'] = df['close'].pct_change().dropna() * 100
                if df['returns'].empty:
                    self.logger.warning({"event": "no_returns_data", "pair": pair})
                    return 0.01

                # GARCH(1,1) model
                if model.lower() == "garch11":
                    vol = await self._fit_garch(df['returns'], model_name="GARCH(1,1)")
                else:
                    vol = await self._fit_realgarch(df, pair, session, interval)

                # Cache and export
                self.volatility_cache[cache_key] = {'volatility': vol, 'timestamp': datetime.now(pytz.UTC)}
                await self._export_metrics(pair, session, model, vol)
                elapsed = asyncio.get_event_loop().time() - start_time
                self.logger.info({"event": "volatility_forecast_complete", "pair": pair, "model": model, "volatility": vol, "elapsed": round(elapsed, 3)})
                return vol

        except Exception as e:
            self.logger.error({"event": "volatility_forecast_error", "pair": pair, "model": model, "error": str(e)})
            return 0.01

    # -----------------------------
    # Model Implementations
    # -----------------------------
    async def _fit_garch(self, returns: pd.Series, model_name: str = "GARCH(1,1)") -> float:
        """Fit basic GARCH(1,1) model."""
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal', rescale=True)
            fitted = await asyncio.to_thread(model.fit, disp='off', options={'maxiter': 50})
            forecast = fitted.forecast(horizon=1)
            vol = float(np.sqrt(forecast.variance.values[-1, :][0]) / 100)
            return vol
        except Exception as e:
            self.logger.error({"event": "fit_garch_error", "model": model_name, "error": str(e)})
            return 0.01

    async def _fit_realgarch(self, df: pd.DataFrame, pair: str, session: str, interval: str) -> float:
        """Fit RealGARCH model with exogenous signals."""
        try:
            if len(df) < 12:
                self.logger.warning({"event": "insufficient_data_for_realgarch", "pair": pair})
                return 0.01

            # Compute realized measure
            realized_measure = await self._compute_bipower_variation(df)

            # Fetch external factors (sentiment, event impact, implied volatility)
            edge_signals, iv = await asyncio.gather(
                self.alt_data_engine.get_edge_signals([pair], session),
                self._fetch_implied_volatility(pair, session)
            )
            sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
            event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0

            returns = df['returns']
            exog = pd.DataFrame({
                'realized_measure': [realized_measure] * len(returns),
                'sentiment': [sentiment] * len(returns),
                'event_impact': [event_impact] * len(returns),
                'implied_volatility': [iv] * len(returns)
            })

            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                mean='Zero',
                dist='normal',
                rescale=True
            )
            fitted_model = await asyncio.to_thread(
                model.fit,
                disp='off',
                options={'maxiter': 50},
                cov_type='robust',
                exog=exog
            )

            forecast = fitted_model.forecast(horizon=1, reindex=False)
            volatility = np.sqrt(forecast.variance.values[-1, :][0]) / 100
            volatility *= (1 + 0.2 * event_impact + 0.1 * abs(sentiment) + 0.15 * iv)
            return volatility

        except Exception as e:
            self.logger.error({"event": "fit_realgarch_error", "pair": pair, "error": str(e)})
            return 0.01

    # -----------------------------
    # Export / Dashboard
    # -----------------------------
    async def _export_metrics(self, pair: str, session: str, model: str, volatility: float):
        """Export forecast results to JSON for visualization."""
        try:
            metrics = {
                "pair": pair,
                "session": session,
                "model": model,
                "volatility": float(volatility),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            async with aiofiles.open(f"copilot/modules/logs/volatility_metrics_{pair}_{session}.json", mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(metrics, indent=4))
        except Exception as e:
            self.logger.error({"event": "export_metrics_error", "pair": pair, "error": str(e)})