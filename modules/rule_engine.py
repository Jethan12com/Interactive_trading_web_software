# rule_engine.py
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import pandas_ta as ta
from prometheus_client import Counter, Gauge
from .utils import Utils
from .config_manager import ConfigManager
from .logger_setup import setup_logger
from .indicators import get_rsi, get_macd, get_bbands, get_obv, get_mfi, get_adx
from .altdata_engine import AlternativeDataEngine

# -------------------------
# Prometheus metrics
# -------------------------
rule_gen_success = Counter(
    'rule_gen_success_total', 'Successful rule-based signal generations', ['strategy', 'pair']
)
rule_gen_failure = Counter(
    'rule_gen_failure_total', 'Failed rule-based signal generations', ['strategy', 'pair']
)
rule_gen_duration = Gauge(
    'rule_gen_duration_seconds', 'Duration of rule-based signal generations', ['strategy', 'pair']
)


class RuleEngine:
    """
    Production-ready Modular Hybrid Rule-based Trading Engine.
    Supports async sentiment integration, dynamic strategy registration, and Prometheus monitoring.
    """

    def __init__(self, config_manager: ConfigManager, data_provider):
        self.config = config_manager
        self.data_provider = data_provider
        self.alt_data_engine = AlternativeDataEngine(data_provider)
        self.utils = Utils()
        self.logger = setup_logger("RuleEngine", "logs/rule_engine.log", to_console=True)
        self.strategies = {}
        self.register_default_strategies()

    # ===========================
    # === STRATEGY REGISTRATION
    # ===========================
    def register_strategy(self, name: str, func):
        """Register a new strategy (sync or async)."""
        if name in self.strategies:
            self.logger.warning(f"Strategy '{name}' is being overwritten")
        self.strategies[name] = func
        self.logger.info(f"Strategy '{name}' registered successfully")

    def register_default_strategies(self):
        """Register default built-in strategies."""
        defaults = {
            'ma_crossover': self._ma_crossover,
            'rsi_momentum': self._rsi_momentum,
            'macd_crossover': self._macd_crossover,
            'bollinger_bands': self._bollinger_bands,
            'rsi_obv_confirmation': self._rsi_obv_confirmation,
            'macd_adx_filter': self._macd_adx_filter,
            'bollinger_mfi_reversal': self._bollinger_mfi_reversal,
            'hybrid': self.generate_hybrid_signal
        }
        for name, func in defaults.items():
            self.register_strategy(name, func)

    # ===========================
    # === SIGNAL GENERATION
    # ===========================
    async def generate(self, pair: str, df: pd.DataFrame, user_profile: dict,
                       strategy: str = 'rsi_momentum', session: str = None) -> dict:
        start_time = asyncio.get_event_loop().time()
        try:
            if strategy not in self.strategies:
                self.logger.error({"event": "unknown_strategy", "strategy": strategy, "pair": pair})
                rule_gen_failure.labels(strategy=strategy, pair=pair).inc()
                return self._default_signal(pair, strategy)

            # --- Fetch sentiment & event impact ---
            edge_signals = await self.alt_data_engine.get_edge_signals([pair], session)
            news_sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
            event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0

            # --- Call strategy (sync or async) ---
            strat_func = self.strategies[strategy]
            if asyncio.iscoroutinefunction(strat_func):
                signal = await strat_func(df, news_sentiment, event_impact)
            else:
                signal = strat_func(df)

            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 1.0 if action != 'HOLD' else 0.0)
            confidence *= 1 + 0.1 * news_sentiment + 0.1 * event_impact

            # --- Risk-adjusted TP/SL ---
            price = df['close'].iloc[-1] if not df.empty else 0.0
            volatility = await self.data_provider.forecast_volatility(df, pair) if hasattr(self.data_provider, 'forecast_volatility') else 0.01
            risk_multiplier = self._risk_multiplier(user_profile.get('risk_profile', 'Medium'))
            trading_cfg = self.config.get_config('trading') or {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
            stop_loss_pct = trading_cfg['stop_loss_pct'] * risk_multiplier * (1 + volatility)
            take_profit_pct = trading_cfg['take_profit_pct'] * risk_multiplier * (1 + volatility)

            signal.update({
                'pair': pair,
                'strategy': strategy,
                'direction': action,
                'confidence': min(confidence, 1.0),
                'sentiment_score': news_sentiment,
                'event_impact': event_impact,
                'tp': price * (1 + take_profit_pct) if action == 'BUY' else price * (1 - take_profit_pct),
                'sl': price * (1 - stop_loss_pct) if action == 'BUY' else price * (1 + stop_loss_pct),
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'session': session
            })

            rule_gen_success.labels(strategy=strategy, pair=pair).inc()
            rule_gen_duration.labels(strategy=strategy, pair=pair).set(asyncio.get_event_loop().time() - start_time)
            self.logger.info({"event": "generate_signal_success", "pair": pair, "strategy": strategy, "session": session})
            return signal

        except Exception as e:
            self.logger.error({"event": "generate_signal_error", "pair": pair, "strategy": strategy, "session": session, "error": str(e)})
            rule_gen_failure.labels(strategy=strategy, pair=pair).inc()
            return self._default_signal(pair, strategy)

    # ===========================
    # === HYBRID SIGNAL GENERATION
    # ===========================
    async def generate_hybrid_signal(self, pair: str, df: pd.DataFrame, user_profile: dict, session: str = None) -> dict:
        start_time = asyncio.get_event_loop().time()
        try:
            if df is None or df.empty:
                return self._default_signal(pair, "hybrid")

            edge_signals = await self.alt_data_engine.get_edge_signals([pair], session)
            news_sentiment = edge_signals[edge_signals['pair'] == pair]['news_sentiment'].iloc[0] if not edge_signals.empty else 0.0
            event_impact = edge_signals[edge_signals['pair'] == pair]['event_impact'].iloc[0] if not edge_signals.empty else 0.0

            # --- Indicators ---
            rsi = get_rsi(df)
            macd_df = get_macd(df)
            obv = get_obv(df)
            adx = get_adx(df)
            mfi = get_mfi(df)
            bb = get_bbands(df)
            close = df['close'].iloc[-1]

            rsi_score = np.interp(rsi.iloc[-1], [0, 100], [-1, 1])
            macd_score = np.sign(macd_df['MACD_12_26_9'].iloc[-1] - macd_df['MACDs_12_26_9'].iloc[-1])
            obv_score = np.sign(obv.iloc[-1] - obv.iloc[-2])
            adx_score = np.interp(adx.iloc[-1], [10, 50], [0, 1])
            mfi_score = np.interp(50 - abs(mfi.iloc[-1] - 50), [0, 50], [-1, 1])
            bb_score = 1 if close < bb['BBL_20_2'].iloc[-1] else -1 if close > bb['BBU_20_2'].iloc[-1] else 0

            weights = {'rsi': 0.25, 'macd': 0.25, 'obv': 0.15, 'adx': 0.2, 'mfi': 0.1, 'bb': 0.05}
            total_score = (
                rsi_score * weights['rsi'] +
                macd_score * weights['macd'] +
                obv_score * weights['obv'] +
                adx_score * weights['adx'] +
                mfi_score * weights['mfi'] +
                bb_score * weights['bb']
            )

            total_score += 0.2 * news_sentiment + 0.1 * event_impact

            if total_score > 0.2:
                action = 'BUY'
            elif total_score < -0.2:
                action = 'SELL'
            else:
                action = 'HOLD'

            confidence = round(min(abs(total_score), 1.0), 3)

            volatility = await self.data_provider.forecast_volatility(df, pair) if hasattr(self.data_provider, 'forecast_volatility') else 0.01
            risk_multiplier = self._risk_multiplier(user_profile.get('risk_profile', 'Medium'))
            trading_cfg = self.config.get_config('trading') or {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
            stop_loss_pct = trading_cfg['stop_loss_pct'] * risk_multiplier * (1 + volatility)
            take_profit_pct = trading_cfg['take_profit_pct'] * risk_multiplier * (1 + volatility)

            tp = close * (1 + take_profit_pct) if action == 'BUY' else close * (1 - take_profit_pct) if action == 'SELL' else 0.0
            sl = close * (1 - stop_loss_pct) if action == 'BUY' else close * (1 + stop_loss_pct) if action == 'SELL' else 0.0

            signal = {
                'pair': pair,
                'strategy': 'hybrid',
                'direction': action,
                'confidence': confidence,
                'indicator_score': round(total_score, 3),
                'sentiment_score': news_sentiment,
                'event_impact': event_impact,
                'tp': tp,
                'sl': sl,
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'session': session
            }

            rule_gen_success.labels(strategy='hybrid', pair=pair).inc()
            rule_gen_duration.labels(strategy='hybrid', pair=pair).set(asyncio.get_event_loop().time() - start_time)
            self.logger.info({"event": "generate_hybrid_signal_success", "pair": pair, "session": session, "score": total_score})
            return signal

        except Exception as e:
            self.logger.error({"event": "generate_hybrid_signal_error", "pair": pair, "session": session, "error": str(e)})
            rule_gen_failure.labels(strategy='hybrid', pair=pair).inc()
            return self._default_signal(pair, 'hybrid')

    # ===========================
    # === DEFAULT STRATEGIES (SYNC)
    # ===========================
    def _ma_crossover(self, df: pd.DataFrame) -> dict:
        try:
            fast = ta.sma(df['close'], length=10)
            slow = ta.sma(df['close'], length=50)
            if len(fast) < 2 or len(slow) < 2:
                return {'action': 'HOLD', 'confidence': 0.0}

            if fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]:
                return {'action': 'BUY', 'confidence': 0.8}
            elif fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]:
                return {'action': 'SELL', 'confidence': 0.8}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"MA Crossover failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _rsi_momentum(self, df: pd.DataFrame, news_sentiment: float = 0.0, event_impact: float = 0.0) -> dict:
        try:
            rsi = get_rsi(df)
            if len(rsi) < 2:
                return {'action': 'HOLD', 'confidence': 0.0}
            rsi_current, rsi_prev = rsi.iloc[-1], rsi.iloc[-2]
            if rsi_current < 30 and rsi_prev >= 30:
                return {'action': 'BUY', 'confidence': 0.9}
            elif rsi_current > 70 and rsi_prev <= 70:
                return {'action': 'SELL', 'confidence': 0.9}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"RSI Momentum failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _macd_crossover(self, df: pd.DataFrame, news_sentiment: float = 0.0, event_impact: float = 0.0) -> dict:
        try:
            macd_df = get_macd(df)
            macd_line = macd_df['MACD_12_26_9']
            signal_line = macd_df['MACDs_12_26_9']
            if len(macd_line) < 2:
                return {'action': 'HOLD', 'confidence': 0.0}

            macd_current, macd_prev = macd_line.iloc[-1], macd_line.iloc[-2]
            sig_current, sig_prev = signal_line.iloc[-1], signal_line.iloc[-2]

            if macd_current > sig_current and macd_prev <= sig_prev:
                return {'action': 'BUY', 'confidence': 0.85}
            elif macd_current < sig_current and macd_prev >= sig_prev:
                return {'action': 'SELL', 'confidence': 0.85}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"MACD Crossover failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _bollinger_bands(self, df: pd.DataFrame) -> dict:
        try:
            bb = get_bbands(df)
            close = df['close'].iloc[-1]
            upper = bb['BBU_20_2'].iloc[-1]
            lower = bb['BBL_20_2'].iloc[-1]
            bb_width = (upper - lower) / close

            if close < lower and bb_width < 0.02:
                return {'action': 'BUY', 'confidence': 0.75}
            elif close > upper and bb_width < 0.02:
                return {'action': 'SELL', 'confidence': 0.75}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"Bollinger Bands failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _rsi_obv_confirmation(self, df: pd.DataFrame) -> dict:
        try:
            rsi = get_rsi(df)
            obv = get_obv(df)
            if rsi is None or obv is None:
                return {'action': 'HOLD', 'confidence': 0.0}
            rsi_val = rsi.iloc[-1]
            obv_slope = obv.iloc[-1] - obv.iloc[-2]

            if rsi_val < 30 and obv_slope > 0:
                return {'action': 'BUY', 'confidence': 0.9}
            elif rsi_val > 70 and obv_slope < 0:
                return {'action': 'SELL', 'confidence': 0.9}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"RSI OBV Confirmation failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _macd_adx_filter(self, df: pd.DataFrame) -> dict:
        try:
            macd_df = get_macd(df)
            adx = get_adx(df)
            if macd_df is None or adx is None:
                return {'action': 'HOLD', 'confidence': 0.0}
            macd_val = macd_df['MACD_12_26_9'].iloc[-1]
            adx_val = adx.iloc[-1]

            if macd_val > 0 and adx_val > 25:
                return {'action': 'BUY', 'confidence': 0.8}
            elif macd_val < 0 and adx_val > 25:
                return {'action': 'SELL', 'confidence': 0.8}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"MACD ADX Filter failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _bollinger_mfi_reversal(self, df: pd.DataFrame) -> dict:
        try:
            bb = get_bbands(df)
            mfi = get_mfi(df)
            if bb is None or mfi is None:
                return {'action': 'HOLD', 'confidence': 0.0}
            close = df['close'].iloc[-1]
            upper = bb['BBU_20_2'].iloc[-1]
            lower = bb['BBL_20_2'].iloc[-1]
            mfi_val = mfi.iloc[-1]

            if close < lower and mfi_val < 20:
                return {'action': 'BUY', 'confidence': 0.85}
            elif close > upper and mfi_val > 80:
                return {'action': 'SELL', 'confidence': 0.85}
            return {'action': 'HOLD', 'confidence': 0.0}
        except Exception as e:
            self.logger.error(f"Bollinger MFI Reversal failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}

    # ===========================
    # === UTILITIES
    # ===========================
    def _risk_multiplier(self, risk_profile: str) -> float:
        return {'Low': 0.5, 'Medium': 1.0, 'Aggressive': 1.5}.get(risk_profile, 1.0)

    def _default_signal(self, pair: str, strategy: str) -> dict:
        return {
            'pair': pair,
            'strategy': strategy,
            'direction': 'HOLD',
            'confidence': 0.0,
            'sentiment_score': 0.0,
            'tp': 0.0,
            'sl': 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }