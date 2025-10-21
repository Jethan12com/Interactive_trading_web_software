import pandas as pd
import pandas_ta as ta
from .logger_setup import setup_logger

logger = setup_logger("Indicators", "logs/indicators.log", to_console=False)

# ============================================================
# === PRICE-BASED INDICATORS ===
# ============================================================

def get_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        rsi = ta.rsi(df['close'], length=period)
        return rsi.fillna(50)
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return pd.Series([50]*len(df), index=df.index)

def get_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD indicator."""
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        return macd.fillna(0)
    except Exception as e:
        logger.error(f"MACD calculation failed: {e}")
        return pd.DataFrame({
            'MACD_12_26_9':[0]*len(df),
            'MACDh_12_26_9':[0]*len(df),
            'MACDs_12_26_9':[0]*len(df)
        }, index=df.index)

def get_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple Moving Average (SMA)."""
    try:
        sma = ta.sma(df['close'], length=period)
        return sma.fillna(df['close'])
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        return df['close']

def get_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    try:
        ema = ta.ema(df['close'], length=period)
        return ema.fillna(df['close'])
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        return df['close']

def get_bbands(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
    """Bollinger Bands (BB)."""
    try:
        bb = ta.bbands(df['close'], length=period, std=std)
        return bb.fillna(method='bfill').fillna(method='ffill')
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        return pd.DataFrame({
            f'BBL_{period}_{std}':[0]*len(df),
            f'BBM_{period}_{std}':[0]*len(df),
            f'BBU_{period}_{std}':[0]*len(df),
            f'BBB_{period}_{std}':[0]*len(df),
            f'BBP_{period}_{std}':[0]*len(df),
        }, index=df.index)

def get_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) – measures volatility."""
    try:
        required_cols = {'high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain {required_cols}")
        atr = ta.atr(df['high'], df['low'], df['close'], length=period)
        return atr.fillna(method='bfill').fillna(method='ffill')
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        return pd.Series([0]*len(df), index=df.index)

# ============================================================
# === VOLUME & TREND-STRENGTH INDICATORS ===
# ============================================================

def get_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)."""
    try:
        if not {'close','volume'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'close' and 'volume'")
        obv = ta.obv(df['close'], df['volume'])
        return obv.fillna(method='bfill').fillna(0)
    except Exception as e:
        logger.error(f"OBV calculation failed: {e}")
        return pd.Series([0]*len(df), index=df.index)

def get_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MFI)."""
    try:
        required_cols = {'high','low','close','volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain {required_cols}")
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=period)
        return mfi.fillna(50)
    except Exception as e:
        logger.error(f"MFI calculation failed: {e}")
        return pd.Series([50]*len(df), index=df.index)

def get_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    try:
        required_cols = {'high','low','close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain {required_cols}")
        adx = ta.adx(df['high'], df['low'], df['close'], length=period)['ADX_14']
        return adx.fillna(25)
    except Exception as e:
        logger.error(f"ADX calculation failed: {e}")
        return pd.Series([25]*len(df), index=df.index)

# ============================================================
# === COMBINED CONVENIENCE FUNCTION ===
# ============================================================

def get_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full indicator set for Backtester."""
    try:
        rsi = get_rsi(df)
        macd = get_macd(df)
        sma20 = get_sma(df, 20)
        ema20 = get_ema(df, 20)
        bb = get_bbands(df, 20, 2)
        obv = get_obv(df)
        mfi = get_mfi(df)
        adx = get_adx(df)
        atr = get_atr(df)

        merged = pd.concat([
            df['close'],
            rsi.rename("RSI_14"),
            macd['MACD_12_26_9'],
            macd['MACDs_12_26_9'].rename("MACD_signal"),
            sma20.rename("SMA_20"),
            ema20.rename("EMA_20"),
            bb[f'BBU_20_2'].rename("BB_upper"),
            bb[f'BBL_20_2'].rename("BB_lower"),
            obv.rename("OBV"),
            mfi.rename("MFI_14"),
            adx.rename("ADX_14"),
            atr.rename("ATR_14")
        ], axis=1)

        return merged
    except Exception as e:
        logger.error(f"get_all_indicators failed: {e}")
        return pd.DataFrame()