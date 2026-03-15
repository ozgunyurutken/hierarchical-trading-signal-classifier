"""
Technical indicator computation.
Computes trend, momentum, volatility, and volume indicators from OHLCV data.
Uses the `ta` library for most indicators.
"""

import pandas as pd
import numpy as np
import ta

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def compute_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend-following indicators (Stage 1 candidates).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        DataFrame with trend indicator columns only.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    features = pd.DataFrame(index=df.index)

    # Simple Moving Averages
    for period in [20, 50, 200]:
        features[f"SMA_{period}"] = ta.trend.sma_indicator(close, window=period)

    # Exponential Moving Averages
    for period in [12, 26]:
        features[f"EMA_{period}"] = ta.trend.ema_indicator(close, window=period)

    # ADX (Average Directional Index)
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    features["ADX_14"] = adx_indicator.adx()
    features["DI_plus_14"] = adx_indicator.adx_pos()
    features["DI_minus_14"] = adx_indicator.adx_neg()

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(high, low, close)
    features["PSAR"] = psar.psar()
    features["PSAR_up"] = psar.psar_up_indicator()
    features["PSAR_down"] = psar.psar_down_indicator()

    # SMA Crossover signals
    features["SMA20_SMA50_cross"] = (
        features["SMA_20"] - features["SMA_50"]
    )
    features["SMA50_SMA200_cross"] = (
        features["SMA_50"] - features["SMA_200"]
    )

    # Price distance from SMAs (%)
    for period in [20, 50, 200]:
        sma_col = f"SMA_{period}"
        features[f"price_dist_SMA{period}_pct"] = (
            (close - features[sma_col]) / features[sma_col] * 100
        )

    logger.info(f"Computed {len(features.columns)} trend indicators")
    return features


def compute_oscillator_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum/oscillator indicators (Stage 3 candidates).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    features = pd.DataFrame(index=df.index)

    # RSI
    features["RSI_14"] = ta.momentum.rsi(close, window=14)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    features["Stoch_K_14"] = stoch.stoch()
    features["Stoch_D_3"] = stoch.stoch_signal()

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    features["MACD_line"] = macd.macd()
    features["MACD_signal"] = macd.macd_signal()
    features["MACD_histogram"] = macd.macd_diff()

    # Williams %R
    features["Williams_R_14"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # CCI (Commodity Channel Index)
    features["CCI_20"] = ta.trend.cci(high, low, close, window=20)

    # ROC (Rate of Change)
    features["ROC_10"] = ta.momentum.roc(close, window=10)

    logger.info(f"Computed {len(features.columns)} oscillator indicators")
    return features


def compute_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility indicators (Stage 1 or 3).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    features = pd.DataFrame(index=df.index)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    features["BB_upper"] = bb.bollinger_hband()
    features["BB_lower"] = bb.bollinger_lband()
    features["BB_bandwidth"] = bb.bollinger_wband()
    features["BB_pct_b"] = bb.bollinger_pband()

    # ATR (Average True Range)
    features["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Historical volatility (rolling std of log returns)
    log_returns = np.log(close / close.shift(1))
    features["hist_volatility_20"] = log_returns.rolling(window=20).std() * np.sqrt(252)

    logger.info(f"Computed {len(features.columns)} volatility indicators")
    return features


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume indicators (Stage 1 or 3).
    """
    close = df["Close"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)

    # OBV (On-Balance Volume)
    features["OBV"] = ta.volume.on_balance_volume(close, volume)

    # Volume SMA
    features["Volume_SMA_20"] = volume.rolling(window=20).mean()

    # Volume rate of change
    features["Volume_ROC"] = volume.pct_change(periods=10) * 100

    logger.info(f"Computed {len(features.columns)} volume indicators")
    return features


def compute_all_technical_indicators(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute all technical indicators grouped by category.

    Returns
    -------
    dict with keys: 'trend', 'oscillator', 'volatility', 'volume'
    """
    logger.info(f"Computing all technical indicators for {len(df)} rows")

    return {
        "trend": compute_trend_indicators(df),
        "oscillator": compute_oscillator_indicators(df),
        "volatility": compute_volatility_indicators(df),
        "volume": compute_volume_indicators(df),
    }
