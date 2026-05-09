"""
V5 Faz 2 — Technical features (proposal §2.2).

Stage 1 + Stage 3 input için technical indicators üretir.

Stage 1 (Trend Classifier) inputs:
  SMA(20,50,200), EMA(12,26), ADX(14), Parabolic SAR
  + cross indicators (SMA20>SMA50, Close>SMA50, Close>SMA200)
  + log_ret_5d (short momentum context)

Stage 3 (Signal Classifier) inputs (oscillators + volume):
  RSI(14), Stochastic %K(14)/%D(3), MACD(12,26,9), Williams %R(14)
  Bollinger Bands(20, 2σ), ATR(14)
  OBV, VWAP, Volume_SMA(20), Volume_zscore(20)

Library: `ta` (pure-python TA-Lib alternative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, PSARIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def build_stage1_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 trend features (long-horizon)."""
    o = pd.DataFrame(index=ohlcv.index)
    close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

    # SMA
    o["SMA_20"] = SMAIndicator(close, 20).sma_indicator()
    o["SMA_50"] = SMAIndicator(close, 50).sma_indicator()
    o["SMA_200"] = SMAIndicator(close, 200).sma_indicator()
    # EMA
    o["EMA_12"] = EMAIndicator(close, 12).ema_indicator()
    o["EMA_26"] = EMAIndicator(close, 26).ema_indicator()
    # Cross indicators (binary, helpful as features)
    o["SMA_20_above_50"] = (o["SMA_20"] > o["SMA_50"]).astype(float)
    o["Close_above_SMA_50"] = (close > o["SMA_50"]).astype(float)
    o["Close_above_SMA_200"] = (close > o["SMA_200"]).astype(float)
    # ADX
    o["ADX_14"] = ADXIndicator(high, low, close, 14).adx()
    # Parabolic SAR (position normalized: (close - sar) / close)
    psar = PSARIndicator(high, low, close).psar()
    o["Parabolic_SAR_position"] = (close - psar) / close
    # Short momentum context
    o["log_ret_5d"] = np.log(close / close.shift(5))

    return o


def build_stage3_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 oscillator + volume features (short-term)."""
    o = pd.DataFrame(index=ohlcv.index)
    close, high, low, volume = (ohlcv["Close"], ohlcv["High"],
                                 ohlcv["Low"], ohlcv["Volume"])

    # Momentum oscillators
    o["RSI_14"] = RSIIndicator(close, 14).rsi()
    stoch = StochasticOscillator(high, low, close, 14, 3)
    o["Stoch_K_14"] = stoch.stoch()
    o["Stoch_D_3"] = stoch.stoch_signal()
    macd_obj = MACD(close, 26, 12, 9)
    o["MACD_line"] = macd_obj.macd()
    o["MACD_signal"] = macd_obj.macd_signal()
    o["MACD_histogram"] = macd_obj.macd_diff()
    o["Williams_R_14"] = WilliamsRIndicator(high, low, close, 14).williams_r()

    # Volatility
    bb = BollingerBands(close, 20, 2)
    o["BB_upper"] = bb.bollinger_hband()
    o["BB_lower"] = bb.bollinger_lband()
    o["BB_pct_b"] = bb.bollinger_pband()   # %B = (close - lower) / (upper - lower)
    o["ATR_14"] = AverageTrueRange(high, low, close, 14).average_true_range()

    # Volume
    o["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    o["VWAP"] = VolumeWeightedAveragePrice(high, low, close, volume, 14).volume_weighted_average_price()
    o["Volume_SMA_20"] = volume.rolling(20).mean()
    o["Volume_zscore_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    return o


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    s1 = build_stage1_features(btc)
    s3 = build_stage3_features(btc)
    print(f"Stage 1 features shape: {s1.shape}, cols: {list(s1.columns)}")
    print(f"Stage 3 features shape: {s3.shape}, cols: {list(s3.columns)}")
    print(f"\nStage 1 NaN per col (warm-up):\n{s1.isna().sum()}")
    print(f"\nStage 3 NaN per col (warm-up):\n{s3.isna().sum()}")
