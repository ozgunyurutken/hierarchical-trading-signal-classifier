"""
V5 Phase 3 — Stage 1 Trend Classifier features.

Sadece BTC/ETH OHLCV girdi (macro feature DAHİL DEĞİL — o Stage 2'nin işi).

14 feature, 6 kategori:
  Returns (3):    log_ret_5d, log_ret_20d, log_ret_60d
  Trend (3):      ADX_14, MA_slope_20, MA_slope_50
  Momentum (2):   RSI_14, MACD_signal_diff
  Mean rev. (2):  Bollinger_pct_b, distance_to_SMA_50
  Volatility (2): ATR_14_pct, realized_vol_20d
  Volume (2):     volume_zscore_20, OBV_zscore_60

Library: `ta` (pure-python). Tree models: no scaling. MLP: StandardScaler
on train fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

STAGE1_FEATURE_COLS = [
    "log_ret_5d", "log_ret_20d", "log_ret_60d",
    "ADX_14", "MA_slope_20", "MA_slope_50",
    "RSI_14", "MACD_signal_diff",
    "Bollinger_pct_b", "distance_to_SMA_50",
    "ATR_14_pct", "realized_vol_20d",
    "volume_zscore_20", "OBV_zscore_60",
]

FEATURE_GROUPS = {
    "Returns":         ["log_ret_5d", "log_ret_20d", "log_ret_60d"],
    "Trend strength":  ["ADX_14", "MA_slope_20", "MA_slope_50"],
    "Momentum":        ["RSI_14", "MACD_signal_diff"],
    "Mean reversion":  ["Bollinger_pct_b", "distance_to_SMA_50"],
    "Volatility":      ["ATR_14_pct", "realized_vol_20d"],
    "Volume":          ["volume_zscore_20", "OBV_zscore_60"],
}


def build_stage1_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build 14-column Stage 1 feature DataFrame from BTC/ETH OHLCV.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns. Daily.

    Returns
    -------
    pd.DataFrame, datetime-indexed, NaN warm-up at start (~60 rows).
    """
    f = pd.DataFrame(index=ohlcv.index)
    close, high, low, volume = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]

    f["log_ret_5d"]  = np.log(close / close.shift(5))
    f["log_ret_20d"] = np.log(close / close.shift(20))
    f["log_ret_60d"] = np.log(close / close.shift(60))

    f["ADX_14"] = ADXIndicator(high, low, close, window=14).adx()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    f["MA_slope_20"] = (sma20 - sma20.shift(5))  / sma20.shift(5)
    f["MA_slope_50"] = (sma50 - sma50.shift(10)) / sma50.shift(10)

    f["RSI_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    f["MACD_signal_diff"] = macd.macd() - macd.macd_signal()

    bb = BollingerBands(close, window=20, window_dev=2)
    f["Bollinger_pct_b"]    = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    f["distance_to_SMA_50"] = (close - sma50) / sma50

    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    f["ATR_14_pct"] = atr / close
    log_ret_1d = np.log(close / close.shift(1))
    f["realized_vol_20d"] = log_ret_1d.rolling(20).std() * np.sqrt(252)

    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    f["volume_zscore_20"] = (volume - vol_mean) / vol_std
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    f["OBV_zscore_60"] = (obv - obv.rolling(60).mean()) / obv.rolling(60).std()

    return f[STAGE1_FEATURE_COLS]


def sma_crossover_trend_label(close: pd.Series,
                              fast_window: int = 20,
                              slow_window: int = 50) -> pd.Series:
    """SMA-crossover trend label (TAUTOLOGY DEMO ONLY — paper'da tartışılır).

    uptrend  : SMA(fast) > SMA(slow)
    downtrend: SMA(fast) < SMA(slow)
    range    : (paper-friendly tek bir ayrım; aralarında küçük fark yok)

    Bu label Stage 1 feature'larıyla (MA_slope_20, MA_slope_50,
    distance_to_SMA_50) yarı-çizgisel ilişki taşır → bilgi sızıntısı = tautoloji.
    Stage 1 modelleri bu label'ı ezbere öğrenir, gerçek prediction skill'i
    kanıtlanmaz. Bu yüzden V5'te ZigZag / forward-return tercih edilir.
    """
    fast = close.rolling(fast_window).mean()
    slow = close.rolling(slow_window).mean()
    label = pd.Series(index=close.index, dtype=object)
    label[fast > slow] = "uptrend"
    label[fast < slow] = "downtrend"
    label[(fast - slow).abs() < 1e-9] = "range"
    return label
