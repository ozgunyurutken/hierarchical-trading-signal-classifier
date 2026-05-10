"""V5 Phase 4 — Stage 3 oscillator + Stage 2 derived features.

Stage 3 input feature catalog (16 total per asset):

  Stage 1 (6 from OOF, raw + smoothed10d):
    P1_down, P1_range, P1_up
    P1_down_smooth10, P1_range_smooth10, P1_up_smooth10

  Stage 2 (4: hard one-hot + days_since_last_transition):
    P2_Bull, P2_Neutral, P2_Bear, regime_age_days

  Oscillator (6, momentum/mean-reversion):
    RSI_14, MACD_signal_diff, Bollinger_pct_b,
    Stochastic_K_14, volume_zscore_20, OBV_change_20d

Causality: all oscillators use rolling/EMA which look back only.
Stage 1 OOF and Stage 2 regime are already causal by construction
(walk-forward OOF for Stage 1, FSM with hysteresis for Stage 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


STAGE3_FEATURE_COLS = [
    # Stage 1 raw
    "P1_down", "P1_range", "P1_up",
    # Stage 1 smoothed (10d rolling mean of probabilities)
    "P1_down_smooth10", "P1_range_smooth10", "P1_up_smooth10",
    # Stage 2 hard one-hot
    "P2_Bull", "P2_Neutral", "P2_Bear",
    # Stage 2 regime tenure
    "regime_age_days",
    # Oscillator
    "RSI_14",
    "MACD_signal_diff",
    "Bollinger_pct_b",
    "Stochastic_K_14",
    "volume_zscore_20",
    "OBV_change_20d",
]

# Regime-conditional interaction features (added Phase 7+ to address the
# "RSI > 70 → Buy" data-driven divergence from classical TA). Provides the
# model with explicit regime-aware oscillator readings, e.g. "MACD value
# inside a Bull regime" vs "inside a Bear regime". Tree-based models can
# learn these via depth-2 splits, but explicit interactions help LGBM/MLP
# converge faster and tighten Bayesian decision-rule semantics.
REGIME_OSC_INTERACTIONS = [
    "RSI_14_x_PBull",         # RSI_14 * P2_Bull
    "RSI_14_x_PBear",         # RSI_14 * P2_Bear
    "MACD_signal_diff_x_PBull",
    "MACD_signal_diff_x_PBear",
    "Bollinger_pct_b_x_PBull",
    "Bollinger_pct_b_x_PBear",
]

# Extended feature list (used by the "regime-aware" Stage 3 variant)
STAGE3_FEATURE_COLS_EXTENDED = STAGE3_FEATURE_COLS + REGIME_OSC_INTERACTIONS

FEATURE_GROUPS = {
    "Stage 1 raw":            ["P1_down", "P1_range", "P1_up"],
    "Stage 1 smoothed (10d)": ["P1_down_smooth10", "P1_range_smooth10", "P1_up_smooth10"],
    "Stage 2 regime":         ["P2_Bull", "P2_Neutral", "P2_Bear", "regime_age_days"],
    "Oscillator":             ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
                               "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"],
    "Regime-conditional (interaction)": REGIME_OSC_INTERACTIONS,
}


def smooth_stage1_oof(oof: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Apply causal rolling mean to Stage 1 OOF probabilities.

    Centered=False to keep causality (only past data).
    Renormalize to sum=1 after smoothing in case of edge effects.
    """
    cols = ["P_downtrend", "P_range", "P_uptrend"]
    smoothed = oof[cols].rolling(window=window, min_periods=1).mean()
    # Renormalize each row (small numerical drift after rolling mean)
    row_sum = smoothed.sum(axis=1)
    smoothed = smoothed.div(row_sum, axis=0)
    smoothed.columns = ["P1_down_smooth10", "P1_range_smooth10", "P1_up_smooth10"]
    return smoothed


def regime_age_from_label(regime_label: pd.Series) -> pd.Series:
    """Days since the last regime transition.

    Day 0 of a new regime = 1, day after = 2, etc.
    First row of dataset = 1 (we don't know history).
    """
    arr = regime_label.to_numpy()
    n = len(arr)
    age = np.empty(n, dtype=int)
    age[0] = 1
    for i in range(1, n):
        if arr[i] == arr[i - 1]:
            age[i] = age[i - 1] + 1
        else:
            age[i] = 1
    return pd.Series(age, index=regime_label.index, name="regime_age_days")


def build_stage3_oscillators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 oscillator features. OHLCV must have Open, High, Low, Close, Volume."""
    f = pd.DataFrame(index=ohlcv.index)
    high, low, close, volume = ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"]

    f["RSI_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    f["MACD_signal_diff"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2.0)
    f["Bollinger_pct_b"] = bb.bollinger_pband()

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    f["Stochastic_K_14"] = stoch.stoch()

    vol_mean20 = volume.rolling(window=20, min_periods=20).mean()
    vol_std20  = volume.rolling(window=20, min_periods=20).std()
    f["volume_zscore_20"] = (volume - vol_mean20) / vol_std20

    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    f["OBV_change_20d"] = (obv - obv.shift(20)) / obv.shift(20).abs().replace(0, np.nan)

    return f


def build_stage3_features(
    ohlcv: pd.DataFrame,
    stage1_oof: pd.DataFrame,        # cols: P_downtrend, P_range, P_uptrend
    stage2_regime: pd.DataFrame,     # cols: regime_label, P_Bull, P_Neutral, P_Bear
    smooth_window: int = 10,
) -> pd.DataFrame:
    """Join all Stage 3 features into one DataFrame.

    Returns DataFrame with STAGE3_FEATURE_COLS, indexed by date.
    NaN warm-up rows at start (~30 rows from oscillators) — caller drops.
    """
    out = pd.DataFrame(index=ohlcv.index)

    # --- Stage 1 raw ---
    s1 = stage1_oof.rename(columns={
        "P_downtrend": "P1_down",
        "P_range":     "P1_range",
        "P_uptrend":   "P1_up",
    })[["P1_down", "P1_range", "P1_up"]]
    out = out.join(s1, how="left")

    # --- Stage 1 smoothed ---
    s1_smooth = smooth_stage1_oof(stage1_oof, window=smooth_window)
    out = out.join(s1_smooth, how="left")

    # --- Stage 2 hard one-hot ---
    s2 = stage2_regime[["P_Bull", "P_Neutral", "P_Bear"]].rename(columns={
        "P_Bull":    "P2_Bull",
        "P_Neutral": "P2_Neutral",
        "P_Bear":    "P2_Bear",
    })
    out = out.join(s2, how="left")

    # --- Stage 2 regime age ---
    age = regime_age_from_label(stage2_regime["regime_label"])
    out = out.join(age, how="left")

    # --- Oscillator ---
    osc = build_stage3_oscillators(ohlcv)
    out = out.join(osc, how="left")

    # --- Regime-conditional interactions ---
    # These are computed AFTER the oscillator + Stage 2 columns are present.
    # MACD scale: BTC daily is ~ -1500..2000. We feed the raw value
    # (no normalization) so that tree splits stay interpretable and the
    # interaction faithfully reflects "MACD reading in this regime".
    out["RSI_14_x_PBull"]            = out["RSI_14"]            * out["P2_Bull"]
    out["RSI_14_x_PBear"]            = out["RSI_14"]            * out["P2_Bear"]
    out["MACD_signal_diff_x_PBull"]  = out["MACD_signal_diff"]  * out["P2_Bull"]
    out["MACD_signal_diff_x_PBear"]  = out["MACD_signal_diff"]  * out["P2_Bear"]
    out["Bollinger_pct_b_x_PBull"]   = out["Bollinger_pct_b"]   * out["P2_Bull"]
    out["Bollinger_pct_b_x_PBear"]   = out["Bollinger_pct_b"]   * out["P2_Bear"]

    # Return BOTH base + extended columns; downstream code can pick either.
    return out[STAGE3_FEATURE_COLS_EXTENDED]
