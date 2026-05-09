"""
V5 Phase 3 — Stage 1 Trend Label Generation.

İki ortogonal yaklaşım:
  1. ForwardReturnTrendLabeler:
     - h-gün forward log return ile yön belirle
     - Adaptive threshold = rolling std × k
     - |ret| > threshold → up/down, aksi → range

  2. ZigZagTrendLabeler:
     - Pivot detection (deviation_pct sapması)
     - Pivot-to-pivot segment yönü → up/down
     - Düşük amplitüd veya kısa süre → range

Causality:
  - Forward return: t+h kullanır → label t için training target,
    inference time t+h sonra hesaplanabilir. Training-only kullanım uygun.
  - ZigZag: pivots retrospective bulunur ama label OFFLINE üretilir,
    Stage 1 model t anındaki features'lardan p̂(trend) tahmin eder.
    Training fold'ları chronological → no look-ahead leakage.

Literature:
  - López de Prado (2018) [N52]: triple-barrier method for path-dependent labeling
  - Krauss et al. (2017) [N48]: deep learning for stock-direction classification
  - ZigZag indicator: classical TA, e.g., Murphy "Technical Analysis of Financial Markets"
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TREND_LABELS = ["uptrend", "downtrend", "range"]


def forward_return_trend_label(
    close: pd.Series,
    horizon: int = 10,
    rolling_window: int = 252,
    threshold_k: float = 1.5,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Label trend via h-day forward log return vs adaptive threshold.

    Parameters
    ----------
    close : pd.Series
        Daily close prices, datetime-indexed.
    horizon : int, default 10
        Forward return horizon in days.
    rolling_window : int, default 252
        Rolling window for std (≈ 1 trading year).
    threshold_k : float, default 1.5
        Threshold multiplier: ε = k × rolling_std(forward_return).
    min_periods : int, default 60
        Min observations for rolling std (warm-up).

    Returns
    -------
    pd.DataFrame with columns:
      forward_return : np.log(close[t+h] / close[t])
      threshold      : adaptive ε
      label          : "uptrend" / "downtrend" / "range" / NaN (warm-up + tail)
    """
    log_ret = np.log(close.shift(-horizon) / close)
    rolling_std = log_ret.rolling(rolling_window, min_periods=min_periods).std()
    threshold = threshold_k * rolling_std

    label = pd.Series(index=close.index, dtype=object)
    valid = log_ret.notna() & threshold.notna()
    label[valid & (log_ret > threshold)] = "uptrend"
    label[valid & (log_ret < -threshold)] = "downtrend"
    label[valid & (log_ret.abs() <= threshold)] = "range"

    return pd.DataFrame({
        "forward_return": log_ret,
        "threshold": threshold,
        "label": label,
    })


def find_zigzag_pivots(
    close: pd.Series,
    deviation_pct: float = 0.10,
) -> pd.DatetimeIndex:
    """ZigZag pivot detection (classical TA).

    Yields a sequence of (date, price) pivots where price reverses by at
    least deviation_pct from the last extreme. Standard offline algorithm.

    Returns datetime index of pivot dates (includes first and last bar).
    """
    if len(close) < 2:
        return pd.DatetimeIndex(close.index)

    pivots = [close.index[0]]
    last_pivot_price = float(close.iloc[0])
    direction = 0
    extreme_idx = close.index[0]
    extreme_price = float(close.iloc[0])

    for t in close.index[1:]:
        price = float(close.loc[t])
        if direction == 0:
            if price > last_pivot_price * (1 + deviation_pct):
                direction = +1
                extreme_idx, extreme_price = t, price
            elif price < last_pivot_price * (1 - deviation_pct):
                direction = -1
                extreme_idx, extreme_price = t, price
        elif direction == +1:
            if price > extreme_price:
                extreme_idx, extreme_price = t, price
            elif price < extreme_price * (1 - deviation_pct):
                pivots.append(extreme_idx)
                last_pivot_price = extreme_price
                direction = -1
                extreme_idx, extreme_price = t, price
        else:  # direction == -1
            if price < extreme_price:
                extreme_idx, extreme_price = t, price
            elif price > extreme_price * (1 + deviation_pct):
                pivots.append(extreme_idx)
                last_pivot_price = extreme_price
                direction = +1
                extreme_idx, extreme_price = t, price

    pivots.append(close.index[-1])
    return pd.DatetimeIndex(pivots)


def zigzag_trend_label(
    close: pd.Series,
    deviation_pct: float = 0.10,
    min_segment_days: int = 10,
    range_amplitude: float = 0.05,
) -> pd.DataFrame:
    """Label trend via ZigZag piecewise segmentation.

    Parameters
    ----------
    close : pd.Series
    deviation_pct : float, default 0.10
        Reversal threshold for pivot detection (e.g. 0.10 = 10%).
    min_segment_days : int, default 10
        Segments shorter than this → labeled "range".
    range_amplitude : float, default 0.05
        Segments with |amplitude| < this → labeled "range".

    Returns
    -------
    pd.DataFrame with columns: label, segment_id
    """
    pivots = find_zigzag_pivots(close, deviation_pct)
    label = pd.Series("range", index=close.index, dtype=object)
    seg_id = pd.Series(0, index=close.index, dtype=int)

    for i in range(len(pivots) - 1):
        start, end = pivots[i], pivots[i + 1]
        last = i == len(pivots) - 2
        if last:
            seg_mask = (close.index >= start) & (close.index <= end)
        else:
            seg_mask = (close.index >= start) & (close.index < end)
        if not seg_mask.any():
            continue

        amp = abs(float(close.loc[end]) - float(close.loc[start])) / float(close.loc[start])
        days = (end - start).days
        seg_id.loc[seg_mask] = i + 1

        if days < min_segment_days or amp < range_amplitude:
            label.loc[seg_mask] = "range"
        elif float(close.loc[end]) > float(close.loc[start]):
            label.loc[seg_mask] = "uptrend"
        else:
            label.loc[seg_mask] = "downtrend"

    return pd.DataFrame({
        "label": label,
        "segment_id": seg_id,
    })


def label_distribution(label: pd.Series) -> pd.Series:
    """Return label proportion (excluding NaN)."""
    return label.dropna().value_counts(normalize=True).reindex(TREND_LABELS).fillna(0.0)
