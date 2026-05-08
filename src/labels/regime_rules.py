"""
v3 — Rule-based market regime detection.

Posterdeki başarılı tasarımı (Multi-Asset Portfolio, Sharpe 1.41) BTC
günlük seriye uyarlıyor. GMM unsupervised approach yerine **şeffaf
threshold tabanlı 3-sınıf rejim tanımı**:

  HIGH_RISK:   VIX > 25  OR  ret_20d < -15%  OR  vol_20d_ann > 100%
  LOW_RISK:    VIX < 16  AND ret_60d > +5%   AND vol_60d_ann <  60%
  MEDIUM_RISK: default

Persistence filter ile tek-günlük dalgalanmalar yumuşatılır
(en az 5 gün aynı sınıfta kalmalı).

Eşikler config.yaml > regime_rules altında parametrize edildi (grid
search yapılabilir).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)

REGIMES = ["Low", "Medium", "High"]


def _annualize_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def detect_regime(
    aligned: pd.DataFrame,
    vix_high: float = 25.0,
    vix_low: float = 16.0,
    ret_20d_low: float = -0.15,
    ret_60d_high: float = 0.05,
    vol_20d_high: float = 1.00,
    vol_60d_low: float = 0.60,
    persistence_days: int = 5,
) -> pd.Series:
    """
    Compute rule-based regime label per day.

    Parameters
    ----------
    aligned : pd.DataFrame
        Must have columns ['Close', 'VIX'].
    vix_high, vix_low : float
        VIX thresholds for High/Low risk gates.
    ret_20d_low, ret_60d_high : float
        Return thresholds (decimal, e.g. -0.15 = -15%).
    vol_20d_high, vol_60d_low : float
        Annualized volatility thresholds (decimal).
    persistence_days : int
        Minimum consecutive days a regime must persist.

    Returns
    -------
    pd.Series of {'Low', 'Medium', 'High'} indexed by date.
    """
    close = aligned["Close"]
    vix = aligned["VIX"]

    ret_20d = close.pct_change(20)
    ret_60d = close.pct_change(60)
    daily_ret = close.pct_change()
    vol_20d = _annualize_vol(daily_ret, 20)
    vol_60d = _annualize_vol(daily_ret, 60)

    high_mask = (vix > vix_high) | (ret_20d < ret_20d_low) | (vol_20d > vol_20d_high)
    low_mask = (vix < vix_low) & (ret_60d > ret_60d_high) & (vol_60d < vol_60d_low)

    regime = pd.Series("Medium", index=close.index, name="regime")
    regime[high_mask.fillna(False)] = "High"
    # Low overrides Medium but NOT High (High wins on conflict)
    regime[low_mask.fillna(False) & ~high_mask.fillna(False)] = "Low"

    if persistence_days > 1:
        regime = _persistence_filter(regime, min_days=persistence_days)

    counts = regime.value_counts()
    pct = (counts / len(regime) * 100).round(1)
    logger.info(
        f"Rule-based regime: vix>{vix_high}/<{vix_low}, "
        f"ret_20d<{ret_20d_low:.0%}, ret_60d>{ret_60d_high:.0%}, "
        f"vol_20d>{vol_20d_high:.0%}, vol_60d<{vol_60d_low:.0%}, "
        f"persistence={persistence_days}d"
    )
    logger.info(f"  Distribution:  Low={pct.get('Low', 0):.1f}%  "
                f"Medium={pct.get('Medium', 0):.1f}%  "
                f"High={pct.get('High', 0):.1f}%")
    return regime


def _persistence_filter(labels: pd.Series, min_days: int) -> pd.Series:
    """
    Eliminate runs shorter than min_days by absorbing them into the
    surrounding regime (whichever has the longer immediate neighbour).
    Simple: if a run is < min_days long, replace with previous valid regime.
    """
    out = labels.copy()
    cur = out.iloc[0]
    run_start = 0
    for i in range(1, len(out)):
        if out.iloc[i] != cur:
            run_len = i - run_start
            if run_len < min_days and run_start > 0:
                # Replace this short run with previous valid value
                prev_regime = out.iloc[run_start - 1]
                out.iloc[run_start:i] = prev_regime
            cur = out.iloc[i]
            run_start = i
    # Last run — leave as-is (would need future info to decide)
    return out


def regime_to_onehot(regime: pd.Series) -> pd.DataFrame:
    """Convert regime str series into 3-column one-hot DataFrame."""
    out = pd.DataFrame(0.0, index=regime.index, columns=[f"P_{r}" for r in REGIMES])
    for r in REGIMES:
        out[f"P_{r}"] = (regime == r).astype(float)
    return out
