"""V5 Phase 4 — Stage 3 signal label generator.

Buy / Sell / Hold based on causal forward returns + volatility-adjusted
adaptive threshold.

Spec (V5_PLAN.md L82):
  ret_h = (close[t+h] - close[t]) / close[t]
  eps_t = k * rolling_std_w(daily_returns)[t]    (causal, t inclusive)
  label =  Buy   if ret_h >  +eps_t
           Sell  if ret_h <  -eps_t
           Hold  otherwise

Default: h=5, k=0.5, w=20.

Causality notes:
  - rolling_std uses ONLY past data up to and including t (no future leak).
  - forward_return uses future close[t+h] — that is the label generation,
    NOT a feature. The label at time t represents "what happens in the
    next h days," which is exactly what we want to predict.
  - Last h rows have NaN forward returns and are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_v5_signal_labels(close: pd.Series,
                              h: int = 5,
                              k: float = 0.5,
                              window: int = 20) -> pd.DataFrame:
    """Generate V5 Stage 3 signal labels.

    Parameters
    ----------
    close : pd.Series, datetime index
        Daily close prices.
    h : int
        Forward horizon in days. Default 5.
    k : float
        Volatility multiplier for threshold. Default 0.5.
    window : int
        Rolling window for daily-return std (causal). Default 20.

    Returns
    -------
    pd.DataFrame columns:
      ['signal_label', 'forward_return', 'eps_threshold']
      Indexed by date. Rows with NaN forward_return or eps dropped.
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be Series, got {type(close)}")

    daily_returns = close.pct_change()
    rolling_std = daily_returns.rolling(window=window, min_periods=window).std()
    forward_return = (close.shift(-h) - close) / close
    eps = k * rolling_std

    label = pd.Series("Hold", index=close.index, name="signal_label", dtype=object)
    label[forward_return > eps] = "Buy"
    label[forward_return < -eps] = "Sell"

    out = pd.DataFrame({
        "signal_label":   label,
        "forward_return": forward_return,
        "eps_threshold":  eps,
    })

    valid = forward_return.notna() & eps.notna()
    out = out[valid].copy()
    return out


def label_distribution(label: pd.Series) -> pd.Series:
    """Frequency table for {Buy, Sell, Hold}, with proportions."""
    counts = label.value_counts().reindex(["Buy", "Hold", "Sell"], fill_value=0)
    props = counts / counts.sum()
    df = pd.concat([counts.rename("count"), props.rename("share")], axis=1)
    return df


# Sanity-check helpers (no leakage)

def assert_no_lookahead_leakage(close: pd.Series, labels_df: pd.DataFrame,
                                h: int, window: int) -> None:
    """Assert that label at index t is computed only from close[<=t-1] for std,
    plus close[t+h] for forward_return. The std uses past-only data via
    rolling().std() — verified by recomputing manually here.
    """
    daily_returns = close.pct_change()
    for t in labels_df.index[::500]:  # sample every 500th row for speed
        if t not in close.index:
            continue
        i = close.index.get_loc(t)
        if i < window:
            continue
        manual_std = daily_returns.iloc[i - window + 1:i + 1].std()
        recorded_eps = labels_df.loc[t, "eps_threshold"]
        if pd.isna(manual_std) or pd.isna(recorded_eps):
            continue
        expected_eps = 0.5 * manual_std  # default k
        # Allow tiny numerical tolerance
        if not np.isclose(recorded_eps, expected_eps, rtol=1e-6, atol=1e-9):
            raise AssertionError(
                f"Leakage check failed at {t}: recorded eps={recorded_eps:.6e} "
                f"vs manual eps={expected_eps:.6e}"
            )
