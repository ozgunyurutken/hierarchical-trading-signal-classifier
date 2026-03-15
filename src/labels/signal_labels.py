"""
Stage 3 — Signal label generation.
Labels: Buy, Sell, Hold
Based on forward returns with fixed or adaptive thresholds.
"""

import pandas as pd
import numpy as np

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def generate_signal_labels_fixed(
    close: pd.Series,
    forward_days: int | None = None,
    threshold: float | None = None,
) -> pd.Series:
    """
    Generate signal labels using fixed threshold on forward returns.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    forward_days : int
        Number of days forward for return calculation.
    threshold : float
        Threshold for Buy/Sell (e.g., 0.01 = 1%).

    Returns
    -------
    pd.Series
        Signal labels (Buy/Sell/Hold).
    """
    config = cfg()
    forward_days = forward_days or config["labels"]["signal"]["forward_return_days"]
    threshold = threshold or config["labels"]["signal"]["fixed_threshold"]

    forward_return = (close.shift(-forward_days) - close) / close

    labels = pd.Series("Hold", index=close.index, name="signal_label")
    labels[forward_return > threshold] = "Buy"
    labels[forward_return < -threshold] = "Sell"

    # Remove rows where forward return is NaN (last `forward_days` rows)
    labels = labels[forward_return.notna()]

    dist = labels.value_counts()
    logger.info(
        f"Fixed-threshold signal labels: forward={forward_days}d, threshold=±{threshold:.2%}\n"
        f"  Distribution: {dist.to_dict()}"
    )

    return labels


def generate_signal_labels_adaptive(
    close: pd.Series,
    forward_days: int | None = None,
    constant: float | None = None,
    rolling_window: int | None = None,
) -> pd.Series:
    """
    Generate signal labels using volatility-adjusted adaptive threshold.

    Threshold = c * rolling_std(returns, window)
    """
    config = cfg()
    forward_days = forward_days or config["labels"]["signal"]["forward_return_days"]
    constant = constant or config["labels"]["signal"]["adaptive_constant"]
    rolling_window = rolling_window or config["labels"]["signal"]["adaptive_rolling_window"]

    daily_returns = close.pct_change()
    rolling_std = daily_returns.rolling(window=rolling_window).std()
    forward_return = (close.shift(-forward_days) - close) / close

    adaptive_threshold = constant * rolling_std

    labels = pd.Series("Hold", index=close.index, name="signal_label")
    labels[forward_return > adaptive_threshold] = "Buy"
    labels[forward_return < -adaptive_threshold] = "Sell"

    # Remove NaN rows
    valid_mask = forward_return.notna() & adaptive_threshold.notna()
    labels = labels[valid_mask]

    dist = labels.value_counts()
    logger.info(
        f"Adaptive-threshold signal labels: forward={forward_days}d, "
        f"c={constant}, window={rolling_window}\n"
        f"  Distribution: {dist.to_dict()}"
    )

    return labels


def generate_signal_labels(
    close: pd.Series,
    method: str = "fixed",
    **kwargs,
) -> pd.Series:
    """
    Generate signal labels using specified method.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    method : str
        'fixed' or 'adaptive'.

    Returns
    -------
    pd.Series
        Signal labels.
    """
    if method == "fixed":
        return generate_signal_labels_fixed(close, **kwargs)
    elif method == "adaptive":
        return generate_signal_labels_adaptive(close, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fixed' or 'adaptive'.")


def verify_no_leakage(features: pd.DataFrame, close: pd.Series, forward_days: int = 5) -> None:
    """
    CRITICAL: Assert that forward return is NOT present in feature columns.
    Call this before training to prevent data leakage.
    """
    forward_return = (close.shift(-forward_days) - close) / close

    for col in features.columns:
        corr = features[col].corr(forward_return)
        assert abs(corr) < 0.99, (
            f"LEAKAGE DETECTED: Feature '{col}' has correlation {corr:.4f} "
            f"with forward return! This feature may contain future information."
        )

    logger.info(f"Leakage check passed: {len(features.columns)} features verified")
