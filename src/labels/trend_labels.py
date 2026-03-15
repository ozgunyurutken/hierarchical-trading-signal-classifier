"""
Stage 1 — Trend label generation.
Labels: Uptrend, Downtrend, Sideways
Based on SMA crossover rules with persistence filter.
"""

import pandas as pd
import numpy as np

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def generate_trend_labels(
    df: pd.DataFrame,
    sma_short: int | None = None,
    sma_long: int | None = None,
    min_persistence: int | None = None,
) -> pd.Series:
    """
    Generate trend labels based on SMA crossover rules.

    Rules:
        - Uptrend:   SMA(short) > SMA(long) AND Close > SMA(long)
        - Downtrend: SMA(short) < SMA(long) AND Close < SMA(long)
        - Sideways:  everything else

    Noise filter: Labels must persist for at least `min_persistence` days.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' column and SMA columns, or raw OHLCV.
    sma_short : int
        Short SMA period. Default from config.
    sma_long : int
        Long SMA period. Default from config.
    min_persistence : int
        Minimum consecutive days for a label to hold.

    Returns
    -------
    pd.Series
        Trend labels indexed by date.
    """
    config = cfg()
    sma_short = sma_short or config["labels"]["trend"]["sma_short"]
    sma_long = sma_long or config["labels"]["trend"]["sma_long"]
    min_persistence = min_persistence or config["labels"]["trend"]["min_persistence_days"]

    close = df["Close"]

    # Compute SMAs if not present
    sma_s = close.rolling(window=sma_short).mean()
    sma_l = close.rolling(window=sma_long).mean()

    # Apply rules
    labels = pd.Series("Sideways", index=df.index, name="trend_label")
    labels[(sma_s > sma_l) & (close > sma_l)] = "Uptrend"
    labels[(sma_s < sma_l) & (close < sma_l)] = "Downtrend"

    # Drop rows where SMA is NaN (warm-up period)
    labels = labels[sma_l.notna()]

    # Apply persistence filter
    labels = _apply_persistence_filter(labels, min_persistence)

    # Report
    dist = labels.value_counts()
    logger.info(
        f"Trend labels generated: {len(labels)} rows, "
        f"SMA({sma_short}/{sma_long}), persistence={min_persistence}\n"
        f"  Distribution: {dist.to_dict()}"
    )

    return labels


def _apply_persistence_filter(labels: pd.Series, min_days: int) -> pd.Series:
    """
    Remove label flips that don't persist for at least `min_days`.
    Reverts short-lived labels to the previous stable label.
    """
    if min_days <= 1:
        return labels

    filtered = labels.copy()
    values = filtered.values
    n = len(values)

    i = 0
    while i < n:
        # Find the start of a run
        current_label = values[i]
        run_start = i
        while i < n and values[i] == current_label:
            i += 1
        run_length = i - run_start

        # If run is too short and not at the start, revert to previous label
        if run_length < min_days and run_start > 0:
            prev_label = values[run_start - 1]
            values[run_start:i] = prev_label

    filtered = pd.Series(values, index=labels.index, name=labels.name)

    changes_before = (labels != labels.shift()).sum()
    changes_after = (filtered != filtered.shift()).sum()
    logger.info(
        f"  Persistence filter: label changes {changes_before} -> {changes_after}"
    )

    return filtered
