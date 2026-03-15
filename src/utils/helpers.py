"""
Utility helpers used across the project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """Save DataFrame to CSV, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_csv(path: str | Path, index_col: str | int = 0, parse_dates: bool = True) -> pd.DataFrame:
    """Load CSV with sensible defaults for time-series data."""
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically (no shuffle)."""
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def report_class_distribution(
    labels: pd.Series,
    name: str = "Labels",
) -> pd.DataFrame:
    """Print and return class distribution."""
    counts = labels.value_counts()
    pcts = labels.value_counts(normalize=True) * 100
    dist = pd.DataFrame({"count": counts, "percentage": pcts.round(2)})
    print(f"\n--- {name} Class Distribution ---")
    print(dist)
    print(f"Total: {len(labels)}")
    return dist


def compute_class_weights(labels: pd.Series) -> dict:
    """Compute balanced class weights inversely proportional to frequency."""
    counts = labels.value_counts()
    total = len(labels)
    n_classes = len(counts)
    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)
    return weights
