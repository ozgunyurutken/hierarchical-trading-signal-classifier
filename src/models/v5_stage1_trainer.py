"""
V5 Phase 3 — Stage 1 Trend Classifier walk-forward CV trainer.

4 classifiers compared on the same expanding-window CV splits:
  - XGBoost (multi:softprob)
  - LightGBM (multiclass)
  - Random Forest (sklearn)
  - MLP (sklearn, with StandardScaler on train fold only)

Walk-forward expanding-window CV:
  fold k:
    train = [t0, t_k]            (expanding)
    gap   = horizon (default 10) (avoid leakage from forward-return-style label)
    val   = [t_k + gap, t_k + gap + val_size]
  step forward by val_size.

Output per (asset, model):
  - OOF probabilities concatenated across folds (DataFrame, 3 cols)
  - Per-fold metrics (accuracy, F1 macro, F1 per class)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

LABEL_TO_IDX = {"downtrend": 0, "range": 1, "uptrend": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
CLASS_NAMES = ["downtrend", "range", "uptrend"]


# ---------- Walk-forward splits ----------

@dataclass
class WalkForwardConfig:
    train_min_size: int = 750     # ~3 trading years initial train
    val_size: int = 200           # ~10 months val window per fold
    step: int = 200               # advance by val_size (non-overlapping)
    gap: int = 10                 # horizon-aware gap


def walk_forward_splits(n: int, cfg: WalkForwardConfig) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) tuples for expanding-window CV."""
    t = cfg.train_min_size
    while t + cfg.gap + cfg.val_size <= n:
        train_idx = np.arange(0, t)
        val_start = t + cfg.gap
        val_idx = np.arange(val_start, val_start + cfg.val_size)
        yield train_idx, val_idx
        t += cfg.step


# ---------- Model factories ----------

def make_xgboost(random_state: int = 42, balanced: bool = False, **hp):
    # XGBoost has no class_weight; we apply sample_weight at fit time.
    defaults = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
    )
    defaults.update(hp)
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        **defaults,
    )


def make_lightgbm(random_state: int = 42, balanced: bool = False, **hp):
    defaults = dict(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=20,
    )
    defaults.update(hp)
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced" if balanced else None,
        **defaults,
    )


def make_random_forest(random_state: int = 42, balanced: bool = False, **hp):
    defaults = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=2,
        max_features="sqrt",
    )
    defaults.update(hp)
    return RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced" if balanced else None,
        **defaults,
    )


def make_mlp(random_state: int = 42, balanced: bool = False, **hp):
    # sklearn MLPClassifier doesn't support class_weight; balanced flag is
    # noted but not applied. Limitation logged in paper.
    defaults = dict(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size="auto",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    defaults.update(hp)
    return MLPClassifier(random_state=random_state, **defaults)


MODEL_FACTORIES: dict[str, Callable] = {
    "xgboost":       make_xgboost,
    "lightgbm":      make_lightgbm,
    "random_forest": make_random_forest,
    "mlp":           make_mlp,
}

TREE_MODELS = {"xgboost", "lightgbm", "random_forest"}


# ---------- Training ----------

@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    accuracy: float
    f1_macro: float
    f1_downtrend: float
    f1_range: float
    f1_uptrend: float


def _balanced_sample_weight(y_idx: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights for balanced loss (used by XGBoost)."""
    classes, counts = np.unique(y_idx, return_counts=True)
    n = len(y_idx)
    n_classes = len(classes)
    freq = {c: counts[i] for i, c in enumerate(classes)}
    return np.array([n / (n_classes * freq[v]) for v in y_idx], dtype=float)


def train_walk_forward(X: pd.DataFrame, y: pd.Series, model_name: str,
                       cfg: WalkForwardConfig | None = None,
                       random_state: int = 42,
                       balanced: bool = False,
                       hp: dict | None = None) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Walk-forward CV training. Returns OOF predictions DataFrame + per-fold results.

    Parameters
    ----------
    balanced : bool, default False
        If True, applies balanced class weighting:
          - LightGBM, RandomForest: class_weight="balanced" via sklearn API
          - XGBoost: inverse-frequency sample_weight at fit time
          - MLP: not supported by sklearn (ignored, paper limitation)
    hp : dict, optional
        Hyperparameter overrides forwarded to the model factory. None -> defaults.

    OOF DataFrame columns: ['P_downtrend', 'P_range', 'P_uptrend', 'pred_label', 'true_label', 'fold']
    indexed by date; rows present only for val periods of any fold.
    """
    cfg = cfg or WalkForwardConfig()
    hp = hp or {}

    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}")

    y_idx = y.map(LABEL_TO_IDX).to_numpy()
    X_arr = X.to_numpy()
    dates = X.index

    oof_records = []
    fold_results: list[FoldResult] = []

    for fold_i, (train_idx, val_idx) in enumerate(walk_forward_splits(len(X), cfg), start=1):
        X_tr, y_tr = X_arr[train_idx], y_idx[train_idx]
        X_val, y_val = X_arr[val_idx], y_idx[val_idx]

        if model_name in TREE_MODELS:
            model = MODEL_FACTORIES[model_name](random_state=random_state, balanced=balanced, **hp)
            if model_name == "xgboost" and balanced:
                sw = _balanced_sample_weight(y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)
        else:
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            model = MODEL_FACTORIES[model_name](random_state=random_state, balanced=balanced, **hp)
            model.fit(X_tr_s, y_tr)
            proba = model.predict_proba(X_val_s)

        # Some classifiers may not see all 3 classes in training; pad columns to 3
        if proba.shape[1] < 3:
            classes_seen = list(model.classes_)
            full = np.zeros((len(val_idx), 3), dtype=float)
            for j, c in enumerate(classes_seen):
                full[:, c] = proba[:, j]
            proba = full

        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average="macro", labels=[0, 1, 2], zero_division=0)
        f1c = f1_score(y_val, pred, average=None, labels=[0, 1, 2], zero_division=0)

        fold_results.append(FoldResult(
            fold=fold_i,
            train_start=dates[train_idx[0]], train_end=dates[train_idx[-1]],
            val_start=dates[val_idx[0]],     val_end=dates[val_idx[-1]],
            n_train=len(train_idx), n_val=len(val_idx),
            accuracy=acc, f1_macro=f1m,
            f1_downtrend=f1c[0], f1_range=f1c[1], f1_uptrend=f1c[2],
        ))

        for k, di in enumerate(val_idx):
            oof_records.append({
                "date":          dates[di],
                "P_downtrend":   proba[k, 0],
                "P_range":       proba[k, 1],
                "P_uptrend":     proba[k, 2],
                "pred_label":    IDX_TO_LABEL[pred[k]],
                "true_label":    IDX_TO_LABEL[y_val[k]],
                "fold":          fold_i,
            })

    oof = pd.DataFrame(oof_records).set_index("date").sort_index()
    return oof, fold_results


def fold_results_to_df(results: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def overall_metrics(oof: pd.DataFrame) -> dict:
    """Compute overall accuracy + F1 macro across all OOF predictions."""
    y_true = oof["true_label"].map(LABEL_TO_IDX).to_numpy()
    y_pred = oof["pred_label"].map(LABEL_TO_IDX).to_numpy()
    return {
        "n":         len(oof),
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
