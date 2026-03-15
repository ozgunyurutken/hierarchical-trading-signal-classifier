"""
Stage 1 — Trend Classifier Trainer.
Handles walk-forward validation, hyperparameter tuning, and model evaluation.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.classifiers import get_classifier, BaseClassifier
from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, compute_class_weights

logger = setup_logger(__name__)


def expanding_window_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_months: int = 6,
    step_months: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding-window walk-forward train/val index splits.
    All splits are chronological with no overlap.

    Returns
    -------
    list of (train_indices, val_indices) tuples
    """
    dates = X.index
    min_date = dates.min()
    max_date = dates.max()

    min_train_end = min_date + pd.DateOffset(months=min_train_months)
    current_val_start = min_train_end

    folds = []
    while current_val_start < max_date:
        val_end = current_val_start + pd.DateOffset(months=step_months)
        if val_end > max_date:
            val_end = max_date

        train_mask = dates < current_val_start
        val_mask = (dates >= current_val_start) & (dates < val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))

        current_val_start = val_end

    logger.info(f"Walk-forward: {len(folds)} folds, min train window = {min_train_months} months")
    return folds


def train_stage1(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str = "xgboost",
    params: dict | None = None,
    save_model: bool = True,
) -> dict:
    """
    Train a Stage 1 trend classifier with walk-forward validation.

    Parameters
    ----------
    X : pd.DataFrame
        Stage 1 features (trend indicators).
    y : pd.Series
        Trend labels (Uptrend/Downtrend/Sideways).
    classifier_name : str
        Classifier name (xgboost, lightgbm, random_forest, mlp).
    params : dict, optional
        Hyperparameters for the classifier.
    save_model : bool
        Whether to save the final model.

    Returns
    -------
    dict with keys:
        'model': fitted model
        'fold_metrics': per-fold metrics
        'oof_predictions': out-of-fold probability predictions
        'feature_importance': importance array
    """
    config = cfg()
    params = params or {}

    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    logger.info(
        f"Stage 1 Training: {classifier_name}, {len(X)} samples, "
        f"{X.shape[1]} features, classes={y.unique().tolist()}"
    )

    # Walk-forward folds
    folds = expanding_window_walk_forward(
        X, y,
        min_train_months=config["training"]["min_train_window_months"],
        step_months=config["training"]["walk_forward_step_months"],
    )

    # OOF predictions
    n_classes = len(y.unique())
    oof_proba = np.full((len(X), n_classes), np.nan)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        clf = get_classifier(classifier_name, **params)

        # Handle NaN
        X_train_clean = X_train_fold.fillna(X_train_fold.median())
        X_val_clean = X_val_fold.fillna(X_train_fold.median())

        clf.fit(X_train_clean, y_train_fold, X_val_clean, y_val_fold)

        # Predictions
        val_pred = clf.predict(X_val_clean)
        val_proba = clf.predict_proba(X_val_clean)
        oof_proba[val_idx] = val_proba

        # Fold metrics
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_val_fold, val_pred)
        f1 = f1_score(y_val_fold, val_pred, average="macro", zero_division=0)

        fold_metrics.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy": acc,
            "f1_macro": f1,
        })
        logger.info(f"  Fold {fold_idx}: acc={acc:.4f}, f1_macro={f1:.4f}")

    # Train final model on all data
    clf_final = get_classifier(classifier_name, **params)
    X_clean = X.fillna(X.median())
    clf_final.fit(X_clean, y)

    # Save model
    if save_model:
        root = get_project_root()
        model_path = root / config["paths"]["models"] / f"stage1_{classifier_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_final, model_path)
        logger.info(f"  Saved model to {model_path}")

    # Average metrics
    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc = metrics_df["accuracy"].mean()
    avg_f1 = metrics_df["f1_macro"].mean()
    logger.info(f"  Average: acc={avg_acc:.4f}, f1_macro={avg_f1:.4f}")

    # OOF predictions DataFrame
    oof_df = pd.DataFrame(
        oof_proba,
        index=X.index,
        columns=[f"trend_prob_{c}" for c in clf_final.classes_],
    )

    return {
        "model": clf_final,
        "fold_metrics": metrics_df,
        "oof_predictions": oof_df,
        "feature_importance": clf_final.get_feature_importance(),
        "avg_accuracy": avg_acc,
        "avg_f1_macro": avg_f1,
    }
