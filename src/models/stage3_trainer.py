"""
Stage 3 — Signal Classifier Trainer.
Uses out-of-fold (OOF) predictions from Stage 1 & 2 as additional features.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.classifiers import get_classifier, BaseClassifier
from src.models.stage1_trainer import expanding_window_walk_forward
from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def generate_oof_predictions(
    X_stage1: pd.DataFrame,
    y_stage1: pd.Series,
    X_stage2: pd.DataFrame,
    y_stage2: pd.Series,
    classifier_name: str = "xgboost",
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate out-of-fold predictions from Stage 1 and Stage 2 for Stage 3 training.

    Uses chronological K-fold splits to prevent information leakage.

    Parameters
    ----------
    X_stage1 : pd.DataFrame
        Features for Stage 1 (trend indicators).
    y_stage1 : pd.Series
        Labels for Stage 1.
    X_stage2 : pd.DataFrame
        Features for Stage 2 (macro indicators).
    y_stage2 : pd.Series
        Labels for Stage 2.
    classifier_name : str
        Classifier to use.
    n_folds : int
        Number of chronological folds.

    Returns
    -------
    tuple of (oof_trend_proba, oof_regime_proba) DataFrames
    """
    # Common index across all data
    common_idx = (
        X_stage1.index
        .intersection(y_stage1.index)
        .intersection(X_stage2.index)
        .intersection(y_stage2.index)
    )

    X_s1 = X_stage1.loc[common_idx]
    y_s1 = y_stage1.loc[common_idx]
    X_s2 = X_stage2.loc[common_idx]
    y_s2 = y_stage2.loc[common_idx]

    n = len(common_idx)
    fold_size = n // n_folds

    logger.info(f"Generating OOF predictions: {n} samples, {n_folds} folds")

    # Initialize OOF arrays
    n_classes_s1 = len(y_s1.unique())
    n_classes_s2 = len(y_s2.unique())
    oof_s1 = np.full((n, n_classes_s1), np.nan)
    oof_s2 = np.full((n, n_classes_s2), np.nan)

    for fold_i in range(n_folds):
        val_start = fold_i * fold_size
        val_end = val_start + fold_size if fold_i < n_folds - 1 else n

        # Chronological: train on everything before val + everything after val
        # BUT to maintain temporal integrity, only train on data BEFORE the val fold
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:val_start] = True
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        if train_mask.sum() == 0:
            # First fold: skip (no training data available)
            logger.warning(f"  Fold {fold_i}: skipping (no training data before this fold)")
            continue

        # Stage 1 OOF
        clf_s1 = get_classifier(classifier_name)
        X_s1_train = X_s1.iloc[train_mask].fillna(X_s1.iloc[train_mask].median())
        X_s1_val = X_s1.iloc[val_mask].fillna(X_s1.iloc[train_mask].median())
        clf_s1.fit(X_s1_train, y_s1.iloc[train_mask])
        oof_s1[val_mask] = clf_s1.predict_proba(X_s1_val)

        # Stage 2 OOF
        clf_s2 = get_classifier(classifier_name)
        X_s2_train = X_s2.iloc[train_mask].fillna(X_s2.iloc[train_mask].median())
        X_s2_val = X_s2.iloc[val_mask].fillna(X_s2.iloc[train_mask].median())
        clf_s2.fit(X_s2_train, y_s2.iloc[train_mask])
        oof_s2[val_mask] = clf_s2.predict_proba(X_s2_val)

        logger.info(f"  Fold {fold_i}: train={train_mask.sum()}, val={val_mask.sum()}")

    # Build DataFrames
    # Get class names from a dummy fit
    clf_dummy_s1 = get_classifier(classifier_name)
    clf_dummy_s1.fit(
        X_s1.fillna(X_s1.median()).iloc[:10],
        y_s1.iloc[:10],
    )
    clf_dummy_s2 = get_classifier(classifier_name)
    clf_dummy_s2.fit(
        X_s2.fillna(X_s2.median()).iloc[:10],
        y_s2.iloc[:10],
    )

    oof_trend_df = pd.DataFrame(
        oof_s1, index=common_idx,
        columns=[f"trend_prob_{c}" for c in clf_dummy_s1.classes_],
    )
    oof_regime_df = pd.DataFrame(
        oof_s2, index=common_idx,
        columns=[f"regime_prob_{c}" for c in clf_dummy_s2.classes_],
    )

    # Drop rows where OOF is NaN (first fold if skipped)
    valid_mask = ~oof_trend_df.isna().any(axis=1)
    oof_trend_df = oof_trend_df[valid_mask]
    oof_regime_df = oof_regime_df[valid_mask]

    logger.info(
        f"OOF predictions generated: {len(oof_trend_df)} valid rows "
        f"(dropped {(~valid_mask).sum()} NaN rows from first fold)"
    )

    return oof_trend_df, oof_regime_df


def train_stage3(
    X_oscillator: pd.DataFrame,
    y_signal: pd.Series,
    oof_trend: pd.DataFrame,
    oof_regime: pd.DataFrame,
    classifier_name: str = "xgboost",
    params: dict | None = None,
    save_model: bool = True,
) -> dict:
    """
    Train Stage 3 signal classifier using oscillator features + OOF predictions.

    Parameters
    ----------
    X_oscillator : pd.DataFrame
        Oscillator features (after selection).
    y_signal : pd.Series
        Signal labels (Buy/Sell/Hold).
    oof_trend : pd.DataFrame
        OOF trend probability predictions from Stage 1.
    oof_regime : pd.DataFrame
        OOF regime probability predictions from Stage 2.
    classifier_name : str
        Classifier to use.
    params : dict
        Hyperparameters.
    save_model : bool
        Whether to save the model.

    Returns
    -------
    dict with training results.
    """
    config = cfg()
    params = params or {}

    # Align all data
    common_idx = (
        X_oscillator.index
        .intersection(y_signal.index)
        .intersection(oof_trend.index)
        .intersection(oof_regime.index)
    )

    X_osc = X_oscillator.loc[common_idx]
    y = y_signal.loc[common_idx]
    trend_probs = oof_trend.loc[common_idx]
    regime_probs = oof_regime.loc[common_idx]

    # Combine features: oscillators + trend probs + regime probs
    X_combined = pd.concat([X_osc, trend_probs, regime_probs], axis=1)

    logger.info(
        f"Stage 3 Training: {classifier_name}, {len(X_combined)} samples, "
        f"{X_combined.shape[1]} features "
        f"({X_osc.shape[1]} oscillator + {trend_probs.shape[1]} trend + {regime_probs.shape[1]} regime), "
        f"classes={y.unique().tolist()}"
    )

    # Walk-forward folds
    folds = expanding_window_walk_forward(
        X_combined, y,
        min_train_months=config["training"]["min_train_window_months"],
        step_months=config["training"]["walk_forward_step_months"],
    )

    n_classes = len(y.unique())
    oof_proba = np.full((len(X_combined), n_classes), np.nan)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = X_combined.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_combined.iloc[val_idx]
        y_val = y.iloc[val_idx]

        clf = get_classifier(classifier_name, **params)

        X_train_clean = X_train.fillna(X_train.median())
        X_val_clean = X_val.fillna(X_train.median())

        clf.fit(X_train_clean, y_train, X_val_clean, y_val)

        val_pred = clf.predict(X_val_clean)
        val_proba = clf.predict_proba(X_val_clean)
        oof_proba[val_idx] = val_proba

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

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
    X_clean = X_combined.fillna(X_combined.median())
    clf_final.fit(X_clean, y)

    if save_model:
        root = get_project_root()
        model_path = root / config["paths"]["models"] / f"stage3_{classifier_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_final, model_path)
        logger.info(f"  Saved model to {model_path}")

    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc = metrics_df["accuracy"].mean()
    avg_f1 = metrics_df["f1_macro"].mean()
    logger.info(f"  Average: acc={avg_acc:.4f}, f1_macro={avg_f1:.4f}")

    return {
        "model": clf_final,
        "fold_metrics": metrics_df,
        "feature_names": X_combined.columns.tolist(),
        "feature_importance": clf_final.get_feature_importance(),
        "avg_accuracy": avg_acc,
        "avg_f1_macro": avg_f1,
    }
