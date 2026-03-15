"""
Stage 2 — Macro Regime Classifier Trainer.
Same walk-forward methodology as Stage 1, applied to macro features.
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


def train_stage2(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str = "xgboost",
    params: dict | None = None,
    save_model: bool = True,
) -> dict:
    """
    Train a Stage 2 macro regime classifier with walk-forward validation.

    Parameters
    ----------
    X : pd.DataFrame
        Macro features (after feature selection).
    y : pd.Series
        Regime labels (Risk-On/Risk-Off/Neutral).
    classifier_name : str
        Classifier to use.
    params : dict, optional
        Hyperparameters.
    save_model : bool
        Whether to save the model.

    Returns
    -------
    dict with same structure as Stage 1 trainer.
    """
    config = cfg()
    params = params or {}

    # Align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    logger.info(
        f"Stage 2 Training: {classifier_name}, {len(X)} samples, "
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

        X_train_clean = X_train_fold.fillna(X_train_fold.median())
        X_val_clean = X_val_fold.fillna(X_train_fold.median())

        clf.fit(X_train_clean, y_train_fold, X_val_clean, y_val_fold)

        val_pred = clf.predict(X_val_clean)
        val_proba = clf.predict_proba(X_val_clean)
        oof_proba[val_idx] = val_proba

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

    if save_model:
        root = get_project_root()
        model_path = root / config["paths"]["models"] / f"stage2_{classifier_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_final, model_path)
        logger.info(f"  Saved model to {model_path}")

    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc = metrics_df["accuracy"].mean()
    avg_f1 = metrics_df["f1_macro"].mean()
    logger.info(f"  Average: acc={avg_acc:.4f}, f1_macro={avg_f1:.4f}")

    oof_df = pd.DataFrame(
        oof_proba,
        index=X.index,
        columns=[f"regime_prob_{c}" for c in clf_final.classes_],
    )

    return {
        "model": clf_final,
        "fold_metrics": metrics_df,
        "oof_predictions": oof_df,
        "feature_importance": clf_final.get_feature_importance(),
        "avg_accuracy": avg_acc,
        "avg_f1_macro": avg_f1,
    }
