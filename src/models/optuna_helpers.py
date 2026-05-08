"""
Optuna-based hyperparameter tuning helpers.

Generic walk-forward CV tuner that maximizes macro-F1 across folds. Designed to
work with any BaseClassifier wrapper from `classifiers.py`.

Search spaces are model-specific; only LDA and MLP are wired up for the MVP.
XGBoost / LightGBM / RandomForest / SVM spaces are stubs ready for the second
iteration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.classifiers import get_classifier
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def suggest_params(trial, classifier_name: str) -> dict:
    """Define the Optuna search space per classifier."""
    name = classifier_name.lower()

    if name == "lda":
        solver = trial.suggest_categorical("solver", ["svd", "lsqr"])
        if solver in ["lsqr", "eigen"]:
            shrinkage = trial.suggest_float("shrinkage", 0.0, 1.0)
        else:
            shrinkage = None
        return {"solver": solver, "shrinkage": shrinkage}

    if name == "mlp":
        hidden_choices = [[64, 32], [128, 64]]
        idx = trial.suggest_int("hidden_idx", 0, len(hidden_choices) - 1)
        return {
            "hidden_layers": hidden_choices[idx],
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "patience": 5,
            "max_epochs": 30,  # MVP: aggressive cap to keep walk-forward tuning under ~10 min
        }

    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }

    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        }

    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 20, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    return {}


def tune_classifier_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 20,
    study_name: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[dict, object]:
    """
    Run Optuna study to find best hyperparameters via walk-forward CV.

    Each trial trains the classifier on every fold's train slice and reports the
    mean macro-F1 across folds. The TPE sampler with seed=42 makes the search
    deterministic.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    classifier_name : str
        Name of classifier (must match `get_classifier` factory).
    folds : list of (train_idx, val_idx) tuples
        Pre-computed walk-forward fold indices.
    n_trials : int
        Number of Optuna trials.
    study_name : str, optional
        Name for the study (logging).
    timeout_seconds : int, optional
        Wall-clock timeout for the whole study.

    Returns
    -------
    tuple of (best_params, study)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute median fill values per fold to avoid recomputation in trials
    fold_data = []
    for train_idx, val_idx in folds:
        X_train = X.iloc[train_idx]
        median = X_train.median()
        fold_data.append({
            "X_train": X_train.fillna(median),
            "y_train": y.iloc[train_idx],
            "X_val": X.iloc[val_idx].fillna(median),
            "y_val": y.iloc[val_idx],
        })

    def objective(trial):
        params = suggest_params(trial, classifier_name)
        f1_scores = []
        for fold in fold_data:
            try:
                clf = get_classifier(classifier_name, **params)
                clf.fit(fold["X_train"], fold["y_train"], fold["X_val"], fold["y_val"])
                y_pred = clf.predict(fold["X_val"])
                f1 = f1_score(fold["y_val"], y_pred, average="macro", zero_division=0)
                f1_scores.append(f1)
            except Exception as e:
                logger.warning(f"Trial fold failed: {e}")
                return 0.0

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name or f"tune_{classifier_name}",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    logger.info(
        f"Optuna [{study_name or classifier_name}] complete: "
        f"best_f1={study.best_value:.4f}, params={study.best_params}"
    )

    return study.best_params, study
