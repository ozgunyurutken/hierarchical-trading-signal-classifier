"""V5 Phase 4 — Stage 3 Optuna hyperparameter tuning.

Same structure as v5_stage1_optuna.py:
  - InnerCVConfig: 5 evenly-spaced WF folds
  - 4 search spaces (XGB 6HP, LGBM 6HP, RF 4HP, MLP 5HP)
  - MedianPruner + TPESampler + SQLite storage
  - Objective: F1 macro mean across inner folds
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.models.v5_stage3_trainer import (
    LABEL_TO_IDX, MODEL_FACTORIES, TREE_MODELS,
    _balanced_sample_weight,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class InnerCVConfig:
    train_min_size: int = 750
    val_size: int = 300
    gap: int = 10
    n_folds: int = 5


def inner_walk_forward_splits(n: int, cfg: InnerCVConfig):
    first_t = cfg.train_min_size
    last_t = n - cfg.gap - cfg.val_size
    if last_t < first_t:
        raise ValueError(
            f"Dataset too small: n={n}, train_min={cfg.train_min_size}, "
            f"val={cfg.val_size}, gap={cfg.gap}"
        )
    if cfg.n_folds == 1:
        ts = [last_t]
    else:
        step = (last_t - first_t) / (cfg.n_folds - 1)
        ts = [int(round(first_t + i * step)) for i in range(cfg.n_folds)]
    for t in ts:
        train_idx = np.arange(0, t)
        val_start = t + cfg.gap
        val_idx = np.arange(val_start, val_start + cfg.val_size)
        yield train_idx, val_idx


def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def suggest_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600, step=50),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127, log=True),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 5, 50),
    }


def suggest_random_forest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth":        trial.suggest_int("max_depth", 6, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),  # leaf>=3 to avoid Phase3-style overfit
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def suggest_mlp(trial: optuna.Trial) -> dict:
    layers = trial.suggest_categorical(
        "hidden_layer_sizes_idx", ["64", "64_32", "128_64", "64_32_16"]
    )
    layer_map = {
        "64":         (64,),
        "64_32":      (64, 32),
        "128_64":     (128, 64),
        "64_32_16":   (64, 32, 16),
    }
    bs = trial.suggest_categorical("batch_size", ["auto", "64", "128"])
    return {
        "hidden_layer_sizes": layer_map[layers],
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "alpha":              trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "batch_size":         bs if bs == "auto" else int(bs),
    }


SUGGEST_FNS = {
    "xgboost":       suggest_xgboost,
    "lightgbm":      suggest_lightgbm,
    "random_forest": suggest_random_forest,
    "mlp":           suggest_mlp,
}


def _eval_fold_f1m(model, X_tr, y_tr, X_val, y_val, model_name: str,
                   balanced: bool) -> float:
    if model_name in TREE_MODELS:
        if model_name == "xgboost" and balanced:
            sw = _balanced_sample_weight(y_tr)
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
    else:
        scaler = StandardScaler().fit(X_tr)
        model.fit(scaler.transform(X_tr), y_tr)
        pred = model.predict(scaler.transform(X_val))
    return f1_score(y_val, pred, average="macro", labels=[0, 1, 2], zero_division=0)


def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
               model_name: str, balanced: bool, inner_cfg: InnerCVConfig,
               random_state: int) -> float:
    hp = SUGGEST_FNS[model_name](trial)
    y_idx = y.map(LABEL_TO_IDX).to_numpy()
    X_arr = X.to_numpy()

    f1s: list[float] = []
    for step_i, (tr_idx, val_idx) in enumerate(
        inner_walk_forward_splits(len(X), inner_cfg), start=1
    ):
        X_tr, y_tr = X_arr[tr_idx], y_idx[tr_idx]
        X_val, y_val = X_arr[val_idx], y_idx[val_idx]
        model = MODEL_FACTORIES[model_name](
            random_state=random_state, balanced=balanced, **hp
        )
        f1 = _eval_fold_f1m(model, X_tr, y_tr, X_val, y_val,
                            model_name=model_name, balanced=balanced)
        f1s.append(f1)
        trial.report(float(np.mean(f1s)), step=step_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(f1s))


def run_study(asset: str, model_name: str, X: pd.DataFrame, y: pd.Series,
              n_trials: int = 30, balanced: bool = True,
              inner_cfg: InnerCVConfig | None = None,
              random_state: int = 42,
              pruner_warmup: int = 5,
              storage: str | None = None) -> tuple[optuna.Study, dict]:
    if model_name not in SUGGEST_FNS:
        raise ValueError(f"Unknown model: {model_name}")

    inner_cfg = inner_cfg or InnerCVConfig()

    study = optuna.create_study(
        study_name=f"v5_stage3_{asset}_{model_name}",
        storage=storage,
        load_if_exists=storage is not None,
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_warmup_steps=1, n_startup_trials=pruner_warmup),
    )

    study.optimize(
        lambda t: _objective(t, X, y, model_name, balanced, inner_cfg, random_state),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    if model_name == "mlp":
        layer_map = {
            "64": (64,), "64_32": (64, 32),
            "128_64": (128, 64), "64_32_16": (64, 32, 16),
        }
        idx = best.pop("hidden_layer_sizes_idx")
        best["hidden_layer_sizes"] = layer_map[idx]
        if isinstance(best.get("batch_size"), str) and best["batch_size"] != "auto":
            best["batch_size"] = int(best["batch_size"])

    return study, best


def study_to_trials_df(study: optuna.Study, asset: str, model_name: str) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        rows.append({
            "asset":      asset,
            "model":      model_name,
            "trial":      t.number,
            "state":      t.state.name,
            "value":      t.value if t.value is not None else float("nan"),
            "duration_s": (t.duration.total_seconds() if t.duration else float("nan")),
            **{f"param_{k}": v for k, v in t.params.items()},
        })
    return pd.DataFrame(rows)
