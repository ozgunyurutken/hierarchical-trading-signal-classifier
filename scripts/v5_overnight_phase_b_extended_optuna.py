"""V5 Overnight Phase B — Extended Optuna for 3-Stage Full Stage 3.

Same 5-fold inner CV but DOUBLED trial budget (60 vs 30), with WIDER
search spaces. Goal: F1m 0.37 -> 0.40 decision gate target.

Wider search spaces:
  XGBoost      max_depth 3->10, learning_rate 0.005-0.20, n_estimators 100-1000
  LightGBM     num_leaves 15-255, min_data_in_leaf 3-100
  RandomForest max_depth 6-30, min_samples_leaf 2-20, max_features wider
  MLP          hidden_layer_sizes more options

Outputs:
  data/processed/{asset}_stage3_oof_{model}_v5_tuned_extended.csv
  reports/Phase5.2_extended_optuna/v5_p5_extended_overall.csv
  reports/Phase5.2_extended_optuna/v5_p5_extended_optuna_best.csv
  reports/Phase5.2_extended_optuna/v5_p5_extended_backtest.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.evaluation.v5_backtester import (
    backtest_stateful, backtest_defensive, backtest_prob_weighted,
    backtest_buy_and_hold,
)
from src.features.v5_stage3_features import STAGE3_FEATURE_COLS
from src.models.v5_stage3_optuna import InnerCVConfig, inner_walk_forward_splits
from src.models.v5_stage3_trainer import (
    LABEL_TO_IDX, MODEL_FACTORIES, TREE_MODELS,
    _balanced_sample_weight, train_walk_forward, overall_metrics,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Wider search spaces
def suggest_xgb_wide(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
    }


def suggest_lgbm_wide(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 3, 100),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
    }


def suggest_rf_wide(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth":        trial.suggest_int("max_depth", 6, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
    }


def suggest_mlp_wide(trial):
    layers = trial.suggest_categorical(
        "hidden_layer_sizes_idx",
        ["32", "64", "128", "32_16", "64_32", "128_64", "256_128",
         "64_32_16", "128_64_32", "256_128_64"]
    )
    layer_map = {
        "32": (32,), "64": (64,), "128": (128,),
        "32_16": (32, 16), "64_32": (64, 32),
        "128_64": (128, 64), "256_128": (256, 128),
        "64_32_16": (64, 32, 16), "128_64_32": (128, 64, 32),
        "256_128_64": (256, 128, 64),
    }
    bs = trial.suggest_categorical("batch_size", ["auto", "32", "64", "128", "256"])
    return {
        "hidden_layer_sizes": layer_map[layers],
        "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-5, 5e-2, log=True),
        "alpha":              trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
        "batch_size":         bs if bs == "auto" else int(bs),
    }


SUGGEST_FNS_WIDE = {
    "xgboost":       suggest_xgb_wide,
    "lightgbm":      suggest_lgbm_wide,
    "random_forest": suggest_rf_wide,
    "mlp":           suggest_mlp_wide,
}


def _eval_fold_f1m(model, X_tr, y_tr, X_val, y_val, model_name, balanced):
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


def _objective(trial, X, y, model_name, balanced, inner_cfg, random_state):
    hp = SUGGEST_FNS_WIDE[model_name](trial)
    y_idx = y.map(LABEL_TO_IDX).to_numpy()
    X_arr = X.to_numpy()

    f1s = []
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


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase5.2_extended_optuna"
    out.mkdir(parents=True, exist_ok=True)

    inner_cfg = InnerCVConfig()
    n_trials = 60
    random_state = 42

    overall_rows = []
    backtest_rows = []
    optuna_best_rows = []

    ASSETS = ["btc", "eth"]
    MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]

    for asset in ASSETS:
        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]
        X = full[STAGE3_FEATURE_COLS]
        y = full["signal_label"]
        print(f"\n[{asset.upper()}] extended Optuna {X.shape}", flush=True)

        for model_name in MODELS:
            t0 = time.time()
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=random_state),
                pruner=MedianPruner(n_warmup_steps=1, n_startup_trials=5),
            )
            study.optimize(
                lambda t: _objective(t, X, y, model_name, True, inner_cfg, random_state),
                n_trials=n_trials, n_jobs=1, gc_after_trial=True,
            )
            best = dict(study.best_params)
            if model_name == "mlp":
                layer_map = {
                    "32": (32,), "64": (64,), "128": (128,),
                    "32_16": (32, 16), "64_32": (64, 32),
                    "128_64": (128, 64), "256_128": (256, 128),
                    "64_32_16": (64, 32, 16), "128_64_32": (128, 64, 32),
                    "256_128_64": (256, 128, 64),
                }
                idx = best.pop("hidden_layer_sizes_idx")
                best["hidden_layer_sizes"] = layer_map[idx]
                if isinstance(best.get("batch_size"), str) and best["batch_size"] != "auto":
                    best["batch_size"] = int(best["batch_size"])

            tune_dt = time.time() - t0

            optuna_best_rows.append({
                "asset": asset, "model": model_name,
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
                "best_params": json.dumps(best, default=lambda o: list(o) if isinstance(o, tuple) else o),
                "elapsed_s": round(tune_dt, 1),
            })

            # Tuned outer retrain
            hp_factory = dict(best)
            if model_name == "mlp" and isinstance(hp_factory.get("hidden_layer_sizes"), list):
                hp_factory["hidden_layer_sizes"] = tuple(hp_factory["hidden_layer_sizes"])
            oof, _ = train_walk_forward(X, y, model_name, balanced=True, hp=hp_factory)

            oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5_tuned_extended.csv"
            oof.to_csv(oof_path)

            ov = overall_metrics(oof)
            overall_rows.append({
                "asset": asset, "model": model_name,
                "n_oof": ov["n"], "accuracy": ov["accuracy"], "f1_macro": ov["f1_macro"],
                "f1_sell": ov["f1_per_class"][0],
                "f1_hold": ov["f1_per_class"][1],
                "f1_buy":  ov["f1_per_class"][2],
            })

            for rule_name, fn in [
                ("stateful",      backtest_stateful),
                ("defensive",     backtest_defensive),
                ("prob_weighted", backtest_prob_weighted),
            ]:
                res, _ = fn(oof, prices, asset=asset, model=model_name)
                backtest_rows.append(res.to_dict())

            print(f"  {model_name:14s} inner {study.best_value:.4f}  "
                  f"outer {ov['f1_macro']:.4f}  ({tune_dt:.0f}s)", flush=True)

        # B&H per asset
        first_oof = pd.read_csv(proc / f"{asset}_stage3_oof_xgboost_v5_tuned_extended.csv",
                                 index_col=0, parse_dates=True)
        bh_res, _ = backtest_buy_and_hold(prices, first_oof.index.min(),
                                          first_oof.index.max(), asset=asset)
        backtest_rows.append(bh_res.to_dict())

    pd.DataFrame(overall_rows).to_csv(out / "v5_p5_extended_overall.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p5_extended_backtest.csv", index=False)
    pd.DataFrame(optuna_best_rows).to_csv(out / "v5_p5_extended_optuna_best.csv", index=False)

    print(f"\n=== Phase B complete ===")
    print(pd.DataFrame(overall_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
