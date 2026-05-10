"""V5 Phase 7+ — Stage 3 walk-forward calibration.

Why calibration?
----------------
Tree-based classifiers (and MLPs without temperature scaling) produce
posterior probabilities that are systematically over- or under-confident.
For Bayesian decision-theoretic soft fusion this matters: the trading
rules depend on the *magnitude* of P_Buy / P_Sell (especially the
probability-weighted rule, where position size = P_Buy − P_Sell).

We apply isotonic regression in a walk-forward-safe manner:

For each outer-CV fold k with training span [0, t_k] and validation span
[t_k+gap, t_k+gap+v]:
  1. Re-split the training span into:
     - inner-train  = [0, t_k − calib_size]  (used to fit the model)
     - inner-calib  = [t_k − calib_size, t_k] (held out for calibration)
  2. Fit the model on inner-train with the tuned hyper-parameters.
  3. Predict on inner-calib → fit one isotonic regression per class
     (one-vs-rest), giving three monotonic mappings p_raw → p_cal.
  4. Predict on val_test → apply the calibrated mappings, renormalize
     so the three probabilities sum to 1.

This procedure has zero look-ahead leakage: calibration is fit on data
that is strictly older than the validation window (separated by the
same `gap` that the outer CV uses).

Output: a calibrated OOF DataFrame with columns
  P_Sell_cal, P_Hold_cal, P_Buy_cal, pred_label_cal (argmax of calibrated)
  + the original raw columns kept for diff comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.models.v5_stage3_trainer import (
    LABEL_TO_IDX, IDX_TO_LABEL, MODEL_FACTORIES, TREE_MODELS,
    WalkForwardConfig, walk_forward_splits, _balanced_sample_weight,
)

warnings.filterwarnings("ignore")


@dataclass
class CalibrationConfig:
    method:      str = "isotonic"   # "isotonic" or "platt" (platt = sigmoid)
    calib_size:  int = 200          # last N days of training span used for calibration
    min_calib:   int = 100          # minimum samples required to fit a calibrator
    eps:         float = 1e-6       # softmax-renormalize floor


def _fit_isotonic_per_class(p_calib: np.ndarray, y_calib: np.ndarray) -> list[IsotonicRegression]:
    """One isotonic regressor per class (one-vs-rest). p_calib (N,3), y_calib (N,) of {0,1,2}."""
    calibrators = []
    for c in range(3):
        y_bin = (y_calib == c).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # Need both classes present; if calibration set is degenerate, fall back to identity
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            ir = None
        else:
            ir.fit(p_calib[:, c], y_bin)
        calibrators.append(ir)
    return calibrators


def _apply_calibrators(calibrators, p_test: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p_test)
    for c in range(3):
        ir = calibrators[c]
        if ir is None:
            out[:, c] = p_test[:, c]
        else:
            out[:, c] = ir.predict(p_test[:, c])
    # Normalize rows to sum 1 (isotonic per-class doesn't conserve sum)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
    return out / row_sum


def calibrated_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cfg: WalkForwardConfig | None = None,
    cal_cfg: CalibrationConfig | None = None,
    random_state: int = 42,
    balanced: bool = True,
    hp: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward fit + isotonic calibration.

    Returns a DataFrame indexed by validation date with columns:
      P_Sell, P_Hold, P_Buy           (raw, for comparison)
      P_Sell_cal, P_Hold_cal, P_Buy_cal
      pred_label, pred_label_cal      (argmax of raw / calibrated)
      true_label, fold
    """
    cfg = cfg or WalkForwardConfig()
    cal_cfg = cal_cfg or CalibrationConfig()
    hp = hp or {}

    y_idx = y.map(LABEL_TO_IDX).to_numpy()
    X_arr = X.to_numpy()
    dates = X.index

    records = []
    for fold_i, (train_idx, val_idx) in enumerate(walk_forward_splits(len(X), cfg), start=1):
        # Split training into inner-train + inner-calibration
        n_train = len(train_idx)
        cal_n = min(cal_cfg.calib_size, max(cal_cfg.min_calib, n_train // 5))
        if n_train - cal_n < cal_cfg.min_calib:
            cal_n = max(0, n_train - cal_cfg.min_calib)
        if cal_n < cal_cfg.min_calib:
            # Calibration impossible; skip calibration on this fold
            cal_n = 0

        inner_tr_idx = train_idx[:n_train - cal_n] if cal_n > 0 else train_idx
        inner_ca_idx = train_idx[n_train - cal_n:] if cal_n > 0 else None

        X_tr  = X_arr[inner_tr_idx]; y_tr = y_idx[inner_tr_idx]
        X_val = X_arr[val_idx];      y_val = y_idx[val_idx]

        # Fit model
        if model_name in TREE_MODELS:
            model = MODEL_FACTORIES[model_name](
                random_state=random_state, balanced=balanced, **hp
            )
            if model_name == "xgboost" and balanced:
                sw = _balanced_sample_weight(y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)
            proba_val = model.predict_proba(X_val)
            if cal_n > 0:
                X_ca = X_arr[inner_ca_idx]; y_ca = y_idx[inner_ca_idx]
                proba_ca = model.predict_proba(X_ca)
        else:
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            model = MODEL_FACTORIES[model_name](
                random_state=random_state, balanced=balanced, **hp
            )
            model.fit(X_tr_s, y_tr)
            proba_val = model.predict_proba(X_val_s)
            if cal_n > 0:
                X_ca   = X_arr[inner_ca_idx]; y_ca = y_idx[inner_ca_idx]
                X_ca_s = scaler.transform(X_ca)
                proba_ca = model.predict_proba(X_ca_s)

        # Pad if class missing in this fold's training
        if proba_val.shape[1] < 3:
            classes_seen = list(model.classes_)
            full = np.zeros((len(val_idx), 3), dtype=float)
            for j, c in enumerate(classes_seen):
                full[:, c] = proba_val[:, j]
            proba_val = full
            if cal_n > 0:
                full_ca = np.zeros((len(inner_ca_idx), 3), dtype=float)
                for j, c in enumerate(classes_seen):
                    full_ca[:, c] = proba_ca[:, j]
                proba_ca = full_ca

        # Calibrate
        if cal_n > 0:
            calibrators = _fit_isotonic_per_class(proba_ca, y_ca)
            proba_val_cal = _apply_calibrators(calibrators, proba_val)
        else:
            proba_val_cal = proba_val.copy()

        pred_raw = np.argmax(proba_val, axis=1)
        pred_cal = np.argmax(proba_val_cal, axis=1)

        for k, di in enumerate(val_idx):
            records.append({
                "date":           dates[di],
                "P_Sell":         proba_val[k, 0],
                "P_Hold":         proba_val[k, 1],
                "P_Buy":          proba_val[k, 2],
                "P_Sell_cal":     proba_val_cal[k, 0],
                "P_Hold_cal":     proba_val_cal[k, 1],
                "P_Buy_cal":      proba_val_cal[k, 2],
                "pred_label":     IDX_TO_LABEL[pred_raw[k]],
                "pred_label_cal": IDX_TO_LABEL[pred_cal[k]],
                "true_label":     IDX_TO_LABEL[y_val[k]],
                "fold":           fold_i,
            })

    return pd.DataFrame(records).set_index("date").sort_index()


def calibration_metrics(oof: pd.DataFrame, n_bins: int = 10) -> dict:
    """ECE (Expected Calibration Error) + Brier score per class, raw vs cal."""
    out = {"raw": {}, "cal": {}}
    y_true = oof["true_label"].map(LABEL_TO_IDX).to_numpy()
    for variant, cols in [("raw", ["P_Sell", "P_Hold", "P_Buy"]),
                          ("cal", ["P_Sell_cal", "P_Hold_cal", "P_Buy_cal"])]:
        proba = oof[cols].to_numpy()
        per_class = []
        for c in range(3):
            y_bin = (y_true == c).astype(int)
            p = proba[:, c]
            # ECE bin-wise
            bins = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for b in range(n_bins):
                mask = (p >= bins[b]) & (p < bins[b + 1])
                if b == n_bins - 1:
                    mask = (p >= bins[b]) & (p <= bins[b + 1])
                if mask.sum() == 0:
                    continue
                acc = y_bin[mask].mean()
                conf = p[mask].mean()
                ece += (mask.sum() / len(p)) * abs(acc - conf)
            brier = brier_score_loss(y_bin, p)
            per_class.append({"class": ["Sell", "Hold", "Buy"][c], "ece": ece, "brier": brier})
        out[variant] = per_class
        out[variant + "_macro_ece"]   = float(np.mean([r["ece"]   for r in per_class]))
        out[variant + "_macro_brier"] = float(np.mean([r["brier"] for r in per_class]))
    return out
