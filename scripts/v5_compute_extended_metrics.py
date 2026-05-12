"""V5 Phase 8 - Extended classification metrics for coursework checklist.

Computes Accuracy, Precision (per-class + macro), Recall (per-class + macro),
F1 (per-class + macro) and ROC-AUC (one-vs-rest per-class + macro) on the OOF
predictions for both stages. Required to satisfy the BBL514E grading rubric
which lists Accuracy, Precision, Recall, F1, ROC-AUC explicitly.

Outputs (under reports/Phase8_coursework_metrics/):
  v5_p8_stage1_extended_metrics.csv   # 4 models x 2 assets
  v5_p8_stage3_extended_metrics.csv   # 4 models x 2 assets (3stage_full default)
  v5_p8_stage1_class_balance.csv      # per-asset class counts and proportions
  v5_p8_stage3_roc_curves.png         # 4 classifiers x 2 assets ROC overlay
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "mlp": "MLP",
}
MODEL_COLORS = {
    "xgboost": "#1f77b4",
    "lightgbm": "#ff7f0e",
    "random_forest": "#2ca02c",
    "mlp": "#d62728",
}

STAGE1_LABEL_TO_IDX = {"downtrend": 0, "range": 1, "uptrend": 2}
STAGE1_CLASS_NAMES = ["downtrend", "range", "uptrend"]
STAGE3_LABEL_TO_IDX = {"Sell": 0, "Hold": 1, "Buy": 2}
STAGE3_CLASS_NAMES = ["Sell", "Hold", "Buy"]


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray,
                              class_names: list[str]) -> dict:
    """Compute Acc / P / R / F1 (per-class + macro) and ROC-AUC (per-class + macro)."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    out["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    out["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = p_per[i]
        out[f"recall_{name}"] = r_per[i]
        out[f"f1_{name}"] = f_per[i]

    n_classes = len(class_names)
    y_onehot = np.eye(n_classes)[y_true]
    auc_per = []
    for i, name in enumerate(class_names):
        try:
            auc = roc_auc_score(y_onehot[:, i], proba[:, i])
        except ValueError:
            auc = float("nan")
        out[f"auc_{name}"] = auc
        auc_per.append(auc)
    out["auc_macro"] = float(np.nanmean(auc_per))
    return out


def stage1_metrics() -> pd.DataFrame:
    proc = PROJECT_ROOT / "data" / "processed"
    rows = []
    for asset in ASSETS:
        for model in MODELS:
            p = proc / f"{asset}_stage1_oof_{model}_v5_tuned.csv"
            if not p.exists():
                print(f"  skip (missing): {p.name}")
                continue
            oof = pd.read_csv(p, index_col=0, parse_dates=True)
            y_true = oof["true_label"].map(STAGE1_LABEL_TO_IDX).to_numpy()
            y_pred = oof["pred_label"].map(STAGE1_LABEL_TO_IDX).to_numpy()
            proba = oof[["P_downtrend", "P_range", "P_uptrend"]].to_numpy()
            metrics = compute_extended_metrics(y_true, y_pred, proba, STAGE1_CLASS_NAMES)
            row = {"asset": asset, "model": model, "n_oof": len(oof)}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def stage3_metrics() -> pd.DataFrame:
    proc = PROJECT_ROOT / "data" / "processed"
    rows = []
    for asset in ASSETS:
        for model in MODELS:
            p = proc / f"{asset}_stage3_oof_{model}_v5_tuned_3stage_full.csv"
            if not p.exists():
                # fallback to default-tuned (which IS the 3stage_full pipeline)
                p = proc / f"{asset}_stage3_oof_{model}_v5_tuned.csv"
            if not p.exists():
                print(f"  skip (missing): {asset} {model}")
                continue
            oof = pd.read_csv(p, index_col=0, parse_dates=True)
            y_true = oof["true_label"].map(STAGE3_LABEL_TO_IDX).to_numpy()
            y_pred = oof["pred_label"].map(STAGE3_LABEL_TO_IDX).to_numpy()
            proba = oof[["P_Sell", "P_Hold", "P_Buy"]].to_numpy()
            metrics = compute_extended_metrics(y_true, y_pred, proba, STAGE3_CLASS_NAMES)
            row = {"asset": asset, "model": model, "n_oof": len(oof)}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def stage1_class_balance() -> pd.DataFrame:
    """Class balance of Stage 1 OOF labels (one canonical model per asset)."""
    proc = PROJECT_ROOT / "data" / "processed"
    rows = []
    for asset in ASSETS:
        # use xgboost OOF as representative (true_label is the same across models)
        p = proc / f"{asset}_stage1_oof_xgboost_v5_tuned.csv"
        oof = pd.read_csv(p, index_col=0, parse_dates=True)
        counts = oof["true_label"].value_counts().reindex(STAGE1_CLASS_NAMES, fill_value=0)
        n = int(counts.sum())
        for name in STAGE1_CLASS_NAMES:
            rows.append({
                "asset": asset,
                "class": name,
                "count": int(counts[name]),
                "proportion": counts[name] / n,
                "n_total": n,
            })
    return pd.DataFrame(rows)


def plot_stage3_roc_curves(out_path: Path) -> None:
    """4-panel ROC curve overlay: 2 assets x (3 classes one-vs-rest) with all 4 models per panel."""
    proc = PROJECT_ROOT / "data" / "processed"

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    for r, asset in enumerate(ASSETS):
        for c, cls in enumerate(STAGE3_CLASS_NAMES):
            ax = axes[r, c]
            cls_idx = STAGE3_LABEL_TO_IDX[cls]
            for model in MODELS:
                p = proc / f"{asset}_stage3_oof_{model}_v5_tuned_3stage_full.csv"
                if not p.exists():
                    p = proc / f"{asset}_stage3_oof_{model}_v5_tuned.csv"
                if not p.exists():
                    continue
                oof = pd.read_csv(p, index_col=0, parse_dates=True)
                y_true = oof["true_label"].map(STAGE3_LABEL_TO_IDX).to_numpy()
                proba = oof[["P_Sell", "P_Hold", "P_Buy"]].to_numpy()
                y_bin = (y_true == cls_idx).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, proba[:, cls_idx])
                auc = roc_auc_score(y_bin, proba[:, cls_idx])
                ax.plot(fpr, tpr, color=MODEL_COLORS[model], linewidth=1.4,
                        label=f"{MODEL_LABELS[model]} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8, alpha=0.7)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_title(f"{asset.upper()} — {cls} vs. rest", fontsize=10)
            if r == 1:
                ax.set_xlabel("False Positive Rate", fontsize=9)
            if c == 0:
                ax.set_ylabel("True Positive Rate", fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.legend(loc="lower right", fontsize=7)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Stage 3 multiclass ROC curves (one-vs-rest) — default 3-Stage Full architecture",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out = PROJECT_ROOT / "reports" / "Phase8_coursework_metrics"
    out.mkdir(parents=True, exist_ok=True)

    print("Computing Stage 1 extended metrics ...")
    s1 = stage1_metrics()
    p1 = out / "v5_p8_stage1_extended_metrics.csv"
    s1.to_csv(p1, index=False)
    print(f"  saved: {p1}")
    print(s1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nComputing Stage 3 extended metrics ...")
    s3 = stage3_metrics()
    p3 = out / "v5_p8_stage3_extended_metrics.csv"
    s3.to_csv(p3, index=False)
    print(f"  saved: {p3}")
    print(s3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nComputing Stage 1 class balance ...")
    cb = stage1_class_balance()
    pcb = out / "v5_p8_stage1_class_balance.csv"
    cb.to_csv(pcb, index=False)
    print(f"  saved: {pcb}")
    print(cb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nGenerating Stage 3 ROC curves figure ...")
    fig_path = out / "v5_p8_stage3_roc_curves.png"
    plot_stage3_roc_curves(fig_path)
    print(f"  saved: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
