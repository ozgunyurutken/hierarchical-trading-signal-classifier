"""V5 Phase 6.5 — ROC-AUC computation for Stage 3 OOF predictions.

Required by coursework checklist (multiclass one-vs-rest).
Reads tuned OOF (3stage_full default) for both assets and computes:
  - Per-class ROC-AUC (Buy vs rest, Hold vs rest, Sell vs rest)
  - Macro-average ROC-AUC
  - Weighted-average ROC-AUC
For all 4 architectures × 4 models × 2 assets.

Output:
  reports/Phase5.3_roc_auc/v5_p5_stage3_roc_auc.csv
  reports/Phase5.3_roc_auc/v5_p5_stage3_roc_curves.png  (4-panel: best per asset)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
ARCHS = ["flat", "2stage_trend", "2stage_macro", "3stage_full"]
LABEL_TO_IDX = {"Sell": 0, "Hold": 1, "Buy": 2}
CLASS_NAMES = ["Sell", "Hold", "Buy"]
CLASS_COLORS = {"Sell": "#cc4444", "Hold": "#999999", "Buy": "#3a8a3a"}


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase5.3_roc_auc"
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    for asset in ASSETS:
        for arch in ARCHS:
            for model in MODELS:
                p = proc / f"{asset}_stage3_oof_{model}_v5_tuned_{arch}.csv"
                if not p.exists():
                    continue
                oof = pd.read_csv(p, index_col=0, parse_dates=True)
                y_true = oof["true_label"].map(LABEL_TO_IDX).to_numpy()
                proba = oof[["P_Sell", "P_Hold", "P_Buy"]].to_numpy()

                # Per-class one-vs-rest AUC
                auc_per = []
                for i, cls in enumerate(CLASS_NAMES):
                    y_bin = (y_true == i).astype(int)
                    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                        auc_per.append(np.nan)
                        continue
                    auc = roc_auc_score(y_bin, proba[:, i])
                    auc_per.append(auc)

                # Macro and weighted
                auc_macro = float(np.nanmean(auc_per))
                # Weighted by class support
                supports = np.array([(y_true == i).sum() for i in range(3)], dtype=float)
                supports = supports / supports.sum()
                auc_weighted = float(np.nansum(np.array(auc_per) * supports))

                rows.append({
                    "asset":       asset,
                    "arch":        arch,
                    "model":       model,
                    "auc_sell":    auc_per[0],
                    "auc_hold":    auc_per[1],
                    "auc_buy":     auc_per[2],
                    "auc_macro":   auc_macro,
                    "auc_weighted": auc_weighted,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "v5_p5_stage3_roc_auc.csv", index=False)
    print(f"Saved: {(out / 'v5_p5_stage3_roc_auc.csv').relative_to(PROJECT_ROOT)}")

    # Print summary
    print("\n=== ROC-AUC summary (macro avg, sorted) ===")
    print(df.sort_values(["asset", "auc_macro"], ascending=[True, False])
            [["asset", "arch", "model", "auc_macro", "auc_buy", "auc_hold", "auc_sell"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ROC curves plot — best per (asset, arch) for default 3stage_full
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for r, asset in enumerate(ASSETS):
        # Best by macro AUC for 3stage_full
        sub = df[(df["asset"] == asset) & (df["arch"] == "3stage_full")]
        best = sub.loc[sub["auc_macro"].idxmax()]
        oof = pd.read_csv(proc / f"{asset}_stage3_oof_{best['model']}_v5_tuned_3stage_full.csv",
                          index_col=0, parse_dates=True)
        y_true = oof["true_label"].map(LABEL_TO_IDX).to_numpy()
        proba = oof[["P_Sell", "P_Hold", "P_Buy"]].to_numpy()

        ax = axes[r, 0]
        for i, cls in enumerate(CLASS_NAMES):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0: continue
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            auc = roc_auc_score(y_bin, proba[:, i])
            ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=1.6,
                    label=f"{cls} (AUC={auc:.3f})")
        ax.plot([0,1], [0,1], "k--", lw=0.7, alpha=0.5, label="chance")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{asset.upper()} 3stage_full / {best['model']}\n"
                     f"macro AUC = {best['auc_macro']:.3f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Bar chart of macro AUC per (arch, model)
        ax2 = axes[r, 1]
        sub2 = df[df["asset"] == asset].copy()
        x_lab = [f"{a}/{m}" for a, m in zip(sub2["arch"], sub2["model"])]
        ax2.barh(x_lab, sub2["auc_macro"],
                  color=[CLASS_COLORS["Buy"] if v > 0.55 else "#888888" for v in sub2["auc_macro"]],
                  edgecolor="black", linewidth=0.4)
        ax2.axvline(0.5, color="black", ls="--", lw=0.7)
        ax2.set_xlabel("Macro ROC-AUC")
        ax2.set_title(f"{asset.upper()} — Macro AUC across 16 (arch, model)",
                       fontsize=11, fontweight="bold")
        ax2.set_xlim(0.45, max(0.65, sub2["auc_macro"].max() + 0.02))
        ax2.tick_params(axis="y", labelsize=7)
        for i, v in enumerate(sub2["auc_macro"]):
            ax2.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)

    fig.suptitle("Phase 5.3 — Stage 3 ROC-AUC analysis (multiclass one-vs-rest)\n"
                 "Per-class AUC + 16-config macro AUC bars",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_path = out / "v5_p5_stage3_roc_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
