"""V5 Phase 1.5 — Derived features for Stage 2 clustering experiments.

Output: reports/Phase1.5/v5_p1.5_clustering_feature_corr.png

Anlatım: "Phase 1.5'te Stage 2 clustering modelleri (K-Means, Constrained
K-Means, HMM, GMM) için 12 derived feature üretildi. Bunların korelasyon
matrisi clustering input space'i karakterize ediyordu. Sonra rule-based
pivot sonrası 5 feature aktif kullanımda kaldı; geri kalan 7 feature
çalışan versiyonda kullanılmıyor (pretrain dataset'te durur, fallback)."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FSM_FEATURES = {
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "Yield_Curve_10Y_2Y",
    "DXY_zscore_long",
    "M2_yoy_change",
}


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                     index_col=0, parse_dates=True).dropna()

    corr = df.corr()
    n = len(corr.columns)

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    xlabels, ylabels = [], []
    for col in corr.columns:
        used = col in FSM_FEATURES
        prefix = "★ " if used else "  "
        xlabels.append(f"{prefix}{col}")
        ylabels.append(f"{prefix}{col}")
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ylabels, fontsize=9)

    # Bold green for used features, grey for unused
    for tick, col in zip(ax.get_xticklabels(), corr.columns):
        if col in FSM_FEATURES:
            tick.set_color("#1a4d1a"); tick.set_fontweight("bold")
        else:
            tick.set_color("#888")
    for tick, col in zip(ax.get_yticklabels(), corr.columns):
        if col in FSM_FEATURES:
            tick.set_color("#1a4d1a"); tick.set_fontweight("bold")
        else:
            tick.set_color("#888")

    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")

    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Phase 1.5 — Derived feature correlation matrix\n"
                 "Initially produced for Stage 2 clustering experiments (K-Means / Constrained KM / HMM / GMM).\n"
                 "★ green-bold = retained by rule-based FSM (5/12).   Grey = clustering-only, archived (7/12).",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()

    out = PROJECT_ROOT / "reports" / "Phase1.5" / "v5_p1.5_clustering_feature_corr.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
