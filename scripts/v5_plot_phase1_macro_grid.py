"""V5 Phase 1 — 12 derived macro feature timeline grid.

Output: reports/Phase1/v5_p1_macro_features_pretrain_grid.png

Stage 2 rule-based FSM'in kullandığı 5 feature panelinde yeşil border + asterisk,
kullanılmayan 7 feature'da gri title. Anlatım: "12 derived macro feature
pretrain üzerinde üretildi; bunlardan 5'i Stage 2 rule-based FSM'de kullanılır,
7'si clustering deneyleri için denenmiş ama rule-based pivot sonrası archived
edildi."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stage 2 Composite Macro FSM (Phase 2.12) features
FSM_FEATURES = {
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "Yield_Curve_10Y_2Y",
    "DXY_zscore_long",
    "M2_yoy_change",
}

# 12 derived feature panel order (FSM features first, then unused)
PANEL_ORDER = [
    # Used by FSM (5)
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "Yield_Curve_10Y_2Y",
    "DXY_zscore_long",
    "M2_yoy_change",
    # Unused by FSM (clustering experiments only, 7)
    "VIX",  # raw level (z-score is FSM input)
    "Gold_log_return_20d",
    "Oil_log_return_20d",
    "FEDFUNDS_change_60d",
    "UNRATE_change_180d",
    "CPI_yoy_change",
    "Gold_Silver_Ratio",
]

USED_COLOR = "#2a7a2a"
UNUSED_COLOR = "#888"


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                     index_col=0, parse_dates=True)

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    for ax, feat in zip(axes.flat, PANEL_ORDER):
        used = feat in FSM_FEATURES
        color = USED_COLOR if used else UNUSED_COLOR
        line_color = "#1a4d1a" if used else "#444"
        ax.plot(df.index, df[feat], color=line_color, lw=0.9)
        marker = " *" if used else ""
        ax.set_title(f"{feat}{marker}", fontsize=11, fontweight="bold" if used else "normal", color=color)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        if used:
            for spine in ax.spines.values():
                spine.set_color(USED_COLOR)
                spine.set_linewidth(1.8)

    fig.suptitle("Phase 1 — 12 Derived Macro Features (pretrain 2000-2025)\n"
                 "★ green-bordered = used by Stage 2 rule-based FSM (5)   |   "
                 "grey = clustering trial only (7, archived)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = PROJECT_ROOT / "reports" / "Phase1" / "v5_p1_macro_features_pretrain_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
