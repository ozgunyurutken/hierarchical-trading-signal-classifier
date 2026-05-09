"""V5 Phase 1.5 — Engineered (derived) features.

Outputs:
  reports/Phase1.5/v5_p1.5_engineering_demo.png       — before/after engineering
  reports/Phase1.5/v5_p1.5_derived_features_grid.png  — 12 engineered features
  reports/Phase1.5/v5_p1.5_derived_correlation.png    — derived correlation matrix

Phase 1.5 = feature engineering uygulanmış derived features. Phase 1'deki
raw veriler manipüle edilerek scale-normalized, momentum, YoY change, spread
gibi türetilmiş feature'lar üretilir. 5'i Stage 2 rule-based FSM'de aktif
kullanılır (vurgulu); 7'si clustering deneyleri için üretildi, archived.
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

PANEL_ORDER = [
    # FSM-used (5)
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "Yield_Curve_10Y_2Y",
    "DXY_zscore_long",
    "M2_yoy_change",
    # Clustering-only / archived (7)
    "VIX",
    "Gold_log_return_20d",
    "Oil_log_return_20d",
    "FEDFUNDS_change_60d",
    "UNRATE_change_180d",
    "CPI_yoy_change",
    "Gold_Silver_Ratio",
]

USED_COLOR = "#2a7a2a"
UNUSED_COLOR = "#888"


def plot_engineering_demo(out_dir: Path):
    """Before/After: 4 örnek feature engineering operation."""
    raw = PROJECT_ROOT / "data" / "raw"
    proc = PROJECT_ROOT / "data" / "processed"
    risk = pd.read_csv(raw / "v5_macro_risk.csv", index_col=0, parse_dates=True)
    yields = pd.read_csv(raw / "v5_macro_yields.csv", index_col=0, parse_dates=True)
    fred = pd.read_csv(raw / "v5_macro_fred_monthly.csv", index_col=0, parse_dates=True)
    derived = pd.read_csv(proc / "macro_derived_pretrain_v5.csv", index_col=0, parse_dates=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=False)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    # Row 1: VIX raw -> VIX_zscore_long (z-score normalization)
    axes[0, 0].plot(risk.index, risk["VIX"], color="#cc4444", lw=0.7)
    axes[0, 0].set_title("VIX (raw level)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Index"); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(derived.index, derived["VIX_zscore_long"], color=USED_COLOR, lw=0.7)
    axes[0, 1].axhline(0, color="grey", ls="--", lw=0.6)
    axes[0, 1].set_title("VIX_zscore_long ★  (expanding mean / std normalization)",
                         fontsize=11, fontweight="bold", color=USED_COLOR)
    axes[0, 1].set_ylabel("z-score"); axes[0, 1].grid(True, alpha=0.3)

    # Row 2: US10Y + US2Y -> Yield_Curve_10Y_2Y (spread)
    axes[1, 0].plot(yields.index, yields["US10Y"], color="#3a6fb0", lw=0.7, label="US10Y")
    axes[1, 0].plot(yields.index, yields["US2Y"], color="#cc4444", lw=0.7, label="US2Y")
    axes[1, 0].set_title("US10Y, US2Y (raw yields)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("%"); axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(derived.index, derived["Yield_Curve_10Y_2Y"], color=USED_COLOR, lw=0.7)
    axes[1, 1].axhline(0, color="red", ls="--", lw=0.8, label="inversion threshold")
    axes[1, 1].set_title("Yield_Curve_10Y_2Y ★  (= US10Y − US2Y)",
                         fontsize=11, fontweight="bold", color=USED_COLOR)
    axes[1, 1].set_ylabel("spread (%)"); axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)

    # Row 3: M2 raw -> M2_yoy_change (YoY % change)
    axes[2, 0].plot(fred.index, fred["WM2NS"] / 1000, color="#557755", lw=0.9)
    axes[2, 0].set_title("M2 Money Supply (raw, $T)", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("$ trillions"); axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(derived.index, derived["M2_yoy_change"] * 100, color=USED_COLOR, lw=0.9)
    axes[2, 1].axhline(0, color="grey", ls="--", lw=0.6)
    axes[2, 1].set_title("M2_yoy_change ★  (year-over-year % change)",
                         fontsize=11, fontweight="bold", color=USED_COLOR)
    axes[2, 1].set_ylabel("%"); axes[2, 1].grid(True, alpha=0.3)

    # Row 4: SP500 raw -> SP500_log_return_5d (5d log return)
    axes[3, 0].semilogy(risk.index, risk["SP500"], color="black", lw=0.7)
    axes[3, 0].set_title("S&P 500 (raw level, log scale)", fontsize=11, fontweight="bold")
    axes[3, 0].set_ylabel("Index (log)"); axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].plot(derived.index, derived["SP500_log_return_5d"] * 100, color=USED_COLOR, lw=0.5)
    axes[3, 1].axhline(0, color="grey", ls="--", lw=0.6)
    axes[3, 1].set_title("SP500_log_return_5d ★  (= log(SP500[t]/SP500[t−5]))",
                         fontsize=11, fontweight="bold", color=USED_COLOR)
    axes[3, 1].set_ylabel("%"); axes[3, 1].grid(True, alpha=0.3)

    fig.suptitle("Phase 1.5 — Feature engineering before / after (★ = used by Stage 2 rule-based FSM)",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "v5_p1.5_engineering_demo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_derived_grid(out_dir: Path):
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / "macro_derived_pretrain_v5.csv", index_col=0, parse_dates=True)

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    for ax, feat in zip(axes.flat, PANEL_ORDER):
        used = feat in FSM_FEATURES
        line_color = "#1a4d1a" if used else "#444"
        title_color = USED_COLOR if used else UNUSED_COLOR
        ax.plot(df.index, df[feat], color=line_color, lw=0.9)
        marker = " ★" if used else ""
        ax.set_title(f"{feat}{marker}", fontsize=11,
                     fontweight="bold" if used else "normal", color=title_color)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        if used:
            for spine in ax.spines.values():
                spine.set_color(USED_COLOR); spine.set_linewidth(1.8)

    fig.suptitle("Phase 1.5 — 12 engineered macro features (pretrain 2000-2025)\n"
                 "★ green = retained by Stage 2 rule-based FSM (5)   |   "
                 "grey = clustering-only, archived (7)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "v5_p1.5_derived_features_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_derived_correlation(out_dir: Path):
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                     index_col=0, parse_dates=True).dropna()
    corr = df.corr()
    n = len(corr.columns)

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    xlabels = [f"{'★ ' if c in FSM_FEATURES else '  '}{c}" for c in corr.columns]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(xlabels, fontsize=9)
    for tick, c in zip(ax.get_xticklabels(), corr.columns):
        if c in FSM_FEATURES: tick.set_color("#1a4d1a"); tick.set_fontweight("bold")
        else: tick.set_color("#888")
    for tick, c in zip(ax.get_yticklabels(), corr.columns):
        if c in FSM_FEATURES: tick.set_color("#1a4d1a"); tick.set_fontweight("bold")
        else: tick.set_color("#888")

    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Phase 1.5 — Engineered feature correlation matrix\n"
                 "★ green-bold = used by Stage 2 rule-based FSM (5/12).   "
                 "Grey = clustering-only, archived (7/12).",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    out = out_dir / "v5_p1.5_derived_correlation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    out_dir = PROJECT_ROOT / "reports" / "Phase1.5"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_engineering_demo(out_dir)
    plot_derived_grid(out_dir)
    plot_derived_correlation(out_dir)


if __name__ == "__main__":
    main()
