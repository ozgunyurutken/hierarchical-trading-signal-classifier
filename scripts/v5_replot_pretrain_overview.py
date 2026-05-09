"""Re-render pretrain overview as 3-decade stacked panels for higher
on-screen readability. Read tool caps display width ~2000px, so a single
wide horizontal panel scales down. Stacking 3 decade panels keeps each
sub-panel's date range readable while the total image stays compact.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240,
                     "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def shade_regimes(ax, regime):
    if regime.empty:
        return
    cur, start = regime.iloc[0], regime.index[0]
    for i in range(1, len(regime)):
        v = regime.iloc[i]
        if v != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime.index[i],
                           color=REGIME_COLORS.get(cur, "white"),
                           alpha=0.30, lw=0, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.30, lw=0, zorder=2)


def plot_panel(ax, ax2, sp500, vix_z, regime, pre_r, start, end, model_thr, title):
    sp_slice = sp500.loc[(sp500.index >= start) & (sp500.index <= end)]
    vix_slice = vix_z.loc[(vix_z.index >= start) & (vix_z.index <= end)]
    reg_slice = regime.loc[(regime.index >= start) & (regime.index <= end)]
    pre_slice = pre_r.loc[(pre_r.index >= start) & (pre_r.index <= end)]

    shade_regimes(ax, reg_slice)
    ax.plot(sp_slice.index, sp_slice.values, color="#1f77b4", lw=1.4, zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4", fontsize=14)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=12)

    ax2.plot(vix_slice.index, vix_slice.values, color="#d62728",
             lw=0.9, alpha=0.85, zorder=4)
    ax2.axhline(model_thr["bear_entry"], color="#d62728", ls="--",
                lw=0.7, alpha=0.5)
    ax2.axhline(model_thr["bull_entry"], color="green", ls="--",
                lw=0.7, alpha=0.5)
    ax2.set_ylabel("VIX z-score", color="#d62728", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=12)
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    sp_max = sp_slice.max()
    bv = pre_slice.index[pre_slice["bull_velocity_entry"]]
    if len(bv):
        ax.scatter(bv, [sp_max * 1.04] * len(bv), marker="^",
                   color="darkgreen", s=70, zorder=5)
    vov = pre_slice.index[pre_slice["velocity_override"]]
    if len(vov):
        ax.scatter(vov, [sp_max * 1.04] * len(vov), marker="v",
                   color="darkred", s=70, zorder=5)
    yc = pre_slice.index[pre_slice["yield_curve_override"]]
    if len(yc):
        ax.scatter(yc, [sp_max * 1.10] * len(yc), marker="s",
                   color="purple", s=60, zorder=5)
    ms = pre_slice.index[pre_slice["macro_stress_override"]]
    if len(ms):
        ax.scatter(ms, [sp_max * 1.10] * len(ms), marker="D",
                   color="brown", s=60, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"
    reports = PROJECT_ROOT / "reports"

    pre_r = pd.read_csv(proc / "macro_pretrain_regime_labels_composite_macro_v3_v5.csv",
                        index_col=0, parse_dates=True)
    derived = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    sp500 = pd.read_csv(raw / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    vix_z = derived["VIX_zscore_long"].dropna()
    regime = pre_r["regime_label"]

    model_thr = {"bear_entry": 1.0, "bull_entry": -0.5}

    fig, axes = plt.subplots(3, 1, figsize=(20, 22))
    decades = [
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"),
         "2000-2010 — Dot-com bust, GFC, recovery"),
        (pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"),
         "2010-2020 — ZIRP era, Trump trade war, COVID Bear + V-shape recovery"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-01-31"),
         "2020-2026 — Pandemic, 2022 Bear, 2024 ATH, 2025 tariff Bear"),
    ]
    twin_axes = [ax.twinx() for ax in axes]
    for (start, end, title), ax, ax2 in zip(decades, axes, twin_axes):
        plot_panel(ax, ax2, sp500, vix_z, regime, pre_r, start, end,
                   model_thr, title)

    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    handles += [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="darkgreen",
                   markersize=10, label="Bull velocity entry (V-shape)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="darkred",
                   markersize=10, label="Bear→Neutral velocity"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="purple",
                   markersize=10, label="Yield curve override"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="brown",
                   markersize=10, label="DXY+M2 macro stress"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V5 Phase 2.10 — Pre-train Overview by Decade\n"
                 "S&P 500 (blue) + VIX z-score (red) + Composite regime",
                 fontsize=16, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])

    out = reports / "Phase2" / "v5_p2.10_01b_pretrain_overview_by_decade.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
