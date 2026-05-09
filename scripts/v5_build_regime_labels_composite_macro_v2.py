"""
V5 Phase 2.9 — Composite VIX + Yield Curve + DXY/M2 Macro Stress (5-rule).

Phase 2.8'in iki düzeltmesi + 1 ek mekanizma:

  Düzeltme A (kullanıcı 2024 bölgesi feedback):
    - Yield curve override Neutral → Bull entry'yi bloke ETMEZ
      (sadece Bull state'inde defensive force)
    - YC persistence window 30 → 60 gün (kısa flicker absorb edilir)

  Yeni Rule 5 (DXY + M2 macro stress composite):
    - DXY_zscore_long rolling 30d mean > +1.0 (persistent dollar strength)
      AND
    - M2_yoy_change rolling 30d mean < 0.023 (low/contracting liquidity, p10)
    → Bull → Neutral defensive force

  Strict AND combine: hem dolar gücülü hem de likidite kısıtlı olmalı.
  Bu 2014-2015 commodity crash + DXY rally + Fed taper, 2018 hike döngusu,
  2022 hike döngusu gibi makro stres dönemlerinde tetiklenir.

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v2_v5.csv
  reports/Phase2/v5_p2.9_composite_macro_v2_state_diagram.png
  reports/Phase2/v5_p2.9_composite_macro_v2_timeline.png
  reports/Phase2/v5_p2.9_composite_macro_v2_distribution.png
  reports/Phase2/v5_p2.9_composite_macro_v2_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, FancyArrowPatch

from src.labels.v5_regime_labels import (
    CompositeVIXRegimeClassifier, BULL_BEAR_LABELS,
)

plt.rcParams.update({"figure.dpi": 220, "savefig.dpi": 220,
                     "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def _shade_regimes(ax, price, regime, log=False, lw=1.0):
    if log:
        ax.semilogy(price.index, price.values, color="black", lw=lw, zorder=3)
    else:
        ax.plot(price.index, price.values, color="black", lw=lw, zorder=3)
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


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, model, out: Path):
    fig, axes = plt.subplots(6, 1, figsize=(28, 26), sharex=True,
                             gridspec_kw={"height_ratios":
                                           [1.4, 1, 1, 0.55, 0.55, 0.4]})

    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()
    yc = derived["Yield_Curve_10Y_2Y"]
    dxy_z = derived["DXY_zscore_long"]
    m2 = derived["M2_yoy_change"]

    # Panel 0 — pretrain context
    ax = axes[0]
    _shade_regimes(ax, pd.Series(np.nan, index=pre_r.index),
                   pre_r["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=1.0,
            label="S&P 500 (left)", zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4", fontsize=12)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=11)
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.8, alpha=0.85,
             label="VIX z-score (right)", zorder=4)
    ax2.axhline(model.bear_entry_threshold, color="#d62728", ls="--", lw=0.7,
                alpha=0.6, label=f"Bear entry (>{model.bear_entry_threshold})")
    ax2.axhline(model.bull_entry_threshold, color="green", ls="--", lw=0.7,
                alpha=0.6, label=f"Bull entry (<{model.bull_entry_threshold})")
    ax2.set_ylabel("VIX z-score", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=11)
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    if "velocity_override" in pre_r.columns:
        d = pre_r.index[pre_r["velocity_override"]]
        if len(d):
            ax.scatter(d, [sp500.reindex(d).max() * 1.07] * len(d),
                       marker="v", color="darkred", s=70, zorder=5,
                       label="Velocity override")
    if "yield_curve_override" in pre_r.columns:
        d = pre_r.index[pre_r["yield_curve_override"]]
        if len(d):
            ax.scatter(d, [sp500.reindex(d).max() * 1.13] * len(d),
                       marker="s", color="purple", s=70, zorder=5,
                       label="YC override")
    if "macro_stress_override" in pre_r.columns:
        d = pre_r.index[pre_r["macro_stress_override"]]
        if len(d):
            ax.scatter(d, [sp500.reindex(d).max() * 1.20] * len(d),
                       marker="D", color="brown", s=70, zorder=5,
                       label="Macro stress override")

    ax.set_title("Pre-train (2000-2025) — S&P 500 + VIX z-score + Composite regime\n"
                 "▼ velocity | ■ yield curve | ◆ DXY+M2 macro stress",
                 fontsize=13, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.2)
    axes[1].set_ylabel("BTC (log)", fontsize=12)
    axes[1].set_title("BTC + Composite regime (Phase 2.9)",
                      fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=11)

    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.2)
    axes[2].set_ylabel("ETH (log)", fontsize=12)
    axes[2].set_title("ETH + Composite regime (Phase 2.9)",
                      fontsize=13, fontweight="bold")
    axes[2].tick_params(labelsize=11)

    # Panel 3 — Yield Curve
    ax3 = axes[3]
    ax3.plot(yc.index, yc.values, color="purple", lw=0.8, label="10Y-2Y spread")
    ax3.axhline(0, color="black", ls="--", lw=0.7, alpha=0.5)
    ax3.fill_between(yc.index, yc.values, 0,
                     where=(yc.values < 0), color="purple", alpha=0.25,
                     label="Inverted (recession leading)")
    ax3.set_ylabel("10Y-2Y (%)", fontsize=11)
    ax3.set_title(f"Yield Curve 10Y-2Y (window: {model.yield_curve_persistence_window}d)",
                  fontsize=11, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=9)
    ax3.tick_params(labelsize=10)

    # Panel 4 — DXY + M2 (twin axis)
    ax4 = axes[4]
    ax4.plot(dxy_z.index, dxy_z.values, color="orange", lw=0.7,
             label="DXY z-score (left)", alpha=0.85)
    ax4.axhline(model.dxy_strong_threshold, color="orange", ls="--", lw=0.6,
                alpha=0.6, label=f"DXY strong (>{model.dxy_strong_threshold})")
    ax4.set_ylabel("DXY z-score", color="orange", fontsize=11)
    ax4.tick_params(axis="y", labelcolor="orange", labelsize=10)
    ax4b = ax4.twinx()
    ax4b.plot(m2.index, m2.values, color="teal", lw=0.7,
              label="M2 yoy change (right)", alpha=0.85)
    ax4b.axhline(model.m2_low_threshold, color="teal", ls="--", lw=0.6,
                 alpha=0.6, label=f"M2 low (<{model.m2_low_threshold})")
    ax4b.set_ylabel("M2 yoy change", color="teal", fontsize=11)
    ax4b.tick_params(axis="y", labelcolor="teal", labelsize=10)
    ax4b.grid(False); ax4b.spines["top"].set_visible(False)
    ax4.set_title("DXY z-score + M2 yoy (macro stress: DXY high AND M2 low)",
                  fontsize=11, fontweight="bold")
    l1, lab1 = ax4.get_legend_handles_labels()
    l2, lab2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)

    # Panel 5 — compact regime band
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[5], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[5].set_yticks([]); axes[5].grid(False)
    axes[5].set_title("Pre-train regime band (compact)", fontsize=11, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(labelsize=11)
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V5 Phase 2.9 — Composite Macro FSM (5 rules)\n"
                 "Hysteresis + Dwell + VIX velocity + Yield curve + DXY/M2 macro stress",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distribution(btc_r, eth_r, pre_r, out: Path):
    rows = []
    for label, df in [("Pre-train", pre_r), ("BTC", btc_r), ("ETH", eth_r)]:
        c = df["regime_label"].value_counts(normalize=True) * 100
        for r in BULL_BEAR_LABELS:
            rows.append({"period": label, "regime": r, "pct": float(c.get(r, 0))})
    pivot = pd.DataFrame(rows).pivot(index="period", columns="regime",
                                      values="pct")[BULL_BEAR_LABELS]
    fig, ax = plt.subplots(figsize=(15, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in BULL_BEAR_LABELS],
               edgecolor="black", lw=0.8)
    ax.set_xlabel("% of days", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.9 — Composite Macro Regime Distribution (5-rule)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in BULL_BEAR_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold")
            cum += v
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_2024_zoom(btc_close, eth_close, btc_r, eth_r, pre_r, out: Path):
    """2024 bölgesi zoom plot — kullanıcının feedback'i için."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")

    fig, axes = plt.subplots(3, 1, figsize=(24, 16), sharex=True)

    btc_slice = btc_close.loc[(btc_close.index >= start) & (btc_close.index <= end)]
    eth_slice = eth_close.loc[(eth_close.index >= start) & (eth_close.index <= end)]
    pre_slice = pre_r.loc[(pre_r.index >= start) & (pre_r.index <= end)]
    btc_r_slice = btc_r.loc[(btc_r.index >= start) & (btc_r.index <= end)]
    eth_r_slice = eth_r.loc[(eth_r.index >= start) & (eth_r.index <= end)]
    sp_slice = sp500.loc[(sp500.index >= start) & (sp500.index <= end)]

    _shade_regimes(axes[0], sp_slice, pre_slice["regime_label"], log=False, lw=1.2)
    axes[0].set_ylabel("S&P 500", fontsize=12)
    axes[0].set_title("2023-2025 zoom — S&P 500 + Composite regime",
                      fontsize=13, fontweight="bold")

    _shade_regimes(axes[1], btc_slice, btc_r_slice["regime_label"], log=True, lw=1.4)
    axes[1].set_ylabel("BTC (log)", fontsize=12)
    axes[1].set_title("BTC 2023-2025 — Composite regime",
                      fontsize=13, fontweight="bold")

    _shade_regimes(axes[2], eth_slice, eth_r_slice["regime_label"], log=True, lw=1.4)
    axes[2].set_ylabel("ETH (log)", fontsize=12)
    axes[2].set_title("ETH 2023-2025 — Composite regime",
                      fontsize=13, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V5 Phase 2.9 — 2023-2025 Zoom (kullanıcı feedback: 2024 boğa kontrolü)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    print("V5 Phase 2.9 — Composite Macro FSM (5-rule)")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)
    pretrain = pretrain.dropna(subset=["VIX_zscore_long",
                                        "SP500_log_return_5d",
                                        "Yield_Curve_10Y_2Y",
                                        "DXY_zscore_long",
                                        "M2_yoy_change"])
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print("\n[1] Fit Composite Macro FSM v2 (5 rules)")
    model = CompositeVIXRegimeClassifier(
        bear_entry_threshold=1.0,
        bear_exit_threshold=0.3,
        bull_entry_threshold=-0.5,
        bull_exit_threshold=0.0,
        bear_min_dwell=20,
        bull_min_dwell=30,
        velocity_window=5,
        velocity_threshold=-0.8,
        velocity_sp500_min=0.0,
        initial_regime="Neutral",
        # Phase 2.8/2.9 yield curve override (yumuşatılmış)
        enable_yield_curve_override=True,
        yield_curve_inverted_threshold=0.0,
        yield_curve_persistence_window=60,        # 30 → 60 (kullanıcı 2024 fix)
        yield_curve_blocks_bull_entry=False,      # Phase 2.9 fix: bloke etme
        # Phase 2.9 macro stress (DXY + M2)
        enable_macro_stress_override=True,
        dxy_strong_threshold=1.0,
        m2_low_threshold=0.023,                   # M2_yoy p10
        macro_stress_window=30,
        macro_stress_combine="AND",               # strict AND
    ).fit(pretrain)
    print(f"  Bear entry: VIX_z > {model.bear_entry_threshold}")
    print(f"  Bear exit:  VIX_z < {model.bear_exit_threshold}")
    print(f"  Bull entry: VIX_z < {model.bull_entry_threshold} AND SP500_5d > 0")
    print(f"            (YC bloke etme: {not model.yield_curve_blocks_bull_entry})")
    print(f"  Bull exit:  VIX_z > {model.bull_exit_threshold} OR YC inverted OR macro_stress")
    print(f"  Bear min dwell: {model.bear_min_dwell}d, Bull min dwell: {model.bull_min_dwell}d")
    print(f"  Velocity override: ΔVIX_z[{model.velocity_window}d] < {model.velocity_threshold}")
    print(f"  YC override: rolling {model.yield_curve_persistence_window}d mean < "
          f"{model.yield_curve_inverted_threshold}")
    print(f"  Macro stress: DXY_z[30d] > {model.dxy_strong_threshold} "
          f"{model.macro_stress_combine} M2_yoy[30d] < {model.m2_low_threshold}")

    print(f"\n[2] Inference (FSM forward pass)")
    pre_r = model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_macro_v2_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_macro_v2_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_macro_v2_v5.csv")
    n_velocity = int(pre_r["velocity_override"].sum())
    n_yc = int(pre_r["yield_curve_override"].sum())
    n_ms = int(pre_r["macro_stress_override"].sum())
    print(f"  Velocity override: {n_velocity} times")
    print(f"  Yield curve override: {n_yc} times")
    if n_yc > 0:
        for d in pre_r.index[pre_r["yield_curve_override"]]:
            print(f"    YC: {d.date()}")
    print(f"  Macro stress override: {n_ms} times")
    if n_ms > 0:
        for d in pre_r.index[pre_r["macro_stress_override"]]:
            print(f"    MS: {d.date()}")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r, model,
                  reports / "Phase2" / "v5_p2.9_composite_macro_v2_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.9_composite_macro_v2_distribution.png")
    plot_2024_zoom(btc["Close"].loc[btc_r.index],
                   eth["Close"].loc[eth_r.index],
                   btc_r, eth_r, pre_r,
                   reports / "Phase2" / "v5_p2.9_composite_macro_v2_2024_zoom.png")

    print("\n" + "=" * 70)
    print("Distribution:")
    distribution = {}
    for label, df, key in [("Pre-train", pre_r, "pretrain"),
                            ("BTC era", btc_r, "btc"),
                            ("ETH era", eth_r, "eth")]:
        c = df["regime_label"].value_counts()
        total = c.sum()
        print(f"  {label:12s} ", end="")
        distribution[key] = {}
        for r in BULL_BEAR_LABELS:
            v = int(c.get(r, 0))
            pct = float(v / total * 100) if total else 0.0
            distribution[key][r] = pct
            print(f"{r}: {v} ({pct:.1f}%)  ", end="")
        print()

    runs = []
    cur, start = pre_r["regime_label"].iloc[0], pre_r.index[0]
    for i in range(1, len(pre_r)):
        v = pre_r["regime_label"].iloc[i]
        if v != cur:
            runs.append({"regime": cur, "start": str(start.date()),
                         "end": str(pre_r.index[i - 1].date()),
                         "duration_days": (pre_r.index[i - 1] - start).days})
            cur, start = v, pre_r.index[i]
    runs.append({"regime": cur, "start": str(start.date()),
                 "end": str(pre_r.index[-1].date()),
                 "duration_days": (pre_r.index[-1] - start).days})
    bear_runs = [r for r in runs if r["regime"] == "Bear"]
    bull_runs = [r for r in runs if r["regime"] == "Bull"]
    neutral_runs = [r for r in runs if r["regime"] == "Neutral"]
    print(f"\nRun-length stats (pretrain):")
    if bear_runs:
        print(f"  # Bear runs: {len(bear_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in bear_runs]):.0f}d, "
              f"max: {max([r['duration_days'] for r in bear_runs]):.0f}d")
    if bull_runs:
        print(f"  # Bull runs: {len(bull_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in bull_runs]):.0f}d, "
              f"max: {max([r['duration_days'] for r in bull_runs]):.0f}d")
    if neutral_runs:
        print(f"  # Neutral runs: {len(neutral_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in neutral_runs]):.0f}d")

    diagnostics = {
        "phase": "V5 Phase 2.9 — Composite Macro FSM (5-rule)",
        "method": "Phase 2.7 + relaxed YC override + DXY/M2 macro stress",
        "thresholds": {
            "bear_entry": model.bear_entry_threshold,
            "bear_exit": model.bear_exit_threshold,
            "bull_entry": model.bull_entry_threshold,
            "bull_exit": model.bull_exit_threshold,
        },
        "dwell_time": {"bear": model.bear_min_dwell, "bull": model.bull_min_dwell},
        "velocity_override": {
            "n_fired": n_velocity,
        },
        "yield_curve_override": {
            "persistence_window": model.yield_curve_persistence_window,
            "blocks_bull_entry": model.yield_curve_blocks_bull_entry,
            "n_fired": n_yc,
            "fired_dates": [str(d.date()) for d in pre_r.index[pre_r["yield_curve_override"]]],
        },
        "macro_stress_override": {
            "dxy_threshold": model.dxy_strong_threshold,
            "m2_threshold": model.m2_low_threshold,
            "window": model.macro_stress_window,
            "combine": model.macro_stress_combine,
            "n_fired": n_ms,
            "fired_dates": [str(d.date()) for d in pre_r.index[pre_r["macro_stress_override"]]],
        },
        "distribution_pct": distribution,
        "bear_runs_summary": {
            "count": len(bear_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bear_runs])) if bear_runs else 0.0,
            "max_days": int(max([r['duration_days'] for r in bear_runs])) if bear_runs else 0,
        },
        "bull_runs_summary": {
            "count": len(bull_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bull_runs])) if bull_runs else 0.0,
            "max_days": int(max([r['duration_days'] for r in bull_runs])) if bull_runs else 0,
        },
    }
    diag_path = reports / "Phase2" / "v5_p2.9_composite_macro_v2_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.9 complete.")


if __name__ == "__main__":
    main()
