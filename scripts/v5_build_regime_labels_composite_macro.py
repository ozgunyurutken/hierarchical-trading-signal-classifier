"""
V5 Phase 2.8 — Composite VIX FSM + Yield Curve Inversion override.

Phase 2.7 (composite VIX FSM) onayli core pattern üzerine Phase 2.8 ek
mekanizma:
  (5) Persistent yield curve inversion (Yield_Curve_10Y_2Y rolling 30d mean < 0):
      - Bull state → Neutral force (defensive bias, recession leading indicator)
      - Neutral → Bull entry blocked (yield curve clear olmadan Bull'a geçme)

Yield curve inversion klasik recession leading indicator (12-24 ay öncesinden):
  - 2000-2001 inversion → 2001 recession
  - 2006-2007 inversion → 2008 GFC
  - 2019 inversion → 2020 recession (covid amplified)
  - 2022-2024 inversion → soft-landing debate

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v5.csv
  reports/Phase2/v5_p2.8_composite_macro_state_diagram.png
  reports/Phase2/v5_p2.8_composite_macro_timeline.png
  reports/Phase2/v5_p2.8_composite_macro_distribution.png
  reports/Phase2/v5_p2.8_composite_macro_diagnostics.json
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

# Larger figures + higher DPI for big readable plots
plt.rcParams.update({"figure.dpi": 160, "font.size": 11,
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


def plot_state_diagram(model, out: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    positions = {"Bull": (0.15, 0.6), "Neutral": (0.5, 0.6), "Bear": (0.85, 0.6)}
    for regime, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.07, color=REGIME_COLORS[regime],
                             ec="black", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, regime, ha="center", va="center",
                fontsize=15, fontweight="bold", zorder=4)
        dwell_d = (model.bull_min_dwell if regime == "Bull"
                   else model.bear_min_dwell if regime == "Bear" else 0)
        ax.text(x, y - 0.16, f"min dwell: {dwell_d}d",
                ha="center", va="center", fontsize=10, color="#444")

    def arrow(p1, p2, label, offset=0.06, color="black", style="->"):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        sx, sy = x1 + ux * 0.075, y1 + uy * 0.075
        ex, ey = x2 - ux * 0.075, y2 - uy * 0.075
        sy += offset; ey += offset
        arr = FancyArrowPatch((sx, sy), (ex, ey),
                               arrowstyle=style, mutation_scale=18,
                               color=color, lw=1.6, zorder=2)
        ax.add_patch(arr)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my + 0.035, label, ha="center", va="bottom",
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.8))

    arrow(positions["Bull"], positions["Neutral"],
          f"VIX_z > {model.bull_exit_threshold}", offset=0.04)
    arrow(positions["Neutral"], positions["Bull"],
          f"VIX_z < {model.bull_entry_threshold} & SP500_5d > 0\n"
          f"& yield curve NOT inverted",
          offset=-0.05)
    arrow(positions["Neutral"], positions["Bear"],
          f"VIX_z > {model.bear_entry_threshold}", offset=0.04)
    arrow(positions["Bear"], positions["Neutral"],
          f"VIX_z < {model.bear_exit_threshold}", offset=-0.04)
    arrow(positions["Bear"], positions["Neutral"],
          f"velocity override:\nΔVIX_z[5d] < {model.velocity_threshold}\n& SP500_5d > 0",
          offset=-0.16, color="darkred", style="->")
    arrow(positions["Bull"], positions["Neutral"],
          f"yield curve override:\nrolling 30d mean < 0\n(persistent inversion)",
          offset=0.16, color="purple", style="->")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V5 Phase 2.8 — Composite VIX + Yield Curve FSM\n"
                 "Hysteresis + Dwell + Velocity override + YC inversion override",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, model, out: Path):
    """Big readable timeline: 5 panels (added yield curve panel)."""
    fig, axes = plt.subplots(5, 1, figsize=(22, 18), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 0.6, 0.5]})

    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()
    yc = derived["Yield_Curve_10Y_2Y"]

    # Panel 0 — Pretrain context (S&P 500 + VIX z-score + regime shading + overrides)
    ax = axes[0]
    _shade_regimes(ax, pd.Series(np.nan, index=pre_r.index),
                   pre_r["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=1.0,
            label="S&P 500 (left, linear)", zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4", fontsize=12)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=11)
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.8, alpha=0.85,
             label="VIX z-score long (right)", zorder=4)
    ax2.axhline(model.bear_entry_threshold, color="#d62728", ls="--", lw=0.7,
                alpha=0.6, label=f"Bear entry (>{model.bear_entry_threshold})")
    ax2.axhline(model.bear_exit_threshold, color="#d62728", ls=":", lw=0.7,
                alpha=0.6, label=f"Bear exit (<{model.bear_exit_threshold})")
    ax2.axhline(model.bull_entry_threshold, color="green", ls="--", lw=0.7,
                alpha=0.6, label=f"Bull entry (<{model.bull_entry_threshold})")
    ax2.set_ylabel("VIX z-score", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=11)
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    if "velocity_override" in pre_r.columns:
        vel_dates = pre_r.index[pre_r["velocity_override"]]
        if len(vel_dates) > 0:
            ax.scatter(vel_dates, [sp500.reindex(vel_dates).max() * 1.07] * len(vel_dates),
                       marker="v", color="darkred", s=70, zorder=5,
                       label="Velocity override")
    if "yield_curve_override" in pre_r.columns:
        yc_dates = pre_r.index[pre_r["yield_curve_override"]]
        if len(yc_dates) > 0:
            ax.scatter(yc_dates, [sp500.reindex(yc_dates).max() * 1.13] * len(yc_dates),
                       marker="s", color="purple", s=70, zorder=5,
                       label="Yield curve override")

    ax.set_title("Pre-train context (2000-2025) — S&P 500 + VIX z-score + Composite regime\n"
                 "Velocity overrides ▼ red | Yield curve overrides ■ purple",
                 fontsize=13, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    # Panel 1 — BTC
    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.2)
    axes[1].set_ylabel("BTC (log)", fontsize=12)
    axes[1].set_title("BTC + Composite regime (Phase 2.8)",
                      fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=11)

    # Panel 2 — ETH
    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.2)
    axes[2].set_ylabel("ETH (log)", fontsize=12)
    axes[2].set_title("ETH + Composite regime (Phase 2.8)",
                      fontsize=13, fontweight="bold")
    axes[2].tick_params(labelsize=11)

    # Panel 3 — Yield Curve 10Y-2Y (with inversion shaded)
    ax3 = axes[3]
    ax3.plot(yc.index, yc.values, color="purple", lw=0.8, label="10Y-2Y spread")
    ax3.axhline(0, color="black", ls="--", lw=0.7, alpha=0.5)
    ax3.fill_between(yc.index, yc.values, 0,
                     where=(yc.values < 0), color="purple", alpha=0.25,
                     label="Inverted (recession leading indicator)")
    ax3.set_ylabel("10Y-2Y (%)", fontsize=11)
    ax3.set_title("Yield Curve 10Y-2Y (purple shading = inversion)",
                  fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=10)
    ax3.tick_params(labelsize=11)

    # Panel 4 — Pretrain band only (compact)
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[4], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[4].set_yticks([]); axes[4].grid(False)
    axes[4].set_title("Pre-train regime band (compact)",
                      fontsize=12, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(labelsize=11)
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V5 Phase 2.8 — Composite VIX + Yield Curve FSM\n"
                 "(Hysteresis + Dwell + Velocity + Yield Curve Inversion)",
                 fontsize=15, fontweight="bold", y=0.998)
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
    ax.set_title("V5 Phase 2.8 — Composite Macro Regime Distribution",
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


def main():
    print("V5 Phase 2.8 — Composite VIX + Yield Curve FSM")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)
    pretrain = pretrain.dropna(subset=["VIX_zscore_long",
                                        "SP500_log_return_5d",
                                        "Yield_Curve_10Y_2Y"])
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print("\n[1] Fit Composite Macro FSM (Phase 2.7 + Yield Curve override)")
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
        enable_yield_curve_override=True,
        yield_curve_inverted_threshold=0.0,
        yield_curve_persistence_window=30,
    ).fit(pretrain)
    print(f"  Bear entry: VIX_z > {model.bear_entry_threshold}")
    print(f"  Bear exit:  VIX_z < {model.bear_exit_threshold}")
    print(f"  Bull entry: VIX_z < {model.bull_entry_threshold} AND SP500_5d > 0 AND YC NOT inverted")
    print(f"  Bull exit:  VIX_z > {model.bull_exit_threshold} OR YC persistent inversion")
    print(f"  Bear min dwell: {model.bear_min_dwell}d, Bull min dwell: {model.bull_min_dwell}d")
    print(f"  Velocity override: ΔVIX_z[{model.velocity_window}d] < {model.velocity_threshold}")
    print(f"  Yield curve override: rolling {model.yield_curve_persistence_window}d mean < "
          f"{model.yield_curve_inverted_threshold}")

    plot_state_diagram(model,
                       reports / "Phase2" / "v5_p2.8_composite_macro_state_diagram.png")

    print(f"\n[2] Inference (FSM forward pass)")
    pre_r = model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_macro_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_macro_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_macro_v5.csv")
    n_velocity = int(pre_r["velocity_override"].sum())
    n_yc = int(pre_r["yield_curve_override"].sum())
    velocity_dates = pre_r.index[pre_r["velocity_override"]]
    yc_dates = pre_r.index[pre_r["yield_curve_override"]]
    print(f"  Velocity override fired: {n_velocity} times")
    print(f"  Yield curve override fired: {n_yc} times")
    if n_yc > 0:
        print(f"    YC override dates:")
        for d in yc_dates:
            print(f"      {d.date()}")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r, model,
                  reports / "Phase2" / "v5_p2.8_composite_macro_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.8_composite_macro_distribution.png")

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
        "phase": "V5 Phase 2.8 — Composite VIX + Yield Curve FSM",
        "method": "Phase 2.7 + persistent yield curve inversion override",
        "thresholds": {
            "bear_entry": model.bear_entry_threshold,
            "bear_exit": model.bear_exit_threshold,
            "bull_entry": model.bull_entry_threshold,
            "bull_exit": model.bull_exit_threshold,
        },
        "dwell_time": {
            "bear_min_days": model.bear_min_dwell,
            "bull_min_days": model.bull_min_dwell,
        },
        "velocity_override": {
            "window_days": model.velocity_window,
            "threshold": model.velocity_threshold,
            "sp500_min": model.velocity_sp500_min,
            "n_fired_pretrain": n_velocity,
            "fired_dates": [str(d.date()) for d in velocity_dates],
        },
        "yield_curve_override": {
            "feature": model.feature_yield_curve,
            "persistence_window_days": model.yield_curve_persistence_window,
            "inverted_threshold": model.yield_curve_inverted_threshold,
            "n_fired_pretrain": n_yc,
            "fired_dates": [str(d.date()) for d in yc_dates],
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
    diag_path = reports / "Phase2" / "v5_p2.8_composite_macro_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.8 complete.")


if __name__ == "__main__":
    main()
