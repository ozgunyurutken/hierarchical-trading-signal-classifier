"""Generate standardized 4-panel timeline plot per Phase checkpoint.

Each phase produces a single PNG with 4 stacked subplots:
  1. S&P 500 + regime shading (pretrain)
  2. BTC + regime shading (log scale)
  3. ETH + regime shading (log scale)
  4. (empty placeholder)

Outputs (one per phase):
  reports/Phase2/v5_phase2.X_<name>_timeline_4panel.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240,
                     "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {
    "Bull": "#7ec27e", "Risk-On": "#7ec27e",
    "Neutral": "#f0c870",
    "Bear": "#e07e7e", "Risk-Off": "#e07e7e",
}

# Each entry: (phase_id, csv_suffix, title)
PHASES = [
    ("v5_phase2.1_constrained_kmeans",
     "constrained_v5",
     "V5 Phase 2.1 — Constrained K-Means (4 macro features)"),
    ("v5_phase2.2_unsupervised_kmeans",
     "unsupervised_v5",
     "V5 Phase 2.2 — Unsupervised K-Means (10 macro features)"),
    ("v5_phase2.3_sparse_kmeans",
     "sparse_v5",
     "V5 Phase 2.3 — Sparse K-Means (feature selection)"),
    ("v5_phase2.4_minimal_kmeans",
     "minimal_kmeans_v5",
     "V5 Phase 2.4 — Minimal K-Means (top-2 features)"),
    ("v5_phase2.5_hmm",
     "hmm_v5",
     "V5 Phase 2.5 — Hidden Markov Model (3 states, Gaussian emissions)"),
    ("v5_phase2.5b_gmm",
     "gmm_v5",
     "V5 Phase 2.5b — GMM (3 components, soft assignment)"),
    ("v5_phase2.6_rule_vix",
     "rule_vix_v5",
     "V5 Phase 2.6 — Single-rule VIX threshold (baseline)"),
    ("v5_phase2.7_composite_vix",
     "composite_rule_v5",
     "V5 Phase 2.7 — Composite VIX FSM (hysteresis + dwell + velocity)"),
    ("v5_phase2.8_yc_override",
     "composite_macro_v5",
     "V5 Phase 2.8 — + Yield Curve persistent inversion override"),
    ("v5_phase2.9_dxy_m2_stress",
     "composite_macro_v2_v5",
     "V5 Phase 2.9 — + DXY/M2 macro stress + YC yumuşatılmış"),
    ("v5_phase2.10_bull_velocity",
     "composite_macro_v3_v5",
     "V5 Phase 2.10 — + Bull velocity entry (V-shape recovery)"),
    ("v5_phase2.11_bear_velocity",
     "composite_macro_v4_v5",
     "V5 Phase 2.11 — + Bear velocity entry (rapid escalation)"),
    ("v5_phase2.12_noise_reduction",
     "composite_macro_v5_v5",
     "V5 Phase 2.12 — + Neutral min dwell (noise reduction)"),
]


def shade_regimes(ax, regime, lw_separator=0):
    if regime.empty:
        return
    cur, start = regime.iloc[0], regime.index[0]
    for i in range(1, len(regime)):
        v = regime.iloc[i]
        if v != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime.index[i],
                           color=REGIME_COLORS.get(cur, "white"),
                           alpha=0.30, lw=lw_separator, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.30, lw=lw_separator, zorder=2)


def plot_phase_timeline(phase_id, csv_suffix, title, sp500, btc_close, eth_close,
                        out_dir: Path):
    proc = PROJECT_ROOT / "data" / "processed"
    pre_csv = proc / f"macro_pretrain_regime_labels_{csv_suffix}.csv"
    btc_csv = proc / f"btc_regime_labels_{csv_suffix}.csv"
    eth_csv = proc / f"eth_regime_labels_{csv_suffix}.csv"
    if not pre_csv.exists():
        print(f"  SKIP {phase_id}: {pre_csv.name} missing")
        return

    pre_r = pd.read_csv(pre_csv, index_col=0, parse_dates=True)
    btc_r = pd.read_csv(btc_csv, index_col=0, parse_dates=True)
    eth_r = pd.read_csv(eth_csv, index_col=0, parse_dates=True)

    fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 0.4]})

    # Subplot 1 — S&P 500 + pretrain regime
    sp_aligned = sp500.reindex(pre_r.index, method="nearest")
    shade_regimes(axes[0], pre_r["regime_label"])
    axes[0].plot(sp_aligned.index, sp_aligned.values, color="black",
                 lw=1.3, zorder=3)
    axes[0].set_ylabel("S&P 500", fontsize=14)
    axes[0].set_title("S&P 500 (pretrain) + Composite Macro Regime",
                       fontsize=13, fontweight="bold")
    axes[0].tick_params(labelsize=12)

    # Subplot 2 — BTC
    shade_regimes(axes[1], btc_r["regime_label"])
    axes[1].semilogy(btc_close.index, btc_close.values, color="black",
                     lw=1.4, zorder=3)
    axes[1].set_ylabel("BTC (log)", fontsize=14)
    axes[1].set_title("BTC + regime shading", fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=12)

    # Subplot 3 — ETH
    shade_regimes(axes[2], eth_r["regime_label"])
    axes[2].semilogy(eth_close.index, eth_close.values, color="black",
                     lw=1.4, zorder=3)
    axes[2].set_ylabel("ETH (log)", fontsize=14)
    axes[2].set_title("ETH + regime shading", fontsize=13, fontweight="bold")
    axes[2].tick_params(labelsize=12)

    # Subplot 4 — boş placeholder (kullanıcı isteği)
    axes[3].set_yticks([])
    axes[3].set_xticks([])
    axes[3].grid(False)
    for spine in axes[3].spines.values():
        spine.set_visible(False)
    axes[3].text(0.5, 0.5, "(reserved)", ha="center", va="center",
                 transform=axes[3].transAxes, fontsize=11, color="#888",
                 style="italic")

    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Compose legend from labels actually present in this phase
    labels_present = set(pre_r["regime_label"].dropna().unique())
    seen_colors = set()
    handles = []
    for label, color in REGIME_COLORS.items():
        if label in labels_present and color not in seen_colors:
            handles.append(Patch(facecolor=color, alpha=0.5, label=label))
            seen_colors.add(color)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=12,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = out_dir / f"{phase_id}_timeline_4panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"
    out_dir = PROJECT_ROOT / "reports" / "Phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    sp500 = pd.read_csv(raw / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)
    btc_close = btc["Close"]
    eth_close = eth["Close"]

    for phase_id, csv_suffix, title in PHASES:
        plot_phase_timeline(phase_id, csv_suffix, title,
                            sp500, btc_close, eth_close, out_dir)

    print("\nAll phase timelines generated.")


if __name__ == "__main__":
    main()
