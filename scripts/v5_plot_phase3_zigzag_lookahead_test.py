"""V5 Phase 3 — ZigZag lookahead bias / label revisability demonstration.

Output: reports/Phase3/v5_p3_zigzag_lookahead_test.png

Aynı tarih için iki ZigZag label hesabını karşılaştırır:
  (a) FULL: tüm dataset (2014..2025) kullanılarak label üretilir (mevcut implement.)
  (b) HISTORICAL: sadece o tarihe kadar olan data kullanılarak label üretilir
      (production realistic — strictly causal)

Eğer bu iki label farklıysa → ZigZag label retrospective revisable, ML modele
gerek var argüman somut kanıt kazanır.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.labels.v5_trend_labels import zigzag_trend_label

LABEL_COLORS = {"uptrend": "#7ec27e", "downtrend": "#e07e7e", "range": "#f0c870"}


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "reports" / "Phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    close = btc["Close"]

    # (a) FULL dataset ZigZag (current offline)
    full_labels = zigzag_trend_label(close, deviation_pct=0.10,
                                      min_segment_days=10, range_amplitude=0.05)["label"]

    # (b) HISTORICAL-only ZigZag — for each cutoff t, run ZigZag on close[:t+1] only
    cutoff_dates = pd.date_range("2017-01-01", "2025-10-01", freq="MS")
    cutoff_dates = [d for d in cutoff_dates if d in close.index]

    historical_label_at_cutoff = {}
    for t in cutoff_dates:
        sub = close.loc[:t]
        if len(sub) < 100:
            continue
        hist_labels = zigzag_trend_label(sub, deviation_pct=0.10,
                                          min_segment_days=10, range_amplitude=0.05)["label"]
        historical_label_at_cutoff[t] = hist_labels.iloc[-1]

    cutoff_df = pd.DataFrame({
        "cutoff": list(historical_label_at_cutoff.keys()),
        "historical_label": list(historical_label_at_cutoff.values()),
    }).set_index("cutoff")
    cutoff_df["full_label"] = full_labels.reindex(cutoff_df.index)
    cutoff_df["disagree"] = cutoff_df["historical_label"] != cutoff_df["full_label"]

    n_total = len(cutoff_df)
    n_disagree = cutoff_df["disagree"].sum()
    print(f"Cutoffs tested: {n_total}")
    print(f"Disagreements (label revised retrospectively): {n_disagree} ({n_disagree/n_total:.0%})")
    print(cutoff_df.head(15).to_string())

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 0.8, 0.8]})
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    # Panel 1: BTC price + offline ZigZag full label
    cur, start = full_labels.iloc[0], full_labels.index[0]
    for i in range(1, len(full_labels)):
        v = full_labels.iloc[i]
        if v != cur:
            axes[0].axvspan(start, full_labels.index[i], color=LABEL_COLORS[cur], alpha=0.30, lw=0)
            cur, start = v, full_labels.index[i]
    axes[0].axvspan(start, full_labels.index[-1], color=LABEL_COLORS[cur], alpha=0.30, lw=0)
    axes[0].semilogy(close.index, close.values, color="#F7931A", lw=1.0)
    axes[0].set_ylabel("BTC (log)")
    axes[0].set_title("FULL ZigZag (uses entire 2014-2025 dataset, retrospective) — current offline label",
                      fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: HISTORICAL ZigZag label per monthly cutoff (causal)
    color_map = {"uptrend": LABEL_COLORS["uptrend"],
                 "downtrend": LABEL_COLORS["downtrend"],
                 "range": LABEL_COLORS["range"]}
    bar_h = 0.6
    for t, row in cutoff_df.iterrows():
        c = color_map.get(row["historical_label"], "white")
        axes[1].barh(0, 30, left=t - pd.Timedelta(days=15), height=bar_h, color=c, alpha=0.85, edgecolor="white", lw=0.5)
    axes[1].set_yticks([])
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_title("HISTORICAL ZigZag (rerun monthly with only past data, strictly causal)",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: disagreements highlighted (where full_label != historical_label)
    for t, row in cutoff_df.iterrows():
        if row["disagree"]:
            axes[2].axvline(t, color="red", lw=1.5, alpha=0.7)
    axes[2].set_yticks([])
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_title(
        f"Retrospective revisions: {n_disagree}/{n_total} ({n_disagree/n_total:.0%}) cutoff dates "
        f"where historical label != full-dataset label",
        fontsize=11, fontweight="bold", color="red")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Phase 3 — ZigZag lookahead bias test\n"
                 "Why we need an ML classifier: ZigZag label is RETROSPECTIVE / REVISABLE, not production-stable",
                 fontsize=13, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "v5_p3_zigzag_lookahead_test.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
