"""
Stage 2 v4 GMM rejim atamasını ve makro girdileri görselleştir.

ÖNEMLİ: GMM unsupervised → cluster idx 0/1/2 rastgele atanır. Bu
script önce cluster'ları VIX ortalamasına ve SP500 realized
volatility'sine göre semantically relabel eder:
  - en düşük VIX + en düşük vol → Calm    (yeşil)
  - en yüksek vol               → Stress   (kırmızı)
  - geri kalan                   → Transition (turuncu)

Output: reports/monthly_fred_overview.png  (4-panel)
  Panel 1: BTC log + cluster shading
  Panel 2: SP500 log + cluster shading + VIX (twin axis)
  Panel 3: Stage 2 posterior probability time series (3 line)
  Panel 4: Cluster sınıf statistik tablosu (panel içi text)
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

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

# Renk şeması (semantic, cluster idx'e değil!)
COLORS = {"Calm": "#a8dca8", "Transition": "#f8d6a4", "Stress": "#f7b3b3"}
ORDER = ["Calm", "Transition", "Stress"]


def relabel_clusters(s2_post: pd.DataFrame, btc: pd.DataFrame) -> tuple[dict, np.ndarray, dict]:
    """
    Returns
    -------
    idx_to_name : dict[int, str]   raw cluster idx → semantic name
    semantic_hard : np.ndarray     test-set hard labels in semantic str form
    stats : dict                   per-cluster stats (n, vix_mean, vol)
    """
    common = s2_post.index.intersection(btc.index)
    s2_c = s2_post.loc[common]
    btc_c = btc.loc[common]
    hard_idx = s2_c.values.argmax(1)
    sp500_ret = btc_c["SP500"].pct_change()

    stats_raw = {}
    for c in sorted(set(hard_idx)):
        mask = hard_idx == c
        stats_raw[c] = {
            "n": int(mask.sum()),
            "vix_mean": float(btc_c["VIX"][mask].mean()),
            "sp500_real_vol": float(sp500_ret[mask].std() * np.sqrt(252)),
        }

    # Map by VIX mean (implied vol — daha güvenilir cluster ayrıştırma):
    #  en düşük VIX → Calm, en yüksek VIX → Stress, orta → Transition
    by_vix = sorted(stats_raw.items(), key=lambda kv: kv[1]["vix_mean"])
    calm_idx = by_vix[0][0]      # en düşük VIX
    stress_idx = by_vix[-1][0]   # en yüksek VIX
    trans_idx = by_vix[1][0]     # ortada
    idx_to_name = {calm_idx: "Calm", trans_idx: "Transition", stress_idx: "Stress"}

    semantic_hard = np.array([idx_to_name[c] for c in hard_idx])
    stats = {idx_to_name[c]: v for c, v in stats_raw.items()}
    return idx_to_name, semantic_hard, stats


def shade_clusters(ax, dates, semantic_hard):
    """Add axvspan blocks coloured by cluster name."""
    cur, start_idx = semantic_hard[0], 0
    for i in range(1, len(semantic_hard)):
        if semantic_hard[i] != cur:
            ax.axvspan(dates[start_idx], dates[i], color=COLORS[cur], alpha=0.35, lw=0)
            cur, start_idx = semantic_hard[i], i
    ax.axvspan(dates[start_idx], dates[-1], color=COLORS[cur], alpha=0.35, lw=0)


def main() -> None:
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                     index_col=0, parse_dates=True)
    common = btc.index.intersection(s2.index)
    btc = btc.loc[common]; s2 = s2.loc[common]

    idx_to_name, semantic_hard, stats = relabel_clusters(s2, btc)
    print(f"Cluster mapping (raw idx → semantic):  {idx_to_name}")
    for name in ORDER:
        s = stats[name]
        print(f"  {name:11s}  n={s['n']:4d}  VIX_mean={s['vix_mean']:5.2f}  "
              f"SP500_real_vol={s['sp500_real_vol']*100:.1f}%")

    # Posterior columns are P_0, P_1, P_2 — remap to semantic
    p_cols = list(s2.columns)
    name_to_col = {idx_to_name[int(c.split("_")[-1])]: c for c in p_cols}

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.5, 1]})

    # ---- Panel 1: BTC log + cluster shading ----
    ax = axes[0]
    ax.semilogy(btc.index, btc["Close"], color="#222", lw=1.0, label="BTC Close")
    shade_clusters(ax, btc.index, semantic_hard)
    ax.set_ylabel("BTC (USD, log)")
    ax.set_title(
        "BTC fiyatı + Stage 2 v4 GMM makro rejim atamasından gelen cluster shading\n"
        "(rejim VIX/SP500/credit-spread/FFR/UNRATE'ten türetilmiş, BTC GMM girdisi DEĞİL)",
        fontsize=10, fontweight="bold",
    )
    handles = [Patch(facecolor=COLORS[n], alpha=0.55, label=f"{n}") for n in ORDER]
    handles.insert(0, plt.Line2D([], [], color="#222", lw=1, label="BTC Close"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=4)

    # ---- Panel 2: SP500 + VIX + cluster shading ----
    ax = axes[1]
    ax.semilogy(btc.index, btc["SP500"], color="#1f5b87", lw=1.2, label="SP500 (log)")
    shade_clusters(ax, btc.index, semantic_hard)
    ax.set_ylabel("SP500 (log)", color="#1f5b87")
    ax.tick_params(axis="y", labelcolor="#1f5b87")
    ax2 = ax.twinx()
    ax2.plot(btc.index, btc["VIX"], color="#d62728", lw=0.9, alpha=0.7, label="VIX")
    ax2.set_ylabel("VIX", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    ax.set_title("SP500 log fiyat + VIX  (Stage 2 GMM'in iki ana girdisi)",
                 fontsize=10, fontweight="bold")

    # ---- Panel 3: Posterior soft probabilities ----
    ax = axes[2]
    for name, color in [("Calm", "#2ca02c"), ("Transition", "#ff7f0e"),
                        ("Stress", "#d62728")]:
        col = name_to_col[name]
        ax.plot(s2.index, s2[col], color=color, lw=0.8, label=f"P({name})")
    ax.set_ylabel("Cluster posterior")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Stage 2 GMM soft posterior probabilities (test'te modelin Stage 3'e gönderdiği vektör)",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Tarih")

    # Stat box on top of figure
    stat_lines = [
        f"Calm        n={stats['Calm']['n']:4d}  VIX={stats['Calm']['vix_mean']:5.2f}  "
        f"realized vol={stats['Calm']['sp500_real_vol']*100:.1f}%",
        f"Transition  n={stats['Transition']['n']:4d}  VIX={stats['Transition']['vix_mean']:5.2f}  "
        f"realized vol={stats['Transition']['sp500_real_vol']*100:.1f}%",
        f"Stress      n={stats['Stress']['n']:4d}  VIX={stats['Stress']['vix_mean']:5.2f}  "
        f"realized vol={stats['Stress']['sp500_real_vol']*100:.1f}%",
    ]
    fig.text(0.5, -0.01, "  |  ".join(stat_lines), ha="center", va="top",
             fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#999"))

    fig.suptitle(
        "Stage 2 v4 GMM Makro Rejim — semantik relabeling sonrası "
        "(en düşük gerçekleşmiş vol = Calm, en yüksek = Stress)",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])

    out = PROJECT_ROOT / "reports" / "monthly_fred_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
