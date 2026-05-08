"""
Render the v4 hierarchical pipeline architecture diagram.

Layout (top → bottom):
  [Data sources]   [Aligned dataset]
        ↓                ↓
  [Feature engineering: technical + macro]
        ↓                ↓
  [Stage 1 (Trend)]  [Stage 2 (Macro Regime)]
        ↓                ↓
        └──────┬─────────┘
               ↓
       [Stage 3 (Signal)]
               ↓
           [Output]

Output: reports/pipeline_v4.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def box(ax, x, y, w, h, title, body, color, title_color="black", body_size=7.5,
        title_size=10, body_pad_top=0.22):
    """Draw a rounded box with a bold title and a multi-line body."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.2, edgecolor="#333",
                       facecolor=color, alpha=0.95)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 0.07, title,
            ha="center", va="top", fontsize=title_size, fontweight="bold", color=title_color)
    ax.text(x + 0.08, y + h - body_pad_top, body,
            ha="left", va="top", fontsize=body_size, color="#222",
            family="monospace", linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, label=None, color="#444", lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        color=color, lw=lw, zorder=5)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.012, label,
                ha="center", va="bottom", fontsize=7.5, fontstyle="italic",
                color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 13))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors
    C_DATA   = "#e8e8e8"
    C_ALIGN  = "#cfe5ff"
    C_FEAT_T = "#cfeed1"
    C_FEAT_M = "#a8dca8"
    C_S1     = "#ffd2a3"
    C_S2     = "#ff9b87"
    C_S3     = "#d3b3e6"
    C_OUT    = "#fde68a"

    # ---------------- ROW 1: Data sources / Aligned / Label gen ----------------
    box(ax, 0.1, 8.55, 3.2, 1.35, "Veri Kaynakları",
        "• Yahoo Finance: BTC, ETH, SP500,\n  VIX, DXY, Gold, Silver, Oil,\n  ^TNX → US10Y\n• FRED daily: DGS2 (US2Y), HY/IG,\n  Treasury20Y, TIPS\n• FRED monthly (v4): FEDFUNDS,\n  CPIAUCSL, UNRATE, WM2NS, ICSA",
        C_DATA)

    box(ax, 3.5, 8.55, 3.0, 1.35, "Hizalanmış Veri (v4)",
        "btc_aligned.csv\n  3,961 sat × 22 kol\n  2014-09-17 → 2025-12-30\neth_aligned.csv\n  2,857 sat × 22 kol\n\nv4: 17 → 22 (5 monthly FRED,\nrelease-lag shifted)",
        C_ALIGN)

    box(ax, 6.7, 8.55, 3.2, 1.35, "Etiket Üretimi",
        "Trend etiketi (Stage 1):\n  SMA(20,50) cross  +  ZigZag\n  → Up / Down / Flat\n\nSinyal etiketi (Stage 3):\n  ATR-adaptive forward return\n  → Buy / Sell / Hold",
        "#fad8e0")

    # arrow Data → Aligned (centered between boxes)
    arrow(ax, 3.3, 9.22, 3.5, 9.22)

    # ---------------- ROW 2: Feature engineering ----------------
    box(ax, 0.1, 6.5, 4.7, 1.75, "Technical Features",
        "src/features/technical_features.py\n  Returns, log-returns\n  RSI, MACD, Stochastic, Williams%R\n  Bollinger %B, ATR\n  SMA/EMA (5, 10, 20, 50, 100, 200)\n  Volume z-score, OBV, lags\n\n→ btc_features_stage3_v2.csv  (~65 feat)",
        C_FEAT_T)

    box(ax, 5.1, 6.5, 4.8, 1.75, "Macro Features (v4)",
        "src/features/macro_features.py\nHer raw kolon × {sma, zscore, roc}\n  pencereler: 20 / 50 / 100 gün\nTürev feature'lar:\n  Yield_Curve_10Y_2Y, Credit_Spread_log\n  Gold_Silver_Ratio, SP500_VIX_ratio\n  macro_real_interest_rate  (Fisher)\n\n→ btc_features_macro.csv  (187 feat)\n  v4: 136 → 187 (+51 FRED türevi)",
        C_FEAT_M)

    # arrows: Aligned → Features
    arrow(ax, 2.0, 8.55, 2.0, 8.25)
    arrow(ax, 6.5, 8.55, 6.5, 8.25)

    # ---------------- ROW 3: Stage 1 + Stage 2 ----------------
    box(ax, 0.1, 3.95, 4.7, 2.25, "STAGE 1 — Trend Sınıflandırıcı",
        "Girdi:  technical features (~65)\n        + smoothed price moments\nAlgoritma:  LDA  (Linear Discriminant)\nDoğrulama:  5-fold OOF (chronological,\n            no shuffle)\nÇıktı:  p̂(trend) ∈ ℝ³\n        [P(Up), P(Down), P(Flat)]\n\nParalel iki sürüm:\n  • SMA cross etiketi  →  Phase A+B\n  • ZigZag etiketi      →  Phase C",
        C_S1, title_size=10.5)

    box(ax, 5.1, 3.95, 4.8, 2.25, "STAGE 2 — Makro Rejim   (v4: 11 feature)",
        "Girdi (11 makro feature):\n  Eski 8:  VIX, VIX_z50, Yield_Curve,\n           Credit_Spread, Gold_Silver,\n           SP500_VIX, DXY_z50, SP500_roc20\n  Yeni 3 (v4):  FEDFUNDS, real_interest_rate,\n                UNRATE\nÖn-işlem:  StandardScaler (train-only fit)\nAlgoritma:  GMM,  n_clusters = 3\nDoğrulama:  5-fold OOF + full-train\nÇıktı:  p̂(macro) ∈ ℝ³\n        [P(Calm), P(Transition), P(Stress)]",
        C_S2, title_size=10.5)

    # arrows: Features → Stages
    arrow(ax, 2.4, 6.5, 2.4, 6.20, label="tech")
    arrow(ax, 7.5, 6.5, 7.5, 6.20, label="macro 11-subset")

    # ---------------- ROW 4: Stage 3 ----------------
    box(ax, 0.6, 1.5, 8.8, 2.15, "STAGE 3 — Sinyal Sınıflandırıcı",
        "Girdi  ≈ 71 feature:    Technical (~65)  +  s1 OOF posterior (3)  +  s2 OOF posterior (3)\nEtiket:  y ∈ {Buy, Sell, Hold}\nModeller (7):\n   Phase A:  LDA, MLP                            (baseline)\n   Phase B:  XGBoost, LightGBM, Random Forest    (tree models, en güçlü)\n   Phase C:  ZZ-XGBoost, ZZ-MLP                  (ZigZag Stage 1 ile)\nTuning:  Optuna 6-8 trial × walk-forward CV   (12-ay min train, 6-ay step)\nArtefakt:  app/models/stage3_*_v2.joblib    (v4 retrain'de yeniden export → demo skew kapatıldı)",
        C_S3, title_size=11)

    # arrows: Stage 1 + Stage 2 → Stage 3
    arrow(ax, 2.4, 3.95, 3.5, 3.65, label="p̂(trend)∈ℝ³")
    arrow(ax, 7.5, 3.95, 6.5, 3.65, label="p̂(macro)∈ℝ³")

    # ---------------- ROW 5: Output ----------------
    box(ax, 0.6, 0.0, 8.8, 1.25, "Çıktı  /  Backtest sonuçları (BTC test seti, 462 gün)",
        " Model      Acc    F1     MCC    Return     Sharpe    MaxDD    Win%\n ───────────────────────────────────────────────────────────────────────────\n LDA       0.344  0.174  -0.004   -0.8%    -0.78    -1.0%   50.0%\n MLP       0.379  0.263   0.080  +42.8%    +1.33   -10.2%   76.5%\n XGB       0.383  0.253   0.093  +42.7%    +1.35   -11.4%   84.6%   ★ best Sharpe\n LGBM      0.390  0.264   0.116  +27.7%    +1.06   -11.4%   90.0%\n RF        0.394  0.264   0.128  +38.1%    +1.25   -11.4%   80.0%\n ZZ-XGB    0.394  0.264   0.123  +40.5%    +1.33   -11.4%   91.7%   ★ best Win%\n ZZ-MLP    0.390  0.275   0.081  +35.6%    +0.95   -13.0%   79.0%\n Buy&Hold    -      -       -    +47.6%    +0.75   -32.1%      -",
        C_OUT, title_size=10.5, body_size=7.0)

    # arrow Stage 3 → Output
    arrow(ax, 5.0, 1.5, 5.0, 1.25)

    # Title
    fig.suptitle(
        "BBL514E — Üç Aşamalı Hiyerarşik Kripto Sinyal Sınıflandırıcı  (v4 mimarisi)",
        fontsize=13, fontweight="bold", y=0.985,
    )
    fig.text(0.5, 0.962,
             "BTC ve ETH ayrı modellenir. Soft fusion: Stage 1 ve Stage 2 olasılık vektörü çıktısı verir, hard label değil.",
             ha="center", va="top", fontsize=9, fontstyle="italic", color="#444")

    out = PROJECT_ROOT / "reports" / "pipeline_v4.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
