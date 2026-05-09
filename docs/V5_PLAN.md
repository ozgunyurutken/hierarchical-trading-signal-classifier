# V5 — From-Scratch Implementation Plan

**Branch:** `v5-from-scratch` (from `v1.0-iter4-final` tag, commit `ab408d5`)
**Onay:** Kullanıcı paket onay verdi 2026-05-08
**Hedef:** Proposal'a %100 uyumlu + literatür-supported, paper'da yer alacak final sürüm.

> Geçmiş iterasyonlar (v1, v2/ablation, v2/bigger-dataset, v2/feature-selection B2 = ZZ-MLP Sharpe 1.68) **paper'da bahsedilmeyecek**. Git history + iç dokümantasyonda fallback olarak kalır (CLAUDE.md "Report scope rule").

---

## 0. Tasarım Sonuçları (13 karar — paket onaylandı)

| # | Karar | Final V5 |
|---|---|---|
| 1 | Veri dönemi | 2018-01 → 2025-12 (~2,800 gün) |
| 2 | Tickers | BTC + ETH ayrı modeller |
| 3 | Stage 1 features | SMA/EMA/ADX/Parabolic SAR + log_ret_5d (11 feat) |
| 4 | Stage 2 method | K-Means k=3 + Elbow + Silhouette + Gap + CH validation |
| 5 | Stage 2 features | 9 (7 raw + 2 derived: yield curve, gold/silver) |
| 6 | Semantic relabeling | Lowest VIX + highest SP → Risk-On; Highest VIX + most negative SP → Risk-Off; rest → Neutral |
| 7 | Stage 3 fusion | One-hot s1 + s2 (proposal-strict) |
| 8 | Stage 3 features | 15 oscillators + volume (RSI/MACD/Stoch/Williams/BB/ATR/OBV/**VWAP**/Volume) |
| 9 | Stage 3 label | **Dual:** primary = adaptive 0.5×rolling_std, ablation = ±1% fixed |
| 10 | Class imbalance | class_weight + scale_pos_weight (SMOTE rejected) |
| 11 | Algorithms | XGB, LGBM, RF, MLP (4) |
| 12 | CV split | 70/15/15 + walk-forward + Optuna 30 trial |
| 13 | Backtest | Long-only primary + regime-sized bonus + B&H benchmark |

---

## 1. Branch + Doküman Stratejisi

```
v5-from-scratch (ana V5 branch)
├── config.yaml          ✅ V5'e göre tamamen yeniden yazıldı
├── docs/V5_PLAN.md      ✅ bu dosya
├── docs/V5_FEATURES.md  ⏳ implementation sırasında yazılır
├── src/                 ⏳ V5 modülleri (aşağıda)
├── notebooks/           ⏳ V5 EDA + analiz
├── data/                ⏳ V5 dataset
├── reports/             ⏳ V5 plotları
└── app/                 ⏳ V5 deployment
```

Geçmiş artefaktlar (v1-v4 + B2) bu branch'te **YOK** — temiz başlangıç. Fallback için ana branch + tag'lar kullanılır.

---

## 2. Source Code Modülleri (yazılacak)

### Veri Pipeline
- `src/data/v5_data_collector.py` — yfinance (BTC/ETH/SP500/VIX/DXY/Gold/Silver/^TNX) + FRED (DGS2 daily, FFR/CPI/UNRATE monthly) fetcher
- `src/data/v5_data_aligner.py` — daily alignment + monthly forward-fill with publication-release lag

### Feature Engineering
- `src/features/v5_technical_features.py` — proposal §2.2 indicators (SMA/EMA/ADX/PSAR/RSI/Stoch/MACD/Williams/BB/ATR/OBV/**VWAP**/Volume)
- `src/features/v5_macro_features.py` — 7 raw + 2 derived + RoC + rolling z-score

### Labels
- `src/labels/v5_trend_labels.py` — SMA(20,50) crossover + 3-day persistence (proposal §2.4)
- `src/labels/v5_regime_labels.py` — K-Means k=3 + cluster validation suite + semantic relabeling
- `src/labels/v5_signal_labels.py` — Dual labeling: adaptive 0.5×rolling_std + fixed ±1%

### Models
- `src/models/v5_classifiers.py` — 4-algorithm wrappers (XGB/LGBM/RF/MLP) with class_weight + scale_pos_weight
- `src/models/v5_pipelines.py` — 3 ablation pipelines (Flat / 2-Stage / 3-Stage)
- `src/models/v5_optuna_tuner.py` — Optuna TPE walk-forward HP search

### Evaluation
- `src/evaluation/v5_metrics.py` — classification metrics + ROC-AUC OvR
- `src/evaluation/v5_mcnemar.py` — McNemar pairwise + Holm correction
- `src/evaluation/v5_shap_analyzer.py` — TreeSHAP + KernelSHAP wrappers
- `src/evaluation/v5_backtester.py` — long-only + regime-sized + transaction cost + DSR

### Notebooks (paper figure reproducibility)
- `notebooks/v5_01_eda.ipynb` — descriptive stats + price/macro plots
- `notebooks/v5_02_kmeans_validation.ipynb` — Elbow + Silhouette + Gap + CH plots, k=3 doğrulaması
- `notebooks/v5_03_label_review.ipynb` — Stage 1/2/3 label distributions, user approval gate
- `notebooks/v5_04_stage1_train.ipynb` — trend classifier 4-algorithm comparison
- `notebooks/v5_05_stage2_train.ipynb` — macro regime classifier 4-algorithm comparison
- `notebooks/v5_06_stage3_train.ipynb` — signal classifier + ablation
- `notebooks/v5_07_evaluation.ipynb` — confusion matrices + ROC + SHAP + McNemar
- `notebooks/v5_08_backtest.ipynb` — long-only + regime-sized + B&H comparison

---

## 3. İcra Planı — Faz Bazlı

### Faz 1 — Foundations (3-4 saat)
1. Branch açıldı ✅
2. config.yaml v5 yazıldı ✅
3. docs/V5_PLAN.md yazıldı ✅
4. requirements.txt güncelle: `optuna shap statsmodels imbalanced-learn ta`
5. src/ V5 modüllerinin iskeleti + helper fonksiyonlar
6. Veri toplama: yfinance + FRED → `data/raw/v5/`
7. Data alignment: BTC/ETH 22 col aligned dataset → `data/processed/btc_aligned_v5.csv`, `eth_aligned_v5.csv`

**Decision gate:** Aligned dataset shape + sanity check + EDA notebook → kullanıcı onayı

### Faz 2 — Feature Engineering + Labels (2-3 saat)
1. Technical features (BTC + ETH) → `btc_features_v5.csv`, `eth_features_v5.csv`
2. Macro features (raw + RoC + long-term static z-score + derived spreads incl. M2 yoy)
3. **Feature correlation re-check (Decision Gate 1.5)** — derived features sonrası
   collinearity raw seviye ~0.9 küme'den ne kadar düştü? Stage 2 K-Means için
   kabul edilebilir mi?
4. Stage 1 trend labels (SMA cross + 3-day persistence)
5. Stage 2 macro regime labels:
   - K-Means k=3 fit on macro feature subset (2000-2025 pre-train data)
   - Elbow + Silhouette + Gap + CH validation plots
   - Semantic relabeling (Risk-On/Off/Neutral)
   - Inference on crypto-aligned period
   - Output `btc_regime_labels_v5.csv` + cluster centroids
6. Stage 3 signal labels (dual: adaptive + fixed)

**Decision gate:** Notebook 02 + 03 review — kullanıcı **collinearity check (1.5) + label dağılımları + cluster görsellerini** onaylar, sonra training'e geçilir.

### Faz 3 — Stage 1 + Stage 2 Training (3-4 saat)
1. Stage 1 (Trend Classifier): 4 algorithm × walk-forward CV × Optuna 30 trial
2. Stage 2 (Macro Regime Classifier): 4 algorithm × walk-forward CV × Optuna 30 trial
3. OOF predictions + Stage 3 input prep (one-hot s1 + s2)
4. Stage 1 + Stage 2 evaluation: confusion matrix + ROC + SHAP per stage

**Decision gate:** Stage 1 ve Stage 2 doğruluk seviyesi yeterli mi? Notebook 04 + 05 review.

### Faz 4 — Stage 3 + Ablation (4-5 saat)
1. Flat baseline: tüm features + 4 algorithm
2. 2-stage Trend: tech + s1 (one-hot) + 4 algorithm
3. 3-stage Full: tech + s1 + s2 (one-hot) + 4 algorithm
4. Total: 3 config × 4 model × 2 coin = **24 deney**
5. McNemar pairwise (3 pair) + Holm correction
6. SHAP per stage per model

**Decision gate:** Notebook 06 review.

### Faz 5 — Backtest + Final Plotlar (2-3 saat)
1. Long-only backtest (24 + B&H)
2. Regime-sized backtest (3-stage × Stage 2 regime)
3. Sharpe + DSR + MaxDD + Calmar + Win rate
4. Final figures: regime detection, equity curves, summary table, CM grid, SHAP global

**Decision gate:** Notebook 07 + 08 review.

### Faz 6 — Deployment + Final Validation (2 saat)
1. Docker + FastAPI rebuild for V5 models
2. Smoke test
3. Paper-ready CSV summary tables
4. v5.0-final tag

---

## 4. Karar Noktaları (User Approval Gates)

Her fazın sonunda kullanıcı onayı şart (CLAUDE.md "Show results first, then wait" + "User is the main decision-maker"):

| Gate | Notebook | Kullanıcı kontrol eder |
|---|---|---|
| 1 | `01_eda.ipynb` | Aligned dataset shape + price plots + macro distributions + train/test split |
| **1.5** | `01b_feature_corr_recheck.py` | **Feature engineering sonrası** korelasyon yeniden hesapla. Faz 1'de raw seviye corr ~0.9 küme (BTC/SP500/Gold/M2). Derived features (z-score, RoC, yoy %) sonrası collinearity gerçekten düştü mü? Stage 2 K-Means cluster için kabul edilebilir mi? |
| 2 | `02_kmeans_validation.ipynb` | Elbow + Silhouette + Gap plot k=3 doğrular mı; cluster centroidleri Risk-On/Off/Neutral semantic mantığa uyuyor mu |
| 3 | `03_label_review.ipynb` | Stage 1/2/3 label dağılımları (Up%/Down%/Side% + Risk-On%/Off%/Neutral% + Buy%/Sell%/Hold%) makul mü |
| 4 | `04_stage1_train.ipynb` + `05_stage2_train.ipynb` | Stage 1+2 accuracy seviyeleri kabul edilebilir mi |
| 5 | `06_stage3_train.ipynb` | Ablation flat/2-stage/3-stage Sharpe trendi makul mü |
| 6 | `07_evaluation.ipynb` + `08_backtest.ipynb` | Final tablolar + plotlar paper'a hazır mı |

---

## 5. Başarı Kriteri ve Fallback

### V5 başarılı sayılır eğer:
- 3-stage Full (en iyi model) test set'te B&H Sharpe'ını geçerse
- McNemar test 3-stage > flat istatistiksel anlamlı
- BTC ve ETH ablation'larda 3-stage > 2-stage sıralaması korunur
- Paper-ready figürler ve summary table üretilir

### Fallback (V5 başarısız çıkarsa):
- v1.0-iter4-final tag (XGB Sharpe 1.35) headline
- v2/feature-selection B2 (ZZ-MLP Sharpe 1.68) "preliminary work" not edilebilir ama paper headline değil
- V5 çıktıları "experimental" appendix olur

---

## 6. Proposal vs V5 — 5 Sapma Savunması (rapor metnine eklenir)

| # | Sapma | Savunma cümlesi (paper §3 Methodology) |
|---|---|---|
| 1 | Veri 2018-2025 (proposal 2021-2025) | "Training window extended to 2018-01 (post-2017 BTC futures market era) to capture multiple bull-bear cycles." |
| 2 | Adaptive label (proposal ±1%) | "We compare proposal-strict ±1% threshold against an adaptive volatility-targeted threshold (0.5×rolling std-20) following Lee et al. 2024 [N84]; both reported in ablation." |
| 3 | class_weight (proposal SMOTE) | "SMOTE was rejected for time-series data due to leakage risk in walk-forward CV (López de Prado 2018 [12]); cost-sensitive weighting applied instead." |
| 4 | `ta` paketi (proposal TA-Lib) | "TA-Lib's TA-Lib alternative `ta` package used for cross-platform compatibility; identical indicator formulas." |
| 5 | sklearn MLP (proposal PyTorch) | "MLP implemented via scikit-learn with early stopping; deep architectures (LSTM/Transformer) left for future work given the tabular feature focus (Shwartz-Ziv & Armon 2022)." |

---

## 7. Naming Convention

V5 için tek standart:
- Pipeline configs: **Flat / 2-Stage Trend / 3-Stage Full** (no A1/A4 codes)
- Stage 2 classes: **Risk-On / Risk-Off / Neutral** (proposal sözü)
- Stage 1 classes: **Uptrend / Downtrend / Sideways**
- Stage 3 classes: **Buy / Sell / Hold**
- Models: **XGBoost, LightGBM, RandomForest, MLP** (full names)
- Coins: **BTC, ETH**
- Artifact suffix: **`_v5`** (örn. `btc_aligned_v5.csv`, `stage3_xgboost_v5.joblib`)

---

_V5 implementation kullanıcı onayıyla başladı. Bir sonraki adım: requirements.txt güncelle + src/ V5 modüllerinin iskeleti + Faz 1 veri toplama._
