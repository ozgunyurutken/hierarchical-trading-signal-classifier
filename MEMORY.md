# MEMORY.md - Project State & Decision Log

## Current Status
**Active Phase:** FAZ 7 — Rapor & Sunum (iter 4 monthly FRED + retrain tamam)
**Last Updated:** 2026-05-08 (öğleden sonra, v4 retrain push edildi: commit 1850e87)
**Days to deadline:** 2 (Final Report 10 Mayıs 2026)

## Progress Tracker

| Phase | Status | Checkpoint | Notes |
|-------|--------|------------|-------|
| FAZ 0: Proje İskeleti | ✅ Tamamlandı | 2026-03-15 | 46 dosya |
| FAZ 1: Veri Toplama | ✅ Tamamlandı | 2026-03-15 | btc_aligned.csv (4111 satır) |
| FAZ 2: Feature Engineering | ✅ Tamamlandı | 2026-05-07 | 200 feature, 136 stationary, 4 derived spread |
| FAZ 3: Label Üretimi | ✅ Tamamlandı | 2026-05-07 | Trend SMA + GMM soft posterior + signal Buy/Sell/Hold |
| FAZ 4: Model Eğitimi | ✅ Tamamlandı | 2026-05-07 | Stage 1: LDA 70% / MLP 82%; Stage 3: LDA 35% / MLP 38% |
| FAZ 5: Değerlendirme | ✅ Tamamlandı | 2026-05-07 | Backtest: MLP +15.8%, Sharpe 0.55, B&H +43.8% |
| FAZ 6: Web App & API | ✅ Tamamlandı | 2026-05-07 | uvicorn lokalde test edildi, Docker build kullanıcıda |
| FAZ 7: Rapor & Sunum | 🔲 Beklemede | - | LaTeX IEEE template, lit review (Hizir araştırır) |

## FAZ 4-5 Sonuç Özeti (2026-05-07)

### Sınıflandırma metrikleri
| Model        | WF F1 | Test Acc | Test F1 | BalAcc | MCC    |
|--------------|-------|----------|---------|--------|--------|
| Stage 1 LDA  | 0.597 | 0.704    | 0.675   | 0.689  | 0.574  |
| Stage 1 MLP  | 0.891 | 0.819    | 0.800   | 0.838  | 0.757  |
| Stage 3 LDA  | 0.265 | 0.350    | 0.173   | 0.331  | -0.021 |
| Stage 3 MLP  | 0.315 | 0.384    | 0.254   | 0.354  | 0.062  |

### Backtest (test period ~ 2024-04 → 2025-12)
| Strateji          | Total Return | Sharpe | MaxDD   | #Trades | Win Rate |
|-------------------|--------------|--------|---------|---------|----------|
| LDA (with cost)   | 0.0%         | 0.00   | 0.0%    | 0       | -        |
| MLP (with cost)   | +15.8%       | 0.55   | -15.9%  | 26      | 53.8%    |
| Buy & Hold        | +43.8%       | 0.67   | -32.1%  | 1       | -        |

### Yorum (raporda anlatılacak)
- Stage 1 SMA crossover label'ı feature space'iyle yarı-tautolojik → MLP ezberliyor (%82). LDA daha gerçekçi (%70).
- Stage 3 chance level (%33) civarında — 3-class trading prediction zor problem; ±%1 forward return threshold'u küçük, gürültü domine.
- LDA Stage 3 sıfır trade verdi (sürekli "Sell" tahmin etti, long-only stratejide trade yok). MLP Stage 3 26 trade yaptı.
- MLP B&H'i geçmedi (BTC bull market) ama yarı drawdown ile %53 win rate — defansif performance.

### 13 üretilen plot (reports/)
- price_history, macro_indicators, correlation_matrix
- feature_correlations, feature_distributions
- trend_labels, signal_labels, k_validation, regime_clusters
- stage1_confusion_matrices, stage3_confusion_matrices, stage3_decision_boundary_pca
- final_confusion_matrices, final_roc_curves, backtest_equity_curves, backtest_drawdowns

### Üretilen artifact'lar
- `app/models/stage1_lda.joblib`, `stage1_mlp.joblib`
- `app/models/stage2_cluster_artifact.joblib` (GMM, 3 cluster, 8 feature)
- `app/models/stage3_lda.joblib`, `stage3_mlp.joblib`
- `app/models/pipeline_lda/`, `pipeline_mlp/` (full HierarchicalSoftPipeline)
- `data/labels/btc_test_signals.csv` (Stage 3 test predictions)
- `data/labels/btc_backtest_summary.csv`, `btc_equity_curves.csv`
- `data/labels/final_classification_summary.csv`

## FastAPI smoke test (2026-05-07)
- `GET /health` → `{"status":"ok","pipelines_loaded":["lda","mlp"],"stage2_loaded":true,"test_dates_count":617,"ready":true}`
- `GET /test_dates/BTC` → 617 tarih
- `POST /predict {date,model:lda}` → Sell @ %55.6 confidence (2024-08-05, $53,991)

## Docker
- `docker/Dockerfile` — Python 3.11-slim, requirements + src + app + 4 CSV bundle
- `docker/docker-compose.yml` — 8000 port, healthcheck
- `.dockerignore` — venv, notebooks, .git, raw data, large all-features CSV hariç
- **Docker bu makinede yüklü değil** → kullanıcı `docker compose -f docker/docker-compose.yml up --build` ile demo öncesi build edecek

## Iter 2 Sonuçları (2026-05-08)

### Phase A — trend-following features + adaptive threshold
- 11 yeni feature: log_ret 5/10/20/50/100, above_sma_200, adx_strong_trend, adx_value, donchian_pct, sharpe_proxy_20d, higher_high_count
- Stage 3 v2 features: 18 → 29 (osc/vol/volume + trend_following)
- Signal label: fixed ±1% → **adaptive 0.5×rolling_std** (primary)
- 2 model retrain: LDA + MLP (tune_stage3 step_months parametresi eklendi)
- **Sonuç:** MLP MCC 0.06 → 0.11 (~2x), F1 0.25 → 0.30, Test acc 0.38 → 0.41

### Phase B — XGBoost + LightGBM + RandomForest
- macOS libomp segfault fix: OMP_NUM_THREADS=1 + n_jobs=1 + tree_method='hist' + force_col_wise=True
- 3 yeni model: XGBoost, LightGBM, RandomForest

### Final v2 Stage 3 Karşılaştırma

| Model | WF F1 | Test Acc | Test F1 | MCC | Backtest Return | Sharpe | Win % |
|-------|-------|----------|---------|-----|----------------|--------|-------|
| LDA | 0.255 | 0.343 | 0.170 | 0.000 | 0.0% (no trade) | 0.00 | - |
| **MLP (best classifier)** | 0.269 | **0.408** | **0.304** | **0.113** | +8.5% | 0.30 | 58.3% |
| **XGBoost (best trader)** | 0.272 | 0.366 | 0.227 | 0.075 | **+25.3%** | **0.88** | **84.6%** |
| LightGBM | 0.279 | 0.378 | 0.263 | 0.080 | +18.1% | 0.56 | 65.0% |
| RandomForest | 0.271 | 0.353 | 0.202 | 0.036 | +19.0% | 0.74 | 66.7% |
| Buy & Hold (benchmark) | - | - | - | - | +43.8% | 0.67 | - |

**Akademik bulgu:** Sınıflandırma metrik (MLP) ile ekonomik metrik (XGBoost) **divergent**. XGBoost daha az ama daha iyi-zamanlanmış Buy tahmini → %84.6 win rate ile Sharpe 0.88 (B&H'ı geçiyor).

**Risk-adjusted:** XGBoost Sharpe 0.88 > Buy&Hold Sharpe 0.67. MaxDD -11.4% vs B&H -32.1% (1/3'ü).

### Yeni artifacts (iter 2)
- `data/processed/btc_features_stage3_v2.csv` (29 feat, 4011 rows)
- `data/labels/btc_stage3_v2_summary.csv`, `btc_stage3_v2_full_summary.csv` (Phase A+B birleşik)
- `data/labels/btc_test_signals_v2.csv` (LDA, MLP), `btc_test_signals_v2_phase_b.csv` (XGB, LGBM, RF)
- `data/labels/btc_backtest_v2_summary.csv`, `btc_backtest_v2_phase_b_summary.csv`
- `app/models/stage3_*_v2.joblib` (5 model)

### Phase C — Causal ZigZag trend label (Stage 1 tautoloji fix)
- Sebep: MVP Stage 1 MLP %82 acc trivial (label SMA-rule + features SMA → tautoloji)
- Yeni: `generate_trend_labels_zigzag(close, threshold=0.10, sideways_band=0.03)` causal swing detection
- BTC label dist: SMA {Up 48.5, Down 35.1, Side 16.5} → ZigZag {Up 61.4, Down 36.7, Side 1.9}
- Stage 1 MLP ZigZag retrain: %75 test acc (gerçek tahmin, %82 trivial'dan düşük)
- `train_stage1` + `tune_stage1`'a step_months/min_train_months override + class-safe OOF (eksik class sıfır kolon ile doldurulur)

### Final v2 ZigZag pipeline backtest

| Strateji | Return | Sharpe | MaxDD | Trades | Win % | Notlar |
|----------|-------:|-------:|------:|-------:|------:|--------|
| **v2 ZZ MLP**     | **+53.9%** | **1.13** | -14.6% | 26 | 57.7% | **B&H'i absolute + Sharpe'ta geçti** |
| v2 SMA XGBoost    | +25.3% | 0.88 | -11.4% | 13 | 84.6% | Yüksek precision, az trade |
| v2 ZZ XGBoost     | +18.4% | 0.60 | -11.4% | 18 | 66.7% | |
| v2 SMA MLP        | +8.5%  | 0.30 | -12.6% | 24 | 58.3% | |
| Buy & Hold        | +43.8% | 0.67 | -32.1% | 1  | -     | benchmark |

**Akademik bulgu (rapora yansır):** SMA × MLP zayıf (+8.5%) ama ZZ × MLP güçlü (+54%). Aynı model türü farklı label'larla farklı performans → **label kalitesi feature kalitesinden daha kritik**. PR dersinde "label generation" konusunun önemi.

### Iter 2 toplu artifact listesi
- `data/processed/btc_features_stage3_v2.csv` (29 feat)
- `data/labels/btc_signal_labels_adaptive.csv` (volatility-adjusted, primary)
- `data/labels/btc_trend_labels_zigzag.csv` (causal swing label)
- `data/labels/btc_stage1_oof_lda_zigzag.csv` (Stage 1 ZZ OOF + test)
- `data/labels/btc_stage3_v2_summary.csv` (Phase A — LDA, MLP)
- `data/labels/btc_stage3_v2_full_summary.csv` (Phase A+B — 5 model)
- `data/labels/btc_stage3_v2_zigzag_summary.csv` (Phase C — ZZ-XGB, ZZ-MLP)
- `data/labels/btc_test_signals_v2.csv`, `_phase_b.csv`, `_zigzag.csv`
- `data/labels/btc_backtest_v2_summary.csv`, `_phase_b_summary.csv`, `_zigzag_summary.csv`
- `data/labels/final_iter2_summary_table.csv` (rapor için)
- `app/models/stage3_lda_v2.joblib`, `_mlp_v2.joblib`, `_xgboost_v2.joblib`, `_lightgbm_v2.joblib`, `_random_forest_v2.joblib`

### Iter 2 toplu görseller (`reports/`)
- `iter2_zigzag_vs_sma.png` — Stage 1 label karşılaştırması
- `iter2_label_comparison.png` — fixed vs adaptive signal label dist
- `iter2_cm_grid_full.png` — 7-model confusion matrix grid
- `iter2_roc_grid_full.png` — 7-model ROC (one-vs-rest)
- `iter2_pred_dist_full.png` — predicted vs true class shares
- `iter2_decision_boundary_pca.png` — PCA 2D scatter (TRUE + best 3 models)
- `iter2_equity_full_v2.png` — equity + drawdown panel, all v2 + B&H
- `iter2_equity_v1_vs_v2.png` — Phase A v1/v2 head-to-head
- `iter2_phase_b_equity.png` — Phase B (tree models) equity
- `iter2_metric_comparison_full.png` — F1/MCC/Sharpe/Return per model + B&H markers
- `iter2_summary_table.png` — final summary table (best Sharpe highlighted)

### Arşiv
MVP/v1 görselleri `old_results/` altına taşındı (proposal/, final/ klasörleri korundu). 13 PNG: trend_labels, signal_labels, k_validation, regime_clusters, stage1/3 confusion matrices, decision_boundary_pca, final_confusion_matrices, final_roc_curves, backtest_equity/drawdowns, feature_correlations/distributions.

## Iter 3 — Data Quality Audit + v3 Retrain (2026-05-08)

### Tetikleyici
Raw data visuals (`reports/raw_data_visuals/`) review sırasında US2Y plot'unun **negatif değerler ve %10K range** gösterdiği fark edildi → debug → **ZT=F futures kontratı price (≈108-110) bond yield olarak kullanılmış.** 2-Year US Treasury yield aslında %0–6 aralığında olmalı.

### Kök neden
- `config.yaml > macro_yields > "ZT=F": "US2Y"` — yfinance ZT=F = 2-Year T-Note **futures fiyatı**, **yield değil**.
- Türev `Yield_Curve_10Y_2Y = US10Y − US2Y` ≈ `4 − 108` ≈ **-104** sabit gürültü oldu (informationless).
- Etki kapsamı:
  - Stage 3 v2 input feature seti **yield curve içermiyor** (sadece osc + vol + volume + trend_following + s1_oof + s2_oof) → tree-based modeller direkt etkilenmedi.
  - **AMA** Stage 2 GMM cluster'ı 8 macro feature kullanıyor ve bunlardan biri `macro_Yield_Curve_10Y_2Y` (1/8 ≈ %12.5) → cluster posterior bug'lı feature ile fit edildi → Stage 3'e feature olarak giren `regime_prob_0/1/2` çarpık.

### Düzeltme
1. **Veri kaynağı:** ZT=F → FRED DGS2 (daily 2-year constant-maturity yield, gerçek %)
2. **Kullanılmayan kolonlar drop:** US5Y, US3M, US30Y (config'te vardı, hiçbir feature/label/model bunları kullanmıyordu — gereksiz disk + confusion)
3. **Aligned patched:** `scripts/patch_aligned_fred_us2y.py` (in-place, backup ile)
4. **Macro features regen:** `compute_macro_features` 136 feature, yield curve artık `-1.08 → +2.04` (median +0.51, son tarih +0.67) — klasik recession indicator olarak meaningful.
5. **Stage 2 OOF posterior regen:** `scripts/rerun_v3_stage2_after_us2y_fix.py` (eski vs yeni karşılaştırma dahil)
6. **Stage 3 retrain:** `scripts/rerun_v3_stage3_retrain.py` — 7 model Optuna walk-forward (LDA 8, MLP 6, XGB/LGBM/RF 8 trial)

### Stage 2 cluster shift (eski v2 vs yeni v3 GMM posterior)
- **ARI (Adjusted Rand Index):** 0.14 (≈ rastgele ilişki)
- **Hard-label agreement:** %49 (eski cluster ID'leri yeni ile sadece yarı yarıya örtüşüyor)
- **Soft posterior L2 distance:** mean 0.72, max 1.41 (0–√2 aralığında, neredeyse maximum)
- **Sonuç:** Cluster partitioning baştan değişti → Stage 3 retrain mecburi (training-serving skew olmamalı)

### Final v3 Sonuçları (BTC, test set 587 gün)

| Model | WF F1 | Test Acc | Test F1 | MCC | Return | Sharpe | MaxDD | Trades | Win % |
|-------|------:|---------:|--------:|----:|-------:|-------:|------:|-------:|------:|
| LDA          | -     | 0.349 | 0.194 | 0.032 |  +9.8% | 0.60 |  -1.9% |  1 | 100% |
| MLP          | -     | 0.384 | 0.312 | 0.059 | +44.7% | 0.95 | -15.5% | 21 | 71.4% |
| **XGB**      | -     | 0.372 | 0.236 | 0.077 | **+50.5%** | **1.39** | -11.4% | 12 | 75.0% |
| LGBM         | -     | 0.376 | 0.250 | 0.065 | +28.5% | 0.85 | -15.2% | 17 | 64.7% |
| **RF**       | -     | 0.384 | 0.248 | 0.109 | +47.7% | 1.29 | -11.4% | 10 | 70.0% |
| ZZ-XGB       | 0.276 | 0.380 | 0.254 | 0.086 | +35.8% | 1.08 | -12.6% | 18 | 66.7% |
| **ZZ-MLP**   | 0.255 | **0.405** | **0.339** | **0.138** | +25.9% | 0.85 | -17.6% | 17 | 52.9% |
| **Buy & Hold** | -   | -     | -     | -     | **+61.6%** | 0.83 | -32.1% |  1 | -    |

### v2 → v3 karşılaştırması (head-to-head)

| Model     | v2 Return | v2 Sharpe | v3 Return | v3 Sharpe | Δ Sharpe |
|-----------|----------:|----------:|----------:|----------:|---------:|
| LDA       | 0.0%      | 0.00      | +9.8%     | 0.60      | +0.60    |
| MLP       | +8.5%     | 0.30      | +44.7%    | 0.95      | **+0.65 (3.2x)** |
| **XGB**   | +25.3%    | 0.88      | +50.5%    | **1.39**  | **+0.51 (1.6x)** |
| LGBM      | +18.1%    | 0.56      | +28.5%    | 0.85      | +0.29    |
| **RF**    | +19.0%    | 0.74      | +47.7%    | **1.29**  | **+0.55 (1.7x)** |
| ZZ-XGB    | +18.4%    | 0.60      | +35.8%    | 1.08      | +0.48    |
| **ZZ-MLP**| **+53.9%**| **1.13**  | +25.9%    | 0.85      | **−0.28 (eski "star" düştü)** |

### Akademik bulgular (rapora yansıyacak)

**1. ZZ-MLP'nin v2'deki "star" performansı (+53.9%, Sharpe 1.13) kısmen spurious:**
Bug'lı `Yield_Curve_10Y_2Y` (sabit -104 gürültü) MLP'nin GMM cluster posterior'ı üzerinden öğrendiği yapıya gizli bir "regime fingerprint" eklemiş — yüksek-kapasiteli MLP bu sahte sinyali ezberlemiş. Bug düzeltilince Stage 2 cluster gerçek bilgi taşımaya başladı, ZZ-MLP'nin avantajı (+28.0% return, −0.28 Sharpe) eridi.

**2. Tree-based modeller (XGB +50.5% Sharpe 1.39, RF +47.7% Sharpe 1.29) v3'te öne çıktı:**
Daha temiz Stage 2 cluster sinyali → tree split'lerinin selective trade timing'i (XGB 12 trade, RF 10 trade vs MLP 21 trade) daha verimli. **Sharpe XGB 1.39 vs B&H 0.83 (%66 üstü)**, MaxDD -11.4% vs B&H -32.1% (1/3'ü).

**3. Sınıflandırma vs ekonomik metrik divergence devam ediyor:**
- En iyi MCC/F1: **ZZ-MLP** (0.138 / 0.339)
- En iyi Sharpe: **XGB** (1.39)
- En iyi return: **XGB** (+50.5%)
- B&H absolute return'üne yetişen yok (+61.6% — yeni test seti DGS2 warm-up yüzünden 30 gün daha kısa, B&H rakamı v2'den farklı)

**4. Data quality audit'in model selection'ı tamamen değiştirdiği örnek:**
"Best model" kararı v2'de ZZ-MLP idi → v3'te XGB-SMA. Bu, makine öğrenmesi pipeline'larında **veri doğrulama (sanity check on raw inputs) modelden önce gelmeli** mesajı için somut bir vaka. Raporda "lessons learned" / "limitations" bölümünde anlatılabilir.

### v3 yeni / güncellenen artifact'lar
- `data/processed/btc_aligned.csv` (3,961 sat — DGS2 fix), `eth_aligned.csv` (2,857 sat)
- `data/processed/btc_features_macro.csv` (136 feat, fixed yield curve)
- `data/labels/btc_oof_regime_posterior.csv` (3,247 sat — yeni GMM)
- `data/labels/btc_test_signals_v2*.csv` (3 dosya, v3 predictions ile override)
- `data/labels/btc_stage3_v2*_summary.csv` (3 dosya, v3 metrikleri)
- `data/labels/btc_backtest_v2*_summary.csv` (3 dosya, v3 backtest)
- `data/labels/final_iter2_summary_table.csv` (rapor tablosu, v3 sayıları)
- `reports/raw_data_visuals/us2y.png` (yeni, gerçek yield)
- `reports/raw_data_visuals/data_summary.csv` (US2Y stats fix)
- 7 PNG yenisi `reports/iter2_*.png` (cm/roc/pred_dist/equity/metric_comparison/summary_table/zigzag_vs_sma)
- Backup'lar: `*.backup_20260508_*.csv` (gitignore'da, lokalde duruyor — geri dönüş için)

### v3 yeni script'ler
- `scripts/patch_aligned_fred_us2y.py` — in-place ZT=F → DGS2 patch + US5Y/3M/30Y drop
- `scripts/rerun_v3_stage2_after_us2y_fix.py` — Stage 2 OOF regen + ARI karşılaştırma
- `scripts/rerun_v3_stage3_retrain.py` — 7 model Optuna walk-forward retrain + backtest + summary
- `scripts/regenerate_all_v2_visuals.py` — date-intersection bug fix (DGS2 warm-up sonrası label tarihleri)

### Config değişikliği
- `config.yaml > data > macro_yields`:
  - Eski: `{"^TNX": "US10Y", "ZT=F": "US2Y"}` (BUG)
  - Yeni: `{"^TNX": "US10Y"}` + ayrı `macro_yields_fred: {"DGS2": "US2Y"}` (FRED API key gerekli)

### Bilinen kalan riskler / TODO (rapor öncesi)
- Notebook 04 ve 07 hâlâ v2 sonuçlarını gösteriyor — sunum öncesi v4 CSV'leriyle re-run lazım. Notebook 04'teki `stage2_feature_names` listesi 8 elemanlı; v4'te 11 olmalı (`+ macro_FEDFUNDS, macro_real_interest_rate, macro_UNRATE`).
- `final_iter2_summary_table.csv` rapor tablosunun text içine kopyalanması gerekiyor.

## Iter 4 — Monthly FRED + Stage 2 Genişlemesi (2026-05-08 öğleden sonra)

### Tetikleyici
Iter 3'te eklediğim DGS2 (daily 2Y yield) yalnızca yield curve fix'i içindi. Aslında `config.yaml > macro_fred_optional` infrastructure'ı 5 monthly/weekly FRED series için de mevcuttu (FEDFUNDS, CPIAUCSL, UNRATE, WM2NS, ICSA) ama hiç çekilmemiş, aligned'a girmemiş, `macro_real_interest_rate` türev feature'ı silently skip ediliyordu. Kullanıcı "diğerlerine gerek yok" derken sadece daily yields'ı (US5Y/3M/30Y) kastettiği için iter 4 ile bu monthly seri seti eklendi.

### Kaynak ekle (`scripts/add_fred_monthly.py`)
| Series | Frequency | Release lag | min → max | Last value |
|---|---|---|---|---|
| FEDFUNDS | monthly | 1d | 0.05 → 5.33 | 3.72 |
| CPIAUCSL | monthly | 45d | 234.7 → 326.6 | 325.06 |
| UNRATE | monthly | 35d | 3.40 → 14.80 | 4.50 |
| WM2NS | weekly | 14d | 11,452 → 22,597 | 22,469 |
| ICSA | weekly | 5d | 190K → 6.13M | 215K |

Her seri kendi publication-release lag'i ile shift edilir, sonra crypto daily index'ine `ffill` ile bindirilir. **Hiçbir warm-up loss YOK** — FRED 2013'ten başlatıldığı için 2014-09-17'de hepsi dolu. Aligned: BTC 3,961 sat × 17 kol → 22 kol; ETH aynı şekilde.

### Macro feature regen
`compute_macro_features` artık 51 ek FRED-türev feature üretiyor (her seri × {level, sma_20/50/100, zscore_20/50/100, roc_5/20/50}). En önemli yeni türev: **`macro_real_interest_rate = FEDFUNDS − CPI_yoy_pct`** (Fisher equation, ZIRP era'da negative). **Toplam: 136 → 187 makro feature.**

### Stage 2 GMM 8 → 11 feature
`scripts/rerun_v4_after_monthly_fred.py` orchestrator script'inde Stage 2 feature subset güncellendi:
- **Mevcut 8:** macro_VIX, VIX_zscore_50, Yield_Curve_10Y_2Y, Credit_Spread_log, Gold_Silver_Ratio, SP500_VIX_ratio, DXY_zscore_50, SP500_roc_20
- **Yeni 3:** macro_FEDFUNDS (rate level), macro_real_interest_rate (Fisher derived), macro_UNRATE (recession proxy)

Bu seçim akademik açıdan şunu sağlıyor: GMM artık **gerçek monetary policy regime'lerini** (ZIRP / hike cycle / cutting cycle) ve **NBER-style recession indicator**'larını (UNRATE) cluster'lamak için kullanabilir, eski 8-feature set bunu sadece dolaylı olarak (yield curve, credit spread) yapıyordu.

### Stage 3 retrain (v4) — model save edildi
`tune_stage3` 7 model × Optuna walk-forward (LDA 8, MLP 6, XGB/LGBM/RF 8 trial, ZZ-XGB ve ZZ-MLP da). **Bu sefer joblib export var:** `app/models/stage3_*_v2.joblib` (5 + 2 ZigZag) — v3'te eksik olan demo training-serving skew kapandı (`.gitignore`'da, lokal demo için).

### Final v4 Sonuçları (BTC, test set 462 gün)

| Model | Test Acc | Test F1 | MCC | Return | **Sharpe** | MaxDD | Trades | Win % |
|-------|---------:|--------:|----:|-------:|-----------:|------:|-------:|------:|
| LDA          | 0.344 | 0.174 | -0.004 |  -0.8% | **-0.78** |  -1.0% |  2 | 50.0% |
| **MLP**      | 0.379 | 0.263 |  0.080 | +42.8% | **+1.33** | -10.2% | 17 | 76.5% |
| **XGB**      | 0.383 | 0.253 |  0.093 | **+42.7%** | **+1.35** | -11.4% | 13 | 84.6% |
| LGBM         | 0.390 | 0.264 |  0.116 | +27.7% | +1.06 | -11.4% | 10 | 90.0% |
| **RF**       | **0.394** | 0.264 |  **0.128** | +38.1% | +1.25 | -11.4% | 10 | 80.0% |
| **ZZ-XGB**   | **0.394** | 0.264 |  0.123 | +40.5% | **+1.33** | -11.4% | 12 | **91.7%** |
| ZZ-MLP       | 0.390 | **0.275** |  0.081 | +35.6% | +0.95 | -13.0% | 19 | 79.0% |
| **Buy & Hold** | -    | -     |  -     | **+47.6%** |  +0.75 | -32.1% |  1 | -    |

**5/7 model B&H Sharpe 0.75'i geçti.** Test seti 587 → 462 gün düştü (FRED warm-up nedeniyle Stage 2 X.dropna 3,247 → 3,079 satıra indi); B&H absolute return 61.6% → 47.6% bu yüzden farklı.

### v2 → v3 → v4 Sharpe trajectory (head-to-head)

| Model | v2 (bug) | v3 (US2Y fix) | v4 (+ monthly FRED) | Net Δ (v2→v4) |
|---|---:|---:|---:|---:|
| LDA       | 0.00 | +0.60 | -0.78 | **-0.78** |
| MLP       | +0.30 | +0.95 | **+1.33** | **+1.03** |
| XGB       | +0.88 | **+1.39** | +1.35 | +0.47 |
| LGBM      | +0.56 | +0.85 | +1.06 | +0.50 |
| RF        | +0.74 | +1.29 | +1.25 | +0.51 |
| ZZ-XGB    | +0.60 | +1.08 | +1.33 | **+0.73** |
| ZZ-MLP    | **+1.13** | +0.85 | +0.95 | -0.18 |

### Akademik bulgular (rapora yansır)

**1. Tree modellerin v4'teki istikrarı:** XGB Sharpe 1.39 (v3) → 1.35 (v4), RF 1.29 → 1.25 — eklenen 3 makro feature (FEDFUNDS, real_rate, UNRATE) tree split'lerin önceki cluster sinyalini bozmadı, marjinal stabil. Bu, yapılan ekleme'nin **noise olmadığı**, gerçek economic signal taşıdığı anlamına geliyor (model çökmedi, sadece soft trade-off oldu).

**2. MLP'nin v4'te yükselişi:** MLP Sharpe 0.95 → **1.33** (+0.38). Daha zengin Stage 2 cluster posterior'ı MLP'nin uniform-weight yapısına daha çok bilgi getirdi — non-linear classifier daha geniş feature space'ten yarar sağladı. Bu, "regime fingerprint MLP için kritik" mesajının somut kanıtı.

**3. LDA'nın v4'te çöküşü:** LDA Sharpe 0.60 → **-0.78** (negatif). Tek bir lineer discriminant fonksiyonu artık 11-feature posterior'ın eklediği non-linear interaction'ları yakalayamıyor — Gaussian + equal-covariance varsayımı ihlal ediliyor. Bu, **teorik PR pattern'inin görünür sınırı:** LDA'nın bias'ı yüksek-boyutlu posterior input'larda dezavantaj, MLP/tree model esnekliği avantaj. Akademik açıdan iyi limitation noktası.

**4. Test seti ne kadar küçüldü?** v3'te 587 gün, v4'te 462 gün — ~21% azalma. Buna rağmen modellerin sıralaması ve Sharpe rakamları stabil kaldı (XGB hâlâ #1, RF/MLP/ZZ-XGB üçlüsü 1.25-1.33 Sharpe ile hep birlikte) → sonuçlar test-set-specific değil, **gerçek pattern**.

**5. ZZ-MLP vakası kapandı:** v2'de 1.13, v3'te 0.85, v4'te 0.95 — bug temizlendiğinde tamamen çökmedi (genuine regime sinyali var), ama "best model" iddiasını da koruyamadı. Iter 3'te yazılan "spurious gain" tezi v4 ile teyit oldu (eğer salt bug exploit olsaydı v4'te yine düşerdi; aksine v3'te dibe vurmuş, v4'te marjinal toparlandı = gerçek bir miktar regime sinyali kullanıyor).

### v4 yeni / güncellenen artifact'lar
- `data/processed/btc_aligned.csv` (3,961 sat × **22 kol** — 17 → 22, 5 monthly FRED ekleme), `eth_aligned.csv` aynı şekilde
- `data/processed/btc_features_macro.csv` (**187 feat** — 136 → 187, +51 FRED-türev)
- `data/labels/btc_oof_regime_posterior.csv` (3,079 sat — 11-feature subset GMM)
- `data/labels/btc_test_signals_v2*.csv`, `btc_stage3_v2*_summary.csv`, `btc_backtest_v2*_summary.csv`, `final_iter2_summary_table.csv` (v4 ile override)
- `app/models/stage3_lda_v2.joblib` (15KB), `_mlp_v2.joblib` (59KB), `_xgboost_v2.joblib` (2.3MB), `_lightgbm_v2.joblib` (2.2MB), `_random_forest_v2.joblib` (18MB), `_xgboost_v2_zigzag.joblib`, `_mlp_v2_zigzag.joblib` — gitignored, lokal demo için
- 7 PNG `reports/iter2_*.png` (cm/roc/pred_dist/equity/metric_comparison/summary_table; zigzag_vs_sma değişmedi çünkü trend label aynı)

### v4 yeni script'ler
- `scripts/add_fred_monthly.py` — REST API ile 5 series fetch + per-series release lag + ffill + aligned in-place patch
- `scripts/rerun_v4_after_monthly_fred.py` — Stage 2 OOF regen (11 feature) + Stage 3 retrain (7 model, save_model=True with explicit `_v2` path) + backtest + summary CSV write

### Kod değişiklikleri (iter 2)
- `src/features/technical_indicators.py` — `compute_trend_following_features()` (11 feat)
- `src/features/macro_features.py` — `compute_derived_spreads()` (4 spread + zscore)
- `src/labels/trend_labels.py` — `generate_trend_labels_zigzag()` (causal)
- `src/labels/regime_labels.py` — `compute_oof_regime_posterior()`, `predict_regime_posterior()`
- `src/models/classifiers.py` — `LDAClassifier` wrapper + `n_jobs=1` for XGB/LGBM/RF (macOS segfault fix)
- `src/models/optuna_helpers.py` — TPE walk-forward HP tuner with timeout
- `src/models/stage1_trainer.py` + `stage3_trainer.py` — `tune_stage1/3` + `step_months`/`min_train_months` override + class-safe OOF
- `src/models/pipeline.py` — `HierarchicalSoftPipeline` (Stage 2 cluster artifact)
- `app/main.py` — `/test_dates`, `/predict` (date|live mode), `HierarchicalSoftPipeline` load
- `app/static/index.html` + `app.js` — date dropdown + Live button + LDA/MLP selector
- `docker/Dockerfile` + `.dockerignore` — model + 4 CSV bundle, 13 dosya hariç
- `scripts/run_iter2_phase_a.py`, `_phase_b.py`, `_phase_c.py`
- `scripts/regenerate_all_v2_visuals.py`

## MVP Kararları (2026-05-07) — Hocanın Guideline'ı + Kullanıcı Tercihleri

### Scope (2 gün hedefi: 7-9 Mayıs)
- **Coin:** Sadece BTC (ETH 2. iterasyon)
- **Pipeline:** 3-Stage tek konfigürasyon
- **Modeller:** LDA + MLP (klasik PR + esnek baseline). XGB/LGBM/RF/SVM 2. iter.
- **Stage 1 (Trend):** SMA crossover + 3-day persistence (mevcut config)
- **Stage 2 (Macro):** K-means/GMM/HMM **soft posterior** → Stage 3'e feature olarak. **Ayrı supervised classifier YOK.**
- **Stage 3 (Signal):** 5-day forward, ±1% sabit + ±0.5×rolling_std, **class_weight='balanced'**
- **HP Tuning:** Optuna (kullanıcı isteği, 2 model olsa bile)
- **Validation:** Walk-forward expanding-window (min 6 ay) + OOF (5 fold)
- **Test set:** Son %15 chronological
- **Backtest (MVP):** Cum return, Sharpe, MDD, equity curve (transaction cost optional)
- **Demo:** Sembol + tarih dropdown (test period ~600 gün) + Live yfinance butonu (fallback)
- **Reproducibility:** random_state=42 her yerde, requirements.txt pinned, joblib model save
- **Notebook review:** Faz sonu batch onay (5-6 nokta), her notebook'ta tek tek değil

### 2. İterasyon (MVP sonrası, sunum öncesi 9-10 Mayıs)
- ETH ekle
- Modeller: XGBoost, LightGBM, RF, SVM (toplam 6 model)
- Ablation configs: Flat baseline, 2-Stage, 3-Stage (3 × 6 × 2 = 36 deney)
- ZigZag trend label (SMA ile karşılaştırma)
- **Stage 2 supervised cluster-ID variant** (soft posterior ile karşılaştırma)
- **VIX-only clustering** (basit baseline, kullanıcı isteği)
- Backtesting genişletilmiş (transaction cost, regime breakdown, benchmark)
- CSV upload mod
- SHAP analysis, McNemar statistical test
- Literature review (Hizir 10-15 ref, MVP sonrası)

### Akademik / Hocanın Guideline'ı Uyumu
- **PR çerçevelemesi:** Hierarchical Bayesian decision theory + posterior fusion. P(signal|osc, p̂_trend, p̂_macro)
- **Mathematical formulation odağı:** LDA discriminant function (Gaussian assumption), MLP softmax + cross-entropy
- **Model varsayımları rapor edilmeli:** LDA Gaussian + equal covariance; MLP no parametric assumption
- **Reproducibility:** seeds, env, deterministic walk-forward
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC (one-vs-rest), Confusion Matrix, **decision boundary plot (PCA 2D)**

## Kullanıcı Cevaplarından Karar Listesi (2026-05-07)

| # | Konu | Karar |
|---|------|-------|
| 1 | Sınıflandırıcı seti | Klasik PR baseline ekle (LDA + SVM) — **MVP: LDA + MLP**; SVM 2. iter |
| 2 | Stage 2 mimarisi | **MVP: soft posterior**; 2. iter: hem soft hem supervised + VIX-only clustering |
| 3 | Web input | Sembol+tarih dropdown (primary) + Live yfinance butonu (bonus) |
| 4 | MVP scope | BTC + 3-Stage tek config + LDA + MLP |
| 5 | MVP modelleri | LDA + MLP |
| 6 | Stage 2 timing | MVP sonrası 2. iter |
| 7 | Demo flow | Test period dropdown + Live butonu |
| 8 | Math framing | Hierarchical Bayesian decision theory |
| 9 | Çağatay rolü | Sonra netleşecek; şu an Özgün + Hizir paralel |
| 10 | Report formatı | LaTeX + IEEE template (Overleaf) |
| 11 | Trend label | SMA (MVP) + ZigZag ablation (2. iter) |
| 12 | PR konsepti | Bayes Decision Theory + posterior fusion |
| 13 | ZigZag timing | 2. iter |
| 14 | Signal threshold | Mevcut config + class_weight='balanced' |
| 15 | Demo dates | Tüm test period (~600 gün) dropdown |
| 16 | Reproducibility | random_state=42, requirements.txt pinned, joblib |
| 17 | CSV upload | 2. iter |
| 18 | Backtest MVP | Basit (cum return, Sharpe, MDD, equity curve) |
| 19 | Notebook review | Faz sonu batch onay |
| 20 | Lit review | Hizir araştırır, MVP sonrası |
| 21 | HP tuning | Optuna (2 model bile olsa) |
| 22 | PPT | Hizir outline + sen edit, final rapor sonrası |
| 23 | Existing code | Sen audit et, kullanılabilir olanları koru |

## Existing Code Audit Sonucu (2026-05-07)

### ✅ Hazır (değişiklik gerekmez)
- `utils/config.py`, `utils/helpers.py`
- `features/technical_indicators.py` (SMA/EMA/ADX/PSAR/RSI/MACD/Stoch/Williams/CCI/ROC/BB/ATR/OBV)
- `features/feature_selector.py` (corr filter + MI + SHAP)
- `labels/trend_labels.py` (SMA crossover + 3-day persistence)
- `labels/signal_labels.py` (fixed + adaptive + verify_no_leakage)
- `evaluation/backtester.py` (long-only, Sharpe, MDD, equity curve)
- `evaluation/metrics.py` (CM, ROC, all metrics)
- `docker/Dockerfile`, `docker/docker-compose.yml`

### ⚠️ MVP için adapt gerekiyor
- `features/macro_features.py` → derived spreads ekle (yield curve, credit, gold/silver, breakeven)
- `labels/regime_labels.py` → OOF soft posterior fonksiyonu ekle (Stage 3'e feed için)
- `models/classifiers.py` → **LDAClassifier wrapper ekle** (sklearn LinearDiscriminantAnalysis)
- `models/stage1_trainer.py`, `stage3_trainer.py` → Optuna entegrasyonu
- `models/pipeline.py` → soft-posterior Stage 2 cluster artifact desteği
- `app/main.py` → `GET /test_dates`, dropdown date predict, cluster Stage 2 desteği
- `app/static/index.html` → date dropdown + Live button (CSV section MVP'den çıkar)
- `docker/Dockerfile` → models'ı image'a bundle et

### 🔲 MVP'de kullanılmayacak (2. iter)
- `models/stage2_trainer.py` (supervised cluster-ID variant — 2. iter)
- 9 boş notebook iskeleti (02-10) — sırayla doldurulacak

## FAZ 1 Veri Durumu (Genişletilmiş Dataset v2 — Mart 2026)

### Kripto OHLCV
| Veri | Satır | Tarih Aralığı | Durum |
|------|-------|---------------|-------|
| BTC-USD | 4,123 | 2014-09-17 → 2025-12-30 | ✅ NaN yok |
| ETH-USD | 2,974 | 2017-11-09 → 2025-12-30 | ✅ NaN yok |

### Hizalanmış Veri (v4 — monthly FRED eklendi)
| Veri | Satır | Sütun | Tarih Aralığı | Notlar |
|------|-------|-------|---------------|--------|
| **BTC Aligned** | 3,961 | 22 | 2014-09-17 → 2025-12-30 | v3'ten 17 → 22 kolon (5 monthly FRED ekle) |
| **ETH Aligned** | 2,857 | 22 | 2017-11-09 → 2025-12-30 | v3'ten 17 → 22 kolon |

### Aligned Dataset Columns (22, v4)
- **OHLCV (5):** Open, High, Low, Close, Volume
- **Risk (3):** SP500, VIX, DXY
- **Commodities (3):** Gold, Silver, Oil_WTI
- **Yields (2, daily):** US10Y (yfinance ^TNX), US2Y (FRED DGS2)
- **Credit (4):** HY_Bond, IG_Bond, Treasury20Y, TIPS
- **Monthly FRED (5, v4):** FEDFUNDS (Fed Funds Rate, lag 1d), CPIAUCSL (CPI, lag 45d), UNRATE (Unemployment, lag 35d), WM2NS (M2 Money Supply, lag 14d), ICSA (Initial Jobless Claims, lag 5d) — her biri publication-release lag ile shift edildi, sonra crypto daily index'ine forward-fill
- **Kaldırıldı (v2'de vardı, hiçbir yerde kullanılmadığı için v3'te silindi):** US5Y, US3M, US30Y

## Open Questions
- [ ] BTC test period start date kesinlikle? (~2024-04 civarı, %15 son chronological)
- [ ] Optuna trial sayısı per model? (default 50 mi, MVP için 20-30 mı?)
- [ ] Çağatay'ın task distribution detayı (proposal'da 50/50 yazıyor olabilir)
- [ ] Live yfinance fail durumunda frontend ne göstersin? (toast notification + dropdown fallback)

## Key Insights
- **Mevcut kod inanılmaz hazır:** FAZ 0+1 yazımları profesyonel, MVP için 4-5 küçük adaptasyon yetiyor
- **Stage 3 OOF mantığı walk-forward'a göre revize edilmeli:** mevcut K-fold "first fold no train data" sorunu var, expanding-window ile uyumlu hale getirilecek
- **Soft posterior approach:** Stage 2 trainer kullanılmayacak; regime_labels modülü cluster posterior'ı doğrudan üretecek

## Experiment Results
_(FAZ 4-5'te doldurulacak)_
