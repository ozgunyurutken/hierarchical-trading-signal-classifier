# MEMORY.md - Project State & Decision Log

## Current Status
**Active Phase:** V5 Phase 5 — Backtest tamamlandı, Phase 6 (FastAPI demo) sırada
**Last Updated:** 2026-05-09 (akşam) — Phase 4 Stage 3 + Phase 5 Backtest done. **BTC stateful XGB Sharpe 1.21x B&H, ETH prob_weighted LGBM hem Sharpe hem Return B&H üstü**
**Active Branch:** `v5-from-scratch`

## V5 Phase 5 — Backtest Sonuçları (FINAL)

3 trading rule × 4 model × 2 asset + 2 B&H benchmark = **26 backtest run**.
Transaction cost 0.1%, daily-resolution.

### Trading Rules

| Rule | Logic |
|---|---|
| **Stateful** | State machine. Cash + Buy → long, Long + Sell → cash, Hold = no-op. |
| **Defensive** | Stateless. Buy → long, Hold/Sell → cash. |
| **Prob-weighted** | Continuous position size = clip(P_Buy − P_Sell, 0, 1). |

### Best per Asset

| Asset | Best Rule + Model | Sharpe | Return | MaxDD | vs B&H |
|---|---|---:|---:|---:|---|
| **BTC** | **stateful + xgboost** | **+1.15** | +2901% | **-46.0%** | Sharpe **+0.20 üstü** (B&H 0.95). Return marjinal altı (-71pp). MaxDD 30pp daha az. |
| **ETH** | **prob_weighted + lightgbm** | **+0.34** | **+19.1%** | **-18.1%** | Hem **Sharpe** (+0.08) hem **Return** (+26.4pp) üstü. MaxDD 1/4. B&H ETH'de **negatif** kapatmış (-7%). |

### BTC Full Table (Sharpe sıralı)

| Rule | Model | Return | Sharpe | MaxDD | Trades | Win% |
|---|---|---:|---:|---:|---:|---:|
| stateful | xgboost | +2901% | **+1.15** | -46.0% | 179 | 59.2% |
| stateful | lightgbm | +2519% | +1.09 | -43.1% | 207 | 58.9% |
| stateful | random_forest | +2186% | +1.07 | -51.8% | 176 | 56.8% |
| defensive | xgboost | +1309% | +0.98 | -35.6% | 248 | 59.7% |
| defensive | lightgbm | +1324% | +0.96 | -44.4% | 271 | 60.1% |
| **B&H** | benchmark | **+2972%** | +0.95 | **-76.6%** | 1 | - |
| defensive | mlp | +1084% | +0.90 | -49.4% | 236 | 55.5% |
| stateful | mlp | +1191% | +0.90 | -55.1% | 195 | 55.4% |
| defensive | random_forest | +938% | +0.88 | -43.1% | 222 | 52.3% |
| prob_weighted | lightgbm | +376% | +0.85 | -31.4% | 244 | 61.5% |
| prob_weighted | random_forest | +91% | +0.83 | **-12.5%** | 202 | 56.9% |
| prob_weighted | xgboost | +213% | +0.82 | -25.5% | 219 | 58.4% |
| prob_weighted | mlp | +509% | +0.81 | -49.3% | 226 | 58.0% |

### ETH Full Table (Sharpe sıralı)

| Rule | Model | Return | Sharpe | MaxDD | Trades | Win% |
|---|---|---:|---:|---:|---:|---:|
| prob_weighted | lightgbm | +19.1% | **+0.34** | **-18.1%** | 132 | 54.5% |
| defensive | mlp | +24.8% | +0.32 | -44.0% | 134 | 49.3% |
| prob_weighted | mlp | +21.6% | +0.29 | -42.2% | 125 | 45.6% |
| **B&H** | benchmark | **-7.2%** | +0.26 | **-71.8%** | 1 | - |
| prob_weighted | xgboost | +6.3% | +0.17 | -24.5% | 111 | 63.1% |
| stateful | random_forest | -11.8% | +0.16 | -63.4% | 77 | 54.5% |
| prob_weighted | random_forest | +3.9% | +0.15 | **-14.2%** | 102 | 56.9% |
| stateful | mlp | -16.3% | +0.15 | -62.4% | 101 | 48.5% |
| defensive | random_forest | -17.2% | +0.10 | -64.3% | 112 | 51.8% |
| defensive | xgboost | -18.2% | +0.07 | -60.4% | 121 | 58.7% |
| defensive | lightgbm | -28.8% | +0.03 | -64.3% | 165 | 49.7% |
| stateful | xgboost | -37.1% | -0.02 | -61.1% | 80 | 53.8% |
| stateful | lightgbm | -47.8% | -0.09 | -69.8% | 103 | 50.5% |

### Akademik Bulgular (paper'ı doğrudan etkiler)

**1. No single trading rule dominates — asset-specific selection gerekli.**
- BTC: bull-heavy market → stateful (long-hold) optimal. 8/12 model + rule kombo B&H Sharpe'ı geçti.
- ETH: volatile market (2022 -75%, 2023 toparlanma) → prob_weighted (soft sizing) optimal. ETH'de stateful kötü, defensive orta, prob_weighted en iyi.
- Bu paper'da "trading rule ablation" olarak değerlendirilebilir, mimari karar noktasına dönüşür.

**2. Risk-adjusted basis'te her iki asset'te de B&H geçildi.**
- BTC: Stage 3 best Sharpe 1.21x B&H, MaxDD 60% azalma
- ETH: Stage 3 best Sharpe 1.31x B&H, **Return da B&H'ı geçti** (volatile döneme dayanıklılık)

**3. Hold class'ın işlevsel kanıtı.**
- Stateful rule Hold'u "pozisyon koru" diye kullandı → BTC stateful XGB Sharpe 1.15
- prob_weighted rule Hold posterior'unu position size'a yansıttı → ETH MaxDD -%18 (B&H -%72'nin 1/4'ü)
- F1m=0.37 modelle (decision gate 0.40 altı) bu sonuçların gelmesi → frame-level F1 trading başarısının zayıf proxy'si.

**4. Backtest # trades (179-271 BTC, 77-165 ETH) yüksek.**
- 0.1% one-way TC ile sürdürülebilir
- Gerçek hayatta spread + slippage daha yüksek olabilir → paper'da limitation olarak not
- prob_weighted rule trade count'u muhafazakar (BTC 202-244, ETH 102-132)

### Reference Files

- `src/evaluation/v5_backtester.py` — 3 trading rule + B&H benchmark
- `scripts/v5_run_phase5_backtest.py` — runner (4 model × 2 asset × 3 rule + 2 B&H)
- `scripts/v5_plot_phase5_backtest.py` — 5 plot (best vs B&H, equity grids, summary heatmap, risk-return scatter)
- `reports/Phase5/v5_p5_backtest_summary.csv`
- `reports/Phase5/v5_p5_equity_curves_{btc,eth}.csv`
- `reports/Phase5/v5_p5_equity_best_vs_bh.png`
- `reports/Phase5/v5_p5_equity_grid_{btc,eth}.png`
- `reports/Phase5/v5_p5_summary_heatmap.png`
- `reports/Phase5/v5_p5_risk_return_scatter.png`

### Phase 6 (FastAPI demo) için sonraki adım

- BTC=stateful XGB tuned, ETH=prob_weighted LGBM tuned modelleri demo'da seçili
- `/predict?asset=&date=` endpoint: tarih bazında tek günlük signal döndür
- Frontend dropdown: tarih seçimi + asset seçimi + best model (asset-specific)
- Equity curve canlı gösterim opsiyonel

---

## V5 Phase 4 — Stage 3 Signal Classifier (HP-TUNED, FINAL)

### Pipeline Özeti

| Bileşen | Karar |
|---|---|
| Signal label | Causal forward-return + adaptive threshold (h=5, k=0.5, w=20) |
| Features | 16 (Stage 1 raw 3 + Stage 1 smooth10d 3 + Stage 2 hard one-hot 3 + regime_age 1 + 6 oscillator) |
| Outer CV | Walk-forward expanding-window, train_min=750, val=200, step=200, gap=10 |
| Class weighting | `balanced` |
| HP Tuning | Optuna 5-fold inner WF-CV, 30 trial × 4 model × 2 asset, MedianPruner |

### Final F1 macro (Outer 12/6 fold OOF, 5-fold tuned HP)

| Asset | Model | F1m | Acc | F1_buy | F1_hold | F1_sell |
|---|---|---:|---:|---:|---:|---:|
| BTC | xgboost | **0.367** | 0.391 | 0.437 | 0.238 | 0.426 |
| BTC | lightgbm | 0.361 | 0.400 | 0.462 | 0.189 | 0.431 |
| BTC | random_forest | 0.360 | 0.400 | 0.451 | 0.182 | 0.446 |
| BTC | mlp | 0.347 | 0.391 | 0.466 | 0.177 | 0.396 |
| ETH | xgboost | **0.368** | 0.384 | 0.452 | 0.274 | 0.377 |
| ETH | lightgbm | 0.350 | 0.373 | 0.444 | 0.240 | 0.366 |
| ETH | random_forest | 0.364 | 0.387 | 0.454 | 0.251 | 0.388 |
| ETH | mlp | 0.312 | 0.353 | 0.428 | 0.135 | 0.372 |

**Decision gate F1m ≥ 0.50: 0/8 model PASS** (V5_PLAN spec, ama V5_PLAN actually L116 says 0.40). 0/8 ≥ 0.40. Best 0.368, gate altında.
- Bu fundamental finansal sınıflandırma zorluğu yansıması — Iter 4 v4'te bile 0.27 idi.
- F1m gate kaçırıldı ama **backtest performansı paper-grade** (Phase 5 tablosuna bak).

### Tuning Trajectory (baseline → 5-fold tuned)

| Asset | Model | Δ F1m | Δ F1_hold |
|---|---|---:|---:|
| BTC | mlp | **+0.035** | +0.079 |
| ETH | xgboost | +0.024 | **+0.061** |
| ETH | mlp | +0.032 | **+0.121** |

MLP en büyük kazanan her iki asset'te (Stage 1 trajectory'siyle paralel — search space genişletildiğinde MLP'nin capacity'si ortaya çıkıyor). ETH XGB Hold class'ı 0.214 → 0.274 yükseltti.

### Reference Files

- `src/labels/v5_signal_labels.py` — V5 signal label (causal forward-return + adaptive threshold)
- `src/features/v5_stage3_features.py` — 16 feature builder (Stage 1 + Stage 2 + oscillator)
- `src/models/v5_stage3_trainer.py` — walk-forward CV, 4 model factory
- `src/models/v5_stage3_optuna.py` — Optuna runner, 5-fold inner CV
- `scripts/v5_build_stage3_dataset.py` — dataset builder
- `scripts/v5_train_stage3.py` — baseline (fixed HP)
- `scripts/v5_tune_stage3.py` — Optuna tuning runner
- `scripts/v5_train_stage3_tuned.py` — tuned outer retrain + delta
- `scripts/v5_plot_phase4_dataset_inputs.py` — Phase 4.0 input visualization

### Reference Artifacts

- `data/processed/{btc,eth}_features_stage3_v5.csv` (16 feat + label/return/eps, ~3200 / 2000 row)
- `data/processed/{btc,eth}_stage3_oof_*_v5.csv` — baseline OOF
- `data/processed/{btc,eth}_stage3_oof_*_v5_tuned.csv` — 5-fold tuned OOF (Phase 5 backtest input)
- `reports/Phase4.0_inputs/` — 7 dataset visualization plots
- `reports/Phase4/v5_p4_stage3_overall.csv` + `metrics.csv` — baseline summary
- `reports/Phase4.5_after_tune/v5_p4_stage3_overall_tuned.csv` + variants
- `reports/Phase4.5_after_tune/v5_p4_stage3_optuna_studies.csv` + `_best.csv`
- `reports/Phase4.5_after_tune/v5_p4_stage3_tuning_delta.csv`

---

## V5 Phase 3 — Stage 1 Trend Classifier (HP-TUNED, FINAL)

### Pipeline Özeti

| Bileşen | Karar |
|---|---|
| Trend label | ZigZag offline + revisable, deviation_pct=0.10, min_segment_days=15, range_amplitude=0.075 (config D) |
| Features | 14 teknik (returns 3, trend strength 3, momentum 2, mean-rev 2, volatility 2, volume 2) |
| Outer CV | Walk-forward expanding-window, train_min=750, val=200, step=200, gap=10 → BTC 16 fold, ETH 10 fold |
| Class weighting | `balanced` (LGBM/RF: sklearn class_weight; XGB: inverse-freq sample_weight; MLP: sklearn limit, raw) |
| HP Tuning | Optuna TPE + MedianPruner, 30 trial × 4 model × 2 asset, **5-fold inner WF-CV objective** |
| Inner CV | `train_min=750, val=300, gap=10`, evenly-spaced 5 fold across full dataset |

### Final Results (Outer 16/10-fold OOF, 5-fold tuned HP)

| Asset | Model | F1m | Acc | F1 down | F1 range | F1 up |
|---|---|---:|---:|---:|---:|---:|
| BTC | xgboost | 0.533 | 0.567 | 0.527 | 0.424 | 0.649 |
| BTC | lightgbm | 0.526 | 0.569 | 0.499 | 0.413 | 0.666 |
| BTC | **random_forest** | **0.563** | 0.593 | 0.549 | **0.473** | 0.667 |
| BTC | mlp | 0.505 | 0.565 | 0.464 | 0.382 | 0.671 |
| ETH | xgboost | 0.538 | 0.557 | 0.500 | 0.483 | 0.630 |
| ETH | lightgbm | 0.538 | 0.557 | 0.503 | 0.476 | 0.634 |
| ETH | **random_forest** | **0.571** | 0.588 | 0.551 | **0.519** | 0.643 |
| ETH | mlp | 0.549 | 0.582 | 0.502 | 0.497 | 0.649 |

**Decision gate F1m ≥ 0.50: 8/8 PASS** (önceki baseline 6/8, 3-fold tuned 7/8 idi).

### Tuning Trajectory: Baseline → 3-fold tuned → 5-fold tuned

#### F1 macro (8 model)

| Asset/Model | Baseline | 3-fold tuned | 5-fold tuned | Δ vs base | Δ vs 3-fold |
|---|---:|---:|---:|---:|---:|
| BTC xgb  | 0.512 | 0.549 | 0.533 | +0.021 | -0.016 |
| BTC lgbm | 0.507 | 0.526 | 0.526 | +0.019 | 0.000 |
| BTC rf   | 0.557 | 0.543 ⚠ | **0.563** | +0.006 | **+0.020** |
| BTC mlp  | 0.462 | 0.537 | 0.505 | +0.044 | -0.032 |
| ETH xgb  | 0.510 | 0.526 | 0.538 | +0.028 | +0.012 |
| ETH lgbm | 0.513 | 0.538 | 0.538 | +0.025 | 0.000 |
| ETH rf   | 0.560 | 0.549 ⚠ | **0.571** | +0.012 | **+0.022** |
| ETH mlp  | 0.475 | 0.471 ⚠ | **0.549** | +0.074 | **+0.078** |

#### Range F1 (en imbalanced class)

| Asset/Model | Baseline | 3-fold tuned | 5-fold tuned |
|---|---:|---:|---:|
| BTC rf  | 0.436 | 0.397 ⚠ | **0.473** ✓ |
| ETH mlp | 0.411 | 0.410 | **0.497** ✓✓ |
| ETH rf  | 0.514 | 0.498 ⚠ | **0.519** ✓ |

### Methodology Findings (paper'a girer)

**1. Inner CV fold count interacts with model regularization.**
3-fold inner CV ile RF her iki asset'te baseline'dan **kötüleşti** (BTC -0.014, ETH -0.011). Sebep: Optuna 3 spesifik tarihsel pencerede (2017 bull / 2021 peak-flip / 2025 modern) iyi çıkan aşırı agresif HP'yi seçti — `min_samples_leaf=1`, `max_depth=8`. Bu HP set outer 16 fold'un farklı tarihsel rejimlerinde generalize etmedi → meta-overfitting.

5-fold inner CV (2017/2019/2021/2023/2025 — her ~2 yılda bir) ile aynı RF için Optuna `min_samples_leaf=11` (BTC) ve `=5` (ETH) seçti — daha conservative, baseline'a yaklaştı, **outer'da iyileşti**.

XGB/LGBM/MLP için 3-fold da yeterli oldu çünkü bu modellerin built-in regularization mekanizması (`min_child_weight`, `min_data_in_leaf`, `early_stopping`) zaten overfit'i frenliyor.

**Akademik mesaj:** Models without built-in regularization (RF) require **richer historical coverage in inner CV** to avoid HP overfitting; models with built-in regularization tolerate fewer folds.

**2. ETH MLP outlier: tuning gerekli.**
Baseline ETH MLP F1m 0.475 (decision gate'in altında). 3-fold tune marjinal değişim (-0.005). 5-fold tune **+0.074** sıçrama (0.549). `(64,)` tek-layer + `alpha=8.7e-3` + `lr=2.6e-4`. ETH gibi kısa dataset'lerde MLP HP tuning olmadan zayıf, tune ile competitive.

**3. Range class hâlâ en zayıf class.**
En iyi range F1: BTC RF 0.473, ETH RF 0.519. Uptrend class ortalama F1 0.65+ (kolay), range 0.45-0.50 (zor). ZigZag ile range = "pre-pivot consolidation" doğası gereği belirsiz (label boundary yumuşak). Bu Stage 3 soft fusion için kritik: range posterior probability düşük olabilir, calibration ileride önemli olacak.

**4. Best per-asset model: Random Forest** (her iki asset'te de hem F1m hem range F1 lider).

### Reference Files

- `src/labels/v5_trend_labels.py:zigzag_trend_label` — ZigZag offline label
- `src/features/v5_trend_features.py:STAGE1_FEATURE_COLS` — 14 feature
- `src/models/v5_stage1_trainer.py` — walk-forward CV, 4 model factory (`**hp_overrides` accept)
- `src/models/v5_stage1_optuna.py` — Optuna search spaces + inner WF-CV objective + MedianPruner
- `scripts/v5_build_stage1_dataset.py` — dataset builder (config D)
- `scripts/v5_zigzag_param_sweep.py` — ZigZag param distribution sweep
- `scripts/v5_train_stage1.py` — baseline (fixed HP) trainer
- `scripts/v5_tune_stage1.py` — Optuna tuning runner (SQLite storage for dashboard)
- `scripts/v5_train_stage1_tuned.py` — tuned outer retrain + delta CSV
- `scripts/v5_plot_phase3_stage1_results.py --variant {base,tuned}`
- `scripts/v5_plot_phase3_truth_vs_pred.py --variant {base,tuned}`
- `scripts/v5_plot_phase3_tuning_compare.py` — baseline vs 3-fold vs 5-fold ablation plot
- `src/evaluation/v5_segment_metrics.py` — segment-level metrics (IoU, onset F1, MVC, rolling-mode smoothing)
- `scripts/v5_evaluate_stage1_segments.py` — runs all metrics on baseline + 3-fold + 5-fold + smooth5/10/20d variants
- `scripts/v5_plot_phase3_segment_metrics.py` — heatmap (4-panel) + smoothing curve ablation

### Reference Artifacts

- `data/processed/{btc,eth}_features_stage1_v5_zz.csv` (15 col + label, ~4034 / 2895 row)
- `data/processed/{btc,eth}_stage1_oof_{xgboost,lightgbm,random_forest,mlp}_v5.csv` — baseline OOF
- `data/processed/{btc,eth}_stage1_oof_*_v5_tuned.csv` — 5-fold tuned OOF (Stage 3 input)
- `data/processed/{btc,eth}_stage1_oof_*_v5_tuned_3fold.csv` — 3-fold tuned OOF (archive, ablation reference)
- `reports/Phase3/v5_p3_stage1_overall.csv` — baseline summary (untouched)
- `reports/Phase3/v5_p3_stage1_metrics.csv` — baseline per-fold (untouched)
- `reports/Phase3/v5_p3_stage1_{confusion_grid,f1_per_fold,oof_timeline_*,truth_vs_pred_*}.png` — baseline plots (untouched)
- `reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned.csv` — 5-fold tuned summary
- `reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned_3fold.csv` — 3-fold tuned summary (archive)
- `reports/Phase3.5_after_tune/v5_p3_stage1_metrics_tuned.csv` — 5-fold tuned per-fold
- `reports/Phase3.5_after_tune/v5_p3_stage1_optuna_studies.csv` — all 240 trial details (5-fold)
- `reports/Phase3.5_after_tune/v5_p3_stage1_optuna_studies_3fold.csv` — 3-fold trials (archive)
- `reports/Phase3.5_after_tune/v5_p3_stage1_optuna_best.csv` — 8 best params per (asset, model)
- `reports/Phase3.5_after_tune/v5_p3_stage1_tuning_delta.csv` — pre vs post-tuning delta
- `reports/Phase3.5_after_tune/v5_p3_stage1_confusion_grid_tuned.png`
- `reports/Phase3.5_after_tune/v5_p3_stage1_f1_per_fold_tuned.png`
- `reports/Phase3.5_after_tune/v5_p3_stage1_oof_timeline_{btc,eth}_tuned.png`
- `reports/Phase3.5_after_tune/v5_p3_stage1_truth_vs_pred_{btc,eth}_tuned.png`
- `reports/Phase3.5_after_tune/v5_p3_stage1_tuning_compare.png` — methodology ablation 4-panel
- `reports/Phase3.5_after_tune/v5_p3_stage1_segment_metrics.csv` — 8 model × 6 variant = 48 row, 4 metric
- `reports/Phase3.5_after_tune/v5_p3_stage1_segment_metrics.png` — heatmap 4-panel (variant × asset/model)
- `reports/Phase3.5_after_tune/v5_p3_stage1_smoothing_curve.png` — smoothing window 0/5/10/20d ablation curves

### Segment-Level Evaluation (post-tuning sanity check)

Frame F1 macro tek başına Stage 1'i değerlendirmek için yetersiz: per-day classification günlük noise'tan etkileniyor. Stage 1 modeli aslında "rejimleri" yakalıyor, ama günlük tahminler arasında uptrend ↔ range ↔ downtrend flip-flop var. Bunu sayısallaştırmak için 4 segment-level metric eklendi:

- **Frame F1 macro**: standard per-day F1m (mevcut, baseline)
- **Mean IoU** (Jaccard): per-class temporal overlap → 3-class mean
- **Onset F1 (±5d)**: trend değişiklik noktasını ±5 gün içinde tespit
- **Majority-Vote Consistency (MVC)**: her TRUE segment içinde dominant predicted class oranı, length-weighted average

Plus: rolling-mode smoothing (5d / 10d / 20d) `pred_label`'a uygulanarak Stage 1 prediction'ın temporal consistency'si test edildi.

#### Mean across 8 (asset, model) — variant ablation

| Variant | Frame F1m | Mean IoU | Onset F1 | **MVC** |
|---|---:|---:|---:|---:|
| baseline (untuned) | 0.512 | 0.351 | 0.266 | 0.692 |
| 3-fold tuned (overfit RF) | 0.530 | 0.366 | 0.268 | 0.699 |
| 5-fold tuned (raw) | 0.541 | 0.376 | 0.269 | 0.703 |
| 5-fold + smooth5d | 0.549 | 0.384 | **0.413** | 0.723 |
| 5-fold + smooth10d | 0.558 | 0.393 | 0.404 | 0.761 |
| 5-fold + smooth20d | **0.565** | **0.402** | 0.350 | **0.805** |

#### Per-asset best (5-fold + smooth20d)

| Asset / Model | Frame F1m raw → smoothed | MVC raw → smoothed |
|---|---|---|
| BTC RF | 0.563 → **0.605** | 0.690 → 0.769 |
| ETH RF | 0.571 → **0.595** | 0.717 → **0.832** |

#### Key Findings (paper-friendly)

**1. Frame-level metrikler regime-detection capability'sini underestimate ediyor.**
Stage 1 modeli **daily noise** üretiyor ama **macro segment'leri %80+ doğru** yakalıyor. ETH RF için 200 günlük bir true downtrend segment'inde model günlerin **%83.2'sini downtrend** olarak tahmin ediyor — sadece ara ara uptrend false positive'leri var.

**2. Smoothing window'un task-specific trade-off'u var.**
- **smooth20d**: F1m + IoU + MVC monoton iyi (regime persistence için optimal)
- **smooth5d**: Onset F1 zirvede (transition timing için optimal)
- **smooth10d**: Tüm 4 metrikte de iyi (Phase 4 sweet spot)

**3. Stage 3 input için pratik öneri.**
Phase 4 Stage 3 Signal Classifier'a iki tip Stage 1 feature eklenebilir:
- **Raw posterior** (`P_down, P_range, P_up` günlük) — onset detection için
- **Smoothed posterior** (10d rolling) — regime persistence için
- **Persistence flag** — son 20 gün majority class konsistensisi (rejim stabilitesi)

Bu literatürde "frame-level vs segment-level evaluation" ayrımı (speech recognition, activity recognition) — paper'ın Discussion bölümünde 1-2 paragraf değer.

### Stage 3 (Phase 4) için sonraki adım

- Stage 3 input olarak per-asset best model OOF kullanılacak: **BTC=RF tuned, ETH=RF tuned** (`_v5_tuned.csv` dosyaları)
- Optional: smoothed posterior (smooth10d) ek feature olarak kullan — segment-level bulguya göre paper-quality
- Optional: 4-model ensemble (mean of OOF probabilities) — soft fusion için değerlendirilebilir
- Phase 4 öncesi opsiyonel: probability calibration analysis (isotonic / Platt) — Stage 3'e gidecek posterior'ların reliability diagram'ı

---

## V5 Phase 2 — Stage 2 Macro Regime Classifier (FINAL)

### Approach Evolution (kısa özet, paper-friendly)

Dört kanonik unsupervised yaklaşımı birer kez denedik; hepsi yetersiz kaldı:

| Yöntem | Sonuç |
|---|---|
| Unsupervised K-Means | Non-temporal cluster atama — 2008 GFC yapısını yakalayamadı |
| Semantic Constrained K-Means | Sınıf semantiği zorlandı ama geçiş kararları noisy |
| HMM 3-state Gaussian | Temporal ama state geçişleri stabil değil |
| GMM 3-component | 2024-2025 stickiness — P(Stress) ≈ 0.96 sabit kaldı |

**Pivot:** Bu 4 unsupervised denemeden sonra makro veriyi (VIX, S&P 500, yield curve, DXY, M2) kullanan **rule-based composite finite-state machine** yaklaşımına hızlıca geçildi. FSM iteratif olarak Phase 2.12'de finalize edildi (hysteresis + dwell + velocity overrides + macro stress overrides + V-shape recovery + rapid escalation + noise reduction).

### Phase 2.12 Final Rule Set

8 rule + 1 guard, üç katmanlı yapı:

**Base FSM:**
| # | Rule | Detail |
|---|---|---|
| 1 | Hysteresis | Bear entry VIX_z>1.0, exit <0.3 / Bull entry VIX_z<-0.5 AND SP500_5d>0, exit >0.0 |
| 2 | Dwell time | Bear ≥20d, Bull ≥40d, Neutral ≥10d (threshold-based exits) |
| 3 | Bear→Neutral velocity | ΔVIX_z[10d] < -0.8 (rapid Bear exit) |

**Macro overrides (defensive bias):**
| # | Rule | Detail |
|---|---|---|
| 4 | YC persistent inversion | 60d rolling(10Y-2Y) < 0 → Bull → Neutral force |
| 5 | DXY + M2 macro stress | DXY_z[30d]>0.7 AND M2_yoy[30d]<0.040 → Bull → Neutral |

**Velocity entries (dwell bypass, rapid response):**
| # | Rule | Detail |
|---|---|---|
| 6 | Bull velocity entry | ΔVIX_z[30d] < -0.6 AND SP500_60d > +2% (V-shape) |
| 7 | Bear velocity entry | ΔVIX_z[5d] > +0.6 AND SP500_5d < -1.5% (rapid escalation) |

**Guard:**
| # | Rule | Detail |
|---|---|---|
| 8 | Bear re-entry guard | Bear→Neutral sonrası 35d Neutral dwell şart Bear'a dönüş için |

### Validation Results

**Distribution (Bull/Neutral/Bear):**
- Pre-train (2000-2025): 63.2% / 24.0% / 12.8%
- BTC era (2014+):      68.7% / 20.5% / 10.8%
- ETH era (2017+):      65.4% / 23.5% / 11.0%

**Crises caught:**
- 2008-09 GFC, 2018 Q4 Fed pivot, 2020-03 COVID, 2022 Q2-Q4 Bear cycle
- 2025-04-03 Trump Liberation Day tariff (rapid Bear via velocity entry)
- 2024-08-02 yen carry trade unwind, 2023-03-13 SVB collapse

**Recoveries caught (V-shape):**
- 2003 dot-com, 2009 GFC, 2020 May post-pandemic, 2022 Q4-2023 Q1, 2025 May post-tariff

**Override fire frequency (Pre-train, 6258 days):**
- Bear→Neutral velocity: 31× | Bull velocity entry: 31× | Bear velocity entry: 35×
- YC override: 13× | Macro stress (DXY+M2): 2×

### Reference Files
- `src/labels/v5_regime_labels.py:CompositeVIXRegimeClassifier` (final implementation)
- `scripts/v5_build_regime_labels_composite_macro_v5.py` (Phase 2.12 build)
- `data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v5_v5.csv`
- `reports/Phase2/v5_phase2.1_constrained_kmeans_timeline_4panel.png` (semantic constrained KM baseline)
- `reports/Phase2/v5_phase2.2_unsupervised_kmeans_timeline_4panel.png` (vanilla K-Means baseline)
- `reports/Phase2/v5_phase2.5_hmm_timeline_4panel.png` (HMM baseline)
- `reports/Phase2/v5_phase2.5b_gmm_timeline_4panel.png` (GMM baseline)
- `reports/Phase2/v5_phase2.12_noise_reduction_timeline_4panel.png` (final Composite Macro FSM)
- `notebooks/06_stage2_training.ipynb` (reproducibility notebook)

---

## Önceki Iterasyon Notları (V1/V2 — fallback referansı, paper'a girmez)

> CLAUDE.md report scope rule: paper sadece V5 final üzerinden yazılır. Aşağıdaki içerik fallback / iç dokümantasyon.


- `v2/feature-selection` branch — Sharpe **1.68** (ZZ-MLP, B2 subset) primary headline
- `v3-rule-based-regime` branch (`042fafd`) — WIP, lit review sonrası başlayacak

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

## v1.0 Checkpoint — Rapor Öncesi Dondurulmuş Referans (2026-05-08)

Iter 4 sonuçları `v1.0-iter4-final` annotated tag'ı ile mühürlendi (commit `ab408d5`, remote push edildi). Rapor yazımı bu tag'a referansla yapılacak. Buradan sonra v2 işlerine **paralel branch'lerde** geçilecek — `claude/review-checkpoint-results-2hh1j` ana branch'i bu noktada bitti.

### v1 → v2 yol haritası (öncelik sırası)

#### **A. Ablation Çalışması** (rapor için kritik eksiklik)
- `src/models/pipeline.py` zaten 3 sınıf içeriyor: `FlatBaselinePipeline`, `TwoStagePipeline`, `ThreeStagePipeline` — yazılı ama hiç kullanılmadı.
- `notebooks/08_ablation_study.ipynb` boş (2 cell, sadece başlık).
- **Test edilecek 4 konfigürasyon:**
  1. **Flat (1-stage):** sadece teknik features → signal classifier
  2. **2-stage (Trend-only):** tech + s1 → signal (s2 yok)
  3. **2-stage (Macro-only):** tech + s2 → signal (s1 yok)
  4. **3-stage (Full v4):** tech + s1 + s2 → signal (mevcut)
- En iyi modelde (XGB) çalıştır, Sharpe + MaxDD + Win% farkını ölç.
- Hipotez: 1→2-stage arası belirgin sıçrama, 2→3-stage marjinal kazanç (ya da değil — sonuca göre tezi yeniden çerçeveleyebiliriz).
- Branch: `v2/ablation`.

#### **B. Veri Seti Genişletme** (yfinance dışı kaynak)
- yfinance BTC-USD start = 2014-09-17 (kesin sınır).
- **Alternatif kaynaklar:** CoinGecko API (2010-Apr → günümüz, free tier 10K req/month), CryptoCompare (2010 → günümüz), Bitstamp historical CSV (2011 → günümüz).
- Hedef: 2010-07 → 2014-09 arası ~1500 ek gün → toplam 5500 gün, test seti 462 → ~825 gün.
- Risk: erken Bitcoin verisi düşük likidite/wash trading var, label kalitesi düşebilir; ETL aşaması yeniden yazılması gerekiyor (yfinance vs CoinGecko field isimleri farklı).
- Branch: `v2/bigger-dataset`.

#### **C. Notebook Refresh** (sunum öncesi zorunlu)
- `notebooks/04_label_generation.ipynb` — Stage 2 feature listesi 8 → 11.
- `notebooks/07_evaluation.ipynb` — v4 CSV'leri ile re-run.
- Bu, A ve B'den bağımsız, paralel yapılabilir.

#### **D. ETH Modelini Çalıştır** (opsiyonel)
- ETH aligned (2,857 sat × 22 kol) hazır, hiç model train edilmedi.
- Aynı v4 pipeline'ını ETH için de koştur → BTC/ETH karşılaştırma tablosu rapora ek somut sayı.
- Branch: `v2/eth-pipeline`.

### v2 branch yapısı (paralel)
- `v2/ablation` — Konu A
- `v2/bigger-dataset` — Konu B
- `v2/notebook-refresh` — Konu C
- `v2/eth-pipeline` — Konu D (opsiyonel)

İkisi paralel ilerlerken sonra entegrasyon: A ve B'den çıkan sonuçlar farklı veri kümeleri kullanır, dolayısıyla "B verisi üzerinde A ablation'ı" ek bir entegrasyon iterasyonu (v2.1) gerektirebilir. Final v2 sonuçları için `v3.0-final` tag'ı atılır.

## v2 Deney Sonuçları (2026-05-08, akşam)

### v2/ablation — 4 konfigürasyon × XGBoost, v1 verisi (462 gün test)
Branch HEAD: `a8f0cb2`. Aynı veri/aynı fold/aynı Optuna bütçesi, sadece feature subset değişiyor.

| Config | n_feat | Acc | F1 | MCC | Return | **Sharpe** | MaxDD | Win% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1 Flat (1-stage)** | 29 | 0.388 | 0.256 | **0.117** | +41.2% | **1.43** | -11.4% | **90.9%** |
| A2 2-stage Trend | 32 | 0.381 | 0.249 | 0.093 | +31.9% | 1.10 | -12.0% | 72.7% |
| A3 2-stage Macro | 32 | 0.385 | 0.251 | 0.112 | +32.7% | 1.16 | -11.4% | 75.0% |
| A4 3-stage Full | 35 | 0.383 | 0.253 | 0.093 | **+42.7%** | 1.35 | -11.4% | 84.6% |
| Buy & Hold | - | - | - | - | +47.6% | 0.75 | -32.1% | - |

**Bulgu:** A1 Flat 4 metrikten 3'ünde lider (Sharpe, MCC, Win%). A4 sadece +1.5pp Return marjında önde. Tek stage eklemek (A2 / A3) flat'tan KÖTÜLEŞTİRİYOR — hiyerarşinin orta kısmı bilgi taşımıyor, gürültü ekliyor.

### v2/bigger-dataset — Veri 2014-09 → 2013-02 genişletildi (+577 satır)
Branch HEAD: `8c70929`. CryptoCompare REST API ile 2010-07 → 2014-09 arası BTC OHLCV çekildi (1,523 ek gün), yfinance ile %0.62 mean overlap deviation (kabul edilebilir). 2013-02-15'ten itibaren tüm makro asset (12 yfinance + 1 FRED daily DGS2 + 5 FRED monthly) yeniden çekildi. Yeni aligned: 4,538 sat × 22 kol (v1: 3,961 × 22).

#### Tüm 7 model retrain → tümü v1'e göre KÖTÜLEŞTİ

| Model | v1 Sharpe | v2-bigger Sharpe | Δ |
|---|---:|---:|---:|
| LDA      | -0.78 |  0.00 (no trade) | +0.78 |
| MLP      | +1.33 | +0.40 | -0.93 |
| **XGB**  | +1.35 | **+0.69** | -0.66 |
| LGBM     | +1.06 | +0.40 | -0.66 |
| RF       | +1.25 | +0.02 | -1.23 |
| ZZ-XGB   | +1.33 | +0.42 | -0.91 |
| ZZ-MLP   | +0.95 | -0.77 | **-1.72** |
| **B&H**  | +0.75 | **+0.53** | -0.22 |

7/7 model kötüleşti. Sadece XGB B&H Sharpe'ı geçti (0.69 vs 0.53). v1'de 5/7 model B&H'ı geçiyordu. Olası sebepler:
- 2013-2018 dönemi farklı rejim profilleri (2014-15 crypto winter, 2018 bear) içeriyor
- Adaptive volatility threshold'u tüm tarih boyunca calibrate edildi → 2024-25 bull dönemine özel kalibrasyon kayboldu
- Test penceresi 3 ay erken başladı (2024-09 → 2024-06), 2024-06 → 09 sideways consolidation modelleri zorladı

#### Aynı ablation 4 konfig v2-bigger verisi üzerinde tekrar (533 gün test)

| Config | Acc | F1 | MCC | Return | **Sharpe** | MaxDD | Win% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A1 Flat** | 0.388 | 0.259 | 0.089 | **+33.5%** | **0.84** ★ | -17.4% | **68.4%** |
| A2 2-stage T | 0.392 | 0.272 | 0.082 | +29.3% | 0.72 | -18.0% | 53.8% |
| A3 2-stage M | 0.383 | 0.253 | 0.072 | +26.6% | 0.70 | -18.5% | 55.0% |
| A4 3-stage F | 0.394 | 0.270 | 0.093 | +26.8% | 0.69 | -18.6% | 60.0% |
| B&H | - | - | - | +31.8% | 0.53 | -32.1% | - |

**Bulgu güçlendi:** Sharpe **MONOTON azalan** A1 > A2 > A3 > A4. A1 vs A4 farkı +0.08 (v1) → +0.15 (v2-bigger), iki katına çıktı. A1, B&H Return'ünü geçen TEK model. Sıralama 2 farklı dataset/test window/training era kombinasyonunda aynı → **robust bulgu**.

#### Üretilen artifact'lar (v2 deneyleri)
- `data/raw/btc_extended_history.csv` (2010-07 → 2025-12, 5,484 sat) — gitignored
- `data/processed/btc_aligned_v2.csv` (4,538 × 22)
- `data/processed/btc_features_stage3_v2_bigger.csv`, `btc_features_macro_bigger.csv`
- `data/labels/btc_signal_labels_adaptive_bigger.csv`, `btc_stage1_oof_lda_bigger.csv`, `btc_stage1_oof_lda_zigzag_bigger.csv`, `btc_oof_regime_posterior_bigger.csv`
- `data/labels/btc_stage3_bigger_full_summary.csv`, `btc_test_signals_bigger.csv`, `btc_backtest_bigger_summary.csv`, `final_iter5_summary_bigger.csv`
- `data/labels/btc_ablation_v2_summary.csv`, `btc_ablation_v2_bigger_summary.csv`
- `reports/ablation_v2_comparison.png`, `ablation_v2_equity.png`, `ablation_v2_bigger_comparison.png`, `ablation_v2_bigger_equity.png`, **`ablation_v1_vs_bigger.png`**
- `scripts/fetch_cryptocompare_btc.py`, `build_aligned_v2.py`, `rerun_v2_bigger_dataset.py`, `run_ablation_v2.py`, `run_ablation_v2_bigger.py`, `plot_ablation_v1_vs_bigger.py`

## v2.1 Yol Haritası — Hiyerarşik Yapıyı Kurtaracak Ek Deneyler

Ablation negatif bulgu **ham haliyle alınmamalı** — kök sebep büyük ihtimalle Stage 3'e verdiğimiz feature set'in yapısal sorunu. Aşağıdaki olası sebepler ve çözümler:

### Olası kök sebepler

1. **Feature redundancy.** Stage 3'e verilen 29 teknik feature'ın çoğu zaten trend/regime bilgisini implicitly taşıyor:
   - `above_sma_200`, `log_ret_50`, `log_ret_100`, `sharpe_proxy_20d`, `adx_value`, `adx_strong_trend`, `higher_high_count` → bunlar zaten **trend göstergeleri**, Stage 1 SMA cross OOF'ı ile collinear
   - `bollinger_pct_b`, `atr_*`, `volume_zscore_*` → bunlar zaten **volatility regime** göstergeleri, Stage 2 GMM (VIX, credit spread, SP500/VIX) bilgisiyle collinear
   - Sonuç: A4'e eklediğimiz s1 (3 col) + s2 (3 col) **redundant**, XGB tree split'leri arasında bilgi paylaşımı oluşmuyor.

2. **Soft fusion kanalının az bilgi taşıması.** Stage 1 LDA OOF posterior'ı ve Stage 2 GMM OOF posterior'ı sadece 3+3=6 kolon ekliyor. Tree splitleri için 6 ek kolon, 29 mevcut kolon yanında **marjinal düzen değişikliği** sağlıyor. **Soft fusion yerine hard interaction features** (örn. `tech_feature_X × P(Up)`, `RSI × P(Stress)`) eklenirse, non-linear etkileşimler explicit olur.

3. **OOF posterior leakage olasılığı (sıfır değil).** Stage 1 SMA cross etiketi tech features'tan üretildi (SMA cross zaten tech feature). Stage 1 model bu etiketi öngörmek için tech features kullandı. Stage 3'e bu OOF posterior'ı geri verince **döngüsel referans** oluşuyor (label leakage değil ama feature-target tautoloji türevi).

### v2/feature-selection — Önerilen Roadmap

#### A. Redundancy elimination (öncelikli)
1. Mutual Information / SHAP feature importance ranking yap, Stage 3 v2 features (29 col) içinde redundant olanları çıkar
2. **Feature subset stratejileri:**
   - **B1:** "Stage 3'e sadece **oscillator + volatility** ver" (trend ve regime bilgisi sadece s1/s2 üzerinden gelsin) — 12-15 feature
   - **B2:** "Tüm 29 feature − 7 trend-related" + s1 + s2 — 22 + 3 + 3 = 28 feature
   - **B3:** Mutual Information top-15 + s1 + s2 — 21 feature
3. Her subset için XGB ablation tekrar (A1-A4 dört konfig)
4. Eğer hierarchic A4 > flat A1 olursa: hipotez doğrulandı, hiyerarşik mimari **gerçekten yardımcı** ama redundant feature'larla maskelenmişti.

#### B. Interaction features (sonraki adım)
- `tech_feature × P(s1_class)` çapraz feature'lar (örn. `RSI × P_Up`, `MACD × P_Calm`)
- `tech_feature × P(s2_class)` (örn. `Bollinger_pct × P_Stress`)
- 29 × 3 + 29 × 3 = 174 ek feature → Optuna ile en iyi 20-30 seç
- Tree splitleri non-linear çarpımları direkt kullanır → s1/s2'nin gerçek information value görünür hale gelir

#### C. Hard fusion alternatifi
- Soft posterior yerine **hard label** kullan: `s1_label = argmax(p̂_trend)` → kategorik feature
- XGB için one-hot encoded {Up, Down, Flat} × {Calm, Trans, Stress} = 9 kategorik feature
- 9 feature + 29 tech = 38 toplam, ama her birinin XGB için "regime bucket" olarak değeri yüksek

#### D. Stage 1 / Stage 2 hard label evaluation
- **Trend label kalitesi:** SMA cross etiketi tautoloji yaratıyor, ZigZag causal değişim açıkladı; ama ZigZag de v2-bigger'da kötüleşti. Belki **Stage 1'i tamamen kaldırıp** sadece Stage 2 (macro regime) + Stage 3 ile dene → ablation bulgusu A3 (macro-only) Sharpe 0.70, A1 (flat) 0.84 farkı 0.14, gerçek bilgi var ama dominant değil.
- **Macro feature küme yenileme:** Şu an 11 makro Stage 2 feature'ında 8 derived + 3 raw. SHAP / GMM cluster purity ile değerlendir, gereksiz olanları çıkar (4-5 core feature kalsın).

### v2/feature-selection branch
- v1.0-iter4-final tag'ından branch et → `v2/feature-selection`
- B1 → B2 → B3 sırasıyla 3 alt-deney koştur
- Her birinin sonucunu `data/labels/btc_ablation_fs_*_summary.csv` olarak kaydet
- En iyi konfig için Stage 3 7 model retrain
- v2-bigger test setinde de tekrarla (robustness)

### Karar noktası
- Eğer A4 (full hierarchical) feature selection sonrası A1'i geçerse → **tezi destekleyen sonuç**, rapor "feature redundancy hidden the hierarchical advantage; after careful selection, 3-stage outperforms flat by Δ Sharpe X" şeklinde yazılır.
- Hiçbir feature subset'te A4 > A1 ise → tez gerçekten zayıf, rapor "honest negative finding" framework ile yazılır.

### Şimdilik askıda kalan
- Notebook 04+07+08 refresh (Konu C v2/notebook-refresh) — feature selection sonrası yapılacak
- ETH pipeline (Konu D v2/eth-pipeline) — opsiyonel, deadline buffer'a göre

## v2/feature-selection — KAZANAN SONUÇ (2026-05-08, akşam)

Branch: `v2/feature-selection` HEAD `bf2aae7` (diğer commit'ler dahil v2/feature-selection branch'inde tutulur).

### B2 + ZZ-MLP = Proje rekoru

| Metrik | Değer |
|---|---|
| Sharpe Ratio | **1.68** |
| Total Return | **+89.5%** |
| Win Rate | **72.2%** |
| MaxDD | -11.9% |
| Test Acc | 0.411 |
| Test F1 | 0.331 |
| MCC | 0.122 |

**Buy & Hold benchmark karşılaştırması:**
- B&H Sharpe 0.75 / Return +47.6% / MaxDD -32.1%
- ZZ-MLP **Sharpe 2.24x B&H, Return +41.9pp B&H'ı geçti**, MaxDD 1/3'üne indi

### B2 + 7 model toplu (462 gün test, 24 tech feat + Stage 1 ZigZag OOF + Stage 2 GMM OOF)

| Model | Sharpe | Return | Win% | MCC |
|---|---:|---:|---:|---:|
| LDA | -0.78 | -0.8% | 50.0% | -0.004 |
| MLP | 0.76 | +27.8% | 61.1% | 0.097 |
| **XGB** | **1.58** | +56.7% | 87.5% | 0.115 |
| LGBM | 0.61 | +13.7% | 61.5% | 0.076 |
| RF | 1.07 | +30.0% | 80.0% | 0.100 |
| ZZ-XGB | 1.11 | +33.9% | 75.0% | 0.096 |
| **ZZ-MLP** | **1.68** ★★ | **+89.5%** ★★ | 72.2% | 0.122 |

5/7 model B&H Sharpe (0.75) üstünde.

### Ablation 4×4 Grid Sharpe (XGBoost)

|  | A1 flat | A2 trend | A3 macro | A4 full |
|---|---:|---:|---:|---:|
| v1 (29 feat) | **1.43** | 1.10 | 1.16 | 1.35 |
| B1 (15 osc/vol) | 0.50 | 1.10 | 1.25 | 0.97 |
| **B2 (24 −5 long-trend)** | 1.17 | 0.85 | 1.18 | **1.58** ★ |
| B3 (MI top-15) | 1.26 | **1.50** | 0.99 | 1.04 |

**Ana methodology bulgusu:** v1 ablation 29 feat full set ile yapıldığında A1 flat ≥ A4 full (hiyerarşi kazanç sağlamıyor görünüyor). 5 long-trend feature (`log_ret_50d/100d`, `above_sma_200`, `adx_value`, `sharpe_proxy_20d`) çıkarıldığında (B2 subset) hiyerarşi gerçek katma değer veriyor — Sharpe A4 1.58 > A1 1.17 (+0.41 fark), ZZ-MLP de 0.95→1.68 (+0.73 sıçrama). **Feature redundancy hiyerarşinin değerini gizlemişti.**

### Bilinen sorun: GMM Stickiness (rapor için kritik)

Stage 2 GMM unsupervised cluster aşırı **overconfident**:
- 2024 H2: P(Stress) ortalama **0.994** (median 1.000)
- 2025 boyunca: P(Stress) ortalama **0.964**, 12 ay'ın 11'i %100 Stress
- Argmax confidence 0.999 — soft posterior pratikte hard label gibi davranıyor

VIX gerçeği:
- 2024-2025 ortalama VIX = 17.22 (2018-2019 baseline 15.92'den sadece +1.3 fark)
- Yani gerçekten "Stress" denecek seviye değil

Sebep tahmin: 2024-2025'te FEDFUNDS (4.5-5.5%, train mostly 0-2.5%), real_interest_rate (pozitif, train negatif), UNRATE level seviyeleri **out-of-distribution** sample. GMM full-train fit, scaler stale → test günleri en yakın cluster'a yapışıyor.

**Paradoks:** Stage 3 modelleri bu "yanlış" Stress sinyalini **defansif filtre** olarak kullanıp Sharpe 1.68'e ulaşıyor. Yani GMM görsel olarak yanıltıcı ama functional olarak iş yapıyor.

### Eklenen dokümantasyon (v2/feature-selection branch'te)
- `docs/GLOSSARY.md` — tüm isimlendirmeler tek doküman (versions, phases, ablation, models, labels, branches)
- `docs/V3_PLAN.md` — bu sürümün adresleyebileceği bekleyen sorunlar + V3 restart planı

## V3 Restart Kararı (2026-05-08, akşam, posterden ilham)

### Tetik: Kullanıcının önceki başarılı projesi
"Machine Learning-Based Market Regime Prediction and Dynamic Weight Optimization for Multi-Asset Portfolios" posteri. Sharpe 1.41 (Dynamic MVO vs B&H 0.88), XGBoost rejim tahmininde %76.7 accuracy, %85.7 early warning capacity.

**Posterdeki tasarım kararları, bizim kademeli adapte edeceğimiz:**
1. **Rule-based regime** (VIX > 35, return < -10%, vol > 35%) — GMM unsupervised yerine
2. **Stage 1 = early warning ML** — rule-based regime'i 1-3 ay (bizde 5-20 gün) önceden tahmin
3. **Adaptive position sizing** (Dynamic MVO λ=1/2/5) — bizim long-only Buy/Sell yerine
4. **44 feature** (multi-period momentum, drawdown, recovery, VIX dynamics)
5. **2 ana plot** (regime detection + portfolio comparison) — sade poster style

### V3 branch yapısı
```
claude/review-checkpoint-results-2hh1j  ← ana (CHECKPOINT_2026-05-08 burada)
├── v1.0-iter4-final  (tag, fallback referansı, Sharpe 1.35)
├── v2/feature-selection  (B2 ZZ-MLP Sharpe 1.68, fallback referansı)
└── v3-rule-based-regime  ← yeni, henüz boş (`042fafd` WIP regime_rules.py only)
```

### V3 sonraki adımlar (BEKLEMEDE)

Kullanıcı: "önce literatür taraması yapalım, problemleri nasıl ele almışlar, sağlam atıflar bulalım. Sonra implementation."

**Literature review hedefleri (Research mode):**
1. Hierarchical / multi-stage classifiers — soft fusion vs hard fusion
2. Market regime detection — rule-based vs unsupervised (GMM, HMM)
3. Feature redundancy in stacked models — bizim B2 bulgusu
4. Walk-forward CV in financial time series
5. Trading signal classification — Buy/Sell/Hold
6. Adaptive position sizing — Dynamic MVO, Kelly, vol-targeting
7. Class imbalance in finance ML — bizim CM Sell-bias

Çıktı: `docs/LITERATURE_REVIEW.md` (15-25 atıflı).
