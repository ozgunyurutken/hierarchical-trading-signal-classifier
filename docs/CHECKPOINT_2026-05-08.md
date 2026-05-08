# Checkpoint — 2026-05-08 Akşam

> Tek-pasta proje durumu. Yeni bir oturum açıldığında önce bu dosya okunur.

---

## 1. Tek satırda durum

Best result: **B2 + ZZ-MLP, Sharpe 1.68, Return +89.5%, Win 72.2%** (462 gün BTC test seti, B&H +47.6% / Sharpe 0.75'i absolute + risk-adjusted geçti). v3-rule-based-regime restart başlatıldı, **literatür taraması safhasında**, training henüz başlamadı.

## 2. Branch tablosu

| Branch / tag | HEAD | Ne içeriyor | Durum |
|---|---|---|---|
| `v1.0-iter4-final` (tag) | `ab408d5` | v4 final (Sharpe 1.35 XGB), monthly FRED dahil 22 col aligned, GMM 11 feat | **Dondurulmuş, fallback** |
| `v2/feature-selection` | `bf2aae7` | B2 24-feat subset, 7-model retrain, ZZ-MLP **Sharpe 1.68**, ablation grid 4×4 | **Kazanan, fallback (rapor headline buradan)** |
| `v2/ablation` | `a8f0cb2` | İlk ablation deneyi (29 feat full set), A1 flat 1.43 vs A4 1.35 | Tarihsel, korunur |
| `v2/bigger-dataset` | `8c70929` | CryptoCompare ile 2013-02 başlangıç (+577 satır), 533 gün test, **kötüleşti** (XGB 0.69) | Negative finding, "limitation" olarak rapora |
| `v3-rule-based-regime` | `042fafd` | WIP — `regime_rules.py` modülü taslağı; lit review beklemede | **Aktif, henüz training yok** |
| `claude/review-checkpoint-results-2hh1j` (ana) | bu commit | Doc'lar + MEMORY + CHECKPOINT, B2 referansı | Ana doc branch |

## 3. Ana sonuçlar (rapora gidecek tablo)

### B2 + 7 model (462 gün BTC test, v4 verisi, B2 24-feat tech subset)

| Model | Test Acc | Test F1 | MCC | Return | Sharpe | MaxDD | Win% |
|---|---:|---:|---:|---:|---:|---:|---:|
| LDA | 0.344 | 0.174 | -0.004 | -0.8% | -0.78 | -1.0% | 50.0% |
| MLP | 0.403 | 0.291 | 0.097 | +27.8% | 0.76 | -12.8% | 61.1% |
| **XGB** | 0.392 | 0.263 | 0.115 | +56.7% | **1.58** | -11.4% | **87.5%** |
| LGBM | 0.375 | 0.247 | 0.076 | +13.7% | 0.61 | -15.2% | 61.5% |
| RF | 0.381 | 0.247 | 0.100 | +30.0% | 1.07 | -11.4% | 80.0% |
| ZZ-XGB | 0.383 | 0.253 | 0.096 | +33.9% | 1.11 | -11.4% | 75.0% |
| **ZZ-MLP** | **0.411** | **0.331** | **0.122** | **+89.5%** ★ | **1.68** ★ | -11.9% | 72.2% |
| Buy & Hold | - | - | - | +47.6% | 0.75 | -32.1% | - |

### Ablation (XGBoost, 4 config × 4 tech subset, Sharpe matrix)

|  | A1 flat | A2 trend | A3 macro | A4 full |
|---|---:|---:|---:|---:|
| v1 (29 feat) | **1.43** | 1.10 | 1.16 | 1.35 |
| B1 (15) | 0.50 | 1.10 | 1.25 | 0.97 |
| **B2 (24)** | 1.17 | 0.85 | 1.18 | **1.58** ★ |
| B3 (15 MI) | 1.26 | **1.50** | 0.99 | 1.04 |

→ Hiyerarşi (A4) sadece B2 subset'te (5 long-trend feat çıkarılmış) net üstünlük gösteriyor. **Methodology katkısı:** "soft fusion in hierarchical classifiers requires careful feature deduplication".

## 4. Bilinen problemler (V3'te ele alınacak)

| # | Problem | V3 çözümü |
|---|---|---|
| 1 | **GMM stickiness** (2025 boyunca P(Stress)=0.99, OOD makro feature'lar) | Rule-based regime detection (VIX/vol/return threshold) |
| 2 | CM Sell-bias (LDA 0 trade, XGB %80+ Sell pred) | Adaptive position sizing (Sell ≠ short, %50/0 weight) |
| 3 | Naming karışık (v1/B2/A4/Phase A) | Tek isimlendirme: Low/Medium/High Risk |
| 4 | Raw FRED viz eksik | 5 yeni raw plot |
| 5 | Feature dökümantasyonu yok | docs/FEATURES.md |
| 6 | Notebook 04/07/08 stale | V3 retrain sonrası re-run |
| 7 | ETH modellenmedi | V3 sonrası opsiyonel |
| 8 | 13 plot karmaşası | V3'te 4 ana plot |
| 9 | Joblib stale (B2 retrain'de save_model=False) | V3 save_model=True |
| 10 | v2-bigger negative finding kullanılmıyor | "Data extension limitation" rapor bölümü |

## 5. V3 plan özeti (`docs/V3_PLAN.md` detayda)

**Posterdeki başarılı tasarımı (Sharpe 1.41 Dynamic MVO) BTC günlük seriye uyarla:**
- Rule-based regime (Low/Med/High Risk via VIX + vol + return threshold + persistence filter)
- Stage 1 = early warning (regime'i 5-20 gün önceden tahmin)
- 32 feature (multi-period mom, drawdown, recovery, VIX dynamics)
- Adaptive position sizing (Low=100%, Med=50%, High=0%)

4-faz icra: Foundations → ML retrain → Visuals → Notebook + decision (12-15h toplam).

## 6. Şu anki adım: Literatür Taraması (Research mode)

Kullanıcı kararı: "training'e başlamadan önce **literatür taraması yapalım**, problemleri nasıl ele almışlar, sağlam atıflar bulalım."

**Hedef:** `docs/LITERATURE_REVIEW.md` (15-25 atıflı, IEEE-style).

**Konular:**
1. Hierarchical / multi-stage classifiers (soft fusion, OOF posterior, stacking)
2. Market regime detection (rule-based vs unsupervised — GMM/HMM)
3. Feature redundancy in stacked / hierarchical models
4. Walk-forward CV in financial time series
5. Trading signal classification (Buy/Sell/Hold, 3-class)
6. Adaptive position sizing (Dynamic MVO, Kelly, vol-targeting)
7. Class imbalance in finance ML
8. Cryptocurrency-specific ML literature (recent, 2020+)

## 7. Yeniden başlama protokolü (yeni Claude oturumu için)

1. **Bu dosyayı oku**: `cat docs/CHECKPOINT_2026-05-08.md`
2. `MEMORY.md` son bölümlerini gözden geçir (V3 Restart Kararı)
3. `docs/V3_PLAN.md` planı kontrol et
4. `docs/GLOSSARY.md` isimlendirme rehberi
5. Aktif branch: `git checkout v3-rule-based-regime` (henüz boş)
6. Bekleyen iş: Literature review → docs/LITERATURE_REVIEW.md
7. Lit review onaylandıktan sonra V3 Faz 1 başlayacak

## 8. Tüm artefakt envanteri (özet)

### Veri
- `data/processed/btc_aligned.csv` (3,961 × 22, v4 monthly FRED dahil)
- `data/processed/eth_aligned.csv` (2,857 × 22)
- `data/processed/btc_features_macro.csv` (187 macro feat)
- `data/processed/btc_features_stage3_v2.csv` (29 tech feat)

### Etiket / OOF (v2/feature-selection branch'inde B2 türevleri)
- `data/labels/btc_signal_labels_adaptive.csv` (Buy/Sell/Hold, ATR-adaptive)
- `data/labels/btc_stage1_oof_lda.csv` (SMA cross trend OOF)
- `data/labels/btc_stage1_oof_lda_zigzag.csv` (ZigZag trend OOF)
- `data/labels/btc_oof_regime_posterior.csv` (Stage 2 GMM 11-feat OOF)
- `data/labels/btc_test_signals_b2*.csv` (B2 final test predictions)
- `data/labels/final_iter6_b2_summary.csv` (B2 7-model summary)

### Model artefakt
- `app/models/stage3_*_v2.joblib` (v4 retrain'den, B2 değil — eskimiş)
- `app/models/pipeline_*` (v1 hierarchical pipeline)

### Plotlar (v2/feature-selection'da temiz hâli)
- `pipeline_v4.png` (mimari)
- `monthly_fred_overview.png` (Stage 2 cluster + makro, 2018+)
- `ablation_fs_b1/b2/b3_comparison.png` + `ablation_fs_combined.png`
- `b2_summary_table/metric_bars/equity_curves/cm_grid/pred_distribution/roc_grid.png`
- `v1_vs_b2_models.png`
- `raw_data_visuals/` (10 ham veri plot, 5 monthly FRED ekleme bekliyor)

### Dokümantasyon
- `MEMORY.md` (gelişim günlüğü, 700+ satır)
- `CLAUDE.md` (AI assistant kuralları, report-writing gate dahil)
- `docs/V2_PLAN.md` (v2 yol haritası, B2 sonuçları)
- `docs/V3_PLAN.md` (V3 restart planı, 16 bölüm)
- `docs/GLOSSARY.md` (isimlendirme rehberi)
- `docs/CHECKPOINT_2026-05-08.md` (bu dosya)

---

_Bu dosya `claude/review-checkpoint-results-2hh1j` ana branch'te tutulur. Her büyük ilerleme adımında güncellenir veya yeni tarihli checkpoint dosyası eklenir._
