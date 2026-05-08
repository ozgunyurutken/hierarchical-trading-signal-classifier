# V3 Restart Planı — Rule-Based Regime + Adaptive Position Sizing

> **Tetik:** Posterdeki başarılı tasarım (Multi-Asset Portfolio Optimization, Sharpe 1.41) ve B2 deneyinde ortaya çıkan GMM stickiness/overconfidence problemi. Posterdeki yaklaşımı bizim tek-asset günlük setine uyarlamak ve **bekleyen tüm sorunları aynı restart'ta** çözmek.

---

## 0. Strateji ve Fallback

v3, v1.0-iter4-final tag'ından (`ab408d5`) yepyeni bir branch'te (`v3-rule-based-regime`) inşa edilir. **v1 (Sharpe 1.35) ve B2 (Sharpe 1.68) referansları korunur** — v3 bu rakamları geçemezse, rapor B2 üzerinden yazılır.

**Başarı kriteri:**
- Hard target: v3 best Sharpe ≥ B2 (1.68) **VE** B&H absolute return'ü geç (≥ +%47.6)
- Soft target: v3 best Sharpe ≥ v1 (1.35), GMM stickiness olmadan, sade rapor

**Fallback eşiği:** v3 Sharpe < 1.20 → v3'ü "deneysel ek bölüm" olarak rapora at, B2'yi headline yap.

---

## 1. Mevcut Tüm Bekleyen Sorunlar — Çözüm Haritası

| # | Problem | v3 çözümü | Sorumlu adım |
|---|---|---|---|
| 1 | **Stage 2 GMM stickiness** (2025 boyunca %97 Stress) | Rule-based regime detection (VIX/vol/drawdown threshold'ları), unsupervised yok | §3 — Rule-Based Regime |
| 2 | **CM Sell-bias** (LDA hep Sell, XGB %80+ Sell tahmin) | Adaptive position sizing (Sell ≠ short, sadece pozisyon küçült) | §5 — Adaptive Sizing |
| 3 | **Naming karışıklığı** (v1/B2/A4/Phase A) | Tek isimlendirme: Low/Med/High Risk + standart model isimleri | §7 — Single naming |
| 4 | **Raw data viz monthly FRED yok** | 5 ek raw plot (FEDFUNDS, CPI, UNRATE, M2, ICSA) | §8 — Visuals |
| 5 | **Feature dokümantasyonu eksik** | docs/FEATURES.md detaylı (poster style, 44 feature listesi) | §4 — Feature Eng |
| 6 | **Notebook 04, 07, 08 stale** | v3 retrain sonrası tek pasta yeniden run | §10 — Notebooks |
| 7 | **ETH modellenmedi** | v3 pipeline'ı ETH'ye de uygula (BTC + ETH karşılaştırma) | §11 — ETH (opsiyonel) |
| 8 | **Plot karmaşası** | 4 ana plot + raw_data_visuals = 14 toplam | §8 — Visuals |
| 9 | **Joblib model artefact'ları stale** | v3 retrain'de save_model=True ile güncel artefakt | §6 — Final retrain |
| 10 | **v2-bigger negative finding kullanılmıyor** | Rapora "data extension limitations" bölümü | §12 — Report |

---

## 2. Posterden Alınan 5 Tasarım Kararı

### 2.1 Rule-Based Regime Detection (GMM yerine)
Önceki posterda (Multi-Asset Portfolio):
```
High Risk:   VIX > 35 OR monthly_return < -10% OR vol > 35%
Low Risk:    VIX < 14 AND strong positive momentum AND low vol
Medium Risk: default
```

**Bizim BTC günlük setine uyarlama:**
```
High Risk:   VIX > 25  OR  20-day return < -15%  OR  20-day annualized vol > 80%
Low Risk:    VIX < 16  AND 60-day return > +5%   AND 60-day annualized vol < 50%
Medium Risk: default
```

Eşikler için **train set'te grid search** yapılacak (her threshold için backtest sonucuna göre).

### 2.2 Stage 1 = Early Warning Predictor
Posterda XGBoost rule-based regime'i **1-3 ay önceden** tahmin ediyordu (%85.7 early warning rate).

Bizim adapte: Stage 1 LDA/XGB rule-based regime'i **5-20 gün önceden** tahmin etsin. Hedef değişken `regime.shift(-h)` (forward looking), feature'lar t-day'e kadar.

### 2.3 Adaptive Position Sizing
Posterda Dynamic MVO `λ` regime-dependent (low=1.0, med=2.0, high=5.0).

Bizim adapte (tek asset olduğu için λ yerine doğrudan position fraction):
| Regime | Position |
|---|---:|
| Low Risk | %100 long |
| Medium Risk | %50 long |
| High Risk | %0 (cash) |

Stage 3 model output'u **sadece sinyal** vermek yerine, **regime + sinyal** kombinasyonu ile pozisyon ölçeklendirir.

### 2.4 Feature Engineering Revamp (~44 feature)
Posterdaki kategorileri taban alıyor:
- **Multi-period momentum**: ret_1d, ret_5d, ret_20d, ret_60d, ret_120d
- **Multi-window vol**: vol_5d, vol_20d, vol_60d (annualized)
- **Drawdown + recovery**: rolling_drawdown_60, recovery_ratio_30
- **VIX dynamics**: VIX_level, VIX_change_5d, VIX_zscore_60
- **Yield curve + credit spread**: 10Y-2Y, HY-IG (zaten var)
- **Risk appetite**: SP500_VIX_ratio, Gold_Silver_ratio, DXY_z
- **Volume**: Volume_zscore_20, OBV_zscore
- **Total: ~30-35 feature** (poster 44'tü ama multi-asset)

### 2.5 Sade Plot Tasarımı (poster style)
4 ana figür:
1. **Regime Detection**: BTC log price + rule-based regime shading (Low/Med/High Risk)
2. **Equity Curve Comparison**: 7 model + B&H (poster Figure 2'ye paralel)
3. **Performance Table**: tek tablo, classification + backtest metrikleri (poster style)
4. **Confusion Matrix Grid**: 7 model panel (zaten var, sadece label fix)

+ raw_data_visuals: 15 plot (10 mevcut + 5 monthly FRED)

---

## 3. Stage 2 — Rule-Based Regime Detection

**Input:** v4 aligned columns (VIX, BTC Close, BTC volatility, vb.)

**Hesaplama:**
```python
def detect_regime(aligned_df) -> pd.Series:
    close = aligned_df["Close"]
    vix = aligned_df["VIX"]
    
    ret_20d = close.pct_change(20)
    vol_20d_ann = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_60d_ann = close.pct_change().rolling(60).std() * np.sqrt(252)
    ret_60d = close.pct_change(60)
    
    high_mask = (vix > 25) | (ret_20d < -0.15) | (vol_20d_ann > 0.80)
    low_mask = (vix < 16) & (ret_60d > 0.05) & (vol_60d_ann < 0.50)
    
    regime = pd.Series("Medium", index=close.index)
    regime[high_mask] = "High"
    regime[low_mask] = "Low"
    
    # Persistence filter: en az 5 gün dayan
    regime = persistence_filter(regime, min_days=5)
    return regime
```

**Persistence filter:** Tek-günlük dalgalanmaları yumuşat, posterdaki "regime persistence" mantığı.

---

## 4. Feature Engineering (§4)

`src/features/v3_features.py` yeni modül. ~32 feature üretir, hepsi BTC OHLCV + makro inputs:

### 4.1 BTC Price/Vol (12)
- `ret_1d, ret_5d, ret_20d, ret_60d, ret_120d` (log returns)
- `vol_5d, vol_20d, vol_60d` (annualized)
- `rolling_drawdown_60, recovery_ratio_30, sharpe_proxy_60`
- `above_sma_50` (binary, momentum proxy)

### 4.2 BTC Volume (3)
- `volume_zscore_20, volume_change_5d, OBV_zscore_60`

### 4.3 Oscillators (5)
- `RSI_14, MACD_histogram, Bollinger_pct_b, Stochastic_K_14, ATR_14_pct` (ATR/Close)

### 4.4 VIX Dynamics (4)
- `VIX_level, VIX_change_5d, VIX_zscore_60, VIX_above_25` (binary)

### 4.5 Macro Spread (4)
- `Yield_Curve_10Y_2Y, Credit_Spread_log, Gold_Silver_Ratio, DXY_zscore_60`

### 4.6 Monthly FRED — sadece **anlamlı 4 tane** (8 değil 11 değil)
- `FEDFUNDS_change_60d` (3-month FFR change — Fed direction)
- `real_interest_rate` (Fisher)
- `UNRATE_change_180d` (6-month unemployment change)
- `M2_yoy` (annual growth)

**Toplam: 32 feature** (44 değil, çünkü tek asset).

`docs/FEATURES.md` — her feature için: hesaplama formülü, niye kullanıldı, hangi stage'de kullanılır.

---

## 5. Stage 3 + Adaptive Position Sizing

**Stage 3 input:** 32 v3 feature + Stage 1 OOF early warning posterior (3 col) = 35 feature

**Stage 3 output:** signal in {Buy, Sell, Hold} (mevcut yapı)

**Position sizing layer (yeni):**
```python
def compute_position(signal, regime):
    # Default position from signal
    base_pos = {"Buy": 1.0, "Hold": current_pos, "Sell": 0.0}
    # Regime cap (posterdaki λ benzeri)
    cap = {"Low": 1.0, "Medium": 0.5, "High": 0.0}
    return min(base_pos[signal], cap[regime])
```

**Backtester revamp:** mevcut sadece in/out pozisyon. Yeni: fractional position [0, 1]. Pozisyon değişimi → trade. Volume cost (0.1%) eklenebilir.

---

## 6. Final Retrain (§6)

**7 model + 4 ablation config × v3 features:**

| Config | Stage 3 input |
|---|---|
| A1 — Flat (1-stage) | 32 v3 feature |
| A2 — 2-stage Trend | 32 v3 + s1 (3) |
| A3 — 2-stage Macro (rule-based) | 32 v3 + s2_rule (3) |
| A4 — Full 3-stage | 32 v3 + s1 + s2_rule = 38 |

7 model: LDA, MLP, XGB, LGBM, RF, ZZ-XGB, ZZ-MLP. Optuna 8 trial walk-forward CV.

**Toplam: 4 × 7 = 28 deney.** Her birinin Sharpe + Return + MaxDD + Win% + early warning rate (Stage 1 vs rule-based regime tahmini başarısı).

**Save:** `app/models/stage3_*_v3.joblib` joblib export, demo skew kapatılır.

---

## 7. Naming Convention (§7)

**Tek standart, GLOSSARY.md güncellemesi:**

| Eski | v3 Yeni |
|---|---|
| v1 / v2 / v3 / v4 | "Original (v4 final)" — referans olarak kalır |
| B1 / B2 / B3 | yok, v3 tek subset kullanır |
| A1 / A2 / A3 / A4 | "Flat" / "Trend-only" / "Regime-only" / "Full-3-stage" |
| Phase A / B / C | yok, v3 tüm 7 modeli tek koşturur |
| Calm / Trans / Stress | "Low Risk" / "Medium Risk" / "High Risk" (poster style) |
| ZZ-XGB | "XGB-ZigZag" — explicit |

---

## 8. Plots (§8)

**Tutulacak (4 ana + raw_data):**
1. `v3_regime_detection.png` — BTC + rule-based regime shading (poster Fig 1 style)
2. `v3_equity_curves.png` — 7 model + B&H + Dynamic Regime (poster Fig 2 style)
3. `v3_summary_table.png` — final tablo (poster table style)
4. `v3_cm_grid.png` — 7-panel confusion matrix
5. `v3_ablation_comparison.png` — 4 config bar chart
6. `raw_data_visuals/*.png` — **15 plot** (10 mevcut + 5 yeni FRED)

**Silinecek:** mevcut 13 plot'tan B2 plot'ları + ablation_fs_* + monthly_fred_overview (mevcut hâli).

---

## 9. CM Bias Çözümü (§9)

Adaptive position sizing **yapısal** olarak çözer:
- Eski: model %80 Sell tahmin → %80 zaman kapalı pozisyon → kayıp Buy fırsatları
- Yeni: model Sell tahmin → regime "Low" ise yine pozisyon kalır → false-Sell etkisi azalır

Ayrıca **threshold tuning** ekleyebiliriz: `predict_proba(Sell) > 0.6` eşiği, `argmax` yerine.

---

## 10. Notebook Refresh (§10)

v3 retrain bittikten sonra:
- `notebooks/04_label_generation.ipynb` — rule-based regime ve adaptive label kodu
- `notebooks/07_evaluation.ipynb` — v3 CSV'leri ile re-run
- `notebooks/08_ablation_study.ipynb` — 4-config ablation görsel

---

## 11. ETH Pipeline (§11, opsiyonel)

v3 pipeline'ı ETH'ye uygula. Tek değişiklik: `eth_aligned.csv` + ETH-specific volatility threshold'ları (BTC vol > ETH vol değil mi? Kontrol edilecek).

ETH train: 2017-11 → 2024-09, test: 2024-09 → 2025-12. Aynı 4-config ablation.

---

## 12. Rapor Stratejisi (§12)

**Final report yapısı:**
1. **Introduction:** problem, multi-stage classification framework
2. **Data:** 22-column aligned dataset (BTC + 12 macro + 5 monthly FRED)
3. **Methodology:**
   - 32 v3 feature (FEATURES.md ek olarak)
   - Rule-based regime (poster style)
   - 3-stage hierarchical with early warning Stage 1
   - Adaptive position sizing
4. **Experiments:**
   - Ablation 4 config × 7 model
   - v1 (Sharpe 1.35) vs B2 (1.68) vs v3 referansları
   - v2-bigger limitation: data extension hurts in OOD
5. **Results:** v3 best (umut: Sharpe ≥ 1.7, Return ≥ B&H)
6. **Limitations:**
   - Tek asset (BTC)
   - Test seti küçük (462 gün)
   - GMM negative finding (eski v4'te), v3'te rule-based ile aşıldı
7. **Future Work:** ETH expansion, intraday data, HMM regime detection

---

## 13. İcra Planı (4 Faz)

### Faz 1 — Foundations (3-4 saat)
- `v3-rule-based-regime` branch açılır
- Rule-based regime detection script + persistence filter
- v3 feature engineering (32 feat)
- `docs/FEATURES.md` yazılır
- `docs/V3_PLAN.md` (bu doc) commit'lenir

### Faz 2 — ML Stage 1 + Stage 3 (4-6 saat)
- Stage 1 early warning model (LDA, XGB) — predict regime t+5d
- Stage 3 retrain × 7 model × 4 config (Optuna walk-forward)
- save_model=True (joblib export)
- Backtest + adaptive position sizing
- Output CSVs + summary tables

### Faz 3 — Visuals + Docs (3 saat)
- 5 v3 plot üret (regime detection, equity, summary table, CM grid, ablation)
- 5 yeni raw FRED plot
- Eski plot'ları sil
- `GLOSSARY.md` v3 ile güncelle

### Faz 4 — Notebook + Decision Gate (2 saat)
- Notebook 04, 07, 08 v3 ile re-run
- v3 vs B2 vs v1 final karşılaştırma
- Eğer v3 ≥ B2 → **rapor v3 üzerinden yazılacak**
- Eğer v3 < B2 → v3 ek deneysel bölüm, B2 headline

**Toplam: 12-15 saat** (1.5-2 takvim günü)

---

## 14. Karar Noktaları

1. **§3 sonrası (rule-based regime):** Plot'ta Low/Med/High dağılımı dengeli mi (örn. her birinin payı %20'den fazla) yoksa High Risk hep yapışık mı kalıyor? Eşikler ayarlanır.

2. **§5 sonrası (adaptive sizing backtest):** Sharpe ≥ 1.5 mi? Yoksa eşikleri tune et.

3. **Faz 4 sonrası (final v3 vs B2):** v3 Sharpe ≥ 1.68 (B2) mi? Karar gate.

4. **Faz 4 sonrası (ETH opsiyonel):** Vakit kalırsa ETH pipeline.

---

## 15. v3 Branch Yapısı

```
claude/review-checkpoint-results-2hh1j  ← ana branch
│
├── v1.0-iter4-final  (tag, ab408d5 — fallback referansı)
│
├── v2/feature-selection  (B2 = 1.68 Sharpe, fallback referansı)
│
└── v3-rule-based-regime  ← yeni
    ├── docs/V3_PLAN.md, FEATURES.md, GLOSSARY.md (güncel)
    ├── src/features/v3_features.py (yeni)
    ├── src/labels/regime_rules.py (yeni)
    ├── src/evaluation/adaptive_backtester.py (yeni)
    ├── scripts/build_v3_features.py
    ├── scripts/run_v3_ablation.py
    ├── scripts/regenerate_v3_visuals.py
    └── data/labels/btc_*_v3.csv (yeni artifact set)
```

`v3-final` tag → final commit'te atılır.

---

## 16. Risk Tablosu

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| Rule-based regime BTC için aşırı volatil olur | Orta | Sharpe < B2 | Eşik grid search |
| Adaptive sizing transaction cost'u yüksek | Düşük | Net return azalır | Persistence filter (5-day) |
| Stage 1 early warning low accuracy | Orta | Stage 3 input zayıf | Multi-window features, MLP'yi de dene |
| 2 günlük deadline yetmez | Orta | Faz 3-4 yarım | Faz 2 sonunda decision: v3 ≥ B2 ise devam, değilse fallback |
| ETH (§11) yetişmez | Yüksek | - | Opsiyonel, atlanabilir |

---

_Bu plan v3-rule-based-regime branch'inde çalışmaya başlamadan önce kullanıcının onayı ile commit'lenir._
