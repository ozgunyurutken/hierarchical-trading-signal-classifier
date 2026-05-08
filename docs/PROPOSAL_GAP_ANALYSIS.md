# Proposal vs Yapılan — Boşluk Analizi

**Tarih:** 2026-05-08
**Amaç:** Hocaya verilen proposal ile fiilen uyguladığımız çalışma arasındaki farkları net belgele. Sıfırdan restart kararının tasarım girdisi olarak kullanılır.

> **Kullanıcı kararı:** "%100 uyumlu kalmak zorunlu değil, özellikle features için." Ama hocanın **temel beklentilerine** (3-stage hierarchy, 4 algorithm comparison, ablation, BTC+ETH, FastAPI/Docker, walk-forward CV, SHAP, McNemar) uymak gerekli.

---

## 1. Proposal Özet (cd3c8b0e PDF)

**Başlık:** "A Three-Stage Hierarchical ML Framework for Cryptocurrency Trading Signal Classification Using Technical and Macroeconomic Indicators"

| Konu | Proposal sözü |
|---|---|
| **Veri** | BTC/USDT + ETH/USDT, **2021-2025**, ~1,400+ daily samples, yfinance/Binance |
| **Stage 1** | Trend (Up/Down/Sideways), long-horizon tech (SMA, ADX, Parabolic SAR), label: SMA crossover + 3-day persistence filter |
| **Stage 2** | Macro Regime (Risk-On/Risk-Off/Neutral), 7 macro feature (FFR, CPI, Unemployment, S&P, Gold, DXY, VIX), label: **K-Means k=3 + semantic relabeling** (Elbow + Silhouette validation) |
| **Stage 3** | Buy/Sell/Hold, oscillators + **one-hot encoded** s1+s2 outputs, label: forward 5-day return **±1%** (will be optimized) |
| **Imbalance** | **SMOTE veya class_weight** |
| **Algorithms** | XGBoost, LightGBM, RF, MLP — 4 primary |
| **Optional** | CatBoost, TabNet, SVM |
| **Ablation** | (i) flat / 2-stage / 3-stage, (ii) 4 algorithms per stage, (iii) BTC vs ETH, (iv) signal vs B&H |
| **CV** | Chronological 70/15/15 split, walk-forward, Optuna Bayesian |
| **Evaluation** | Accuracy/Precision/Recall/F1, Confusion Matrix, ROC-AUC, Backtest (Sharpe), **SHAP**, **McNemar's test** |
| **Deployment** | Docker + FastAPI + HTML/JS frontend |

---

## 2. Yapılan vs Proposal — Madde Madde Karşılaştırma

| Konu | Proposal | Yapılan | Durum |
|---|---|---|---|
| **Veri dönemi** | 2021-2025 (~1,400 gün) | 2014-09 → 2025-12 (3,961 gün) | **Genişletildi** (savunulabilir, daha çok train data) |
| **Veri kaynağı** | yfinance + Binance | yfinance + FRED REST API + CryptoCompare | **Genişletildi** (FRED daily DGS2 + monthly FFR/CPI/UNRATE/M2/ICSA, CryptoCompare 2010-2014) |
| **Stage 1 label** | SMA cross + 3-day persistence | SMA cross + ZigZag (causal swing 10%) | **Genişletildi** (ZigZag eklenmiş, Stage 1 tautoloji fix) |
| **Stage 2 method** | **K-Means k=3 hybrid** | **GMM k=3** (unsupervised) | **DEĞİŞTİRİLDİ** — proposal'a UYUMSUZ |
| **Stage 2 features** | 7 macro (FFR, CPI, Unemployment, S&P, Gold, DXY, VIX) | 11 macro (VIX, VIX_z50, Yield_Curve, Credit_Spread, Gold_Silver, SP500_VIX, DXY_z50, SP500_roc20, FEDFUNDS, real_interest_rate, UNRATE) | **Değiştirildi/genişletildi** (CPI çıkarılmış, türev 4 spread eklenmiş) |
| **Stage 2 label semantic** | "Risk-On/Risk-Off/Neutral" + Elbow/Silhouette validation | Cluster idx 0/1/2 → "Calm/Trans/Stress" hatalı semantic atama (yanlış renklendirme) | **Eksik** — Elbow/Silhouette analizi yapılmamış |
| **Stage 3 label** | ±1% fixed (optimize edilebilir) | Fixed ±1% **+** adaptive 0.5×rolling_std(20) | **Genişletildi** (iki versiyonu var) |
| **Stage 3 input** | **one-hot encoded** s1+s2 + oscillators | **soft posterior** s1+s2 (ℝ³) + tech features | **Değiştirildi** — soft fusion proposal'da yok ama "iyileştirme" olarak savunulabilir |
| **Imbalance handling** | **SMOTE veya class_weight** | **Yapılmadı** | **EKSİK** — CM Sell-bias problemi tam buradan |
| **Algorithms** | 4: XGB, LGBM, RF, MLP | 5 + 2 ZigZag = **7** (LDA, MLP, XGB, LGBM, RF, ZZ-XGB, ZZ-MLP) | **Genişletildi** — LDA eklenmiş, ZigZag versiyonları eklenmiş |
| **Ablation flat/2-stage/3-stage** | Belirtilmiş | **Yapıldı** (v2/ablation, v2/feature-selection B1/B2/B3) | ✅ |
| **Algorithm comparison per stage** | Belirtilmiş | **Yapıldı** Stage 3 için | ✅ kısmen |
| **BTC vs ETH** | Belirtilmiş | **Sadece BTC** modellendi, ETH aligned hazır ama train edilmedi | **EKSİK** |
| **Signal vs B&H** | Belirtilmiş | **Yapıldı** (Sharpe, Return, MaxDD karşılaştırma) | ✅ |
| **Walk-forward** | Belirtilmiş | **Yapıldı** (5-fold expanding window) | ✅ |
| **Optuna Bayesian** | Belirtilmiş | **Yapıldı** (8 trial × 4 config × 7 model) | ✅ |
| **70/15/15 split** | Belirtilmiş | **15% test only** (train+val ayrımı yok, walk-forward içinde validation) | **Kısmen sapma** — 70/15/15 yerine 85/15 (CV içinde validation) |
| **SHAP** | Belirtilmiş | **Yapılmadı** | **EKSİK** |
| **McNemar's test** | Belirtilmiş | **Yapılmadı** | **EKSİK** |
| **Confusion Matrix** | Belirtilmiş | **Yapıldı** (b2_cm_grid.png) | ✅ |
| **ROC-AUC** | Belirtilmiş | **Yapıldı** (b2_roc_grid.png) | ✅ |
| **Backtesting Sharpe** | Belirtilmiş | **Yapıldı** | ✅ |
| **Docker** | Belirtilmiş | **Yapıldı** (v1, demo için Dockerfile) | ✅ kısmen — v3 için yeniden yapılması gerek |
| **FastAPI backend** | Belirtilmiş | **Yapıldı** (v1) | ✅ |
| **HTML/JS frontend** | Belirtilmiş | **Yapıldı** (v1) | ✅ |
| **TA-Lib** | Belirtilmiş | `ta` Python paketi (TA-Lib alternatifi) | ✅ kısmen sapma |
| **PyTorch** | Belirtilmiş | sklearn MLP kullanıldı, PyTorch yok | **EKSİK** (ama sklearn MLP yeterli derin öğrenme yerine — savunulabilir) |

---

## 3. Kritik Boşluklar Özeti — V3 Restart'ta Çözülecekler

### Yüksek öncelik (rapor için ciddi risk)
1. **Stage 2: GMM yerine K-Means + semantic relabeling** — proposal sözü, mutlaka yapılacak
2. **SMOTE veya class_weight** — CM Sell-bias düzeltmesi proposal'da var
3. **SHAP feature importance** — interpretability proposal'da kritik söz
4. **McNemar's test** — istatistiksel anlamlılık proposal sözü
5. **ETH modeli** — BTC vs ETH karşılaştırması proposal'da
6. **Stage 2 Elbow + Silhouette validation** — proposal'da k=3 doğrulaması bekliyor

### Orta öncelik (savunulabilir sapmalar, raporda açıklanır)
7. **Veri dönemi** 2021-2025 yerine 2014-2025 — daha çok data avantaj olarak savun
8. **Soft fusion vs one-hot** — biz soft yaptık, "improvement over proposal" olarak frame
9. **Stage 2 features** 7 yerine 11 — derived spread'ler fazlası, savunulabilir
10. **70/15/15 split** — biz 85/15 + walk-forward içi validation yaptık

### Düşük öncelik (esneklik dahilinde)
11. **PyTorch yerine sklearn MLP** — fonksiyonel olarak eşdeğer
12. **TA-Lib yerine `ta` paketi** — kütüphane farkı

---

## 4. Sıfırdan Yeni Tasarım — Proposal Çatısı + V3 İlhamı

Yeni başlangıç noktası: **proposal'a temel uyum + V3 plan'ından gelen iyileştirmeler.**

### Mimari Çatı (proposal-uyumlu)
```
Stage 1 (Trend)        Stage 2 (Macro Regime)
SMA + ADX +            K-Means k=3 + semantic
Parabolic SAR          relabeling (Risk-On/Off/Neutral)
→ Up/Down/Sideways     → Risk-On/Risk-Off/Neutral
        ↓                       ↓
        └───────┬───────────────┘
                ↓
       Stage 3 (Signal)
       Oscillators (RSI/MACD/Stoch/BB)
       + ONE-HOT s1 (3) + ONE-HOT s2 (3)
       → Buy/Sell/Hold
                ↓
       4 algorithm: XGB / LGBM / RF / MLP
       (+ optional CatBoost / TabNet / SVM)
                ↓
   Ablation: flat / 2-stage / 3-stage
   Both BTC and ETH
   SHAP + McNemar test
```

### V3 İlhamından eklenecekler (proposal-uyumlu olarak)
- **Adaptive position sizing** opsiyonel ek bölüm (proposal'da yok ama poster'dan ilham, "future work" veya bonus deney)
- **Multi-window momentum + drawdown** Stage 1/3 feature'lara ekleme
- **Persistence filter** Stage 1 (proposal'da var ama 3 gün; biz tune edebiliriz)

### Yeni başlangıç için ne yapılır?
1. **`docs/LITERATURE_REVIEW_v2.md`** — daha kapsamlı tarama (proposal'daki 7 atıf detayı + ek 50+ atıf, toplam ~100)
2. **`config.yaml` v2** — yeniden yazılacak, K-Means + semantic relabeling parametreleri, SHAP + McNemar konfigürasyonu
3. **Yeni branch:** `v3-from-scratch` (`v1.0-iter4-final` tag'ından) — proposal-uyumlu kod tabanı
4. **`src/` modülleri:** K-Means hybrid labeling, semantic relabeler, SHAP wrapper, McNemar test runner
5. **Notebook'lar yeniden yazılır:** EDA, K-Means validation (Elbow + Silhouette), SHAP analysis
6. **BTC + ETH paralel pipeline** her şey iki coin için
7. **Ablation:** sadeleştirilmiş 3 config (flat / 2-stage / 3-stage), 4 model × 2 coin × 3 config = 24 deney

---

## 5. Yeni İcra Planı (Tahmini Süre: 2-3 gün)

| Faz | İçerik | Süre |
|---|---|---|
| **0** | Bu boşluk analizi + kullanıcı onayı + kapsamlı lit review (10-12 paralel Agent) | 1-2h |
| **1** | Yeni `config.yaml`, `src/` refactor, K-Means + semantic + SHAP + McNemar modülleri | 4h |
| **2** | EDA + K-Means validation notebook, Stage 1 + Stage 2 train | 3-4h |
| **3** | Stage 3 train × 4 model × 3 config × 2 coin = 24 deney + walk-forward | 6-8h |
| **4** | Backtest + SHAP analysis + McNemar test + plotlar | 3h |
| **5** | Notebook'lar + Docker + FastAPI rebuild + final raporu hazırlık | 2h |

---

## 6. Önemli Karar Noktaları

1. **Stage 2 metodolojisi:** K-Means saf mi yoksa K-Means + GMM ensemble mı? Proposal saf K-Means ama biz GMM bilgisinden faydalanabiliriz.
2. **Soft vs hard fusion:** Proposal one-hot, biz soft posterior yaptık. **Soft fusion'ı koruyalım mı yoksa proposal-strict one-hot'a mı dönelim?**
3. **Veri dönemi:** 2014-2025 (mevcut) vs 2021-2025 (proposal). **Daha geniş data savunmak istiyoruz mu?**
4. **B2 ZZ-MLP Sharpe 1.68 sonucunu rapor headline yap mı yoksa V3 sıfırdan setup yap mı?**

---

_Bu analiz, V3 restart kararının temel girdisi. Kullanıcı onayı ile sıfırdan tasarım başlayacak._
