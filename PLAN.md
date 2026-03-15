# Uygulama Planı: Kripto Para Alım-Satım Sinyal Sınıflandırması
# 3 Aşamalı Hiyerarşik ML Pipeline

## Context

BBL514E Pattern Recognition dersi dönem projesi. Teslim tarihi: 10 Mayıs 2026.
Mevcut durumda proje klasörü yok, sıfırdan oluşturulacak.
Mevcut varlıklar: `/Users/yurutkenozgun/Desktop/coin_data/` altında 61 kripto para CSV dosyası var ama yfinance/FRED ile taze veri çekilecek.

**Proje konumu:** `/Users/yurutkenozgun/Desktop/crypto-signal-classifier/`

---

## FAZ 0: Proje İskeleti & Ortam Kurulumu

### 0.1 Dizin Yapısı Oluşturma
```
crypto-signal-classifier/
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_label_generation.ipynb
│   ├── 05_stage1_training.ipynb
│   ├── 06_stage2_training.ipynb
│   ├── 07_stage3_training.ipynb
│   ├── 08_ablation_study.ipynb
│   ├── 09_backtesting.ipynb
│   └── 10_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── price_collector.py
│   │   ├── macro_collector.py
│   │   └── data_aligner.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── macro_features.py
│   │   └── feature_selector.py
│   ├── labels/
│   │   ├── __init__.py
│   │   ├── trend_labels.py
│   │   ├── regime_labels.py
│   │   └── signal_labels.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   ├── stage1_trainer.py
│   │   ├── stage2_trainer.py
│   │   ├── stage3_trainer.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── backtester.py
│   │   ├── shap_analysis.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── app/
│   ├── main.py
│   ├── models/
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── templates/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── reports/
│   ├── proposal/
│   └── final/
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

### 0.2 Ortam Kurulumu
- Python 3.11+ sanal ortam (venv)
- `requirements.txt` oluşturma:
  ```
  pandas, numpy, yfinance, fredapi, ta, scikit-learn,
  xgboost, lightgbm, torch, hmmlearn, shap, matplotlib,
  seaborn, plotly, fastapi, uvicorn, joblib, pyyaml,
  scipy, statsmodels, optuna, python-multipart
  ```
- `config.yaml`: Tüm sabitler, yollar, hiperparametreler merkezi config

### 0.3 Git Repo Başlatma
- `.gitignore` (data/raw/, .venv, __pycache__, .env, *.joblib)
- İlk commit

---

## FAZ 1: Veri Toplama & Ön İşleme

### 1.1 Fiyat Verisi Toplama (`src/data/price_collector.py`)
- **Kaynak:** yfinance
- **Varlıklar:** BTC-USD, ETH-USD
- **Periyot:** 2021-01-01 → 2025-12-31
- **Alanlar:** Open, High, Low, Close, Volume (OHLCV)
- **Çıktı:** `data/raw/btc_ohlcv.csv`, `data/raw/eth_ohlcv.csv`
- Her coin ayrı dosya, tarih index olarak
- İndirme sırasında eksik gün kontrolü ve loglama

### 1.2 Makroekonomik Veri Toplama (`src/data/macro_collector.py`)
- **FRED API (aylık):**
  - FEDFUNDS (Federal Funds Rate)
  - CPIAUCSL (CPI - Enflasyon)
  - UNRATE (İşsizlik Oranı)
- **yfinance (günlük):**
  - ^GSPC (S&P 500)
  - GC=F (Altın Vadeli)
  - DX-Y.NYB (DXY Dolar Endeksi)
  - ^VIX (Volatilite Endeksi)
- **Çıktı:** `data/raw/macro_monthly.csv`, `data/raw/macro_daily.csv`

### 1.3 Veri Hizalama & Temizleme (`src/data/data_aligner.py`)
- **Temporal hizalama kuralları:**
  1. FRED aylık veri → gerçek yayın tarihine göre hizalama (release date)
  2. Forward-fill: aylık → günlük (yayın tarihinden sonra), hafta sonu boşlukları → son işlem gününden
  3. Timezone: Kripto UTC 00:00, Piyasa NYSE kapanış (21:00 UTC)
- **Çıktı:** `data/processed/btc_aligned.csv`, `data/processed/eth_aligned.csv`
- NaN satır sayısını logla ve dokümante et

### 1.4 Notebook: `01_data_collection.ipynb`
- Veri toplama kodlarını çalıştır
- Temel istatistikler: satır sayısı, tarih aralığı, eksik veri oranı
- Veri kalitesi kontrolleri

### 1.5 Notebook: `02_eda.ipynb`
- Fiyat grafikleri (BTC, ETH zaman serisi)
- Volume dağılımı
- Makro göstergeler zaman serisi
- Korelasyon matrisi (tüm ham veriler arası)
- Temel istatistiksel özetler (describe)
- Dağılım histogramları

---

## FAZ 2: Feature Engineering

### 2.1 Teknik İndikatörler (`src/features/technical_indicators.py`)

**Trend indikatörleri (Stage 1 adayları):**
- SMA(20), SMA(50), SMA(200)
- EMA(12), EMA(26)
- ADX(14)
- Parabolic SAR
- SMA crossover sinyalleri (SMA20-SMA50, SMA50-SMA200)
- Fiyatın SMA'lardan uzaklığı (%)

**Momentum/Osilatör indikatörleri (Stage 3 adayları):**
- RSI(14)
- Stochastic %K(14), %D(3)
- MACD line, MACD signal, MACD histogram
- Williams %R(14)
- CCI(20)
- ROC(10)

**Volatilite indikatörleri (Stage 1 veya 3):**
- Bollinger Bands(20,2): upper, lower, bandwidth, %B
- ATR(14)
- Historical volatility (rolling std of returns)

**Volume indikatörleri (Stage 1 veya 3):**
- OBV
- Volume SMA(20)
- Volume rate of change

### 2.2 Makro Feature'lar (`src/features/macro_features.py`)
- Ham: FFR, CPI, Unemployment, S&P close, Gold close, DXY, VIX
- Rate-of-change: Δx_t = (x_t - x_{t-n}) / x_{t-n}
- Rolling z-score: z_t = (x_t - μ_rolling) / σ_rolling
- Rolling pencereler: 20, 50, 100 gün
- Türetilmiş: real interest rate (FFR - CPI yıllık değişim)

### 2.3 Durağanlık Testi
- Tüm feature'lara ADF testi uygula
- Durağan olmayanlara differencing
- Sonuçları tablo olarak kaydet

### 2.4 Feature Selection (`src/features/feature_selector.py`)
Her stage'in feature havuzu için AYRI AYRI:
1. **Korelasyon analizi:** |r| > 0.95 olanlardan birini çıkar
2. **Mutual Information:** MI skoru ile sıralama
3. **SHAP-based importance:** Ön XGBoost ile SHAP değerleri
4. **Final seçim:** Birleşik sıralamadan top-k feature

### 2.5 Ölçekleme Stratejisi
- Tree modeller (XGBoost, LightGBM, RF): Ölçekleme YOK
- MLP: StandardScaler (sadece train üzerinde fit, val/test transform)
- K-Means: StandardScaler

### 2.6 Notebook: `03_feature_engineering.ipynb`
- Tüm feature hesaplamaları
- ADF test sonuçları tablosu
- Feature selection sonuçları ve görselleştirme
- Feature importance plotları
- Seçilen feature'ların her stage için listesi
- **Toplam aday feature:** ~40-60 → seçim sonrası her stage için final set

### 2.7 Çıktılar
- `data/processed/btc_features_stage1.csv`
- `data/processed/btc_features_stage2.csv`
- `data/processed/btc_features_stage3.csv`
- (aynıları ETH için)

---

## FAZ 3: Label (Etiket) Üretimi

### 3.1 Stage 1 — Trend Etiketleri (`src/labels/trend_labels.py`)
- **Sınıflar:** Uptrend, Downtrend, Sideways
- **Kural:**
  ```python
  if SMA(20) > SMA(50) and close > SMA(50): → "Uptrend"
  elif SMA(20) < SMA(50) and close < SMA(50): → "Downtrend"
  else: → "Sideways"
  ```
- **Gürültü filtresi:** Minimum 3 gün persistence — 3 gün dayanmayan etiket öncekine döner
- **Rapor notu:** "Rule distillation" limitasyonunu dokümante et

### 3.2 Stage 2 — Makro Rejim Etiketleri (`src/labels/regime_labels.py`)
- **Sınıflar:** Risk-On, Risk-Off, Neutral
- **Yöntem 1: K-Means (k=3)**
  - Tüm makro feature normalize (StandardScaler)
  - K-Means fit
  - Elbow method (k=2..6) + Silhouette score ile k doğrulama
  - Centroid incelemesiyle semantik etiket atama:
    - En düşük VIX + en yüksek S&P return → Risk-On
    - En yüksek VIX + en negatif S&P → Risk-Off
    - Kalan → Neutral
- **Yöntem 2: GMM** — soft cluster, Adjusted Rand Index ile K-Means karşılaştırma
- **Yöntem 3: HMM** (hmmlearn) — temporal geçişleri modeller
- **Karşılaştırma kriterleri:**
  - Etiket stabilitesi (az flip)
  - Downstream Stage 3 performansı
  - Rejim yorumlanabilirliği
- **Görselleştirme:** PCA/t-SNE 2D projeksiyonla küme görselleştirme

### 3.3 Stage 3 — Sinyal Etiketleri (`src/labels/signal_labels.py`)
- **Sınıflar:** Buy, Sell, Hold
- **Yöntem 1: Sabit eşik**
  ```python
  forward_return = (close[t+5] - close[t]) / close[t]
  if forward_return > +0.01: → "Buy"
  elif forward_return < -0.01: → "Sell"
  else: → "Hold"
  ```
- **Yöntem 2: Volatilite-ayarlı eşik**
  ```python
  rolling_std = returns.rolling(20).std()
  c = 0.5
  if forward_return > c * rolling_std[t]: → "Buy"
  elif forward_return < -c * rolling_std[t]: → "Sell"
  else: → "Hold"
  ```
- **KRİTİK ASSERTION:** Forward return ASLA input feature olarak kullanılmayacak — kodda assert ile doğrula

### 3.4 Sınıf Dengesi Analizi
- Her etiket seti için sınıf dağılımı (sayı ve %)
- Dengesiz ise: class weighting (SMOTE değil — zaman serisi yapısını bozar)
- Sınıf ağırlıklarını dokümante et

### 3.5 Notebook: `04_label_generation.ipynb`
- Tüm etiket üretim kodları
- Sınıf dağılımı grafikleri (bar chart)
- K-Means vs GMM vs HMM karşılaştırma tablosu
- PCA/t-SNE küme görselleştirmesi
- Elbow + Silhouette grafikleri
- Label zaman serisi görselleştirmesi (trend, rejim, sinyal zaman içinde nasıl değişiyor)

### 3.6 Çıktılar
- `data/labels/btc_trend_labels.csv`
- `data/labels/btc_regime_labels.csv`
- `data/labels/btc_signal_labels.csv`
- (aynıları ETH için)

---

## FAZ 4: Model Eğitimi

### 4.1 Veri Bölme Stratejisi
```
|<---------- %85 Train + Val ---------->|<-- %15 Test -->|
| Expanding-window walk-forward         | Final eval      |
```
- Son %15 kronolojik test seti — eğitim sırasında ASLA dokunulmaz
- Train+Val (%85): expanding-window walk-forward validation
  - Minimum eğitim penceresi: 6 ay
  - Her fold'da 1 ay genişle
  - Her zaman sonraki 1 ayı tahmin et
- **Random shuffle YOK** — tüm bölmeler kronolojik

### 4.2 Sınıflandırıcılar (`src/models/classifiers.py`)
4 temel model wrapper:
1. **XGBoost** — gradient boosting
2. **LightGBM** — leaf-wise boosting
3. **Random Forest** — bagging ensemble
4. **MLP** (PyTorch) — neural network

Her model için:
- `fit(X_train, y_train, X_val, y_val)`
- `predict(X)` → hard labels
- `predict_proba(X)` → probability vectors
- `get_feature_importance()` → importance array

### 4.3 Stage 1 Eğitimi (`src/models/stage1_trainer.py`, `05_stage1_training.ipynb`)
- **Input:** Trend feature seti (seçim sonrası)
- **Target:** Uptrend / Downtrend / Sideways
- **Hiperparametre arama:**
  - XGBoost: n_estimators[100,300,500], max_depth[3,5,7], lr[0.01,0.05,0.1], subsample[0.7,0.8,1.0]
  - LightGBM: n_estimators[100,300,500], num_leaves[15,31,63], lr[0.01,0.05,0.1]
  - RF: n_estimators[100,300,500], max_depth[5,10,20,None], min_samples_leaf[1,5,10]
  - MLP: hidden_layers[(64,32),(128,64),(128,64,32)], lr[0.001,0.0005], batch_size[32,64], early stopping patience=10
- **Tuning:** Optuna + walk-forward validation
- **Kayıt:** Best HP, fold bazlı metrikler, olasılık tahminleri, feature importance (SHAP)

### 4.4 Stage 2 Eğitimi (`src/models/stage2_trainer.py`, `06_stage2_training.ipynb`)
- **Input:** Makro feature seti (seçim sonrası)
- **Target:** Risk-On / Risk-Off / Neutral
- Aynı 4 sınıflandırıcı, aynı HP arama, aynı walk-forward

### 4.5 Stage 3 Eğitimi — OOF ile (`src/models/stage3_trainer.py`, `07_stage3_training.ipynb`)
- **Input:**
  - Osilatör feature'ları (seçim sonrası)
  - p̂(z_trend): Stage 1'den 3-boyutlu olasılık vektörü
  - p̂(z_macro): Stage 2'den 3-boyutlu olasılık vektörü
  - Toplam input dim: num_oscillator_features + 3 + 3
- **Target:** Buy / Sell / Hold
- **KRİTİK — Out-of-Fold (OOF) Tahminler:**
  ```
  1. Train verisini K kronolojik fold'a böl
  2. Her fold i için:
     a. Stage 1'i fold i HARİÇ tümünde eğit
     b. Fold i'yi tahmin et → p̂(z_trend) for fold i
     c. Stage 2'yi fold i HARİÇ tümünde eğit
     d. Fold i'yi tahmin et → p̂(z_macro) for fold i
  3. Tüm fold tahminlerini birleştir → tam train seti için OOF tahminler
  4. Bu OOF tahminleri Stage 3 eğitim feature'ı olarak kullan
  ```
- **Test zamanında:** Stage 1-2 tam train setinde eğitilir, test tahminleri doğal olarak Stage 3'e akar

### 4.6 Ablation Study (`08_ablation_study.ipynb`)
3 konfigürasyon × 4 sınıflandırıcı × 2 coin = **24 deney**

| Config | Pipeline | Feature'lar |
|--------|----------|-------------|
| A | Flat baseline | TÜM feature'lar → tek model → Buy/Sell/Hold |
| B | 2-Stage | Stage 1 (trend) → Stage 3 (osilatör + trend prob). Makro yok. |
| C | 3-Stage (önerilen) | Stage 1 + Stage 2 → Stage 3 (osilatör + trend + makro prob) |

### 4.7 Pipeline Orkestrasyon (`src/models/pipeline.py`)
- `FullPipeline`: 3 stage'i zincirleme çalıştıran sınıf
- `TwoStagePipeline`: ablation Config B
- `FlatBaseline`: ablation Config A
- Her pipeline: `fit()`, `predict()`, `predict_proba()`, `save()`, `load()`

---

## FAZ 5: Değerlendirme & Backtesting

### 5.1 Sınıflandırma Metrikleri (`src/evaluation/metrics.py`, `10_evaluation.ipynb`)
Her deney için:
- Accuracy (genel)
- Precision, Recall, F1-Score (sınıf bazlı + macro-averaged)
- Confusion matrix (heatmap)
- ROC-AUC (one-vs-rest, sınıf bazlı)
- PR-AUC (özellikle Hold baskınsa)
- Balanced Accuracy
- MCC (Matthews Correlation Coefficient)

### 5.2 İstatistiksel Anlamlılık (`src/evaluation/statistical_tests.py`)
- **McNemar testi:** Flat baseline vs 3-stage pipeline
- Tüm kritik karşılaştırmalar için p-value raporlama

### 5.3 Feature Importance Analizi (`src/evaluation/shap_analysis.py`)
- **SHAP summary plot:** Her stage'in en iyi modeli için
- **SHAP dependence plot:** Her stage'in top-5 feature'ı için
- Finansal mantıklılık tartışması

### 5.4 Karar Sınırı Görselleştirmesi
- PCA ile feature'ları 2D'ye projekte et
- Her stage'den en az 1 sınıflandırıcı için decision boundary çiz
- Rapor gereksinimine doğrudan yanıt

### 5.5 Backtesting (`src/evaluation/backtester.py`, `09_backtesting.ipynb`)
- **Strateji:**
  ```python
  for each day in test period:
      signal = pipeline.predict(features_t)
      if signal == "Buy" and position != "long": buy at close
      elif signal == "Sell": sell/close long
      # Hold = no action
  ```
- **Metrikler:**
  - Kümülatif getiri (equity curve grafiği)
  - Yıllık getiri
  - Sharpe Ratio (yıllıklandırılmış)
  - Maximum Drawdown
  - Win Rate (karlı trade %)
  - Toplam trade sayısı / Turnover
  - Profit Factor
- **İşlem maliyetleri:**
  - %0.1 ücret + %0.05 slippage per trade
  - Maliyetli ve maliyetsiz karşılaştırma
- **Benchmark:** Buy-and-Hold

### 5.6 Rejim Analizi (Bonus)
- Test setini Stage 2 tahminlerine göre segmentlere ayır
- Rejim bazlı metrikler raporla

---

## FAZ 6: Web Uygulaması & API

### 6.1 FastAPI Backend (`app/main.py`)
```
POST /predict
  Input: {"symbol": "BTC", "date": "optional"}
  Output: {signal, confidence, trend_probs, macro_regime_probs, signal_probs}

GET /health
  Output: {"status": "ok"}

GET /indicators/{symbol}
  Output: son hesaplanmış indikatörler
```
- Model yükleme: startup event'de joblib ile
- İndikatör hesaplama: on-the-fly veya önceden hesaplanmış
- CORS middleware
- Hata yönetimi (try/except, HTTP exception'lar)

### 6.2 Web Frontend (`app/static/`)
- **index.html:** Ana sayfa
  - Dropdown: BTC veya ETH seçimi
  - Feature input formu: anahtar indikatör değerlerini manuel girme
  - CSV dosya yükleme (OHLCV verisi) — guideline gereksinimine yanıt
  - "Get Prediction" butonu
- **style.css:** Temiz, profesyonel tasarım
  - Renk kodlu sinyal gösterimi (Buy=yeşil, Sell=kırmızı, Hold=sarı)
  - Confidence bar/gauge
- **app.js:**
  - API çağrıları (fetch)
  - Sonuç görüntüleme: sinyal + güven skoru
  - Stage breakdown: trend + makro rejim + final sinyal ayrıntısı
  - Opsiyonel: basit fiyat grafiği (Chart.js veya Plotly.js)

### 6.3 Docker Deployment (`docker/`)
- **Dockerfile:**
  ```dockerfile
  FROM python:3.11-slim
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY app/ /app/
  COPY src/ /src/
  WORKDIR /app
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- **docker-compose.yml:** Tek komutla ayağa kaldırma
- **Demo checklist:**
  - [ ] Container hatasız başlıyor
  - [ ] Web arayüz tarayıcıda açılıyor
  - [ ] Coin seçip prediction alınıyor
  - [ ] Confidence skorları gösteriliyor
  - [ ] <5 saniye yanıt süresi

---

## FAZ 7: Final Rapor & Sunum

### 7.1 Final Rapor (6-12 sayfa, IEEE format)
1. **Başlık** — ekip üyeleri ile
2. **Özet** (150-250 kelime, geçmiş zaman)
3. **Giriş + Literatür Taraması** — 10+ akademik referans
4. **Materyal ve Yöntemler:**
   - Veri seti: kaynak, örnek sayısı, feature sayısı, sınıf sayısı, denge, ön işleme
   - Matematiksel formülasyon
   - Model açıklamaları + varsayımlar
   - Eğitim prosedürü: walk-forward, OOF
5. **Deneysel Kurulum:**
   - Train/test bölme stratejisi
   - HP tuning yöntemi
   - Yazılım ortamı
   - Değerlendirme metrikleri
6. **Sonuçlar:**
   - Confusion matrisleri
   - Performans tabloları
   - Yöntem karşılaştırmaları
   - ROC eğrileri
   - Decision boundary görselleştirmesi (PCA 2D)
   - SHAP plotları
   - Backtesting equity curve'leri
   - İstatistiksel anlamlılık testleri
7. **Teorik Analiz:**
   - Model varsayımları
   - Neden bu modeller
   - Bias-variance tradeoff
   - Pattern Recognition dersi bağlantısı
8. **Sonuç** — bulgular, en iyi yöntem, limitasyonlar, gelecek çalışma
9. **Referanslar** (IEEE format, [1], [2], ...)

### 7.2 Sunum (5+5 dakika)
- **PowerPoint (5 dk):** 7 slide
  1. Başlık + ekip
  2. Problem tanımı + motivasyon
  3. Pipeline mimari diyagramı
  4. Veri seti + feature özeti
  5. Anahtar sonuçlar (tablo)
  6. Backtesting sonuçları (equity curve, Sharpe)
  7. Sonuç + gelecek çalışma
- **Canlı Demo (5 dk):**
  - `docker-compose up` gösterimi
  - Web arayüzde BTC → prediction
  - ETH → prediction
  - Feature input / CSV upload → prediction
- **Yedek:** Çalışan demo video kaydı

---

## UYGULAMA SIRASI ve BAĞIMLILIKLAR

```
FAZ 0 ──→ FAZ 1 ──→ FAZ 2 ──→ FAZ 3 ──→ FAZ 4 ──→ FAZ 5
  │                                                    │
  │         (paralel çalışılabilir)                     ▼
  │                                               FAZ 6
  │                                                    │
  └────────────────────────────────────────────────→ FAZ 7
```

**Tahmini süre:**
- FAZ 0: ~1 saat (iskelet + ortam)
- FAZ 1: ~3-4 saat (veri toplama + hizalama)
- FAZ 2: ~4-5 saat (feature engineering + selection)
- FAZ 3: ~3-4 saat (label generation + kümeleme)
- FAZ 4: ~8-10 saat (24 deney + OOF + tuning)
- FAZ 5: ~4-5 saat (değerlendirme + backtesting)
- FAZ 6: ~4-5 saat (API + frontend + Docker)
- FAZ 7: ~6-8 saat (rapor + sunum)

---

## DOĞRULAMA

Her faz sonunda kontrol:
- [ ] Kod çalışıyor, hata yok
- [ ] Notebook'lar temiz çıktı veriyor
- [ ] Veri sızıntısı (leakage) yok — assertion'lar geçiyor
- [ ] Test seti hiçbir aşamada eğitime karışmamış
- [ ] Tüm metrikler anlamlı ve rapor edilebilir
- [ ] Docker container ayağa kalkıyor
- [ ] Web arayüz çalışıyor
- [ ] <5 saniye yanıt süresi sağlanıyor
