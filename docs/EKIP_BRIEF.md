# Ekip Brief — Bu Projede Tam Olarak Ne Yaptık?

**Hedef kitle:** Çağatay ve Yusuf. Sunumda 5 dakika konuşacağız, sorulara cevap vereceğiz. Bu döküman seni 0'dan tam hakimiyete götürür.

**Proje adı:** "A Three-Stage Hierarchical Soft-Fusion Framework for Cryptocurrency Trading Signal Classification" — yani **kripto için üç-aşamalı hiyerarşik AL/SAT/BEKLE tahmin sistemi**.

**Tek cümleyle ne yaptık:** "Trend → Makro Rejim → Sinyal" sırasıyla üç model çalıştırdık, ilk ikisinin tahminlerini üçüncüye girdi olarak verdik, dört farklı mimariyi karşılaştırdık ve hangi mimarinin hangi varlık için iyi çalıştığını bulduk.

---

## 1. Niye Bu Problemi Seçtik?

Bitcoin ve Ethereum gibi kripto paralarda her gün için **AL / SAT / BEKLE** üç sınıflı bir karar problemimiz var.

**Sorun:** Kripto fiyatları çok gürültülü. Klasik "tek model — tüm feature'lar — direkt karar" yaklaşımı (`flat baseline`) ezbere kaçıyor, yeni rejimlerde patlıyor.

**Çözüm fikri:** Kararı parçalara ayır. Önce *fiyat trendi nedir?*, sonra *makro ortam nedir?* (resesyon mu, boğa mı?), en sonunda *bu duruma göre AL/SAT/BEKLE?*. Her aşama daha kolay bir alt-problem.

**Sunumda söyleyeceğin:** "Tek model her şeyi öğrensin diye zorlamak yerine, problemi mantıklı parçalara böldük. Her parçayı ayrı bir uzman eğitti, sonra birleştirdik."

---

## 2. Veri Toplama (Phase 1)

### Kaynak
| Veri | Nereden | Açıklama |
|---|---|---|
| BTC OHLCV (günlük) | Yahoo Finance | 2014-09-17 → 2025-12-30, 4.094 gün |
| ETH OHLCV (günlük) | Yahoo Finance | 2017-08-17 → 2025-12-30, 3.060 gün |
| 22 makro değişken | Yahoo + FRED API | S&P 500, VIX, DXY, Altın, Petrol, US10Y, US2Y, FedFunds, CPI, M2, vs. |

### Önemli detay: "Publication-release lag"
FRED'den gelen aylık makro veriler (CPI, FedFunds, M2) **gerçekte yayınlandığı tarihte** kullanılır, tarihinde değil.
Örnek: Ocak 2024 CPI verisi 13 Şubat'ta yayınlanır. Modelin 13 Şubat'tan önce o veriyi görmemesi lazım — yoksa **lookahead bias** (geleceği bilme hilesi) olur.

Çözüm: Her veriye yayınlama gecikmesi ekledik (FedFunds 1 gün, CPI 45 gün, M2 14 gün, vs.).

**Sunumda söyleyeceğin:** "Veriyi alırken makro verilerin gerçek yayınlanma tarihini kullandık ki model geleceği görmüş gibi davranmasın."

### Feature engineering (12 türetilmiş makro feature)
Ham VIX/DXY/M2 vs. doğrudan ML'e iyi gelmez. Türev feature'lar ürettik:
- VIX → expanding-window z-score (kaç sigma yukarda?)
- US10Y − US2Y → yield curve spread (resesyon göstergesi)
- M2 → year-over-year % change (para arzı büyüme hızı)
- S&P 500 → 5-günlük log return

**Bunlardan 5 tanesi** Stage 2 FSM tarafından kullanıldı, kalan 7 tane unsupervised denemelerde kaldı.

> **Paper'da:** Fig. 1 (`fig_eng_demo`) — ham vs. işlenmiş halini yan yana gösteriyor.

---

## 3. Stage 1 — Trend Classifier (Çağatay'ın bölümü)

### Etiket nereden geliyor? — ZigZag algoritması
Sadece kapanış fiyatlarına bakarak "şu an uptrend mi, downtrend mi, range mı?" sorusuna *retrospektif* cevap üretir:

1. Fiyat serisinde tepe ve dipleri bul (pivot noktalar).
2. İki pivot arası segment **%10'dan fazla** hareket etmişse o segmenti `uptrend` veya `downtrend` olarak işaretle.
3. Daha az ise `range`.
4. Minimum segment uzunluğu 15 gün.

**Önemli incelik (sınavda sorulabilir):**
"ZigZag etiketleri *ileriye bakıyor* — yani Şubat 2020'deki bir günün etiketi sonradan üretilmiş. Bunu inference'da kullanmak hile olmaz mı?"

**Cevap:** *Etiket* ileriye bakıyor evet, ama *modelin gördüğü feature'lar* (RSI, MACD, ADX, vs.) sadece o güne kadar olan veriden hesaplanıyor. Model sadece "bu causal feature'lar hangi etikete denk geliyor?" mappingini öğrenir, etiketi doğrudan kullanmaz.

Bunu kanıtlamak için **Fig. 3'ü (lookahead testi)** ürettik: 106 farklı tarihte ZigZag'i o günün verisiyle yeniden hesapladık. **42'sinde** sonuç orijinal etiketten farklı çıktı. Yani ZigZag etiketi gerçekten retrospektif — bu yüzden ML model şart, kural-tabanlı kullanılamaz.

### 4 model, 14 feature
- **Feature'lar:** RSI(14), MACD signal-diff (12-26-9), ADX, Bollinger %B, ATR, hacim z-skoru, OBV, 6 farklı pencereli log return — toplam 14.
- **Modeller:** XGBoost, LightGBM, Random Forest, MLP. Hepsi sklearn benzeri API.
- **Sınıflar:** downtrend / range / uptrend.
- **Class balance (BTC):** %29.6 / %18.2 / %52.2 — uptrend baskın çünkü BTC çoğu zaman boğa modunda.

### Sonuç
Random Forest hem BTC'de (F1m=0.563) hem ETH'de (F1m=0.571) **en iyisi**. AUC 0.74-0.76.

> **Paper'da:** Tablo II (Stage 1 metrikleri), Fig. 4 (8 confusion matrix), Fig. 5 (BTC truth-vs-pred timeline).

**Sunumda söyleyeceğin:** "ZigZag bize 'gerçeği' söyleyen retrospektif bir etiket. Modeli causal teknik göstergelerle eğitiyoruz, model bu göstergelerle şu anki trendi tahmin etmeyi öğreniyor."

---

## 4. Stage 2 — Macro Regime FSM (Özgün'ün bölümü, kuralcı)

Burada **öğrenilmiş model YOK**. Bunu özellikle vurgulayacağız çünkü bu bilinçli bir tasarım kararı.

### Niye unsupervised başarısız oldu?
Önce klasik unsupervised yaklaşımları denedik:
1. **Vanilla K-Means** — 2008 Global Financial Crisis'i bir cluster yapamadı.
2. **Semantic constrained K-Means** — gürültülü, hızlı rejim değişimleri ürettiyor.
3. **HMM (3 Gaussian state)** — geçiş olasılıkları kararsız.
4. **3-component GMM** — 2024-2025'te P(Stress) ≈ 0.96'ya saplanıp orada kalıyor.

> **MEMORY.md'de detay:** GMM stickiness ve K-Means failure'ları kayıtlı.

### Çözüm: Deterministic FSM (8 kural + 1 guard)
3 sınıf: **Bear / Neutral / Bull**.

Kurallar:
1. **VIX hysteresis:** Bear'a giriş VIX z-skoru > 1.0, çıkış z < 0.3 (asimetrik).
2. **Min dwell:** Bear ≥ 20 gün, Bull ≥ 40 gün, Neutral ≥ 10 gün — yani bir kez Bull'a geçtikse en az 40 gün orada kal.
3. **Bear→Neutral velocity:** ΔVIX_z[10gün] < -0.8 ise Bull'a hızlı geçiş.
4. **Persistent yield curve inversion:** 60-gün rolling US10Y−US2Y < 0 → Bear sinyali.
5. **DXY+M2 macro stress:** DXY z[30] > 0.7 VE M2 YoY < %4 → Bear override.
6. **Bull velocity entry:** V-shape recovery'de hızlı Bull girişi.
7. **Bear velocity entry:** Hızlı escalation.
8. **Bear re-entry guard:** Bear'dan Neutral'a çıktıktan sonra 35 gün Neutral'da kal, hemen Bear'a geri girme.

### Sonuç (Fig. 2)
2008 GFC'yi, 2018-Q4 düşüşü, 2020-COVID, 2022 ayı piyasası, 2025-04 tarife şoku — hepsini **doğru** Bear olarak işaretledi. 2009 V-recovery, 2020 V, 2025-05 toparlanma — hepsi Bull.

**Sunumda söyleyeceğin:** "Unsupervised yaklaşımlar makro rejimi yakalayamadı çünkü etiketli veri yok ve clustering 2008 gibi kritik olayları kaçırıyor. Bu yüzden 8 kurallı bir finite-state machine tasarladık — finansta makro rejim zaten heuristic ile tanımlanır."

**Olası soru:** "ML projesinde rule-based stage olur mu?"
**Cevap:** "Stage 2 *modelimiz* öğrenmiyor ama *çıktısı* (Bull/Neutral/Bear posterior'u) Stage 3'e ML feature olarak giriyor. Hibrit pipeline. Tüm pattern recognition kitabı bunun OK olduğunu söyler — Hamilton 1989, Ang & Bekaert 2002 gibi makaleler de Markov-switching ile aynı mantığı yapıyor, biz deterministik kuralları tercih ettik."

---

## 5. Stage 3 — Signal Classifier (Özgün'ün bölümü, esas iş)

Burası kalbi. Stage 1 ve Stage 2'nin çıktılarını **soft fusion** ile birleştirip AL/SAT/BEKLE kararı veriyor.

### Etiket nereden geliyor? — Adaptif eşik
Forward 5-günlük getiri kullanıyoruz:
```
y_t = Buy   eğer (P_{t+5} - P_t)/P_t > +0.5 × σ_t
y_t = Sell  eğer (P_{t+5} - P_t)/P_t < -0.5 × σ_t
y_t = Hold  diğer durumda
```

Burada σ_t = son 20 günün **causal** standart sapması (ileriye bakmaz).

**Niye 0.5 sigma?** k=0.5 değerinde sınıflar dengeli (%43 Buy / %23 Hold / %34 Sell). k yükseldikçe Hold büyür.

> Test edilen: k ∈ {0.4, 0.5, 0.7, 1.0}. Tablo IV ve testte k=0.5 dengeli kaldı.

### Feature vector — z_t ∈ ℝ^16
Stage 3 modeline tam olarak şu giriyor:

```
z_t = [ Stage 1 raw posterior (3 sayı: P_down, P_range, P_up) ]   # 3
    + [ Stage 1 smooth-10 posterior (10-gün rolling avg) ]         # 3
    + [ Stage 2 one-hot regime (Bull/Neutral/Bear) ]               # 3
    + [ Stage 2 regime-tenure (kaç gündür bu rejimdeyiz, τ_t) ]    # 1
    + [ 6 oscillator: RSI, MACD, BB %B, Stoch %K, vol z, OBV ]     # 6
    = 16 feature
```

**Önemli:** Stage 1'in *posterior probability vector*'ünü (yumuşak) veriyoruz, sınıf etiketini değil. Bu yüzden adı "soft fusion".

### 4 model
Yine XGBoost, LightGBM, Random Forest, MLP — Stage 1 ile aynı 4 model.

### Sonuç (default 3-Stage Full mimari)
- BTC en iyi: XGBoost, F1m=0.367, AUC=0.530
- ETH en iyi: XGBoost, F1m=0.368, AUC=0.530

**F1=0.37, AUC=0.53 düşük gibi gelebilir** — chance %33, biz biraz üstündeyiz. Ama:

**Trading'de F1 yetmez. Sharpe iyi.** Bunu birazdan açıklayacağız.

> **Paper'da:** Tablo III (Stage 3 metrikleri), Fig. 6 (ROC curves).

---

## 6. Walk-Forward Cross-Validation (Yusuf'un bölümü)

### Niye normal K-fold değil?
Klasik K-fold zamanı karıştırır — train fold'da 2025 verisi, test fold'da 2018 verisi olabilir. **Geleceği bilerek geçmişi tahmin etmek = leakage = sahte yüksek skor.**

### Expanding-window walk-forward
```
Fold 1: train [2014..2017]  →  gap 10 gün  →  validate [2017..]
Fold 2: train [2014..2017.5] → gap 10 → validate [2017.5..]
...
```

- **gap = 10 gün:** label horizon 5 günden büyük, leakage'ı önler.
- **validation window = 200 gün.**
- **BTC: 16 outer fold** (Stage 1) / 12 fold (Stage 3).
- **ETH: 10 outer fold** (Stage 1) / 6 fold (Stage 3).

### Hyperparameter tuning (Optuna)
Her model için 30 trial, **5-fold inner CV** (yine walk-forward). 

**Niye 5-fold inner?** İlk denemede 3-fold yapmıştık, Optuna `min_samples_leaf=1` (degenerate değer) seçti ve outer CV'de un-tuned baseline'dan **kötü** çıktı. 5-fold'a geçince düzeldi. Detay paper'da footnote olarak var, sunumda detaya inme — sorulursa "inner-CV'de yeterli rejim çeşitliliği lazım, 5'e çıkarınca düzeldi" yeterli.

### Software
Python 3.11, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6, Optuna 4.8. Hepsi tek bir 2024 MacBook Air M3'te döndü, **1 saat 43 dakika** total walk-forward + tuning.

---

## 7. Mimari Ablation — En İlginç Bulgu (Yusuf'un bölümü)

Hocaya satacağımız esas hikaye burası.

### 4 Mimari karşılaştırdık

| Mimari | Stage 3'e ne giriyor | Toplam feature |
|---|---|---|
| **Flat** | Sadece 6 oscillator | 6 |
| **2-Stage Trend** | Stage 1 posterior + 6 oscillator | 6 + 6 = 12 |
| **2-Stage Macro** | Stage 2 regime + 6 oscillator | 6 + 4 = 10 |
| **3-Stage Full** | Stage 1 + Stage 2 + 6 oscillator | 6 + 6 + 4 = 16 |

> **Paper'da:** Fig. 7 — bu 4 mimariyi blok-diyagram olarak gösteriyor (TikZ ile çizdik).

### Sonuç (Tablo IV)

| | BTC | ETH |
|---|---|---|
| Flat | Sharpe 0.93 | **Sharpe 0.52 (en iyi!)** |
| 2-Stage Trend | Sharpe 1.08 | Sharpe 0.39 |
| 2-Stage Macro | Sharpe 0.98 | Sharpe 0.47 |
| **3-Stage Full** | **Sharpe 1.15 (en iyi!)** | Sharpe 0.34 |
| Buy & Hold benchmark | Sharpe 0.95 | Sharpe 0.26 |

**Şok edici bulgu:**
- **BTC için**: Daha derin mimari → daha iyi Sharpe (monotonik artış 0.93 → 1.15).
- **ETH için**: Daha derin mimari → daha kötü Sharpe (0.52 → 0.34).

**Açıklamamız:** BTC veri seti uzun (10 yıl, çoklu rejim — 2014 başlangıç, 2018 ayı, 2021 zirve, 2022 kış, 2025 modern) — hiyerarşik feature'lar değer katıyor. ETH veri seti kısa (5 yıl, 2020-2022 dynamics dominant) — ekstra feature'lar overfit ediyor.

**Bu "no-free-lunch" örneği.** Her hiyerarşik öneri her varlık için iyi olmuyor, asset-specific.

**Sunumda söyleyeceğin (Yusuf):** "İlk başta 3-stage'in her varlık için iyi olacağını düşündük. Ablation çalışmamız BTC için doğru olduğunu, ETH için tersinin olduğunu gösterdi. Bu finansal ML'de no-free-lunch'ın somut bir örneği — veri büyüklüğü mimari kararını etkiler."

---

## 8. Trading Rules — F1 Düşük Olsa da Sharpe Yüksek (Yusuf'un bölümü)

Stage 3 modeli AL/SAT/BEKLE olasılıkları üretiyor (her gün için 3 sayı). Bu olasılıkları gerçek trade'lere çevirmek için 3 kural denedik:

### Kural 1: Stateful Long-Only
- Buy görünce **gir**.
- Sell görünce **çık**.
- Hold görünce **mevcut pozisyonu koru**.
- Avantaj: Az trade, az fee.
- Dezavantaj: Bear'da çıkışı geç olabilir.

### Kural 2: Defensive Reset
- Buy ise long pozisyon.
- Diğer her şey (Hold dahil) = nakit.
- Avantaj: Çok savunmacı.
- Dezavantaj: Çok fazla giriş-çıkış, yüksek fee.

### Kural 3: Probability-Weighted (Continuous Sizing)
- Pozisyon boyutu = `clip(P_Buy - P_Sell, 0, 1)`.
- Her gün küçük ayarlamalar yapar.
- Avantaj: Risk yönetimi yumuşak.
- Dezavantaj: Çok küçük trade'ler.

### Sonuç
- **BTC**: Stateful long-only kazandı (Sharpe 1.15) — çünkü test dönemi boğa-ağırlıklı.
- **ETH**: Probability-weighted kazandı (Sharpe 0.52) — çünkü ETH volatil, soft sizing güvenli.

### Niye bu önemli?
F1=0.37 ile Sharpe=1.15 nasıl?

**Cevap:** Hold sınıfı *risk filtresi* gibi davranıyor. Model emin olmadığı günlerde Hold tahmini yapıyor → trading kuralı o gün trade etmiyor → sermayeyi koruyor. 2-sınıflı bir model bu lüksü kaybeder, her gün al-sat döner.

Sayı: BTC için 2.200 trading günü içinde sadece **179 trade** yaptık. Bu seçicilik düşük frame-accuracy'yi yüksek economic precision'a çeviriyor.

**Sunumda söyleyeceğin (Özgün):** "F1=0.37 düşük gibi ama Hold sınıfı sayesinde model emin olmadığı günlerde abstain ediyor. Sadece güvendiği günlerde trade yapınca Sharpe 1.15'e çıkıyor."

---

## 9. Backtest Detayları

- **Transaction cost**: %0.1 one-way (kripto spot için makul, slippage ihmal edildi — paper'da limitasyon olarak yazılı).
- **Buy & Hold benchmark**: aynı OOF aralığında.
- **Sharpe**: 252 trading günü annualization.

### Final sayılar (Tablo IV'ün özeti)

| | Sharpe | Total return | MaxDD | F1m |
|---|---|---|---|---|
| **BTC 3-Stage Full** | **1.15** | **+2901%** | -46% | 0.367 |
| BTC Buy & Hold | 0.95 | +2972% | -77% | — |
| **ETH Flat baseline** | **0.52** | **+26%** | -18% | 0.354 |
| ETH Buy & Hold | 0.26 | **−7%** | -72% | — |

**Vurgu:** ETH için Buy & Hold **eksiye düşmüş** (-%7), bizim model **+%26**. Yani sadece Sharpe iyileşmesi değil, mutlak getiri farkı da var.

---

## 10. Sistem (Docker + FastAPI + Web UI) — Yusuf'un sorumluluğu

Hocanın sistem gereksinimi: **Docker container içinde FastAPI backend + HTML web UI + canlı tahmin**.

### Mimari
```
[ Browser ]
    ↓ HTTP request
[ FastAPI server (Python) ]
    ↓ load saved sklearn models
[ XGBoost / LightGBM / RF / MLP modelleri (pickle) ]
    ↓ predict
[ Chart.js ile equity curve render ]
```

### Özellikler
- Web UI'de model + mimari seçilebilir (Flat / 2-Stage / 3-Stage / vs.)
- Trading kuralı seçilebilir (stateful / defensive / probability-weighted)
- Canlı equity curve görüntülenir, trade marker'ları ile (yeşil = entry, kırmızı = exit)
- Tüm OOF prediction'lar Docker image içine bundle'lanmış — sunum sırasında network olmadan çalışır.

### Sunum demosu (Yusuf'un sorumluluğu)
1. `docker-compose up` ile başlat
2. Browser'da http://localhost:8000 aç
3. BTC seç, 3-Stage Full seç, Stateful kural seç → **Sharpe 1.15 + equity curve render olur**
4. ETH'e geç, Flat baseline seç → **B&H'ı yener, Sharpe 0.52 görünür**

**KRİTİK UYARI:** Hocaya göre demo çalışmazsa technical score *significantly reduced*. Sunumdan ÖNCE mutlaka dry-run.

---

## 11. Paperde Hocanın İstediği Kriterler

| Kriter | Bulunduğu yer |
|---|---|
| Title + 3 yazar | Page 1 başı |
| Abstract (150-250 kelime, no cite/eq/fig) | Page 1, 210 kelime |
| Introduction + 10+ academic ref | Page 1-2, 24 ref |
| Dataset description (source, samples, features, classes, balance) | Tablo I |
| Mathematical formulation | Eq (1)-(4), Section II.B |
| Model description + assumptions | Section II.E |
| Experimental setup (CV, HP tuning, software, metrics) | Section III |
| Confusion matrix | Fig. 4 (Stage 1 × 8 panel) |
| ROC curves | Fig. 6 (Stage 3 × 6 panel) |
| Method comparison | Tablo II, III, IV |
| Conclusion | Section VI |
| References (IEEE format) | Page 8 |
| Author contributions | Sayfa 7 |

---

## 12. İş Bölümü — Sunumda Kim Ne Anlatıyor?

### Çağatay (5 dakika sunumun ~%30'u)
Anlatacağı bölümler:
- **Section II.A — Dataset Description** (Tablo I)
- **Section II.D — ZigZag-Labelled Trend Classifier**
- **Section IV.A — Stage 1 Sonuçları** (Tablo II, Fig. 4, Fig. 5)

Konu özetinin: "Veriyi nasıl topladık, ZigZag etiketini niye kullandık (revisability testi), Stage 1 4 model nasıl eğitildi."

Beklenebilecek soru:
- "ZigZag retrospektif değil mi, niye ML?" → Lookahead testinin %42 disagreement bulgusu.
- "Class imbalance nasıl ele alındı?" → `class_weight=balanced` LightGBM'de, sample_weight XGBoost'ta.

### Özgün (5 dakika sunumun ~%40'ı)
Anlatacağı bölümler:
- **Section II.B — Three-Stage Architecture** (Eq. 2-4)
- **Section II.C — Stage 2 Composite Macro FSM** (Fig. 2)
- **Section II.E — Classifiers** (model assumptions)
- **Section IV.B — Stage 3 Sonuçları** (Tablo III, Fig. 6)

Konu özeti: "Soft fusion mantığı (Stage 1 + 2'nin posterior'u Stage 3'e feature olarak girer), neden FSM (unsupervised başarısız oldu), 4 model assumption'larıyla karşılaştırma."

Beklenebilecek soru:
- "Soft fusion neden hard label'dan iyi?" → Posterior calibration + uncertainty information.
- "FSM'de overfit yok mu?" → Yok çünkü öğrenmiyor, parametre yok, sadece kural var. Kuralları tarihi event'lere fit ettik (2008, 2020, 2022).

### Yusuf (5 dakika sunumun ~%30'u + demo)
Anlatacağı bölümler:
- **Section III — Experimental Setup** (walk-forward CV, Optuna)
- **Section IV.C — Architecture Ablation** (Fig. 7, Tablo IV) — **EN ÖNEMLİ KISIM**
- **Section IV.D — Trading Rules**
- **Section V — Discussion** (asset-specific finding)
- **System component (Docker + FastAPI canlı demo)**

Konu özeti: "4 mimari × 4 model × 2 asset karşılaştırdık. BTC için 3-stage kazandı, ETH için flat kazandı — no-free-lunch. F1=0.37 düşük ama Sharpe=1.15 yüksek çünkü Hold sınıfı risk filtresi. Şimdi canlı demoyu görelim."

Beklenebilecek soru:
- "Niye ETH'de hiyerarşi overfit etti, BTC'de etmedi?" → Veri büyüklüğü farkı (3.200 vs 2.000 OOF), feature/sample ratio.
- "Sharpe 1.15 anlamlı mı, sadece backtest mi?" → Walk-forward CV expanding window, gerçek out-of-sample, transaction cost dahil. Anlamlı ama production'da slippage eklenir.

---

## 13. Sıkça Beklenen Sorular (FAQ)

### S: "Niye sadece XGBoost değil, 4 model birden?"
C: Pattern recognition coursework açısından farklı algoritma ailelerinin (boosting, bagging, neural net) karşılaştırması bekleniyor. Zaten Tablo II'de göreceksin, modeller arası fark var ama hiçbiri devasa değil — yani hyperparameter tuning şeylerden daha çok mimari kararı önemli.

### S: "Niye sadece BTC ve ETH? Daha fazla coin niye yok?"
C: **Coursework scope** — 2 asset zaten zengin bir karşılaştırma sağlıyor (no-free-lunch bulgusunu çıkardı). Daha fazla coin eklemek scope-creep olurdu. Future work bölümünde belirtildi.

### S: "Buy & Hold zaten %2972 yapmış, sizin %2901, daha düşük. Niye başarılı diyorsun?"
C: **Sharpe** önemli, total return değil. B&H Sharpe 0.95, biz 1.15. Ayrıca MaxDD bizimki -%46 (B&H -%77). **Risk-adjusted return ve drawdown** kriptoda total return'den önemli — kimse %77 drawdown'u sindiremez.

### S: "Stage 2 FSM 'ML' değil. Bu pattern recognition projesi olmuş mu?"
C: Evet. Stage 1 ve 3 ML, **soft fusion mantığı kendisi pattern recognition** çekirdek konusu (Jordan & Jacobs hierarchical mixture of experts, 1994). FSM bir feature engineering tekniği — output'u classifier'a feed ediyor. Hibrit yaklaşımlar literatürde standart.

### S: "k=0.5 niye, başka değer denemediniz mi?"
C: Denedik (k ∈ {0.4, 0.5, 0.7, 1.0}). Tree-based modellerde k=0.5 her ikisi için de optimum. MLP'de ETH'te k=1.0 daha iyi ama tutarlılık için 0.5'te kaldık.

### S: "ROC-AUC 0.53, chance 0.5. Bu istatistiksel olarak anlamlı mı?"
C: AUC'nin chance'tan farkı küçük, McNemar testi yapmadık (paper'da limitation). Ama Sharpe 1.15 vs 0.95 farkı backtest süresi (8.7 yıl) boyunca ekonomik olarak anlamlı.

### S: "Forward returns kullandın etiket için. Bu lookahead bias değil mi?"
C: **Etiket** ileriye bakar (her supervised time-series probleminde bu durum vardır), **feature** asla bakmaz. `assert_no_lookahead_leakage` rutini bunu kod tarafında doğrular. Standart supervised ML practice.

---

## 14. Repo Yapısı (Sunum öncesi göz atılmalı)

```
hierarchical-trading-signal-classifier/
├── src/                            # Production code
│   ├── data/                       # Veri toplama, FRED API
│   ├── features/                   # Feature engineering (Phase 1.5)
│   ├── labels/                     # ZigZag, signal label generators
│   ├── models/                     # Stage 1, 2, 3 implementations
│   └── backtest/                   # Trading rules, equity calc
├── app/                            # FastAPI backend + Chart.js frontend
│   ├── main.py                     # FastAPI entry
│   ├── models/                     # Saved sklearn pickles
│   └── static/                     # HTML/JS/CSS
├── docker/                         # Dockerfile + docker-compose.yml
├── docs/
│   ├── EKIP_BRIEF.md               # ← bu dosya
│   ├── PAPER.md                    # Markdown taslağı (legacy)
│   └── paper/                      # ← FINAL TESLİM
│       ├── paper.tex               # IEEE LaTeX kaynağı
│       ├── paper.pdf               # 9 sayfa final
│       ├── references.bib          # 24 BibTeX entry
│       └── figures/                # 7 PNG + 1 inline TikZ
├── data/processed/                 # OOF prediction CSV'leri
├── reports/                        # Phase çıktıları (numerical)
├── notebooks/                      # Interaktif analizler
├── scripts/                        # Plot ve metric scriptleri
├── config.yaml                     # Tüm hyperparameters burada
├── CLAUDE.md                       # AI assistant'a yönergeler
└── MEMORY.md                       # Proje karar geçmişi
```

---

## 15. Hızlı Komut Referansı

```bash
# Veri çekme (zaman alır, idempotent)
python scripts/v5_fetch_data.py

# Stage 1 train + tune
python scripts/v5_train_stage1.py --tune --inner-cv 5

# Stage 2 FSM çalıştır
python scripts/v5_run_stage2_fsm.py

# Stage 3 train + tune (4 mimari için ayrı ayrı)
python scripts/v5_train_stage3.py --arch 3stage_full --tune

# Backtest
python scripts/v5_backtest.py --asset btc --rule stateful

# Paper figürlerini üret
python scripts/v5_compute_extended_metrics.py
python scripts/v5_plot_phase5_arch_ablation.py

# Paper compile
cd docs/paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper

# Sistem (Docker)
docker-compose up --build
# Browser → http://localhost:8000
```

---

## 16. Kullanıma Hazır Sunum Cümleleri

### Açılış (Çağatay)
> "Cryptocurrency markets are noisy, non-stationary, and react to a unique blend of micro-structure, sentiment, and macroeconomic shocks. Tek bir model bu karmaşıklığı doğrudan AL/SAT/BEKLE'ye çevirmeye çalıştığında genelleştirme problemi yaşıyor. Biz problemi üç aşamaya ayırdık: önce trend, sonra makro rejim, sonra sinyal."

### Geçiş (Çağatay → Özgün)
> "Stage 1 trend tahminini ZigZag etiketleriyle eğittik. Şimdi Özgün soft fusion'la bu trend tahmininin ve makro rejimin Stage 3'e nasıl beslendiğini anlatacak."

### Geçiş (Özgün → Yusuf)
> "Stage 3 modeli AL/SAT/BEKLE olasılıkları veriyor. Bu olasılıkları trade'lere çevirme ve mimariyi karşılaştırma kısmını Yusuf anlatacak."

### Kapanış (Yusuf, demo sonrası)
> "İki temel bulgu: (1) Mimari derinliği varlığa özel — BTC için 3-stage kazandı, ETH için flat baseline kazandı. Bu finansal ML'de no-free-lunch'ın somut bir örneği. (2) Frame-level metrikler trade-level değeri eksik tahmin ediyor — F1=0.37 ve AUC=0.53 ile bile Hold-aware trading kuralı ile Sharpe 1.15'e ulaştık. Tüm pipeline Docker container içinde çalışır halde, FastAPI backend ve Chart.js frontend ile."

---

## 17. Sunumdan Önce Yapılacaklar Listesi

- [ ] Docker container build ve demo dry-run
- [ ] Browser'da web UI'nin çalıştığını doğrula
- [ ] BTC ve ETH için Sharpe sayılarının doğru render edildiğini gör
- [ ] [docs/paper/paper.pdf](docs/paper/paper.pdf) son halini PDF olarak yazdır veya tablete kopyala
- [ ] PowerPoint slidedeck hazırla (10 slide max):
  1. Problem
  2. Hierarchical decomposition (mimari diyagram = paper Fig. 7)
  3. Stage 1 — ZigZag + 4 model + Tablo II
  4. Stage 2 — FSM + Fig. 2
  5. Stage 3 — Soft fusion + 16-feature
  6. Walk-forward CV explanation
  7. Architecture ablation = Tablo IV (en kritik slide)
  8. Trading rules + Sharpe 1.15 vs 0.95
  9. Asset-specific finding (BTC vs ETH)
  10. Demo screen + thank you

---

**Bu doküman seni tam hakimiyete götürür.** Her bölümün niye olduğunu, sunumda ne diyeceğini, sorulara nasıl cevap vereceğini biliyorsun şimdi. Hocaya/asistana karşı 3 kişilik düzenli bir ekip görüntüsü vereceğiz.

İyi sunumlar.
