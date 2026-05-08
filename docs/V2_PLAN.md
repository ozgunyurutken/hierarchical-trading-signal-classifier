# V2 Plan — BBL514E Pattern Recognition Term Project

> **Statü:** Aktif planlama. v1.0 rapor öncesi dondurulmuş referans olarak `v1.0-iter4-final` tag'ında bekliyor (commit `ab408d5`). v2 işleri tamamlanınca rapor stratejisine karar verilecek.

---

## 0. Strateji ve Fallback

v1 (`iter4-final`) sonuçları **rapor edilebilir** bir noktada — XGB Sharpe 1.35, 5/7 model B&H'i risk-adjusted bazda geçiyor. Ancak **methodology açısından 4 önemli eksik** var:

1. **Ablation çalışması yok** (`pipeline.py`'da kod yazılı, `notebooks/08` boş)
2. **Test seti küçük** (462 gün, tek bir piyasa rejimi)
3. **Notebook'lar stale** (04 ve 07 hâlâ v2 sonuçlarını gösteriyor)
4. **ETH modellenmedi** (data hazır, pipeline koşturulmadı)

v2 bu eksikleri gideren bir sürüm. İki olası sonuç:

- **v2 daha iyi (veya en az eşit):** Final rapor v2 üzerinden yazılır.
- **v2 daha kötü veya başarısız:** Rapor `v1.0-iter4-final` tag'ı üzerinden yazılır. v2 çıktıları "deneysel/keşifsel" notuyla eklenir veya tamamen dışarda bırakılır.

Bu nedenle v1 tag'ına dokunulmaz — her zaman geri dönülebilir referans noktası.

---

## 1. v1 Referans Özeti (fallback durumunda)

| Bilgi | Değer |
|---|---|
| Tag | `v1.0-iter4-final` |
| Commit | `ab408d5` |
| Best model | XGBoost (Phase B) |
| Best Sharpe | 1.35 (Return +42.7%, Win 84.6%, MaxDD -11.4%) |
| Best Win% | ZZ-XGB 91.7% |
| Test seti | 462 gün (2024-09-25 → 2025-12-30 yaklaşık) |
| 5/7 model | B&H Sharpe 0.75 üstünde |
| BTC absolute return | Hiçbir model B&H +%47.6'yı geçemedi |

---

## 2. v2 Branch Yapısı

Paralel çalışmak üzere 3 (+1 opsiyonel) branch:

```
main (yok — şu an claude/review-checkpoint-results-2hh1j)
│
├── v2/ablation              ← Konu A: Flat / 2-stage / 3-stage karşılaştırması
├── v2/bigger-dataset        ← Konu B: 2014 öncesi BTC verisi (alt kaynak)
├── v2/notebook-refresh      ← Konu C: notebook 04 + 07 v4 CSV ile re-run
└── v2/eth-pipeline          ← Konu D: ETH modeli (opsiyonel)
```

Hepsi `v1.0-iter4-final` tag'ından (= ab408d5) branch eder. Her biri kendi commit'lerini birikitirir, v2 entegrasyonu (A+B birleşimi) ayrı bir aşama.

---

## 3. Konu A — Ablation Çalışması (öncelikli)

### Amaç
Hiyerarşik 3-stage mimarinin gerçek katma değerini sayısal olarak göstermek. Rapor için en kritik methodology kanıtı.

### Hipotez
H1: 3-stage Sharpe > 2-stage Sharpe > flat Sharpe.
H0 (null): aralarında fark yok → mimari overengineered.

### Test Konfigürasyonları (4 adet)

| Config | Stage 1 (s1) | Stage 2 (s2) | Stage 3 girdi |
|---|---|---|---|
| A1 — Flat (1-stage) | yok | yok | tech (~65) |
| A2 — 2-stage Trend-only | var | yok | tech + s1 (3) = ~68 |
| A3 — 2-stage Macro-only | yok | var | tech + s2 (3) = ~68 |
| A4 — 3-stage Full | var | var | tech + s1 + s2 = ~71 (mevcut v4) |

### Yöntem
- En iyi model: **XGBoost** (v1'de Sharpe 1.35) — diğer 4 model çıkar, sadece bir model üzerinden temiz karşılaştırma.
- Aynı v4 verisi (BTC 2014-09 → 2025-12, test 462 gün) → kontrollü kıyas.
- `src/models/pipeline.py`'daki `FlatBaselinePipeline` ve `TwoStagePipeline` sınıfları zaten yazılı, sadece çalıştırma scripti eksik.
- Optuna 8 trial × walk-forward CV (12-ay min train, 6-ay step) → 4 config × ~5 dk = ~20 dk wall-clock.

### Çıktılar
- `data/labels/btc_ablation_summary.csv` — 4 satır (config × {Sharpe, MaxDD, Return, Win%, F1, MCC})
- `reports/ablation_comparison.png` — bar plot (Sharpe + Return)
- `reports/ablation_equity.png` — 4 equity curve + B&H

### Başarı Kriteri
- **Strict success:** A4 Sharpe ≥ A3, A2 Sharpe ≥ A1, ve A4 - A1 farkı ≥ +0.20 (yani 3-stage gerçekten katma değer).
- **Soft success:** A4 ≥ A1 (en azından mimarinin zararı yok).
- **Failure:** A1 (flat) en yüksek Sharpe — bu durumda tezi gözden geçir, rapor "ablation revealed limited gain from hierarchical structure" çerçevesinde yazılır.

### Risk
**A1 flat baseline aynı veya daha iyi olabilir.** Crypto teknik göstergeleri zaten rejim bilgisini implicitly içerir (RSI extremes, Bollinger band squeeze vb.). Bu durumda akademik framing dürüst yapılmalı, gizlenmemeli.

---

## 4. Konu B — Veri Seti Genişletme

### Amaç
Test setini büyütmek (462 → ~825+ gün) ve modelin farklı piyasa rejimleri (2014 öncesi düşük likidite, 2017 ICO mania, 2018 bear) üzerinde test edilmesini sağlamak.

### Veri Kaynağı Seçimi

| Kaynak | Başlangıç | Free tier | Format | Notlar |
|---|---|---|---|---|
| **CoinGecko** | 2013-04-28 | 10K req/month | JSON REST | OHLC günlük endpoint, history 365 gün limit ücretsiz; full history pro |
| **CryptoCompare** | 2010-07-17 | 100K req/month | JSON REST | `histoday` endpoint full history free |
| **Bitstamp** historical CSV | 2011-09 | unlimited | CSV (raw trades) | Daily aggregation manuel |
| Coinbase pro / Kraken API | 2014-12 / 2013-09 | API limits | JSON | yfinance kapsamına yakın |

**Tercih:** CryptoCompare `histoday` endpoint — 2010 başına kadar geri gider, free tier yeterli, OHLCV veriyor.

### Yöntem
1. `src/data/cryptocompare_loader.py` — yeni modül, `histoday` çağrısı, OHLCV → DataFrame.
2. yfinance verisi ile **çakışma kontrolü** (2014-09-17 sonrası): aynı tarihlerde Close fiyatı ±%2 fark içinde mi? Eğer varsa, CryptoCompare'ı baz al, sonra yfinance overlay (likidite normalleştikten sonra Yahoo daha güvenilir kabul edilir; 2014 öncesi sadece CryptoCompare).
3. FRED daily/monthly hâlâ 2013'ten başlıyor → 2010-2013 BTC için makro feature **NaN olur** → ya warm-up drop (500 gün kaybı), ya da makro feature'lar zero-fill / sentinel değer ile gönder (Stage 2 GMM'in alıştırılmasında risk).
4. Aligned'ı yeniden üret: `btc_aligned_v2.csv` (~5500 sat × 22 kol).
5. Stage 1/2/3 hepsini yeniden train.

### Risk Analizi
- **2010-2013 likidite çok düşük** — günlük volume bazen <1M USD, fiyat manipülasyonu / wash trading common.
- **Label kalitesi:** ATR-adaptive forward return label'ı 2010-2013 dönemde anlamlı mı? Volatility ortamı çok farklı (ATR/price ratio bugünden 5-10x büyük olabilir). `config.yaml`'daki `k=0.7` adaptive sabiti yeniden tune gerekebilir.
- **FRED warm-up:** monthly FRED 2013'ten itibaren var, 2010-2013 için NaN. İki seçenek: ya bu dönemi drop et (B'nin kazancını azaltır), ya da makro feature'ları "missing as a feature" ile encode et (XGB nan-aware, ama Stage 2 GMM nan-aware değil → cluster posterior bozulur).

### Başarı Kriteri
- Test seti ≥ 700 gün ve modelde Sharpe v1'den **kötüleşmiyor**.
- 2010-2013 dönemi train'e dahil → 2014 sonrası test'te **eski rejimleri görmemiş** model daha esnek olmalı (transfer benefit).

### Fallback
2010-2013 verisi yararlı sonuç vermezse, sadece 2013-04 (CoinGecko start) sonrasını kullan → +500 gün kazanç, FRED warm-up sorunu çözülür.

---

## 5. Konu C — Notebook Refresh (kritik, bağımsız)

### Amaç
Sunum / teslim için notebook'ların v4 sonuçlarını göstermesi.

### Yapılacaklar
1. **`notebooks/04_label_generation.ipynb`:**
   - `stage2_feature_names` listesi 8 → 11 (FEDFUNDS, real_interest_rate, UNRATE eklenecek)
   - GMM cluster çıktıları yeniden render
   - Cluster characteristic plot'lar regen
2. **`notebooks/07_evaluation.ipynb`:**
   - `data/labels/btc_test_signals_v2.csv`, `btc_backtest_v2_*.csv`, `final_iter2_summary_table.csv` yeniden okunmalı
   - Confusion matrix, ROC, equity curve plot'ları regen (`reports/iter2_*.png` zaten v4)
3. **`notebooks/08_ablation_study.ipynb`:**
   - Konu A tamamlanınca buradan görselleştirme çalışacak.

### Süre
~2-3 saat. Konu A ve B'den bağımsız → paralel.

---

## 6. Konu D — ETH Pipeline (opsiyonel)

### Amaç
İkinci bir kripto için aynı pipeline'ı koştur → BTC'ye specific olmadığını göster.

### Yapılacaklar
- `eth_aligned.csv` zaten 2,857 sat × 22 kol hazır.
- `notebooks/03_data_alignment.ipynb` ETH için zaten çalıştırılmış.
- Stage 1, 2, 3 hepsi ETH için ayrı çalıştırılacak (CLAUDE.md "BTC ve ETH ayrı modellenir").
- `scripts/rerun_v4_after_monthly_fred.py`'in ETH versiyonu yazılmalı (path'ler `eth_*` olacak).

### Süre
~4-6 saat (BTC v4 retrain'in yarısı kadar — daha az veri).

### Önem
Düşük öncelik. Eğer A ve B yeterince güçlü çıktı → bu opsiyonel. Eğer A/B sönükse → ETH karşılaştırmasıyla rapor zenginleştirilebilir.

---

## 7. Entegrasyon Stratejisi

### Senaryo 1: A başarılı, B başarılı (en iyi durum)
- v2.1 entegrasyonu: B'nin geniş verisi üzerinde A'nın ablation'ını yeniden koştur.
- Final tag: `v3.0-final` (A+B birleştirilmiş, full retrain).
- Rapor v3.0 üzerinden yazılır.

### Senaryo 2: A başarılı, B başarısız
- A'yı `claude/review-checkpoint-results-2hh1j` ana branch'ine merge et.
- B'yi `v2/bigger-dataset` branch'inde "future work" olarak bırak.
- Final tag: `v2.0-final`.

### Senaryo 3: A başarısız, B başarılı
- B'yi merge et. A çıktıları "ablation reveals hierarchical structure provides only marginal gain" olarak rapora dürüstçe yansıt.
- Final tag: `v2.0-final`.

### Senaryo 4: İkisi de başarısız
- v1'e dön. Rapor `v1.0-iter4-final` üzerinden yazılır.
- v2 çıktıları silinmez ama rapora konmaz.

---

## 8. Süre Tahmini

| Konu | Wall-clock | Compute |
|---|---|---|
| A. Ablation | 4-6 saat | ~30 dk retrain |
| B. Bigger dataset | 1-2 gün | 1-2 saat retrain (yeni veri pipeline + warm-up sorunu) |
| C. Notebook refresh | 2-3 saat | ~10 dk re-run |
| D. ETH pipeline | 4-6 saat | ~1 saat retrain |

**Toplam (paralel çalışma):** ~3 takvim günü. v1 deadline 10 Mayıs, bugün 8 Mayıs → **2 gün buffer var.**

---

## 9. Risk Özeti

| Risk | Olasılık | Etki | Mitigasyon |
|---|---|---|---|
| A1 (flat) Sharpe ≥ A4 | Orta | Tez zayıflar | Dürüstçe rapora yansıt; "marginal gain" dili kullan |
| 2010-2013 erken BTC verisi label'ı bozar | Yüksek | B kazanç sıfır | CoinGecko 2013-04'ten başla → 2010-2013'ü drop et |
| FRED warm-up B'de problem yaratır | Yüksek | Stage 2 NaN | Drop warm-up + raporda yöntem notu |
| Notebook 04 rerun'da kernel crash | Düşük | C gecikir | Cell cell çalıştır, intermediate state kaydet |
| Süre buffer yetersiz | Orta | v2 yarım kalır | C'yi her şart altında yap (en kısa); D'yi opsiyonel bırak; A>B sırası |

---

## 10. Karar Noktaları (decision gates)

1. **A tamamlandığında:**
   - A1 ≥ A4 mi? → tezi gözden geçir, B'ye devam et ama rapor odağı değişir
   - A4 net üstünse → B sonrası entegrasyona git

2. **B tamamlandığında:**
   - Yeni veri ile Sharpe v1'den iyi mi? → entegre et
   - Eşit veya kötü mü? → v1'in fallback olduğunu hatırla, B'yi ileri sürüme bırak

3. **C tamamlandığında:**
   - Sunum/teslim için her halükârda gerekli, başarı/başarısızlık kavramı yok

4. **Final karar:** A + B sonuçları sentezle, rapor yazımı başlamadan önce hangi tag'ı baz alacağına net karar.

---

_Bu plan v1 rapor öncesi dondurulmuş checkpoint hâliyle birlikte git'e işlenecek (`docs/V2_PLAN.md`)._
