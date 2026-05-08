# Sözlük — İsimlendirme Rehberi

Proje boyunca kullanılan tüm sürüm/phase/config/feature isimlendirmelerinin
tek-yer açıklaması. Yeni bir sonuç yazılırken / rapor yazımı sırasında
referans olarak kullanılacak.

---

## 1. Veri/Pipeline Sürümleri (v1, v2, v3, v4)

Her sürüm aynı veri pipeline + Stage 2 GMM üzerine yapılan **veri kalitesi
düzeltmeleri**. Aralarındaki tek fark hangi macro veri çekildiği ve hangi
bug'ın düzeltildiği.

| Sürüm | Düzeltme | Aligned satır | Stage 2 feat |
|---|---|---:|---:|
| **v1** | İlk MVP — bug'lı US2Y (yfinance ZT=F futures), monthly FRED yok | 4,111 | 8 |
| **v2** | İlk iter2 (Phase A+B+C, 7 model) — hâlâ US2Y bug'ı | 4,011 | 8 |
| **v3** | DGS2 fix (US2Y artık FRED), yield curve Düzeltildi | 3,961 | 8 |
| **v4** | Monthly FRED (FEDFUNDS, CPI, UNRATE, M2, ICSA), Stage 2 GMM 8→11 feat | 3,961 | 11 |

**`v1.0-iter4-final` tag** = v4'ün dondurulmuş hâli (commit `ab408d5`).
Rapor fallback referansı.

---

## 2. Phase A / B / C (iter2 model gruplaması)

Iter2 Stage 3 retrain'leri 3 partide yapıldı, isimler oradan kaldı:

| Phase | Modeller | Stage 1 etiketi |
|---|---|---|
| **Phase A** | LDA, MLP (baseline) | SMA cross |
| **Phase B** | XGBoost, LightGBM, Random Forest (tree models) | SMA cross |
| **Phase C** | ZZ-XGBoost, ZZ-MLP | ZigZag |

Bu yüzden modellerde **`ZZ`** prefix'i = ZigZag Stage 1 OOF kullanan;
**non-ZZ** = SMA-cross Stage 1 OOF kullanan.

---

## 3. Ablation Konfigürasyonları (A1, A2, A3, A4)

Hiyerarşik mimarinin **kademe sayısını test eden** 4 farklı Stage 3 girdi
tasarımı:

| Config | Stage 3 girdisi | Anlamı |
|---|---|---|
| **A1 — Flat** | sadece tech features | 1-stage baseline (kontrol) |
| **A2 — 2-stage Trend** | tech + s1 (Stage 1 OOF) | Sadece trend kanalı eklenmiş |
| **A3 — 2-stage Macro** | tech + s2 (Stage 2 OOF) | Sadece macro kanalı eklenmiş |
| **A4 — 3-stage Full** | tech + s1 + s2 | Tam hiyerarşik (ana tasarım) |

Soru: A4 gerçekten A1'den iyi mi? Bu, ablation çalışmasının temel sorusu.

---

## 4. Feature Subset'leri (B1, B2, B3)

Stage 3'e verilen **teknik feature setinin redundancy'sini** test eden 3
farklı seçim. Sadece `v2/feature-selection` branch'inde mevcut.

| Subset | n_feat | Strateji |
|---|---:|---|
| (orig v1) | 29 | Iter2'de seçilen 29 feature (osc + vol + volume + trend) |
| **B1** | 15 | **Aggressive** — sadece kısa-vade osc+volatility, tüm trend feature kaldır |
| **B2** | 24 | **Moderate** — sadece 5 long-term trend feature kaldır (log_ret_50/100, above_sma_200, adx_value, sharpe_proxy_20d) |
| **B3** | 15 | **MI top-15** — Mutual Information ile veri-driven seçim |

**Kazanan:** B2 + A4 Full → Sharpe 1.68 (ZZ-MLP) / 1.58 (XGB) — proje rekoru.

---

## 5. Modeller (Stage 3'te 7 algoritma)

| Kısa | Açılım | Phase | Stage 1 etiketi |
|---|---|---|---|
| LDA | Linear Discriminant Analysis | A | SMA |
| MLP | Multi-Layer Perceptron | A | SMA |
| XGB | XGBoost | B | SMA |
| LGBM | LightGBM | B | SMA |
| RF | Random Forest | B | SMA |
| **ZZ-XGB** | XGBoost (ZigZag Stage 1 OOF) | C | ZigZag |
| **ZZ-MLP** | MLP (ZigZag Stage 1 OOF) | C | ZigZag |

---

## 6. Etiketler (Stage 1, 2, 3'te)

| Etiket | Hangi sınıflandırıcı | Sınıflar | Üretim kuralı |
|---|---|---|---|
| **Trend** (Stage 1) | LDA | Up/Down/Flat | SMA(20) > SMA(50) cross |
| **Trend** (Stage 1, alt) | LDA (ZigZag) | Up/Down/Flat | %10 swing kuralı (causal) |
| **Macro regime** (Stage 2) | GMM 3-cluster | Calm/Trans/Stress | 11 makro feature üzerinde |
| **Signal** (Stage 3) | LDA / MLP / Tree models | Buy/Sell/Hold | 5-gün forward return ± 0.5×rolling_std(20) |

---

## 7. Branch Yapısı

```
claude/review-checkpoint-results-2hh1j  ← ana branch (v1/v2/v3/v4 entegrasyonu)
│
├── v1.0-iter4-final  (tag, v4 dondurulmuş, commit ab408d5)
│
├── v2/ablation             — A1-A4 ilk denemesi (v1 verisi, 29 feat tech)
├── v2/bigger-dataset       — CryptoCompare ile +577 satır, 2013-02 başlangıç
├── v2/feature-selection    — B1/B2/B3 redundancy temizliği ⭐ KAZANAN
├── v2/notebook-refresh     — boş, henüz yapılmadı
└── v2/eth-pipeline         — boş, opsiyonel
```

---

## 8. Teknik Sözlük

| Terim | Anlam |
|---|---|
| **OOF** | Out-of-fold. K-fold CV'de validation tahminleri = train satırları için leakage olmadan üretilen tahminler |
| **Soft fusion** | Stage 1/2 olasılık vektörü (ℝ³) çıktısı — hard label değil |
| **Walk-forward CV** | Zaman-sıralı k-fold; train fold testten önce, asla shuffle yok |
| **Adaptive label** | Sabit ±%1 yerine `0.5 × rolling_std(returns, 20)` — volatilite-uyumlu |
| **B&H (Buy & Hold)** | Pasif benchmark: ilk gün al, son gün sat |
| **ATR** | Average True Range — volatilite ölçüsü |
| **MI** | Mutual Information — feature × target ilişki ölçüsü (sklearn) |

---

## 9. Final Sonuçların Özeti

| Konfigürasyon | Veri | Tech feat | Stage 1 | Stage 3 | Sharpe | Return |
|---|---|---:|---|---|---:|---:|
| v1 (en iyi) | 462g | 29 | SMA OOF | XGB | 1.35 | +42.7% |
| v2-bigger | 533g | 29 | SMA OOF | XGB | 0.69 | +26.8% |
| **B2 + A4 (kazanan)** | 462g | 24 | **ZZ OOF** | **ZZ-MLP** | **1.68** | **+89.5%** |
| Buy & Hold | 462g | - | - | - | 0.75 | +47.6% |
