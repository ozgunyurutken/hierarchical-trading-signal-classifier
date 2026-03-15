# MEMORY.md - Project State & Decision Log

## Current Status
**Active Phase:** FAZ 1 (Veri Toplama) — genişletilmiş veri seti, kullanıcı incelemesi bekleniyor
**Last Updated:** 2026-03-15

## Progress Tracker

| Phase | Status | Checkpoint | Notes |
|-------|--------|------------|-------|
| FAZ 0: Proje İskeleti | ✅ Tamamlandı | 2026-03-15 | 46 dosya, tüm modüller hazır |
| FAZ 1: Veri Toplama | 🔍 İnceleme Bekleniyor | 2026-03-15 | Genişletilmiş: 15 makro ticker, timezone lag, 2014→ |
| FAZ 2: Feature Engineering | 🔲 Beklemede | - | - |
| FAZ 3: Label Üretimi | 🔲 Beklemede | - | - |
| FAZ 4: Model Eğitimi | 🔲 Beklemede | - | - |
| FAZ 5: Değerlendirme | 🔲 Beklemede | - | - |
| FAZ 6: Web App & API | 🔲 Beklemede | - | - |
| FAZ 7: Rapor & Sunum | 🔲 Beklemede | - | - |

## Decisions Made

### FAZ 0 Kararları (2026-03-15)
- **Proje adı:** hierarchical-trading-signal-classifier
- **Repo:** Public (arkadaş da Claude Code ile katkı verecek)
- **Proje konumu:** `/Users/yurutkenozgun/Desktop/hierarchical-trading-signal-classifier/`
- **Python sürümü:** 3.11.9
- **Modüler yapı:** src/ altında data, features, labels, models, evaluation, utils
- **Config:** Tüm parametreler config.yaml'da merkezi

### FAZ 1 Kararları (2026-03-15)
- **Veri kaynağı:** Tamamen yfinance (API key gerekmedi)
- **FRED verisi:** API key yok → yfinance ile 15 makro ticker kullanıldı
- **Tarih aralığı (kripto):** BTC: 2014-09-17 →, ETH: 2017-11-09 → (ayrı ayrı)
- **Tarih aralığı (makro):** 2007-04-11 → (HYG en genç ticker)
- **Timezone lag:** 1 gün (NYSE 21:00 UTC vs kripto 00:00 UTC)
- **Hizalama:** Forward-fill + dropna (başlangıç NaN'ler)

### FAZ 1 Veri Durumu — Genişletilmiş Dataset (v2)

#### Kripto OHLCV
| Veri | Satır | Tarih Aralığı | Durum |
|------|-------|---------------|-------|
| BTC-USD OHLCV | 4,123 | 2014-09-17 → 2025-12-30 | ✅ NaN yok |
| ETH-USD OHLCV | 2,974 | 2017-11-09 → 2025-12-30 | ✅ NaN yok |

#### Makro Veriler (4 Kategori, 15 Ticker)
| Kategori | Tickers | Ham Satır | Tarih Aralığı |
|----------|---------|-----------|---------------|
| Risk Appetite | SP500, VIX, DXY | ~4,712 | 2007-04-11 → 2025-12-30 |
| Commodities | Gold, Silver, Oil_WTI | ~4,711 | 2007-04-11 → 2025-12-30 |
| Bond Yields | US10Y, US5Y, US3M, US30Y, US2Y | ~4,710 | 2007-04-11 → 2025-12-30 |
| Credit | HY_Bond, IG_Bond, Treasury20Y, TIPS | ~4,712 | 2007-04-11 → 2025-12-30 |

#### Hizalanmış Veri (Aligned)
| Veri | Satır | Sütun | Tarih Aralığı | Kayıp | NaN |
|------|-------|-------|---------------|-------|-----|
| **BTC Aligned** | **4,111** | **20** | 2014-09-17 → 2025-12-30 | 12 satır (%0.3) | 0 |
| **ETH Aligned** | **2,967** | **20** | 2017-11-09 → 2025-12-30 | 7 satır (%0.2) | 0 |

#### Aligned Dataset Columns (20)
- **OHLCV (5):** Open, High, Low, Close, Volume
- **Risk (3):** SP500, VIX, DXY
- **Commodities (3):** Gold, Silver, Oil_WTI
- **Yields (5):** US10Y, US5Y, US3M, US30Y, US2Y
- **Credit (4):** HY_Bond, IG_Bond, Treasury20Y, TIPS

#### Önceki vs Yeni Karşılaştırma
| Metrik | Eski (v1) | Yeni (v2) | Artış |
|--------|-----------|-----------|-------|
| BTC satır | 1,817 | 4,111 | 2.3x |
| ETH satır | 1,817 | 2,967 | 1.6x |
| Makro sütun | 8 | 15 | 1.9x |
| Toplam sütun | 13 | 20 | 1.5x |
| Timezone lag | Yok | 1 gün | ✅ |

## Gemini Önerileri (Gelecek Fazlar İçin)
- **FAZ 2:** Log returns, rolling z-scores, spread variables (yield curve, credit spread, gold/silver ratio, breakeven inflation)
- **FAZ 3:** ZigZag trend labels alternatifi, hysteresis threshold consideration
- **FAZ 4:** Sample weighting experiment, MLP Stage 3 tuning focus
- **FAZ 5:** Confidence threshold sweep backtester'da, rejim-bazlı drawdown analizi
- **FAZ 6:** Manuel feature input + "fill defaults" butonu, absurd value test
- **FAZ 7:** ALFRED notu raporda, temporal alignment dokümantasyonu

## Open Questions
- [x] ~~FRED API key mevcut mu?~~ → Hayır, 15 yfinance ticker yeterli
- [x] ~~Coin veri kaynağı: yfinance mi Binance mı?~~ → yfinance
- [x] ~~Gold için GC=F mi GLD mi?~~ → GC=F
- [x] ~~BTC/ETH ayrı start date mi?~~ → Evet, BTC: 2014, ETH: 2017
- [x] ~~Timezone lag gerekli mi?~~ → Evet, 1 gün shift eklendi
- [ ] 4,111 satır (BTC) / 2,967 satır (ETH) yeterli mi?
- [ ] FRED API key bulunmalı mı yoksa mevcut 15 ticker yeterli mi?

## Key Insights
_(Kullanıcı notebook incelemesinden sonra doldurulacak)_

## Experiment Results
_(FAZ 4-5'te doldurulacak)_
