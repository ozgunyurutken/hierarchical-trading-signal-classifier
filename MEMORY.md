# MEMORY.md - Project State & Decision Log

## Current Status
**Active Phase:** FAZ 1 (Veri Toplama) — kullanıcı incelemesi bekleniyor
**Last Updated:** 2026-03-15

## Progress Tracker

| Phase | Status | Checkpoint | Notes |
|-------|--------|------------|-------|
| FAZ 0: Proje İskeleti | ✅ Tamamlandı | 2026-03-15 | 46 dosya, tüm modüller hazır |
| FAZ 1: Veri Toplama | 🔍 İnceleme Bekleniyor | 2026-03-15 | Veriler toplandı, notebook hazır, kullanıcı onayı bekleniyor |
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
- **FRED verisi:** API key yok → yfinance Treasury Yield proxy'leri kullanıldı (^TNX, ^IRX, ^FVX)
- **Gold ticker:** GC=F (futures) seçildi
- **Tarih aralığı:** 2021-01-01 → 2025-12-30

### FAZ 1 Veri Durumu
| Veri | Satır | Tarih Aralığı | Durum |
|------|-------|---------------|-------|
| BTC-USD OHLCV | 1,825 | 2021-01-01 → 2025-12-30 | ✅ NaN yok |
| ETH-USD OHLCV | 1,825 | 2021-01-01 → 2025-12-30 | ✅ NaN yok |
| S&P 500 | 1,254 | 2021-01-04 → 2025-12-30 | ✅ (weekday only) |
| Gold Futures | 1,256 | 2021-01-04 → 2025-12-30 | ✅ |
| DXY | 1,256 | 2021-01-04 → 2025-12-30 | ✅ |
| VIX | 1,254 | 2021-01-04 → 2025-12-30 | ✅ |
| US 10Y/5Y/3M Yield | 1,254 | 2021-01-04 → 2025-12-30 | ✅ |
| **BTC Aligned** | **1,817** | 2021-01-04 → 2025-12-30 | ✅ 13 columns, 0 NaN |
| **ETH Aligned** | **1,817** | 2021-01-04 → 2025-12-30 | ✅ 13 columns, 0 NaN |

**Alignment kaybı:** 8 satır (%0.4) — ilk birkaç gün makro verinin henüz mevcut olmaması

### Aligned Dataset Columns (13)
OHLCV (5): Open, High, Low, Close, Volume
Macro Daily (4): S&P_500, Gold_Futures, DXY_Dollar_Index, VIX_Volatility
Yield Proxies (4): US10Y_Yield, US3M_Yield, US5Y_Yield, Yield_Spread_10Y_3M

## Open Questions
- [x] ~~FRED API key mevcut mu?~~ → Hayır, yield proxy'ler yeterli
- [x] ~~Coin veri kaynağı: yfinance mi Binance mı?~~ → yfinance
- [x] ~~Gold için GC=F mi GLD mi?~~ → GC=F
- [ ] 1,817 satır yeterli mi? Daha geniş tarih aralığı gerekli mi?
- [ ] Yield proxy'ler rejim tespiti için yeterli mi, FRED API key bulunmalı mı?

## Key Insights
_(Kullanıcı notebook incelemesinden sonra doldurulacak)_

## Experiment Results
_(FAZ 4-5'te doldurulacak)_
