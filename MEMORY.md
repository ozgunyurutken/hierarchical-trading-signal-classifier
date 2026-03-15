# MEMORY.md - Project State & Decision Log

## Current Status
**Active Phase:** FAZ 0 ✅ → FAZ 1 (Veri Toplama) beklemede
**Last Updated:** 2026-03-15

## Progress Tracker

| Phase | Status | Checkpoint | Notes |
|-------|--------|------------|-------|
| FAZ 0: Proje İskeleti | ✅ Tamamlandı | 2026-03-15 | 46 dosya, tüm modüller hazır |
| FAZ 1: Veri Toplama | ⏳ Sırada | - | - |
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
- **Python sürümü:** 3.11+
- **Modüler yapı:** src/ altında data, features, labels, models, evaluation, utils
- **Config:** Tüm parametreler config.yaml'da merkezi

## Open Questions
- [ ] FRED API key mevcut mu? (FAZ 1'de gerekecek)
- [ ] Coin veri kaynağı: yfinance mi Binance mı? (Plan: yfinance)
- [ ] Gold için GC=F (futures) mı GLD (ETF) mı? (Plan: GC=F)

## Key Insights
_(Henüz veri analizi yapılmadı, ilerleyen fazlarda doldurulacak)_

## Experiment Results
_(FAZ 4-5'te doldurulacak)_
