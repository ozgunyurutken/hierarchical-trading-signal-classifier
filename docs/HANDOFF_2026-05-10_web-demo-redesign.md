# Handoff — Web Demo Redesign Session (2026-05-10)

**Branch:** `v5-from-scratch` · **Last commit:** `4251575` Phase 7 IEEE LaTeX paper draft

> **TL;DR** — Web demo'su Bloomberg-style dark Time Machine olarak sıfırdan yeniden yazıldı. Backend'e 3 yeni endpoint, "What-If Lab" interactive feature explorer eklendi. **Phase 5.1 OOF sonuçları HİÇ değişmedi**, paper rakamları aynı (BTC Sharpe 1.15 vs B&H 0.95). Henüz commit yapılmadı, browser'da kullanıcı tarafından test edilmedi.

---

## 1. Başlangıç durumu (session açılışı)

- Phase 6.1 (interactive trade timeline) commit edilmişti.
- `app/main.py`, `app/static/{app.js,index.html,style.css}` üzerinde uncommitted değişiklikler vardı — yarım kalmış "Decision Explanation" paneli (genel hatları: top reasons, outcome grid, trade simulation; UI kütlesel + form-tarzı).
- Kullanıcı **mevcut UI'ı beğenmedi** ("dropdown menü seçimi çok kötü, buy/sell grafiğinden hiçbir şey anlaşılmıyor, WOW etkisi lazım").

## 2. Verilen kararlar (kullanıcı onaylı)

| Soru | Karar |
|---|---|
| Konsept | **Time Machine (Bloomberg-style)** — tek hero chart + scrubber + tüm paneller real-time animate |
| Kitle | Yüksek lisans dersi hocası + öğrenciler (akademik) |
| Tema | **Dark trading terminal** — neon mint buy / neon red sell, JetBrains Mono + Inter |
| Yarım UI | **Sıfırdan yaz** — `*.v2.bak` zaten arşivlendi |
| Play hızları | **1× = 10 g/s, 3× = 30 g/s, 6× = 60 g/s** |
| **Yeni özellik** | Kullanıcı session ortasında istedi: **"feature'larla oynayıp sinyallerin anlık değiştiğini görsek"** → What-If Lab |
| What-If inference | **Hybrid** = final-fit modeller + UI'da disclaimer |
| What-If yerleşim | **Toggle button (default kapalı)** — sağ alt pulse'lu mor `🧪 WHAT-IF LAB` butonu |

## 3. Yapılan iş

### 3.1 Phase F-prep — Final-fit modeller (yeni)

`app/models/v5/` altında 32 yeni `.joblib` dosyası (BTC + ETH × 4 arch × 4 model). 48 saniyede fit edildi.

- **Script:** `scripts/v5_save_final_models.py` (149 satır, idempotent)
- **HP kaynağı:** `reports/Phase5.1_arch_ablation/v5_p5_arch_optuna_best.csv` (zaten vardı)
- **Manifest:** `app/models/v5/manifest.csv` (32 satır + header)
- **Bundle yapısı:** `{model, scaler, feature_cols, classes, asset, arch, model_name, n_train, trained_at}`
- Tree models için scaler=None, MLP için StandardScaler. balanced=True (XGB sample_weight, LGBM/RF class_weight, MLP yok-sayılır — overnight script ile aynı semantik).

> ⚠️ **KRITIK:** Bu modeller **walk-forward DEĞİL**, tüm training set'le fit. **Sadece interactive what-if** için. Backtest/paper için kullanılan `data/processed/*_stage3_oof_*_v5_tuned_*.csv` dosyaları DOKUNULMADI (mtime: 10 May 00:22, Phase 5.1 overnight'tan).

### 3.2 Backend (app/main.py — 927 satır, +400 satır eklendi)

**Yeni importlar:** `joblib`, `numpy`, `GZipMiddleware`.

**state[] genişletildi:**
- `ohlcv_cache` (full OHLCV, eskiden sadece Close vardı)
- `bt_summary` (Phase 5.1 backtest summary CSV)
- `whatif_models` — `(asset, arch, model) → bundle dict`

**Startup'a eklenen:**
- Phase 5.1 backtest summary CSV yüklemesi
- 32 what-if model bundle yüklemesi (loop ile)

**Yeni endpoint'ler:**

| Endpoint | Görevi |
|---|---|
| `GET /bundle?asset=&arch=&model=&rule=` | **Mega payload** — frontend'in tek atışta tüm seriyi alması için. Dönüş: `{asset, arch, model, rule, label, n, dates, ohlc{open,high,low,close,volume}, regime{label,P_Bull,P_Neutral,P_Bear,age_days}, stage1{P_down,P_range,P_up}, osc{6 oscillator}, active_signals, active_probs{Buy,Hold,Sell}, votes{4 model × 4 alan}, positions, equity, bh_equity, trades[], stats{sharpe,total_return,max_drawdown,bh_sharpe,bh_total_return,n_trades,win_rate}}`. Tipik 1.4 MB ham, gzip ile ~300 KB. |
| `POST /predict_custom` | What-If Lab — body: `{asset, arch, model, features:{16 feature dict}}`. Bundle'dan modeli yükle, `predict_proba()` çağır, dönüş `{probs:{Sell,Hold,Buy}, pred_label, confidence}`. ~50ms. |
| `GET /heatmap` | Phase 5.1 ablation grid — `{cells:[{asset,arch,model,rule(best),sharpe,total_return,max_drawdown}], buy_hold:[{asset,sharpe,total_return}]}`. 32 hücre + 2 B&H. |

**Helper'lar:**
- `_positions_for_rule(oof, rule)` — stateful/defensive/prob_weighted
- `_resolve_defaults(asset, arch, model, rule)` — "default" → BEST_PER_ASSET map'i

**Eskiler korundu:** `/health, /assets, /test_dates, /predict, /equity, /timeline, /explain` — frontend artık çoğunu kullanmıyor ama silmedim (geriye uyumluluk).

**Smoke test sonuçları (verified):**
- BTC bundle: n=2400, **Sharpe 1.147** vs B&H 0.951 (Phase 5.1 birebir ✓)
- ETH bundle: n=1200, Sharpe 0.52 vs B&H 0.26 (✓)
- Heatmap: BTC 3stage_full xgb = 1.15, ETH flat lgbm = 0.52 (✓)
- predict_custom: 16 feature dict → Hold/0.456 (model gerçek inference yapıyor)

### 3.3 Frontend — sıfırdan yeniden yazıldı

#### `index.html` (259 satır)
- Topbar: ⚡V5 logosu + BTC/ETH ticker (canlı fiyat) + arch/model/rule dropdown'lar + sağda yeşil ● LIVE OOF LED
- 8-panelli `grid` (12-col):
  - Hero chart (lightweight-charts, span 12)
  - Scrubber + play controls + mini equity ghost (span 12)
  - Signal card (span 3) · Stage 1 gauge (span 3) · Stage 2 orb (span 3) · Model votes (span 3)
  - Why this signal (span 6) · What happened (span 6)
  - Heatmap (span 12)
- What-If Lab drawer (default `hidden`, sağdan slide-in)
- Sağ alt fixed `🧪 WHAT-IF LAB` toggle butonu
- Footer (sticky, fixed bottom)
- CDN: lightweight-charts@4.2.0 + Chart.js 4.4 (Chart.js artık kullanılmıyor ama HTML'de kaldı, çıkartılabilir)
- Fonts: JetBrains Mono + Inter (Google Fonts)

#### `style.css` (846 satır)
- CSS custom properties: `--bg-0 #06090f`, `--buy #00ff9f`, `--sell #ff3860`, `--hold #ffb547`, `--regime-bull/neutral/bear`, mono+sans variables
- Animasyonlar: `bolt-pulse`, `led-pulse`, `lab-pulse` (pulse halka), prob-fill `width transition 0.6s cubic-bezier`
- Responsive: 1280px ve 800px breakpoints
- Dark scrollbar
- `.signal-label.buy/.sell/.hold` → text-shadow glow + bg-tint

#### `app.js` (973 satır, vanilla ES6 modüler)
**Modül yapısı (tek dosya, semantik bölümler):**
- `init()` → bootstrap, event listeners, hero chart kurulumu
- **Bundle/asset:** `switchAsset`, `loadBundle`
- **Hero chart:** `buildHeroChart` (candles + regime ribbon `priceScaleId="regime"` scaleMargins top:0.93 + volume `priceScaleId="volume"`), `renderHero`, `highlightHeroDate` (selected date için circle marker)
- **Scrubber:** `renderScrubber` (mini equity canvas çizimi), `seek`, `togglePlay`, `startPlay/stopPlay`, `setSpeed`
- **Date-driven:** `refreshAll`, `updateSignal`, `updateStage1` (+ `drawGauge` SVG yarım-daire), `updateStage2` (orb + halka rengi), `updateVotes` (4 satır segment bar), `updateReasons` (+ `computeReasons` heuristik), `updateOutcomes` (5/10/30d cells + trade sim)
- **Heatmap:** `loadHeatmap`, `renderHeatmap` (Sharpe gradient `#182039 → #00ff9f`), `renderHeatmapHighlight`, `activateHeatmapCell`
- **What-If Lab:** `toggleLab`, `setLabMode` (actual/custom), `buildLabSliders` (4 grup × N feature), `syncLabFromActual`, `readActualFeatures` (bundle'dan 16-feature okur), `runLabDebounced` (140ms), `runLabPredict` (POST /predict_custom), `paintLabOutput` (delta vs actual), `labResetToActual`, `labRandomize` (random softmax + osc)

**State (`ST`):**
```js
{ asset, arch, model, rule, bundle, idx, playing, speed, playerHandle,
  hero{chart,candle,volume,regime,marker}, miniCanvas, labOpen, labMode,
  labFeatures, labDebounce, heatmap }
```

**Önemli:** `lightweight-charts.subscribeClick` chart click → seek(idx). Selected date marker `position:'inBar', shape:'circle', color:'#4d8aff'`. Markers monotonic order için sort ediliyor.

### 3.4 Arşivlendi
- `app/static/{app.js,index.html,style.css}.v2.bak` → `docs/legacy_ui/` (eski v2 era frontend)

## 4. Açık iş (kullanıcı tarafından)

1. **Browser'da test edilmedi** — sayfa açılıp denedikten sonra bug/iyileştirme listesi alınacak. Kullanıcı tam test etmeden yeni session'a geçti.
2. **Henüz commit yok** — `git status` 4 modified + 8 untracked dosya (bunlardan `docs/paper/paper.aux/.bbl/.blg/.out/.pdf` LaTeX build artifacts, gitignore'a alınabilir).
3. **Server düştü** (exit 144 = SIGTERM, muhtemelen başka bir sebepten). Yeni session'da tekrar `uvicorn app.main:app --host 127.0.0.1 --port 8765` ile başlatılır.

## 5. Devam etmek için (yeni session'da yapılacaklar)

```bash
# 1) Server başlat
cd /Users/yurutkenozgun/Projects/hierarchical-trading-signal-classifier
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level info

# 2) Tarayıcıda aç
open http://127.0.0.1:8765

# 3) Test akışı
# - Sayfa açılışında: BTC en güncel tarih, paneller dolu, hero chart 10 yıl
# - Scrubber sürükle: chart'taki mavi nokta hareket eder, paneller anlık güncellenir
# - Play: 1x = 10 gün/saniye zaman makinesi
# - Heatmap hücresine tıkla: tüm dashboard o kombo'ya geçer
# - 🧪 What-If Lab: RSI'ı 28→75 çek, signal anlık değişir
```

## 6. Bilinen / olası iyileştirme noktaları

- **Bundle size:** 1.4 MB ham JSON. gzip middleware aktif (~300 KB inecek). İlk yükleme yavaşsa daha agresif endpoint splitting yapılabilir (osc ayrı, votes ayrı, on-demand).
- **Smoothed P1 features:** Bundle'da `P1_*_smooth10` yok, what-if'te `readActualFeatures` raw P1'i smoothed slot'a kopyalıyor (proxy). İdeal: bundle'a smoothed da eklensin.
- **Selected date vertical line:** Şu an chart'ta "circle" marker olarak. Gerçek vertical line için custom plugin lazım — şu hali yeterince WOW.
- **Mobile:** 800px breakpoint var ama drawer mobile'da tüm ekranı kaplıyor — UX şu an'ki testte belirsiz.
- **Chart.js CDN HTML'de duruyor** ama kullanılmıyor — temizlenebilir.
- **Eski endpoint'ler** (`/predict`, `/timeline`, `/explain`, `/equity`) frontend tarafından çağrılmıyor — silinebilir, ama backward-compat için kalabilir.
- **Kullanıcı denemeden** WOW seviyesinin gerçekten karşılandığı doğrulanmadı.

## 7. Kritik akademik dürüstlük notu

> Demo'daki **hero chart, scrubber, signal panel, votes, equity, heatmap, "what happened" outcomes**:
> Tümü **Phase 5.1 walk-forward OOF**'tan beslenir. Bu paper'da rapor edilen rakamlardır.
>
> **Sadece "What-If Lab" panelinde** final-fit (full-data) model kullanılır. UI'da `lab-hint` div'inde net disclaimer var:
> *"what-if uses a final-fit model trained on all data; backtest curves above remain walk-forward OOF"*

## 8. Önemli dosyalar (yeni session referansı)

| Yol | Satır | Rol |
|---|---:|---|
| `app/main.py` | 927 | Backend, 8 endpoint + warmup |
| `app/static/index.html` | 259 | Topbar + 8 panel grid + lab drawer |
| `app/static/style.css` | 846 | Dark terminal theme |
| `app/static/app.js` | 973 | Vanilla JS, lightweight-charts hero, scrubber, lab |
| `scripts/v5_save_final_models.py` | 149 | 32 final-fit model fit + pickle (one-shot) |
| `app/models/v5/*.joblib` | 32 dosya (~100 MB) | What-If model bundles |
| `app/models/v5/manifest.csv` | 33 satır | Hangi modeller hangi dosyada |
| `reports/Phase5.1_arch_ablation/v5_p5_arch_optuna_best.csv` | 33 satır | HP kaynağı (DOKUNULMADI, sadece okundu) |
| `data/processed/*_stage3_oof_*_v5_tuned_*.csv` | 32 dosya | Walk-forward OOF (DOKUNULMADI) |
| `docs/legacy_ui/*.v2.bak` | 3 dosya | Eski v2 era frontend, arşivlendi |

## 9. Henüz dokunulmamış todolar (referans için)

- Eski `Decision Explanation` panelinin Phase 7 paper'la entegrasyonu
- Mobile UX testi
- Production Docker bundle güncellemesi (`app/models/v5/` 100 MB image'a eklenir)
- README.md / paper update — yeni demo screenshot'ları, what-if disclosure

---

**Yeni session'a açılış prompt önerisi:**

> Session devam — `docs/HANDOFF_2026-05-10_web-demo-redesign.md`'i oku. Web demo'sunu Bloomberg-style sıfırdan yeniden yazdık (Time Machine + What-If Lab). Backend + frontend hazır, henüz browser'da test etmedim. `uvicorn app.main:app --host 127.0.0.1 --port 8765` ile başlat, tarayıcıda açıp deneyeyim, bug/iyileştirme listesini sana vereyim.
