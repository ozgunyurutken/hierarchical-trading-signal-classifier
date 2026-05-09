# V5 Plan — Hierarchical Trading Signal Classifier

> **Aktif plan.** V3_PLAN.md legacy/fallback referansı olarak kalır.

## Architecture

```
Stage 1 (Trend, BTC/ETH-spesifik) ──► p̂(uptrend/downtrend/range) ──┐
                                                                    │
Stage 2 (Macro Regime) ───────────► p̂(Bull/Neutral/Bear) ──────────► Stage 3 (Signal) ──► Buy/Sell/Hold
                                                                    │
Oscillator + price features ────────────────────────────────────────┘
```

Soft fusion: Stage 1 ve Stage 2, hard label yerine **olasılık vektörü** (P_class) verir; Stage 3 bu olasılıkları feature olarak kullanır.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Data pipeline (BTC/ETH OHLCV + 22 macro features aligned) | DONE |
| 2 | Stage 2 Macro Regime (Composite FSM v5, Phase 2.12) | DONE |
| 3 | **Stage 1 Trend Classifier** | NEXT |
| 4 | Stage 3 Signal Classifier (soft fusion) | TODO |
| 5 | Walk-forward CV + 3-arch ablation (Flat / 2-Stage / 3-Stage) | TODO |
| 6 | FastAPI demo backend + minimal web frontend | TODO |
| 7 | Final report (IEEE 4-6 page, V5 final only) | TODO |

## Phase 3 — Stage 1 Trend Classifier

### Goal

BTC/ETH için ayrı trend sınıflandırıcı. Output: `p̂(uptrend, downtrend, range)` günlük olasılık vektörü.

### Trend Label (causal)

**Tasarım kararı:** SMA crossover **kullanmayacağız** (önceki iter'larda tautoloji yarattı). İki seçenek:

- **(A) ZigZag-based:** Piecewise linear segmentation. Pivot point'ler arası yönü etiket olarak ata.
- **(B) Forward return + threshold:** ret_h = (close[t+h] / close[t]) - 1. |ret_h| < ε → range, ret_h > +ε → up, ret_h < -ε → down.

İkisini de dene, daha causal ve daha balanced label dağılımı vereni seç.

### Features (BTC/ETH crypto-spesifik, 12-15 feature)

| Kategori | Features |
|---|---|
| Returns | log_ret_5d, log_ret_20d, log_ret_60d |
| Trend strength | ADX_14, MA_slope_20, MA_slope_50 |
| Momentum | RSI_14, MACD_signal_diff |
| Mean reversion | Bollinger_pct_b, distance_to_SMA_50 |
| Volatility | ATR_14_pct, realized_vol_20d |
| Volume | volume_zscore_20, OBV_zscore_60 |

### Methodology

- 4 classifier ablation: XGBoost, LightGBM, Random Forest, MLP
- Walk-forward expanding-window CV: train [T0, Ti], val [Ti+1, Ti+gap], step forward; gap = 5 gün
- Tree models: no scaling. MLP: StandardScaler (fit on train fold only)
- OOF predictions: P_class (3 olasılık) → Stage 3'e soft fusion input
- Optuna 20 trial, val F1 macro objektif

### Validation

Test accuracy, F1 macro, MCC, 3-class confusion matrix, probability calibration.

### Deliverables

- src/labels/v5_trend_labels.py
- src/features/v5_trend_features.py
- src/models/v5_stage1_trainer.py
- scripts/v5_train_stage1_trend.py
- notebooks/05_stage1_training.ipynb
- reports/Phase3/v5_stage1_*.png

## Phase 4 — Stage 3 Signal Classifier

BTC/ETH ayrı: günlük Buy/Sell/Hold sinyali.

**Label (causal):** ret_h > +ε → Buy, ret_h < -ε → Sell, |ret_h| ≤ ε → Hold. ε adaptive (rolling std × k). h = 5 gün.

**Features (~12-15):** Stage 1 OOF (3) + Stage 2 OOF (3) + RSI, MACD, Bollinger, Stochastic, Volume z-score, OBV change.

Aynı walk-forward + 4 classifier + Optuna setup.

## Phase 5 — Walk-forward CV + 3-Architecture Ablation

| Architecture | Stage 3 input |
|---|---|
| Flat (1-stage) | Sadece oscillator + price |
| 2-Stage (Trend or Macro) | + Stage 1 OOF veya + Stage 2 OOF |
| 3-Stage (Full) | + Stage 1 OOF + Stage 2 OOF |

3 arch × 4 classifier × 2 asset (BTC, ETH) = 24 runs.

Backtest: long-only, 0.1% TC, Sharpe/return/MaxDD/win, B&H benchmark.

## Phase 6 — FastAPI Demo

- app/main.py FastAPI
- /health, /predict?date=&asset=&model= endpoints
- Stage 1/2/3 joblib artifacts
- Minimal HTML frontend
- docker/Dockerfile + docker-compose.yml

## Phase 7 — Final Report (IEEE 4-6 page)

**Scope rule (CLAUDE.md):** Sadece V5 final. V1/V2/V3 paper'a girmez.

Bölümler: Introduction, Related work, Data, Methodology (Stage 2 FSM + Stage 1 + Stage 3 + WF-CV), Experiments (ablation + classifier compare + BTC vs ETH + backtest), Discussion, Conclusion.

## Decision Gates

- Phase 3: Stage 1 F1 ≥ 0.50 (chance 0.33).
- Phase 4: Stage 3 F1 ≥ 0.40.
- Phase 5: 3-Stage > Flat?
- Phase 6: Demo end-to-end çalışıyor mu?
- Phase 7 öncesi: tüm coursework DONE + kullanıcı onay.

## Branch Strategy

- v5-from-scratch ana V5 branch
- Phase sonrası commit + push
- claude/review-checkpoint-results-2hh1j Phase 2.7-2.12 merge sonrası kapatıldı
