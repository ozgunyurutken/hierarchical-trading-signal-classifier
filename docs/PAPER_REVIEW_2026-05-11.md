# Paper Review — 2026-05-11

**Reviewer**: backend session (Hızır)
**Paper**: `docs/paper/paper.tex` (commit `f82cd68`)
**Cross-checked against**: `reports/Phase5.1_arch_ablation/` + `reports/Phase3.5_after_tune/` + `reports/Phase8_coursework_metrics/`
**Phase G/H/I/J references in paper**: NONE found (good — those are explored-but-not-adopted, see `MEMORY.md`).

## Summary

| # | Severity | Type | Location |
|---|---|---|---|
| 1 | 🔴 Critical | Logic error | §4.4 line 362 "monotonic" claim |
| 2 | 🔴 Critical | Number mismatch | Table 5 line 384 — ETH 2-Stage Macro MaxDD |
| 3 | 🔴 Critical | Internal inconsistency | §4.2 line 206 — MLP tuning gain numbers |
| 4 | 🟡 Needs source check | Table column | Table 5 F1m column — 6/8 cells |
| 5 | 🟡 Number verify | Saturation claim | §4.3 line 248 |
| 6 | 🟢 Typo | Future work scope | §5.3 line 422 |

---

## 🔴 #1 — "Monotonic" claim is false (Section 4.4)

**Paper text (line 362):**

> For BTC the relationship between architectural depth and Sharpe is **monotonic**: Sharpe rises from $0.93$ (Flat) through $1.08$ (2-Stage Trend) and $0.98$ (2-Stage Macro) to $1.15$ (3-Stage Full)

**Problem**: Sequence is `0.93 → 1.08 → 0.98 → 1.15`. **2-Stage Macro (0.98) is lower than 2-Stage Trend (1.08)** — this is NOT monotonic. A reviewer will catch this immediately.

**Suggested fix**: Replace "monotonic" with neutral language. Options:

- "broadly increases with architectural depth"
- "trends upward with depth, with the full 3-stage variant winning"
- "non-monotonic but with the 3-Stage Full as the global maximum"

---

## 🔴 #2 — Table 5 ETH 2-Stage Macro MaxDD wrong (line 384)

**Paper text:**

```latex
ETH & 2-Stage Macro   & state., mlp     & $0.47$ & $69\%$    & $-55\%$ & $0.342$ \\
```

**Problem**: MaxDD column shows `-55%`. CSV gives `-0.505116` → should be `-51%` (or `-50%` if floor).

**CSV evidence** (`reports/Phase5.1_arch_ablation/v5_p5_arch_backtest_summary.csv`):

```
asset=eth, arch=2stage_macro, rule=stateful, model=mlp
annualized_sharpe=0.472408, total_return=0.692938,
max_drawdown=-0.505116, n_trades=106, win_rate=0.547
```

**Suggested fix**: `-55%` → `-51%`

---

## 🔴 #3 — MLP tuning numbers from 3-fold but Table is 5-fold (§4.2 line 206)

**Paper text:**

> The MLP benefited the most from tuning (BTC F1m **$0.462\to 0.537$**, range-class F1 **$0.302\to 0.421$**).

**Problem**: These before/after numbers come from `reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned_3fold.csv` (the 3-fold experimental tuning round that was later abandoned). The **final Table 3 numbers** in the same section come from `v5_p3_stage1_overall_tuned.csv` (5-fold). Internal inconsistency: Table reports BTC MLP F1m=$0.505$ but prose says it tuned to $0.537$.

The paper itself acknowledges 3-fold was abandoned (Section II-F footnote: *"An earlier 3-fold inner configuration tended to select degenerate hyperparameters..."*). So citing 3-fold numbers in the body contradicts the methodology.

**CSV evidence:**

| Source | BTC MLP F1m | BTC MLP range-F1 |
|---|---|---|
| Untuned (`Phase3/v5_p3_stage1_overall.csv`) | 0.4619 | 0.3025 |
| 3-fold tuned (abandoned) | 0.5365 | 0.4210 |
| **5-fold tuned (FINAL, used in Table 3)** | **0.5055** | **0.3818** |

**Suggested fix**: Replace numbers with 5-fold final values:

> The MLP benefited the most from tuning (BTC F1m $0.462\to 0.505$, range-class F1 $0.302\to 0.382$).

---

## 🟡 #4 — Table 5 F1m column: 6 of 8 cells inconsistent (need to confirm source)

**Paper Table 5 F1m column** vs. **CSV (`v5_p5_arch_overall.csv`) F1m of the model named in row 'Best'**:

| Row | Best | Paper F1m | CSV F1m | Δ |
|---|---|---|---|---|
| BTC Flat | state.xgb | 0.354 | **0.349** | −0.005 |
| BTC 2-Stage Trend | probw.lgbm | 0.358 | **0.347** | −0.011 |
| BTC 2-Stage Macro | state.rf | 0.361 | **0.356** | −0.005 |
| BTC 3-Stage Full | state.xgb | 0.367 | 0.367 | ✓ |
| ETH Flat | probw.lgbm | 0.354 | **0.347** | −0.007 |
| ETH 2-Stage Trend | defens.xgb | 0.368 | **0.361** | −0.007 |
| ETH 2-Stage Macro | state.mlp | 0.342 | 0.343 | ✓ rounding |
| ETH 3-Stage Full | probw.lgbm | 0.336 | **0.350** | +0.014 |

**Question for paper-writer**: Which CSV was used to populate the F1m column?

- Hypothesis A: F1m of the Sharpe-best model → doesn't match (6/8 differ)
- Hypothesis B: Best F1m per arch regardless of Sharpe → doesn't match either (BTC 2-Stage Trend best F1m = 0.369 RF)
- Hypothesis C: Mean F1m across all 4 models per arch → not tested but might match

**Suggested action**: Either:

1. Replace the F1m column with the Sharpe-best model's F1m (consistent with the "Best" column semantics), or
2. Rename the column "Best F1m" (i.e., maximum F1m across the 4 models in that arch), or
3. Add a footnote explaining what the F1m column represents.

---

## 🟡 #5 — Optuna saturation claim (§4.3 line 248)

**Paper text:**

> Doubling Optuna's trial budget to 60 and widening the search space added only **$+0.013$** F1m, suggesting saturation.

**CSV check** (`reports/Phase5.2_extended_optuna/v5_p5_extended_overall.csv` vs 5.1 baseline):

| Asset | Model | 30-trial F1m | 60-trial F1m | Δ |
|---|---|---|---|---|
| BTC | xgboost | 0.367 | 0.363 | −0.004 |
| BTC | lightgbm | 0.361 | 0.367 | +0.006 |
| BTC | random_forest | 0.360 | 0.352 | −0.008 |
| BTC | mlp | 0.347 | 0.335 | −0.011 |
| ETH | xgboost | 0.368 | 0.356 | −0.012 |
| ETH | lightgbm | 0.350 | 0.368 | +0.018 |
| ETH | random_forest | 0.364 | 0.377 | +0.012 |
| ETH | mlp | 0.312 | 0.320 | +0.008 |

- **mean Δ = +0.001** (not +0.013)
- **max Δ = +0.018**

**Saturation argument is still valid** (mean ≈ 0, half the deltas are negative), but the cited **+0.013 doesn't match any single statistic**.

**Suggested fix**: Either:

- "added an average of **$+0.001$** F1m (max $+0.018$, min $-0.012$), suggesting saturation"
- "with **no consistent improvement** (mean $\Delta=+0.001$ across 8 configurations), suggesting saturation"

---

## 🟢 #6 — Future Work typo: "Stage-1 calibration" → "Stage-3 calibration" (§5.3 line 422)

**Paper text:**

> Future work will add **Stage-1 calibration**, dynamic Kelly-style position sizing, ablation on the Stage-2 FSM rule set...

**Problem**: Stage-1 already has the strongest classification metrics in the pipeline (RF F1m=$0.563$, AUC=$0.76$). The stage that most needs calibration is **Stage-3** (AUC=$0.53$, near chance; raw ECE 0.15–0.24 for XGB/MLP from internal experiments). The intent is almost certainly Stage-3.

**Suggested fix**: `Stage-1 calibration` → `Stage-3 calibration`

---

## What I verified and is correct

✓ Stage-1 Table 3 (all 8 rows): accuracy, P/R, F1m, AUC all match CSV
✓ Stage-3 Table 4 (all 8 rows): accuracy, P/R, F1m, AUC all match CSV
✓ Table 5 Sharpe column: all 10 rows match CSV (paper rounds to 2 dp)
✓ Table 5 Return column: all 10 rows match CSV
✓ Table 5 MaxDD column: 9/10 rows match (only ETH 2-Stage Macro is wrong, error #2)
✓ B&H benchmarks: BTC Sharpe 0.95, ETH Sharpe 0.26, ETH return −7% all match CSV
✓ Abstract "40% drawdown reduction" math: B&H MaxDD −77%, model MaxDD −46% → (77−46)/77 = 40.3% ✓
✓ Section IV-A claim "all 8 configurations cleared F1m ≥ 0.50 gate, improvement on 6/8 untuned" → verified, untuned has 6/8 ≥ 0.50

## Phase G/H/I/J scan

`grep -E "regime-aware|isotonic|oversample|RandomOverSampler|calibrated|Phase G|Phase H|Phase I|Phase J|22 feat|interaction"` on `docs/paper/paper.tex` returns **only standard usages** (regime detection, regime-aware features in standard sense). Nothing references the rolled-back experiments. ✓
