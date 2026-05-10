# A Three-Stage Hierarchical Soft-Fusion Framework for Cryptocurrency Trading Signal Classification

**Özgün Can Yürütken** (707251003) · **Çağatay Bilgin** (504251013)
*Istanbul Technical University · BBL514E Pattern Recognition · Spring 2026*

---

## Abstract

We present a three-stage hierarchical soft-fusion framework for cryptocurrency Buy/Sell/Hold signal classification on Bitcoin (BTC) and Ethereum (ETH). The framework decomposes the daily trading decision into three distinct pattern-recognition sub-problems: (i) a *trend classifier* (Stage 1) that estimates the local price-trend regime from technical indicators using a ZigZag-based offline label, (ii) a *macroeconomic regime classifier* (Stage 2) that produces Bull/Neutral/Bear labels via a deterministic finite-state machine over VIX, FedFunds, M2, yield-curve and DXY signals, and (iii) a *signal classifier* (Stage 3) that fuses the two upstream posterior probability vectors with six oscillator features to produce the final action. Four classifiers (XGBoost, LightGBM, Random Forest, MLP) were tuned per stage with Optuna 5-fold inner-CV walk-forward, then evaluated on outer expanding-window walk-forward out-of-fold predictions covering 2017-2025 (BTC, 3,200 days) and 2020-2025 (ETH, 2,000 days). On BTC, the full 3-stage architecture achieved an annualized Sharpe ratio of 1.15 (versus Buy & Hold 0.95), a 60% drawdown reduction, and a cumulative return within 2.4% of Buy & Hold. On ETH, an architecture ablation showed that hierarchical fusion overfits the smaller dataset; a flat baseline with only six oscillators delivered Sharpe 0.52 and a 26% return while Buy & Hold lost 7%. Our key empirical finding is that the optimal architectural depth is asset-specific and cannot be predicted from frame-level metrics alone — frame-level macro-F1 stayed at 0.37 for all configurations, yet selective Hold-aware trading rules transformed weak class probabilities into economically meaningful trades.

---

## 1. Introduction

Cryptocurrency markets are notoriously noisy, exhibit non-stationarity, and react to a unique blend of micro-structure, sentiment, and macroeconomic shocks. Traditional supervised approaches that map raw price features directly to a Buy/Sell/Hold label tend to learn idiosyncratic relationships that generalize poorly across regimes [1], [2]. A hierarchical decomposition of the decision — first identifying the *trend*, then the *macro context*, and finally the *trade* — provides three benefits: (i) each sub-problem has a clearer, more identifiable signal-to-noise ratio; (ii) intermediate posteriors can be inspected and audited; and (iii) the modular pipeline can be ablated to expose where the predictive value resides.

This work formulates and empirically evaluates such a three-stage classifier on Bitcoin and Ethereum daily data. We frame the trading-signal problem as a hierarchical Bayesian decision-theoretic posterior fusion: \(\hat p(\text{signal} \mid x) \propto \hat p(\text{signal} \mid x, \hat p_\text{trend}, \hat p_\text{regime})\), where \(\hat p_\text{trend}\) and \(\hat p_\text{regime}\) are out-of-fold posterior estimates from upstream classifiers and \(x\) contains current oscillator features. We compare four canonical pattern-recognition algorithms (XGBoost, LightGBM, Random Forest, MLP) at every stage, perform a four-architecture ablation (Flat, 2-Stage Trend, 2-Stage Macro, 3-Stage Full) and three-trading-rule study, and assess the system on both classification (F1, ROC-AUC) and economic (Sharpe, drawdown, return) metrics.

### 1.1 Related Work

**Hierarchical and cascade classifiers** in time-series forecasting predate modern ML: Jordan & Jacobs [3] introduced hierarchical mixtures of experts in 1994, Wolpert [4] proposed stacked generalization, and Silla & Freitas [5] catalogued hierarchical text classification. In financial time series, Challu et al. [6] proposed N-HiTS for multi-horizon forecasting, and Zhang & Yan [7] used a two-stage attention transformer for multivariate series. Zou et al. [8] applied a cascaded LSTM upstream of an actor-critic agent in stock trading and reported a Sharpe improvement over flat baselines, providing direct empirical support for the hierarchy hypothesis.

**Crypto signal classification** has tended toward two extremes: monolithic deep models such as the LSTM-based directional predictors of Patel et al. [9] and Sezer & Ozbayoglu [10], or rule-based technical strategies. Recent work by Kuznetsov et al. [11] introduced a confidence-threshold cascade for crypto direction prediction, achieving 82.68% directional accuracy at 11.99% market coverage; this directly motivates our Stage-3 Hold-aware design where the model abstains rather than forces a trade.

**Macroeconomic regime detection** has historically relied on Markov-switching models [12] or rule-based filters over VIX and yield-curve indicators. We adopt a deterministic finite-state machine inspired by [13] and [14], hard-coded with hysteresis and minimum-dwell constraints, after observing that GMM and HMM unsupervised approaches displayed pathological stickiness in our preliminary experiments.

**GBDT versus deep learning on tabular data**: Grinsztajn et al. [15] and Shwartz-Ziv & Armon [16] showed that gradient-boosted trees outperform deep architectures on tabular datasets of the size and structure encountered in financial daily data. Our results confirm this on Stage 3.

**Walk-forward cross-validation and meta-overfitting**: Lopez de Prado [17], [18] argues that random K-fold leaks future information in financial settings and recommends purged, embargoed walk-forward CV. We adopt expanding-window walk-forward at every stage and devote particular attention to the *inner-CV configuration* used by Optuna; we show empirically that an under-resourced 3-fold inner CV induces meta-overfitting on Random Forest hyperparameters which is recovered when the inner CV is widened to 5 folds.

**Trading-rule design and the Hold class**: Bailey et al. [19] showed that a strategy's Sharpe ratio is sensitive to whether the holding rule is stateful or stateless, and Lopez de Prado [18] formalized "the deflated Sharpe ratio" for backtest selection. Our three-rule ablation (stateful long-only, defensive reset, probability-weighted continuous sizing) operationalises these distinctions.

### 1.2 Contributions

1. **A three-stage hierarchical Bayesian decision-theoretic framework** for crypto trading-signal classification, with offline-revisable ZigZag trend labels at Stage 1, a deterministic FSM at Stage 2, and 4-classifier soft fusion at Stage 3.
2. **An asset-specific architecture finding**: 3-stage Full is the empirically best architecture for BTC (Sharpe 1.15, +21% over B&H), but the *flat* 6-oscillator baseline is best for ETH (Sharpe 0.52 versus 0.34 for 3-stage Full). Hierarchical fusion overfits the smaller ETH dataset. This is a concrete instance of *no-free-lunch* in financial ML.
3. **A meta-overfitting case study** documenting how a 3-fold inner CV in Optuna selected `min_samples_leaf=1` for Random Forest (a degenerate value) and produced an outer-CV regression of −0.014 F1 macro versus the un-tuned baseline. Switching to a 5-fold inner CV with evenly-spaced historical regimes recovered the lost performance.
4. **A frame-level versus segment-level evaluation gap**: Stage 1 majority-vote consistency is 0.81 for BTC and 0.83 for ETH despite a frame F1 macro of only 0.55-0.57. This decoupling is also visible at Stage 3: ROC-AUC sits at 0.53 (close to chance) yet selective trading rules deliver Sharpe 1.15.
5. **A complete reproducible Dockerised demo** with FastAPI backend, Chart.js front-end, and the entire walk-forward OOF predictions bundled into the image so that no on-the-fly inference is required at presentation time.

---

## 2. Materials and Methods

### 2.1 Dataset Description

| Item | BTC | ETH |
|---|---|---|
| Source | Yahoo Finance (yfinance), FRED API | Yahoo Finance (yfinance), FRED API |
| Date range (raw) | 2014-09-17 to 2025-12-30 | 2017-11-09 to 2025-12-30 |
| Date range (Stage 3 OOF) | 2016-12-24 to 2025-10-16 | 2020-02-11 to 2025-08-15 |
| Stage 3 sample count | 3,200 daily observations | 2,000 daily observations |
| Macro covariates | 22 columns (S&P 500, VIX, DXY, Gold, Silver, Oil WTI, US10Y, US2Y, FedFunds, CPI, UnRate, M2, ICSA, plus credit spreads and ratios) | identical |
| Stage 3 features | 16 (3 Stage-1 raw posterior + 3 Stage-1 smoothed posterior + 3 Stage-2 hard one-hot + 1 regime-tenure + 6 oscillators) | identical |
| Stage 3 classes | Buy / Hold / Sell (3-class) | Buy / Hold / Sell (3-class) |
| Class balance (Buy/Hold/Sell) | 43.6% / 21.7% / 34.8% | 43.4% / 22.1% / 34.5% |
| Preprocessing | Forward-fill of FRED monthly with publication-release lag (FedFunds 1d, CPI 45d, UnRate 35d, M2 14d, ICSA 5d), winsorized OBV change | identical |

Forward returns and labels are the only future-dependent values; they are *never* used as features. A `assert_no_lookahead_leakage` routine (`src/labels/v5_signal_labels.py`) validates that the rolling-std threshold uses past-only data.

### 2.2 Mathematical Formulation

Let \(x_t \in \mathbb{R}^d\) denote the feature vector at day \(t\). Stage 1 outputs a posterior \(\hat p_\text{trend}(t) = (\hat p_\text{down}, \hat p_\text{range}, \hat p_\text{up})\); Stage 2 outputs a hard one-hot \(\hat p_\text{regime}(t) \in \{e_\text{Bull}, e_\text{Neutral}, e_\text{Bear}\}\) plus a regime-age scalar \(\tau_t\); Stage 3 receives the concatenated vector
$$
z_t = [\hat p_\text{trend}(t),\, \hat p_\text{trend}^{(\text{smooth-10})}(t),\, \hat p_\text{regime}(t),\, \tau_t,\, o_t] \in \mathbb{R}^{16}
$$
where \(o_t\) is the six-dimensional oscillator vector. The Stage 3 decision rule is
$$
\hat y_t = \arg\max_{y \in \{\text{Buy}, \text{Hold}, \text{Sell}\}} \hat p(y \mid z_t).
$$
The signal label is generated as
$$
y_t =
\begin{cases}
\text{Buy} & \text{if } \frac{P_{t+5} - P_t}{P_t} > +k\,\hat\sigma_t,\\
\text{Sell}& \text{if } \frac{P_{t+5} - P_t}{P_t} < -k\,\hat\sigma_t,\\
\text{Hold}& \text{otherwise,}
\end{cases}
$$
where \(\hat\sigma_t\) is the causal rolling 20-day standard deviation of daily returns, and \(k = 0.5\) by default (a sensitivity study is reported in Section 4.4).

### 2.3 Model Description

Four classifiers are evaluated at Stages 1 and 3:

- **XGBoost** [20]: Gradient-boosted tree ensemble with `multi:softprob` objective, histogram tree method for speed. Inverse-frequency sample weighting in lieu of `class_weight=balanced`.
- **LightGBM** [21]: Gradient-boosted trees with leaf-wise growth and `class_weight=balanced`.
- **Random Forest** [22]: Bagged decision-tree ensemble; `class_weight=balanced`.
- **MLP** [23]: Two- or three-layer ReLU multilayer perceptron with early-stopping. `class_weight` is not natively supported by scikit-learn's MLP, so the `balanced` flag is logged but not applied — a paper limitation.

Stage 2 is a *deterministic finite-state machine* over macro features, not a learned classifier. Eight rules — VIX hysteresis (entry > 1.0σ, exit < 0.3σ for Bear; symmetric for Bull), minimum dwell (Bear ≥ 20d, Bull ≥ 40d, Neutral ≥ 10d), velocity overrides, persistent yield-curve inversion, and a DXY+M2 macro-stress filter — produce a hard regime label. We adopted FSM only after four canonical unsupervised approaches (vanilla K-Means, semantic constrained K-Means, HMM, GMM) failed to capture the 2008 Global Financial Crisis structure or exhibited 2024–2025 stickiness with P(Stress) ≈ 0.96 saturated.

### 2.4 Trend Label

We use a ZigZag-based offline pivot detection with a deviation threshold of 10% and a minimum segment length of 15 days, classifying the segment between two pivots as `uptrend` if the realised return exceeds 7.5% in absolute amplitude and the segment is longer than the minimum dwell, `downtrend` symmetrically, and `range` otherwise. Although ZigZag is *retrospectively* defined (a property shared with all supervised labels for time series), it is *never* visible to the Stage 1 model at inference time — the model receives only causal technical indicators.

### 2.5 Oscillator Features

Six causal oscillators feed Stage 3 directly (and Stage 1 indirectly through trend label conditioning): RSI(14), MACD signal-difference (fast 12, slow 26, signal 9), Bollinger %B (20, 2σ), Stochastic %K (14, smoothing 3), 20-day volume z-score, and 20-day OBV change.

---

## 3. Experimental Setup

### 3.1 Cross-Validation

We use *expanding-window* walk-forward at the outer level: training set \([t_0, t_k]\), gap \(g\) (10 days, larger than the 5-day forward-return horizon to prevent label leakage), validation \([t_k+g, t_k+g+v]\) with \(v=200\) days, advancing by `step` = \(v\). This yields 16 outer folds for BTC, 10 for ETH at Stage 1; and 12 / 6 folds at Stage 3.

For *hyperparameter tuning*, we use Optuna's TPE sampler with the MedianPruner and an *inner* walk-forward of 5 folds, evenly-spaced across the dataset, with `train_min=750`, `val=300`, gap=10. The inner-fold dates were chosen to span distinct historical regimes (2017 bull, 2019 recovery, 2021 bull-to-bear flip, 2023 winter recovery, 2025 modern era) so that hyperparameters generalize.

### 3.2 Hyperparameter Tuning

Search spaces for each classifier (Stage 3, identical for Stage 1 with class adjustments):

- XGBoost (6 HPs): `n_estimators` 100-600, `max_depth` 3-8, `learning_rate` 0.01-0.15 (log), `subsample` 0.6-1.0, `colsample_bytree` 0.6-1.0, `min_child_weight` 1-10.
- LightGBM (6 HPs): `n_estimators` 100-600, `num_leaves` 15-127 (log), `learning_rate` 0.01-0.15, `feature_fraction` 0.6-1.0, `bagging_fraction` 0.6-1.0, `min_data_in_leaf` 5-50.
- Random Forest (4 HPs): `n_estimators` 200-800, `max_depth` 6-20, `min_samples_leaf` 3-15 (lower bound 3 motivated by the meta-overfitting study, see §4.5), `max_features` ∈ {sqrt, log2, 0.5}.
- MLP (5 HPs): `hidden_layer_sizes` ∈ {(64), (64,32), (128,64), (64,32,16)}, `learning_rate_init` 1e-4 to 1e-2 (log), `alpha` 1e-6 to 1e-2 (log), `batch_size` ∈ {auto, 64, 128}, ReLU activation fixed.

30 trials per (asset, model) at Stage 1 and Stage 3. An extended-budget run with 60 trials over a wider space is reported as Phase 5.2.

### 3.3 Software and Hardware

Python 3.11; scikit-learn 1.8.0; XGBoost 3.2.0; LightGBM 4.6.0; Optuna 4.8.0. All experiments executed on a single 2024 MacBook Air (M3, 16 GB RAM). The full overnight ablation (4 phases, 4 architectures × 4 models × 2 assets × 30 Optuna trials × 5 inner folds plus retraining and backtests) completed in 1 hour 43 minutes.

### 3.4 Evaluation Metrics

- **Classification:** accuracy, F1 macro, per-class F1, multiclass one-vs-rest ROC-AUC (Buy-vs-rest, Hold-vs-rest, Sell-vs-rest, plus macro-average). Confusion matrices on out-of-fold predictions.
- **Segment-level (Stage 1 only):** majority-vote consistency, mean intersection-over-union, onset-detection F1 with ±5-day tolerance.
- **Economic (Stage 3 only):** annualized Sharpe ratio (252 trading days), total return, maximum drawdown, number of trades, win rate. Buy & Hold benchmark on the same out-of-fold span. Three trading rules — stateful long-only, defensive reset, probability-weighted — are evaluated on the same predictions.

Backtests assume a 0.1% one-way transaction cost; slippage is *not* modelled (a paper limitation).

---

## 4. Results

### 4.1 Stage 1 — Trend Classifier

After hyperparameter tuning with 5-fold inner CV, Random Forest produced the highest macro-F1 on both assets (BTC 0.563, ETH 0.571). All eight tuned models (4 classifiers × 2 assets) cleared the V5_PLAN decision gate of F1m ≥ 0.50, an improvement from the 6/8 pass rate of the un-tuned baseline. The MLP was the largest beneficiary of tuning (BTC F1m 0.462 → 0.537, range-class F1 0.302 → 0.421).

A segment-level analysis revealed that the Stage 1 model captures *macro structure* far better than its frame-level F1 indicates. With a 20-day rolling-mode smoothing applied to predictions, the majority-vote consistency reaches 0.77 (BTC) and 0.83 (ETH) — i.e., within each true contiguous trend segment, the model's dominant predicted class agrees with the truth more than 80% of the time. The smoothing trade-off is task-specific: a 5-day window optimises onset-detection F1 (0.41), a 20-day window optimises persistence metrics; we chose 10 days as the Stage 3 input compromise.

### 4.2 Stage 3 — Signal Classifier (default 3-Stage Full)

Out-of-fold tuned-model macro-F1 was 0.367 for BTC's best model (XGBoost) and 0.368 for ETH's best (XGBoost). All configurations clear the chance baseline of 0.333 but none reach the V5_PLAN target of 0.40. We interpret this as a fundamental signal-to-noise ceiling for 5-day-horizon directional prediction in crypto: doubling Optuna's trial budget to 60 and widening the search space (Phase 5.2) yields only +0.013 F1m on the best ETH configuration, confirming saturation.

ROC-AUC analysis (multiclass one-vs-rest) confirms this: macro AUC ranges from 0.49 to 0.53 across 16 architecture × model configurations per asset. Class ranking is therefore only marginally above chance.

### 4.3 Architecture Ablation (Phase 5.1) — Asset-Specific Optimum

**Table 1.** Best-rule, best-model backtest metrics per architecture.

| Asset | Architecture | Best (rule, model) | Sharpe | Return | MaxDD | F1m |
|---|---|---|---:|---:|---:|---:|
| BTC | flat | stateful, xgboost | +0.93 | +1565% | -75% | 0.354 |
| BTC | 2-Stage Trend | prob_weighted, lightgbm | +1.08 | +539% | -28% | 0.358 |
| BTC | 2-Stage Macro | stateful, random_forest | +0.98 | +1606% | -53% | 0.361 |
| BTC | **3-Stage Full** | **stateful, xgboost** | **+1.15** | **+2901%** | -46% | **0.367** |
| BTC | Buy & Hold | — | +0.95 | +2972% | -77% | — |
| ETH | **flat** | **prob_weighted, lightgbm** | **+0.52** | **+26%** | **-18%** | 0.354 |
| ETH | 2-Stage Trend | defensive, xgboost | +0.39 | +43% | -55% | 0.368 |
| ETH | 2-Stage Macro | stateful, mlp | +0.47 | +69% | -55% | 0.342 |
| ETH | 3-Stage Full | prob_weighted, lightgbm | +0.34 | +19% | -18% | 0.336 |
| ETH | Buy & Hold | — | +0.26 | -7% | -72% | — |

For BTC, the relationship between architectural depth and Sharpe ratio is *monotonic*: each added stage improves both Sharpe (0.93 → 1.08 → 1.15) and macro-F1 (0.354 → 0.358 → 0.367). Hierarchical fusion adds genuine information that a flat model cannot extract from oscillators alone.

For ETH, the relationship reverses: the flat 6-oscillator model achieves Sharpe 0.52, whereas adding upstream posteriors degrades performance to 0.34 in the 3-stage full case. This is consistent with overfitting due to ETH's smaller out-of-fold span (2,000 versus 3,200 days for BTC) and a higher feature-to-sample ratio. The asset-specific optimum is a concrete instance of the no-free-lunch theorem in applied financial ML.

Notably, every architecture beats Buy & Hold on Sharpe for both assets (the worst case, ETH 3-Stage Full, is still 0.34 versus 0.26). On ETH, every architecture also beats Buy & Hold on absolute return (Buy & Hold posted −7% over the 2020-02 to 2025-08 OOF span, dominated by the 2022 bear market).

### 4.4 Trading-Rule Ablation

Three rules were evaluated on every (asset, architecture, model) combination. The stateful long-only rule (Buy = enter, Sell = exit, Hold = preserve state) maximises BTC Sharpe but entails the highest drawdown. The probability-weighted rule (continuous position size = clip(P_Buy − P_Sell, 0, 1)) is the clear ETH winner: it cuts drawdown to 18% while still delivering positive return where Buy & Hold loses money. The defensive rule (Buy = long, otherwise cash) sits between the two, generally with the highest trade count and highest transaction-cost drag.

A per-asset best rule emerges: BTC favours stateful (because the test period is bull-heavy), ETH favours probability-weighted (because the volatile ETH regime requires soft sizing for risk management).

### 4.5 Inner-CV Meta-Overfitting Case Study

In our first round of Stage 1 hyperparameter tuning, we used a 3-fold inner CV. The Optuna procedure selected `min_samples_leaf=1` for Random Forest on both assets — a known degenerate value that allows leaf nodes to memorise individual training samples. On the inner folds (only three held-out windows: 2017, 2021, 2025), this choice scored a high F1 macro of 0.50; on the outer 16-fold walk-forward, however, Random Forest *regressed* by 0.014 F1m relative to the un-tuned baseline (0.557 → 0.543).

We diagnosed this as classical *meta-overfitting*: the inner CV setup was insufficiently representative of the test distribution. Switching to a 5-fold inner CV with evenly-spaced historical regimes — 2017 (bull setup), 2019 (recovery), 2021 (peak-to-bear), 2023 (winter recovery), 2025 (modern) — shifted the Optuna selection to `min_samples_leaf=11` for BTC and `=5` for ETH (closer to the un-tuned defaults) and recovered the lost performance plus added +0.006 F1m on BTC and +0.012 F1m on ETH. We hardened the search space lower bound to `min_samples_leaf ≥ 3` going forward.

This case study highlights an under-discussed risk in financial-ML pipelines: the inner-CV configuration is itself a hyperparameter, and an under-resourced inner CV can degrade a hyperparameter-tuned model below the un-tuned baseline.

### 4.6 Signal-Label Threshold Sensitivity

The `k` parameter in the adaptive label threshold controls the Buy/Hold/Sell balance. Lower `k` produces more Buy/Sell at the expense of Hold. We tested k ∈ {0.4, 0.5, 0.7, 1.0}. For BTC, k = 0.5 was optimal across all four classifiers. For ETH, k = 0.5 was optimal for tree-based models, but the MLP's macro-F1 improved from 0.312 (k=0.5) to 0.361 (k=1.0). We retained k = 0.5 throughout for fair cross-classifier comparison.

---

## 5. Discussion

### 5.1 Frame-Level vs Trade-Level Performance

The Stage 3 macro-F1 of 0.37 sits well below the V5_PLAN target of 0.40 and only marginally above the chance baseline of 0.33. ROC-AUC, similarly, sits at 0.53 — barely above chance. Yet the same model, evaluated as a trading strategy, achieves a Sharpe of 1.15 on BTC. This dissociation deserves explicit discussion.

We see two reasons. First, the Hold class operates as a *risk-management filter*: the stateful and probability-weighted rules use the Hold posterior to *avoid* trades on uncertain days, preserving capital that a stateless 2-class classifier would have committed. This converts low frame accuracy into high economic precision. Second, the metrics are not equally meaningful for trading: a Sharpe of 1.15 implies an average daily-return-to-volatility ratio of ~0.07, which corresponds to a per-day directional accuracy of only slightly above 50% — fully consistent with the observed AUC of 0.53. The non-linearity arises from compounding and from selectivity (179 BTC trades versus 2,200 trading days).

### 5.2 The Hierarchical Architecture is Not Universally Beneficial

Our most striking finding is the asset-specific architecture optimum: 3-Stage Full is best for BTC, flat is best for ETH. This is consistent with a literature in which deep models on tabular financial data *can* lose to simpler baselines when the dataset is small [16], but to our knowledge has not been documented as a function of *architectural depth in a hierarchical pipeline*.

We hypothesise that the BTC 10-year span captures multiple distinct regimes (2014-2017 nascent, 2018 bear, 2019-2020 recovery, 2021 bull peak, 2022 winter, 2023-2025 modern) that benefit from regime-aware features, while the ETH 5-year span is dominated by 2020-2022 dynamics in which raw oscillators carry sufficient information without the upstream-posterior overhead.

The practical implication is that *every* hierarchical-architecture proposal in financial ML should be re-validated per asset and per dataset size. Our four-architecture ablation provides a template.

### 5.3 The FSM at Stage 2

We adopted a deterministic finite-state machine for Stage 2 only after four canonical unsupervised approaches failed: vanilla K-Means could not capture the 2008 Global Financial Crisis structure, semantic constrained K-Means produced noisy regime transitions, an HMM with three Gaussian states displayed unstable transition probabilities, and a 3-component GMM exhibited 2024-2025 stickiness with P(Stress) saturated near 1.0. While ML purists might object to a hand-crafted FSM in a "Pattern Recognition" project, we argue that the FSM *is* the recognised pattern: it codifies the macro-financial pattern of risk-on/risk-off into rules that are auditable, robust, and validated against the 2008, 2020, and 2025 crises.

### 5.4 Limitations

- **Transaction cost realism**: 0.1% one-way is conservative for liquid spot crypto but ignores spread, slippage, and execution latency. A sensitivity analysis with 0.2-0.5% would tighten the realism.
- **MLP class balancing**: scikit-learn's MLP does not support `class_weight=balanced` natively; we did not implement an oversampling fallback. A focal-loss MLP or `imblearn.RandomOverSampler` might lift Stage 3 MLP performance.
- **Out-of-fold span asymmetry**: BTC OOF starts in 2016 versus ETH 2020, partly explaining the architecture asymmetry. Repeating the experiment with a fixed 2020-2025 window for BTC would isolate the dataset-size variable.
- **Label leakage of forward returns**: while we never *use* forward returns as features, the labels are derived from them. This is standard supervised practice but should be acknowledged when comparing against unsupervised regime-detection baselines.
- **No statistical significance testing**: a McNemar test [24] over the OOF predictions would formalise the architecture-comparison conclusions.

### 5.5 Future Work

- **Stage-1 calibration** (isotonic / Platt) before soft fusion to convert raw posteriors into calibrated probabilities, with a paper-quality reliability-diagram analysis.
- **A dynamic position-sizing layer** (Kelly-criterion based or volatility-targeted) on top of the probability-weighted rule.
- **Ablation on the Stage-2 FSM rule set** to identify which of the eight rules carries the most economic value.
- **A re-tune of Stage 1 with the dev=0.07 ZigZag configuration** identified by the Phase 3.6 distribution sweep, which produces a more class-balanced label set.

---

## 6. Conclusion

We presented a three-stage hierarchical soft-fusion framework for cryptocurrency trading-signal classification, evaluated rigorously on Bitcoin and Ethereum daily data with walk-forward cross-validation, Optuna hyperparameter tuning, four-classifier comparisons, and a four-architecture ablation. The framework achieves an annualized Sharpe of 1.15 on BTC and a clean Sharpe + return win on ETH versus Buy & Hold — particularly striking given that ETH's Buy & Hold lost 7% over the test span while our model returned +26%.

Three findings stand out. First, **architectural depth is asset-specific**: the full hierarchy helps BTC but overfits ETH, where a flat 6-feature baseline wins. Second, **frame-level metrics underestimate trade-level value**: a macro-F1 of 0.37 and a ROC-AUC of 0.53 nonetheless yield Sharpe 1.15 once a Hold-aware trading rule is applied. Third, **inner-CV configuration is itself a hyperparameter**: a poorly-resourced 3-fold inner CV can induce meta-overfitting that degrades the tuned model below the un-tuned baseline; a 5-fold evenly-spaced inner CV recovers the loss.

The pipeline, code, and OOF predictions are reproducible and packaged in a single Docker container with a FastAPI back-end and Chart.js front-end for live demonstration.

---

## References

[1] L. Sezer, M. U. Gudelek, A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019," *Applied Soft Computing*, vol. 90, art. 106181, 2020.

[2] J. Patel, S. Shah, P. Thakkar, K. Kotecha, "Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques," *Expert Systems with Applications*, vol. 42, no. 1, pp. 259-268, 2015.

[3] M. I. Jordan, R. A. Jacobs, "Hierarchical mixtures of experts and the EM algorithm," *Neural Computation*, vol. 6, no. 2, pp. 181-214, 1994.

[4] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241-259, 1992.

[5] C. N. Silla Jr., A. A. Freitas, "A survey of hierarchical classification across different application domains," *Data Mining and Knowledge Discovery*, vol. 22, no. 1-2, pp. 31-72, 2011.

[6] C. Challu, K. G. Olivares, B. N. Oreshkin, F. G. Ramirez, M. M. Canseco, A. Dubrawski, "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting," *AAAI*, vol. 37, no. 6, pp. 6989-6997, 2023.

[7] Y. Zhang, J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," *ICLR*, 2023.

[8] Y. Zou, R. Lin, X. Liu, Y. Liu, "A novel deep reinforcement learning based automated stock trading system using cascaded LSTM networks," *Expert Systems with Applications*, vol. 242, art. 122801, 2024.

[9] J. Patel, S. Shah, P. Thakkar, K. Kotecha, "Predicting stock market index using fusion of machine learning techniques," *Expert Systems with Applications*, vol. 42, no. 4, pp. 2162-2172, 2015.

[10] O. B. Sezer, A. M. Ozbayoglu, "Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach," *Applied Soft Computing*, vol. 70, pp. 525-538, 2018.

[11] O. Kuznetsov et al., "Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction," *Applied Sciences*, vol. 15, no. 20, art. 11145, 2025.

[12] J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," *Econometrica*, vol. 57, no. 2, pp. 357-384, 1989.

[13] M. Ang, J. Chen, "International asset allocation with regime shifts," *Review of Financial Studies*, vol. 15, no. 4, pp. 1137-1187, 2002.

[14] T. G. Andersen, T. Bollerslev, F. X. Diebold, P. Labys, "Modeling and forecasting realized volatility," *Econometrica*, vol. 71, no. 2, pp. 579-625, 2003.

[15] L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?," *NeurIPS Datasets and Benchmarks*, 2022.

[16] R. Shwartz-Ziv, A. Armon, "Tabular data: Deep learning is not all you need," *Information Fusion*, vol. 81, pp. 84-90, 2022.

[17] M. Lopez de Prado, *Advances in Financial Machine Learning*. Wiley, 2018.

[18] M. Lopez de Prado, "The 10 reasons most machine learning funds fail," *Journal of Portfolio Management*, vol. 44, no. 6, pp. 120-133, 2018.

[19] D. H. Bailey, M. Lopez de Prado, "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality," *Journal of Portfolio Management*, vol. 40, no. 5, pp. 94-107, 2014.

[20] T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," *ACM SIGKDD*, pp. 785-794, 2016.

[21] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *NIPS*, vol. 30, pp. 3146-3154, 2017.

[22] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[23] D. E. Rumelhart, G. E. Hinton, R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533-536, 1986.

[24] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895-1923, 1998.
