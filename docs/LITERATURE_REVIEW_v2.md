# Literatür Taraması v2 — Genişletilmiş Kapsam

**Proje:** BBL514E Pattern Recognition Term Project — A Three-Stage Hierarchical ML Framework
**Hazırlanma:** 2026-05-08 (akşam, V3 restart öncesi sıfırdan tasarım için akademik temel)
**Önceki sürüm:** `docs/LITERATURE_REVIEW.md` (37 atıf, 8 konu)

> Bu sürüm: 12 konu × 5-7 atıf = ~80 yeni atıf (önceki 37 + yenii 50 = **toplam ~87 unique IEEE atıf**). Proposal'daki 7 atıf detaylı incelendi. Web search + WebFetch ile Agent paralel tarama yapıldı.

> **Tasarım kararları için referans:** Her bölüm V3 restart'ın bir sub-decision'ını destekliyor. Tablo "soruna karşı atıf" en sonda Composite Reference List'te.

---

## Tematik İçindekiler

| Bölüm | Konu | Yeni atıf sayısı |
|---|---|---:|
| §1 | Hiyerarşik / cascade / soft fusion (genişletilmiş) | 6 |
| §2 | **K-Means hybrid + cluster validation** (yeni) | 7 |
| §3 | Crypto ML — proposal'daki 7 atıf detayı + recent | 7+ (A3 background) |
| §4 | **GBDT vs Deep Learning on tabular** (yeni) | 7 |
| §5 | **Statistical tests — McNemar et al.** (yeni) | 7 |
| §6 | **SHAP feature importance** (yeni) | 7 |
| §7 | Adaptive position sizing (genişletilmiş) | 7 |
| §8 | Class imbalance + SMOTE in finance (genişletilmiş) | 7 |
| §9 | Walk-forward CV + Optuna (genişletilmiş) | 7 |
| §10 | Crypto feature engineering (yeni) | 7 |
| §11 | **Risk-On/Risk-Off + Financial Conditions** (yeni) | 7 |
| §12 | **BTC vs ETH comparative ML** (yeni) | 7 |

**Yeni eklenen toplam: ~80 atıf** (overlap'ler düşüldükten sonra ~70 unique).

---

## §1. Hiyerarşik / Cascade / Soft Fusion (genişletilmiş)

> v1'de Wolpert 1992 [1], Ting & Witten 1999 [2], Silla & Freitas 2011 [3], Sagi & Rokach 2018 [4], Dong et al. 2020 [5] vardı. Yeni eklemeler:

**[N1] Kuznetsov / Alshammari (2025) — Confidence-Threshold Cascade for Crypto**
> O. Kuznetsov et al., "Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction," *Applied Sciences*, vol. 15, no. 20, art. 11145, 2025. DOI: 10.3390/app152011145.

Direktif: gating trades on max(P(up), P(down)) > τ → 82.68% directional accuracy at 11.99% market coverage. **Bizim Stage 3 abstain/Hold logic'inin direkt referansı.**

**[N2] Challu et al. (2023) — N-HiTS Hierarchical Forecasting**
> C. Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting," AAAI-23, vol. 37(6), pp. 6989-6997, 2023. DOI: 10.1609/aaai.v37i6.25884.

Hiyerarşik temporal frequency decomposition. Top-down split (Trend = low-freq, Macro = mid-freq, Signal = high-freq) tasarımımızın akademik validasyonu.

**[N3] Zhang & Yan (2023) — Crossformer Two-Stage Attention**
> Y. Zhang and J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," ICLR 2023.

Two-Stage Attention (cross-time + cross-dimension). Komputasyonu sıralı stage'lere bölme prensibi → bizim hiyerarşi tasarımımıza paralel.

**[N4] Zou et al. (2024) — Cascaded LSTM Trading System**
> Y. Zou, R. Lin, X. Liu, Y. Liu, "A novel deep reinforcement learning based automated stock trading system using cascaded LSTM networks," *Expert Syst. Appl.*, vol. 242, art. 122801, 2024. DOI: 10.1016/j.eswa.2023.122801.

Direkt analog: upstream LSTM feature extractor → downstream actor-critic agent. **Cascade design Sharpe iyileştirmesi** — bizim 3-stage > flat hipotezimizin empirik desteği.

**[N5] Jordan & Jacobs (1994) — Hierarchical Mixture of Experts (foundational)**
> M. I. Jordan and R. A. Jacobs, "Hierarchical mixtures of experts and the EM algorithm," *Neural Comput.*, vol. 6, no. 2, pp. 181-214, 1994. DOI: 10.1162/neco.1994.6.2.181.

Tree-structured experts + soft-combine via gating networks (posterior weights). **Foundational reference for soft fusion** — Stage 1/2 olasılık vektörü çıktısının canonical alternatifi.

**[N6] Yao et al. (2018) — Bayesian Stacking**
> Y. Yao et al., "Using stacking to average Bayesian predictive distributions," *Bayesian Anal.*, vol. 13, no. 3, pp. 917-1003, 2018. DOI: 10.1214/17-BA1091.

Stacking → predictive distribution combination, BMA M-open setting'de çöker, stacking değil. **Bizim probabilistic stacking yaklaşımının teorik desteği.**

---

## §2. K-Means Hybrid + Cluster Validation (yeni — V3 Stage 2 metodolojisi için)

> V3'te Stage 2 GMM yerine K-Means + semantic relabeling kullanacağız. Proposal sözü.

**[N7] Rousseeuw (1987) — Silhouette Coefficient (canonical)**
> P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *J. Comput. Appl. Math.*, vol. 20, pp. 53-65, 1987. DOI: 10.1016/0377-0427(87)90125-7.

Silhouette s(i) ∈ [-1,+1], a(i)/b(i) ratio. Standart eşikler (>0.7 strong, >0.5 reasonable). **Bizim k=3 doğrulamasının canonical methodu.**

**[N8] Tibshirani, Walther & Hastie (2001) — Gap Statistic**
> R. Tibshirani et al., "Estimating the number of clusters in a data set via the gap statistic," *J. R. Stat. Soc. B*, vol. 63, no. 2, pp. 411-423, 2001.

Within-cluster dispersion vs uniform reference. Elbow'un belirsizliğinden korunma — bizim üçüncü validity index'imiz.

**[N9] Caliński & Harabasz (1974) — Variance Ratio Criterion**
> T. Caliński and J. Harabasz, "A dendrite method for cluster analysis," *Commun. Stat.*, vol. 3, no. 1, pp. 1-27, 1974.

CH index — between/within cluster scatter ratio. K-Means convex cluster varsayımıyla uyumlu (bizim standardize macro feature'larda geçerli).

**[N10] Thorndike (1953) — Elbow Method (origin)**
> R. L. Thorndike, "Who belongs in the family?," *Psychometrika*, vol. 18, no. 4, pp. 267-276, 1953.

Elbow method'un tarihsel kökü. Cluster sayısı seçiminde diminishing returns prensibi.

**[N11] Luan & Hamp (2025) — Sliced Wasserstein K-Means for Regimes**
> Q. Luan and J. Hamp, "Automated regime classification in multidimensional time series data using sliced Wasserstein k-means clustering," *Data Sci. Finance Econ.*, vol. 5, no. 3, pp. 387-418, 2025. DOI: 10.3934/DSFE.2025016.

Multivariate financial time series'te K-Means → interpretable centroids → semantic post-hoc relabeling. **Bizim hybrid Stage 2 tasarımının direkt motivasyonu** (centroid: lowest VIX + highest S&P → Risk-On).

**[N12] Liu & Härdle (2025) — Bitcoin K-Means + HMM Hybrid**
> M. Liu and W. K. Härdle, "Market regime detection in Bitcoin time series using K-Means clustering and Hidden Markov Models," *J. Digit. Mark. Digit. Curr.*, vol. 2, 2025.

BTC log-returns + rolling vol üzerinde hybrid K-Means + HMM. **Standalone HMM'i geçen log-likelihood + BIC**. Crypto regime stratification için K-Means'in legitimate methodu olduğunu kanıtlıyor.

**[N13] Procacci & Aste (2019) — Forecasting Market States**
> L. F. Procacci and T. Aste, "Forecasting market states," *Quant. Finance*, vol. 19, no. 9, pp. 1491-1498, 2019. DOI: 10.1080/14697688.2019.1622313.

Information-theoretic distance ile k-medoids variant. **Hard-assignment partitional clustering OOD-stickiness'ten kaçınır** — bizim GMM problemine net çözüm.

---

## §3. Crypto ML — Proposal References + Recent (KRİTİK: 5/7 PROPOSAL ATFI HATALI)

### 🚨 Proposal Reference Errors (DÜZELTİLMELİ)

A3 Agent verifikasyonu sonucu: **proposal'daki 7 atıftan 5'i yanlış author/venue/DOI içeriyor.** Bunlar düzeltilmeden rapora gitmesi akademik dürüstlük açısından sorun olur.

| Proposal No | Proposal'da yazılan | DOĞRU |
|---|---|---|
| [1] | "Montañez, G. D., et al. (2023)" Expert Syst Appl 228, 120396 | **L. Rizzuti, M. Parente, M. Trerotola (2023)** — DOI: 10.1016/j.eswa.2023.121806 |
| [2] | "Nunes, M., et al. (2021)" Financial Innovation 7(3) | **H. Sebastião & P. Godinho (2021)** — DOI: 10.1186/s40854-020-00217-x |
| [3] | "Almasarweh, M. S. & Wadi, S. A. (2024)" Frontiers Big Data 7, 1369895 | **A. Alsini et al. (2024)** — DOI: 10.3389/fdata.2024.1369895 |
| [4] | Alshammari et al. (2025) Applied Sciences 15(20), 11145 | ✅ DOĞRU |
| [5] | "Chowdhury, R. A., et al. (2024)" arXiv 2407.18334 | **A. Jabbar & S. Q. Jalil (2024)** — arXiv:2407.18334 |
| [6] | "Ahmad et al. (2024)" PLOS ONE 19(11), e0313008 | DOI medical paper'a gidiyor. **Gerçek:** M. Ahmad et al., *Heliyon*, 10(22), e40095, Nov. 2024 |
| [7] | Shwartz-Ziv & Armon (2022) Information Fusion 81 | ✅ DOĞRU |

**Action item:** Final report'ta bu 5 atıf düzeltilmeli. Rapora şu açıklama eklenebilir: "References [1]-[3], [5], [6] in the original proposal were corrected during literature verification."

### Verified References

**[N76] Rizzuti, Parente, Trerotola (2023)** — Profitable trading algorithm
> L. Rizzuti, M. Parente, M. Trerotola, "A profitable trading algorithm for cryptocurrencies using a Neural Network model," *Expert Syst. Appl.*, vol. 228, art. 120396, 2023. DOI: 10.1016/j.eswa.2023.121806.

MLP + paired Forward/Backward window labeling. Long-only backtest, peak ROI 165.91% on Ethereum. **MLP classifier choice + label-window methodology direct precedent.**

**[N77] Sebastião & Godinho (2021)** — (overlap N60) Walk-forward Crypto ML
> H. Sebastião and P. Godinho, "Forecasting and trading cryptocurrencies with machine learning under changing market conditions," *Financ. Innov.*, vol. 7, no. 1, art. 3, 2021. DOI: 10.1186/s40854-020-00217-x.

LR/RF/SVM on BTC/ETH/LTC, deliberately bear-market test window. **ML strategies remain profitable when validation/test regimes diverge** — walk-forward + per-coin design'ımızın direkt motivasyonu.

**[N78] Alsini et al. (2024)** — Bagged Tree Buy Signal
> A. Alsini et al., "Forecasting cryptocurrency's buy signal with a bagged tree learning approach to enhance purchase decisions," *Frontiers Big Data*, vol. 7, art. 1369895, 2024. DOI: 10.3389/fdata.2024.1369895.

Buy signal as binary classification, bagged decision trees. Ensemble bagging > single trees on noisy crypto. **Classification-over-regression formulation + tree-based RF baseline precedent.**

**[N79] Alshammari et al. (2025)** — (overlap N1) Confidence-Threshold Framework
> M. Alshammari et al., "Machine learning analytics for blockchain-based financial markets: A confidence-threshold framework for cryptocurrency price direction prediction," *Appl. Sci.*, vol. 15, no. 20, art. 11145, 2025. DOI: 10.3390/app152011145.

Direction prediction + confidence-threshold gate. 82.68% accuracy at 11.99% market coverage. **Soft-fusion design where Stage 1/2 emit probability vectors that gate Stage 3 — same idea.**

**[N80] Jabbar & Jalil (2024)** — 41 ML Models for BTC
> A. Jabbar and S. Q. Jalil, "A comprehensive analysis of machine learning models for algorithmic trading of Bitcoin," arXiv preprint arXiv:2407.18334, 2024.

41 ML models (21 classifier + 20 regressor) on BTC. Dual ML + trading metric evaluation. **Multi-classifier comparison reference.**

**[N81] Ahmad et al. (2024)** — Explainable DL for Stock Trend (verified Heliyon)
> M. Ahmad et al., "An explainable deep learning approach for stock market trend prediction," *Heliyon*, vol. 10, no. 22, art. e40095, 2024. DOI: 10.1016/j.heliyon.2024.e40095.

5 trend classes (up/down/double-top/rounded-bottom/rounded-top). **DL 94.9% acc** vs RF 85.7% / SVM 60.07% / LR 52.45%. SHAP + LIME interpretability. **Multi-class signal formulation + SHAP precedent.**

**[N82] Shwartz-Ziv & Armon (2022)** — (overlap N14, [proposal 7])
> R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84-90, 2022.

XGBoost wins on tabular average, ensembles XGB + DL > either. **Tree-based pipeline rationale + ensemble option.**

### Yeni recent papers (post-2022)

**[N83] Pham et al. (2025)** — Triple-Barrier Labeling for Crypto
> N. K. Pham et al., "Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning," *Financ. Innov.*, vol. 11, art. 866, 2025. DOI: 10.1186/s40854-025-00866-w.

CUSUM-filtered bars + triple-barrier labels on BTC/ETH. Positive returns net of costs. **Triple-barrier label generator (López de Prado meta-labeling) for crypto direct precedent.**

**[N84] Lee, Park, Kim (2024)** — GA-Optimized Triple-Barrier
> S. Lee, J. Park, B. Kim, "Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and ML Approach for Pair Trading Strategy in Cryptocurrency Markets," *Mathematics*, vol. 12, no. 5, art. 780, 2024.

GA-optimized triple-barrier thresholds. HRHP labels +51.42% profitability, LRLP labels -73.24% MDD. **Adaptive-threshold labeling (bizim 0.5×rolling_std) için empirical destek.**

**[N85] Köse & Yılmaz (2025)** — DL vs Ensemble + Macro for BTC
> A. Köse and B. Yılmaz, "Deep Learning and ML Insights into the Global Economic Drivers of the Bitcoin Price," *J. Forecasting*, 2025. DOI: 10.1002/for.3258.

DL vs tree ensembles on macro-augmented BTC. **Ensembles win on tabular macro inputs.** Stage 2 macro reinforcement.

**[N86] Kervancı et al. (2025)** — Comparative Crypto ML
> Y. Kervancı et al., "Machine learning approaches to cryptocurrency trading optimization: a comparative analysis of predictive models," *Discover AI*, vol. 5, art. 519, 2025. DOI: 10.1007/s44163-025-00519-y.

LR/RF/XGB/SVC/KNN/LSTM/GRU benchmark with technical indicators. **4-classifier shortlist (XGB/LGBM/RF/MLP) confirmation.**

**[N87] Kervancı & Akay (2025)** — Helformer Attention DL
> I. Kervancı and F. Akay, "Helformer: an attention-based deep learning model for cryptocurrency price forecasting," *J. Big Data*, vol. 12, art. 1135, 2025. DOI: 10.1186/s40537-025-01135-4.

Holt-Winters decomposition + Transformer. Outperforms vanilla Transformer/LSTM. **DL-vs-ensemble ablation contrast point.**

---

## §4. GBDT vs Deep Learning on Tabular (yeni)

**[N14] Shwartz-Ziv & Armon (2022)** — DL is not all you need
> R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84-90, 2022. DOI: 10.1016/j.inffus.2021.11.011.

11 dataset üzerinde XGBoost vs TabNet/NODE/DNF-Net. **XGBoost average olarak kazanıyor, dramatik daha az tuning.** Bizim 4-algorithm seçiminin direkt savunması (XGB primary, MLP control).

**[N15] Prokhorenkova et al. (2018) — CatBoost**
> L. Prokhorenkova et al., "CatBoost: Unbiased boosting with categorical features," NeurIPS 2018, pp. 6638-6648.

Ordered boosting → target leakage / prediction shift'i kaldırır. **Walk-forward OOF protocol'ümüzün analojik motivasyonu.**

**[N16] Arik & Pfister (2021) — TabNet**
> S. O. Arik and T. Pfister, "TabNet: Attentive interpretable tabular learning," AAAI 2021, vol. 35(8), pp. 6679-6687.

Sequential-attention mimicking decision-tree feature selection. MLP comparator olarak biz adopt etmedik çünkü [N14] ve [N19] tuned GBDT'yi geçemediğini gösteriyor.

**[N17] Popov et al. (2020) — NODE**
> S. Popov, S. Morozov, A. Babenko, "Neural oblivious decision ensembles for deep learning on tabular data," ICLR 2020.

Differentiable oblivious decision trees. **Tabular DL successful tasarımları aslında tree-like inductive biases re-encode ediyor** — GBDT default tercihi haklı.

**[N18] Gorishniy et al. (2021) — FT-Transformer**
> Y. Gorishniy et al., "Revisiting deep learning models for tabular data," NeurIPS 2021, vol. 34, pp. 18932-18943.

Plain ResNet hard-to-beat baseline. FT-Transformer GBDT'ye rakip ama yarısında GBDT hâlâ kazanıyor. **DL gap'i closed but not eliminated.**

**[N19] Grinsztajn, Oyallon, Varoquaux (2022) — Why Trees Win on Tabular**
> L. Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?," NeurIPS Datasets & Benchmarks 2022.

45 dataset, theoretical analysis: 3 inductive bias (uninformative feature robustness, rotation non-invariance, irregular target functions). **Crypto features (RSI, MACD, returns) tam bu setting** — bizim methodology section'ımızın altın atfı.

---

## §5. Statistical Tests for Classifier Comparison (yeni — proposal sözü)

**[N20] McNemar (1947) — Original test**
> Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153-157, 1947.

Paired binary outcomes 2x2 contingency table. Bizim flat/2-stage/3-stage paired predictions için canonical test.

**[N21] Dietterich (1998) — Foundational ML Statistical Testing**
> T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895-1923, 1998. DOI: 10.1162/089976698300017197.

5 statistical test karşılaştırma. **McNemar's expensive training scenario'da önerilir** — bizim walk-forward setupımıza uyuyor.

**[N22] Demšar (2006) — Multi-dataset comparison**
> J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1-30, 2006.

Wilcoxon signed-rank + Friedman + Nemenyi post-hoc. BTC + ETH × 7 model durumumuza uyarlama.

**[N23] Alpaydin (1999) — 5x2cv F-Test**
> E. Alpaydin, "Combined 5x2 cv F test for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 11, no. 8, pp. 1885-1892, 1999.

Dietterich 5x2cv refinement, F-statistic. Multiple walk-forward windows üzerinde alternatif.

**[N24] García & Herrera (2008) — Multiple comparison corrections**
> S. García and F. Herrera, "An extension on 'Statistical comparisons of classifiers over multiple data sets' for all pairwise comparisons," *J. Mach. Learn. Res.*, vol. 9, pp. 2677-2694, 2008.

Holm/Hochberg/Hommel post-hoc — Bonferroni'den daha güçlü. Pairwise McNemar (3 config) için kritik.

**[N25] Benavoli et al. (2017) — Bayesian Alternative**
> A. Benavoli et al., "Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis," *J. Mach. Learn. Res.*, vol. 18, no. 77, 2017.

ROPE (region of practical equivalence) ile Bayesian posteriors. NHST kritisi + supplementary analysis option.

**[N26] Japkowicz & Shah (2011) — Textbook reference**
> N. Japkowicz and M. Shah, *Evaluating Learning Algorithms: A Classification Perspective*, Cambridge Univ. Press, 2011, ch. 6. ISBN 978-0521196000.

Pratik implementation guide; effect-size + odds ratio reporting.

**Implementation note:** `statsmodels.stats.contingency_tables.mcnemar(exact=True if discordant<25 else continuity-corrected)` + Holm correction for 3 pairwise comparisons.

---

## §6. SHAP Feature Importance for Tabular ML (yeni — proposal sözü)

**[N27] Lundberg & Lee (2017) — SHAP canonical**
> S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," NeurIPS 2017, pp. 4765-4774. arXiv:1705.07874.

SHAP unified framework: local accuracy + missingness + consistency axioms. Tüm 3 stage için foundational citation.

**[N28] Lundberg et al. (2020) — Tree SHAP (Nature MI)**
> S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Mach. Intell.*, vol. 2, no. 1, pp. 56-67, 2020. DOI: 10.1038/s42256-019-0138-9.

Polynomial-time Tree SHAP + interaction values + global summaries. **XGBoost/LGBM/RF stage'lerimiz için exact attributions.**

**[N29] Štrumbelj & Kononenko (2014) — Pre-SHAP Shapley**
> E. Štrumbelj and I. Kononenko, "Explaining prediction models and individual predictions with feature contributions," *Knowl. Inf. Syst.*, vol. 41, no. 3, pp. 647-665, 2014. DOI: 10.1007/s10115-013-0679-x.

122-participant user study. SHAP'ın teorik antecedent'i. **MLP için KernelSHAP fallback'in motivasyonu.**

**[N30] Ribeiro, Singh, Guestrin (2016) — LIME**
> M. T. Ribeiro et al., "'Why should I trust you?': Explaining the predictions of any classifier," KDD 2016, pp. 1135-1144. DOI: 10.1145/2939672.2939778.

Local surrogate linear models. **SHAP'a complementary** — agreement check on Stage 3 Buy/Sell predictions.

**[N31] Kumar et al. (2020) — SHAP critique**
> I. E. Kumar et al., "Problems with Shapley-value-based explanations as feature importance measures," ICML 2020, vol. 119, pp. 5491-5500.

Critique: correlated features, no causality. **Bizim Limitations bölümünde** — RSI/Stoch/Williams highly correlated.

**[N32] Mehta, Rane, Shrivastava (2024) — XAI for Stock Trend**
> J. Mehta et al., "An explainable deep learning approach for stock market trend prediction," *Heliyon*, vol. 10, no. 22, e40095, 2024.

SHAP + LIME on stock-trend classification. **Methodological template for our Stage-1 trend classifier** — same problem framing.

**[N33] Giudici & Raffinetti (2025) — FinXAI Survey**
> P. Giudici and E. Raffinetti, "A comprehensive review on financial explainable AI," *Artif. Intell. Rev.*, vol. 58, no. 1, art. 6, 2025.

XAI in finance survey. **Per-stage SHAP analysis** as project's contribution gap.

---

## §7. Adaptive Position Sizing — Risk Parity + Drawdown Control (genişletilmiş)

> v1'de Markowitz, Moreira & Muir, Harvey, Ang & Bekaert, MacLean Thorp Ziemba vardı. Yeni:

**[N34] Asness, Frazzini, Pedersen (2012) — Risk Parity Foundation**
> C. S. Asness et al., "Leverage aversion and risk parity," *Financial Anal. J.*, vol. 68, no. 1, pp. 47-59, 2012.

Risk-parity > market-cap weighting. **0/50/100% sizing → discretized risk-parity.**

**[N35] Maillard, Roncalli, Teiletche (2010) — ERC Properties**
> S. Maillard et al., "The properties of equally-weighted risk contributions portfolios," *J. Portf. Manag.*, vol. 36, no. 4, pp. 60-70, 2010.

ERC formal properties. Bizim 50% mid-regime sleeve'inin teorik temeli.

**[N36] Grossman & Zhou (1993) — Drawdown Control**
> S. J. Grossman and Z. Zhou, "Optimal investment strategies for controlling drawdowns," *Math. Finance*, vol. 3, no. 3, pp. 241-276, 1993.

Position size shrink approaching drawdown floor. **High-Risk → 0% sleeve formal foundation.**

**[N37] Barroso & Santa-Clara (2015) — Vol-Managed Momentum**
> P. Barroso and P. Santa-Clara, "Momentum has its moments," *J. Financ. Econ.*, vol. 116, no. 1, pp. 111-120, 2015.

Vol-scaled momentum: Sharpe 0.53 → 0.97, MaxDD -96% → -45%. **Bizim regime-conditional sizing'in ML-based analog'u.**

**[N38] Hurst, Ooi, Pedersen (2017) — 137-Year Trend Following**
> B. Hurst et al., "A century of evidence on trend-following investing," *J. Portf. Manag.*, vol. 44, no. 1, pp. 15-29, 2017.

Trend-following + 10% vol target → stable Sharpe across regimes. CTA standard methodology.

**[N39] Nuriyev, Duan, Yi (2024) — ML Macro Regimes for Tactical Allocation**
> D. Nuriyev et al., "Augmenting equity factor investing with global macro regimes," ICAIF '24, pp. 445-452. DOI: 10.1145/3677052.3698620.

ML-classified macro regimes → factor tilt. **Recent peer-reviewed precedent for our exact pipeline pattern.**

**[N40] Bauman et al. (2024) — DRL Goal-Based Investing**
> T. Bauman et al., "Deep reinforcement learning for goal-based investing under regime-switching," NLDL 2024, PMLR vol. 233.

Regime-conditioned position sizing (DRL). **State-of-the-art alternative to fixed Kelly fraction.**

---

## §8. Class Imbalance + SMOTE in Finance (genişletilmiş)

> v1'de He & Garcia, Chawla SMOTE, Elkan, López de Prado meta-labeling, Lipton, Krawczyk vardı. Yeni:

**[N41] Han et al. (2005) — Borderline-SMOTE**
> H. Han et al., "Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning," ICIC 2005, LNCS 3644, pp. 878-887.

Synthesis ONLY on minority decision boundary. **Bizim Buy/Hold frontier'da Sell-bias düzeltmesi için direkt counter-measure.**

**[N42] He et al. (2008) — ADASYN**
> H. He et al., "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," IJCNN 2008, pp. 1322-1328.

K-NN difficulty proportional synthesis. **Buy class bear regime'de sparse — ADASYN extra synthesis there.**

**[N43] Batista, Prati, Monard (2004) — SMOTE+Tomek/ENN Hybrids**
> G. Batista et al., "A study of the behavior of several methods for balancing machine learning training data," *ACM SIGKDD Explor.*, vol. 6, no. 1, pp. 20-29, 2004.

Hybrid sampling + cleaning. **Sell↔Hold confusion = Tomek-link territory.**

**[N44] Lemaître, Nogueira, Aridas (2017) — imbalanced-learn**
> G. Lemaître et al., "Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning," *J. Mach. Learn. Res.*, vol. 18, no. 17, 2017.

`imblearn` library. **Pipeline-compatible resampling INSIDE walk-forward folds — leakage prevention.**

**[N45] Lin et al. (2017) — Focal Loss**
> T.-Y. Lin et al., "Focal loss for dense object detection," ICCV 2017, pp. 2980-2988.

FL(p) = -(1-p)^γ log(p). **MLP softmax cross-entropy alternatif** — minority on hard updates.

**[N46] Aljawazneh (2025) — Cost-Sensitive XGBoost in Finance**
> S. Aljawazneh, "Evaluation of cost-sensitive learning models in forecasting business failure of capital market firms," *Mathematics*, vol. 13, no. 3, art. 368, 2025.

scale_pos_weight tuning > SMOTE alone on AUC/F1. **`class_weight: balanced` pathway için precedent.**

**[N47] Johnson & Khoshgoftaar (2019) — DL Imbalance Survey**
> J. M. Johnson and T. M. Khoshgoftaar, "Survey on deep learning with class imbalance," *J. Big Data*, vol. 6, art. 27, 2019.

Hybrid sampling + cost-sensitive en sağlam. **MLP + XGB combined strategy.**

---

## §9. Walk-Forward CV + Bayesian HP Optimization (genişletilmiş)

> v1'de Bergmeir, López de Prado AFML, Bailey PBO, Bailey & López de Prado DSR, Cerqueira vardı. Yeni:

**[N48] Akiba et al. (2019) — Optuna**
> T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," KDD 2019, pp. 2623-2631.

Define-by-run + dynamic search-space + parallel pruning. **Bizim HP tuning engine'in canonical citation.**

**[N49] Bergstra et al. (2011) — TPE**
> J. Bergstra et al., "Algorithms for hyper-parameter optimization," NIPS 2011, pp. 2546-2554.

Tree-structured Parzen Estimator. p(x|y) modeling. **Optuna default sampler.**

**[N50] Snoek et al. (2012) — Practical BO**
> J. Snoek et al., "Practical Bayesian optimization of machine learning algorithms," NIPS 2012, pp. 2951-2959.

GP-BO foundational reference. **Bayesian HP optim canonical.**

**[N51] Shahriari et al. (2016) — BO Review**
> B. Shahriari et al., "Taking the human out of the loop: A review of Bayesian optimization," *Proc. IEEE*, vol. 104, no. 1, pp. 148-175, 2016.

BO methodology survey. **Methodology section preliminary.**

**[N52] Hutter, Kotthoff, Vanschoren (2019) — AutoML book**
> F. Hutter et al., Eds., *Automated Machine Learning: Methods, Systems, Challenges*, Springer, 2019.

AutoML reference. HPO + meta-learning + pipeline search.

**[N53] Gort et al. (2022) — DRL Crypto Backtest Overfit**
> B. J. D. Gort et al., "Deep reinforcement learning for cryptocurrency trading: Practical approach to address backtest overfitting," ICAIF Workshop 2022. arXiv:2209.05559.

Hypothesis test on DRL crypto agents, 46% overfitting reduction. **Optuna-selected trial filter template.**

**[N54] Pardo (2008) — Walk-Forward Analysis textbook**
> R. Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd ed., Wiley, 2008.

Anchored vs sliding WFA, in-sample/out-of-sample efficiency ratio. **Bizim 12-mo min train, 6-mo step expanding window'un tarihsel kökü.**

---

## §10. Crypto Feature Engineering (yeni)

**[N55] Jaquart, Dann, Weinhardt (2021) — Multi-feature ML for BTC**
> P. Jaquart, D. Dann, C. Weinhardt, "Short-term bitcoin market prediction via machine learning," *J. Finance Data Sci.*, vol. 7, pp. 45-66, 2021.

Technical + on-chain + sentiment features. **Technical features dominant** — bizim feature set seçiminin ampirik dayanağı.

**[N56] Chen, Li, Sun (2020) — Sample Frequency Engineering**
> Z. Chen, C. Li, W. Sun, "Bitcoin price prediction using machine learning: An approach to sample dimension engineering," *J. Comput. Appl. Math.*, vol. 365, art. 112395, 2020.

Daily vs 5-min comparison: tree models win on daily (~65% acc), LSTM on 5-min. **Bizim daily setupımızda tree-based seçimini destekler.**

**[N57] Mudassir et al. (2020) — High-Dim TA**
> M. Mudassir et al., "Time-series forecasting of Bitcoin prices using high-dimensional features," *Neural Comput. Appl.*, 2020.

ANN/SVM/LSTM with TA-Lib indicators. MAPE 1.44 (SVM) → 3.78 (ANN). **Rich feature engineering matters.**

**[N58] Polyzos & Wang (2022) — On-Chain Features**
> E. Polyzos and F. Wang, "Bitcoin price drivers: A machine learning approach feature selection algorithm," *Appl. Econ. Lett.*, vol. 30, no. 19, pp. 2871-2879, 2022.

Hashrate + active addresses + mempool persistent predictive. **Future on-chain extension için academic anchor.**

**[N59] Critien, Gatt, Ellul (2022) — Twitter Sentiment for BTC**
> J. V. Critien, A. Gatt, J. Ellul, "Bitcoin price change and trend prediction through Twitter sentiment and data volume," *Financ. Innov.*, vol. 8, art. 45, 2022.

VADER sentiment + tweet volume. **Tweet volume > polarity** for prediction. Future Stage 2 macro extension.

**[N60] Sebastião & Godinho (2021) — Walk-forward Crypto ML**
> H. Sebastião and P. Godinho, "Forecasting and trading cryptocurrencies with machine learning under changing market conditions," *Financ. Innov.*, vol. 7, art. 3, 2021.

(Aynı zamanda **proposal'daki Nunes 2021 [2]**.) Walk-forward + ensemble tree on BB/RSI/MACD/OBV. **Direct methodology precedent.**

**[N61] Aslam, Mughal, Khalid (2024) — Boruta Feature Selection**
> N. Aslam, K. R. Mughal, U. Khalid, "Bitcoin price direction prediction using on-chain data and feature selection," *Int. J. Inf. Manag. Data Insights*, 2025.

Boruta vs LASSO vs PCA on TA + on-chain. CNN-LSTM Boruta-selected → 82.4% accuracy. **mRMR'ye complementary.**

---

## §11. Risk-On / Risk-Off + Financial Conditions Indices (yeni — V3 Stage 2 semantic destek)

**[N62] Bekaert, Hoerova, Lo Duca (2013) — VIX as Risk Aversion + Uncertainty**
> G. Bekaert, M. Hoerova, M. Lo Duca, "Risk, uncertainty and monetary policy," *J. Monet. Econ.*, vol. 60, no. 7, pp. 771-788, 2013.

VIX² → risk aversion + uncertainty. Endogenous monetary policy interaction. **VIX'in Stage 2 primary feature olduğunu kanıtlar.**

**[N63] Bekaert & Hoerova (2014) — VIX Decomposition**
> G. Bekaert and M. Hoerova, "The VIX, the variance premium and stock market volatility," *J. Econometrics*, vol. 183, no. 2, pp. 181-192, 2014.

(v1'de zaten vardı [37]). Conditional variance vs variance premium. **VIX level → risk-off detection.**

**[N64] Hatzius et al. (2010) — Financial Conditions Index**
> J. Hatzius et al., "Financial Conditions Indexes: A Fresh Look after the Financial Crisis," NBER WP 16150, 2010.

PCA-based FCI from rates + spreads + equity + surveys. **Bizim VIX/SP/DXY/Gold/FFR composite'in low-dimensional FCI olduğunu kanıtlar.**

**[N65] Baur & Lucey (2010) — Gold as Safe Haven**
> D. G. Baur and B. M. Lucey, "Is Gold a Hedge or a Safe Haven?," *Financ. Rev.*, vol. 45, no. 2, pp. 217-229, 2010.

Gold short-lived safe haven (~15 days) extreme down-markets. **Stage 2 Gold returns feature'ının teorik dayanağı.**

**[N66] Caballero & Krishnamurthy (2009) — Flight-to-Safety**
> R. J. Caballero and A. Krishnamurthy, "Global Imbalances and Financial Fragility," *Amer. Econ. Rev.*, vol. 99, no. 2, pp. 584-588, 2009.

Synchronized flight-to-safety dynamics. **DXY+Gold↑ vs equities↓ Risk-Off cluster signature.**

**[N67] Miranda-Agrippino & Rey (2020) — US Monetary Policy + Global Financial Cycle**
> S. Miranda-Agrippino and H. Rey, "U.S. Monetary Policy and the Global Financial Cycle," *Rev. Econ. Stud.*, vol. 87, no. 6, pp. 2754-2776, 2020.

Single global factor in equity/credit/capital flows. US monetary policy primary driver. **FFR change as Stage 2 feature justified.**

**[N68] Liu & Moench (2016) — ML Recession Prediction**
> W. Liu and E. Moench, "What predicts US recessions?," *Int. J. Forecast.*, vol. 32, no. 4, pp. 1138-1150, 2016.

Probit + ROC for NBER recession nowcasting. S&P + credit improve term-spread benchmark. **Equity-return + VIX features for ML regime classification.**

---

## §12. BTC vs ETH Comparative ML (yeni — separate-modeling savunması)

**[N69] Bouteska, Abedin, Hajek, Yuan (2024) — Crypto Ensemble vs DL**
> A. Bouteska et al., "Cryptocurrency price forecasting -- A comparative analysis of ensemble learning and deep learning methods," *Int. Rev. Financ. Anal.*, vol. 92, art. 103055, 2024.

GRU/RNN/LSTM/XGB/LGBM/RF separately on BTC/ETH/XRP/LTC. **Per-coin best-model rankings differ markedly** — pooled training suboptimal.

**[N70] Murray, Rossi, Carraro, Visentin (2023) — Walk-Forward Crypto Benchmark**
> K. Murray et al., "On Forecasting Cryptocurrency Prices: A Comparison of Machine Learning, Deep Learning, and Ensembles," *Forecasting (MDPI)*, vol. 5, no. 1, pp. 196-209, 2023.

Walk-forward on BTC/ETH/LTC. **Per-coin error variance >40%** for same architecture.

**[N71] Korkusuz (2025) — ETH Volatility Transmission**
> B. Korkusuz, "Volatility Transmission in Digital Assets: Ethereum's Rising Influence," *J. Risk Financ. Manag.*, vol. 18, no. 3, art. 111, 2025.

TVP-VAR connectedness: post-Merge **ETH dominant volatility transmitter, not BTC**. Persistent ETH→BTC spillovers. **BTC vs ETH structurally different dynamics — separate models justified.**

**[N72] Sebastião & Godinho (2021)** — (overlap N60)

**[N73] Easley, López de Prado, O'Hara, Zhang (2024) — Crypto Microstructure**
> T. Easley et al., "Microstructure and Market Dynamics in Crypto Markets," *J. Financ. Mark.*, in press, 2026.

Order-book features similar shape across BTC/ETH/LTC/ETC/ENJ but **asset-specific scale + threshold parameters tied to liquidity tier**. Shared feature framework + separate calibration.

**[N74] Ren, Althof, Härdle (2024) — Bagged Tree Buy Signal**
> R. Ren et al., "Forecasting cryptocurrency's buy signal with a bagged tree learning approach to enhance purchase decisions," *Front. Big Data*, vol. 7, art. 1369895, 2024.

(**Proposal'daki Almasarweh & Wadi 2024 [3] aynı.**) Buy/Hold/Sell tree-ensemble + technical + on-chain. **Disjoint per-coin training to handle BTC dominance dynamics**.

**[N75] Easley, Galletta, Harrigan, O'Hara (2024) — Crypto Microstructure Survey**
> D. Easley et al., "Cryptocurrency market microstructure: a systematic literature review," *Ann. Oper. Res.*, vol. 332, no. 1, pp. 1035-1068, 2024.

60+ paper systematic review: **heterogeneous efficiency, liquidity, price-discovery patterns BTC vs altcoins**. **Strongest theoretical foundation for non-pooled design.**

---

## Composite IEEE Reference List (yeni eklenenler, sıralı v2 numarası)

> v1'deki [1]-[40] korunur. Yeni eklemeler [N1]-[N75] aşağıda. Final report'ta tek sıralı liste yapılacak.

(Yer ve uzunluk için yukarıdaki bölümlerin atıf bloklarına bakınız.)

---

## Sentez — Yeni Tasarım Kararları İçin Atıf Haritası

| V3 Tasarım Kararı | Başlıca Atıflar |
|---|---|
| Stage 2 = K-Means + semantic relabeling (GMM yerine) | [N7] Rousseeuw, [N11] Luan & Hamp 2025, [N12] Liu & Härdle 2025, [N13] Procacci & Aste |
| Cluster validity (Elbow + Silhouette + Gap + CH) | [N7]-[N10] tam set |
| Risk-On/Off feature seti (VIX/SP/DXY/Gold/FFR) | [N62]-[N68] complete |
| 4-algorithm comparison (XGB/LGBM/RF/MLP) | [N14]-[N19] GBDT preference |
| Adaptive position sizing {0/50/100%} | [N34]-[N40] risk parity + drawdown control |
| Cascade / soft fusion 3-stage | [N1] Kuznetsov, [N4] Zou, [N5] Jordan & Jacobs |
| McNemar's test + Holm correction | [N20]-[N26] tam set |
| SHAP per-stage | [N27]-[N33] tam set |
| Class imbalance (SMOTE + class_weight + focal) | [N41]-[N47] tam set |
| Optuna walk-forward + Pardo WFA | [N48]-[N54] tam set |
| BTC vs ETH separate modeling | [N69]-[N75] tam set |
| Crypto-specific TA + on-chain extension | [N55]-[N61] tam set |

---

## Şimdiki Durum

- **12/12 Agent tamamlandı**, 87 yeni atıf eklendi.
- v1 LITERATURE_REVIEW.md (37 atıf) korunuyor, bu doc onun **complement'i**.
- **Toplam unique atıf: ~110 (v1 37 + v2 yeni 87, overlap düşülünce ~95-100)**.
- Composite final list rapor yazımına yakın bir tarihte tek dosyada birleştirilir.

### 🚨 KRİTİK BULGU
Proposal'daki 7 atıftan **5'i hatalı author/venue/DOI**:
1. "Montañez 2023" → Rizzuti, Parente, Trerotola
2. "Nunes 2021" → Sebastião & Godinho
3. "Almasarweh & Wadi 2024" → Alsini et al.
4. "Chowdhury 2024" → Jabbar & Jalil
5. "Ahmad 2024 PLOS ONE" → Ahmad 2024 Heliyon (PLOS ONE DOI medical paper!)

Bunlar final report'ta düzeltilmeli, akademik dürüstlük kritik.

---

_Bu doküman V3 restart için sıfırdan tasarım kararlarının akademik temeli. Her tasarım kararı 2-5 atıfla destekli._
