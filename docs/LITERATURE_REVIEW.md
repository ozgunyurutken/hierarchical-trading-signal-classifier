# Literatür Taraması — Hiyerarşik ML Tabanlı Kripto Sinyal Sınıflandırması

**Proje:** BBL514E Pattern Recognition Term Project
**Hazırlanma tarihi:** 2026-05-08

> Bu doküman, projede karşılaşılan 8 ana methodology problemi için literatür taraması içerir. Her bölüm: (a) problem ifadesi, (b) 3-6 anahtar atıf, (c) bizim çözümümüzle ilişkisi. IEEE-style atıf formatı kullanılır.

---

## Özet Eşleme Tablosu — Soruna Karşı Atıflar

| Proje sorunu | Anahtar atıflar |
|---|---|
| 3-stage soft fusion mimarisi savunması | Wolpert (1992) [1], Ting & Witten (1999) [2], Silla & Freitas (2011) [3] |
| Feature redundancy / B2 ablation bulgusu | Sagi & Rokach (2018) [4], Dong et al. (2020) [5], Peng et al. (2005) [22] |
| Stage 2 GMM stickiness / OOD problemi | Hamilton (1989) [6], Tu (2010) [9], Nystrup et al. (2021) [10] |
| Walk-forward CV methodology | Bergmeir et al. (2018) [11], López de Prado (2018) [12], Cerqueira et al. (2020) [15] |
| Backtest overfitting + Sharpe inflation | Bailey et al. (2014) [13], Bailey & López de Prado (2014) [14] |
| 3-class Buy/Sell/Hold sınıflandırması | Krauss et al. (2017) [16], Sezer et al. (2020) [17], Stawarz & Stasiak (2025) [19] |
| Crypto ML benchmark çalışmaları | Jaquart et al. (2024) [20], Goutte et al. (2023) [21] |
| Adaptive position sizing (V3 plan) | Markowitz (1952) [25], Moreira & Muir (2017) [26], MacLean et al. (2010) [29] |
| Regime-switching asset allocation | Ang & Bekaert (2002) [7], Guidolin & Timmermann (2007) [8] |
| CM Sell-bias / class imbalance | He & Garcia (2009) [30], Elkan (2001) [32], López de Prado meta-labeling [33] |
| Sharpe ratio değerlendirmesi | Sharpe (1994) [34], Lo (2002) [35] |

---

## 1. Hiyerarşik Sınıflandırma & Stacking ile Soft Posterior Fusion

**Bizim problem:** 3-aşamalı pipeline'ımız (Stage 1 LDA trend → Stage 2 GMM macro regime → Stage 3 XGBoost signal) **soft fusion** kullanıyor; her ara aşama olasılık vektörü çıktısı veriyor. Ablation sonucu şu sürpriz: **flat XGBoost (29 feature)** = Sharpe 1.43, **hiyerarşik full A4** = Sharpe 1.35 (v1). 5 redundant trend feature çıkarıldıktan sonra (B2 subset) hiyerarşi gerçek üstünlük gösterdi (A4 1.58 > A1 1.17).

### [1] Wolpert (1992) — Stacked Generalization (canonical)
> D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241-259, 1992. DOI: 10.1016/S0893-6080(05)80023-1.

Wolpert stacking'in kuralını koyar: level-0 öğrencilerin *out-of-sample* tahminlerini, level-1 meta-öğrencinin girdisi olarak kullan. Bizim 3-stage tasarımı doğrudan bu şemanın bir uygulaması.

### [2] Ting & Witten (1999) — Probability outputs as meta-features
> K. M. Ting and I. H. Witten, "Issues in stacked generalization," *J. Artif. Intell. Res.*, vol. 10, pp. 271-289, 1999. DOI: 10.1613/jair.594.

Empirik bulgu: meta-öğrenci **olasılık vektörlerini** girdi olarak kullandığında en iyi sonuç. Bizim soft posterior tasarımının (hard label yerine ℝ³ vektör) referansı.

### [3] Silla & Freitas (2011) — Hierarchical classification taxonomy
> C. N. Silla and A. A. Freitas, "A survey of hierarchical classification across different application domains," *Data Min. Knowl. Discov.*, vol. 22, pp. 31-72, 2011. DOI: 10.1007/s10618-010-0175-9.

Hiyerarşik sınıflandırmanın kapsamlı taksonomisi (flat / local-per-node / local-per-level / global). Cascade error-propagation problemi vurgulanır — bizim "Stage 1 OOF redundant tech feature ile çakışırsa cascade etkisi kaybolur" bulgumuzun teorik çerçevesi.

### [4] Sagi & Rokach (2018) — Modern ensemble survey
> O. Sagi and L. Rokach, "Ensemble learning: A survey," *WIREs Data Min. Knowl. Discov.*, vol. 8, e1249, 2018. DOI: 10.1002/widm.1249.

**Bizim B2 bulgusunun altın atıf:** "stacking effectiveness is bounded by **diversity and non-redundancy** of base-learner outputs." Bizim "29 feat'ten 24'e indirip A4 1.35 → 1.58'e çıkarttık" sonucumuzun teorik açıklaması.

### [5] Dong, Yu, Cao, Shi & Ma (2020) — Stacking in deep era
> X. Dong, Z. Yu, W. Cao, Y. Shi, and Q. Ma, "A survey on ensemble learning," *Front. Comput. Sci.*, vol. 14, pp. 241-258, 2020. DOI: 10.1007/s11704-019-8208-z.

Redundant base-learner pruning prensipleri. Bizim B1/B2/B3 ablation'ının methodology referansı.

---

## 2. Market Regime Detection — Rule-Based vs Unsupervised (GMM, HMM)

**Bizim problem:** Stage 2 GMM 11 makro feature üzerinde 3-cluster fit. Test setinde (2024-2025) **P(Stress) = 0.964 ortalaması, 12 ay'ın 11'i %100**. VIX gerçeği: 17.22 (baseline 15.92'den +1.3 fark). OOD sample (FFR 5.5% vs train 0-2.5%) → cluster overconfident. V3'te rule-based regime detection ile değiştirilecek.

### [6] Hamilton (1989) — Foundational HMM for regimes
> J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," *Econometrica*, vol. 57, no. 2, pp. 357-384, 1989. DOI: 10.2307/1912559.

Markov-switching framework. Bizim GMM problemine alternatif: HMM transition probability ile **temporal persistence** dayatır, OOD nokta en yakın cluster'a yapışmaz.

### [7] Ang & Bekaert (2002) — Regime shifts in asset allocation
> A. Ang and G. Bekaert, "International asset allocation with regime shifts," *Rev. Financ. Stud.*, vol. 15, no. 4, pp. 1137-1187, 2002. DOI: 10.1093/rfs/15.4.1137.

Bear regime'de korelasyonlar artar — pasif Buy & Hold'un dezavantajı buradan. Bizim adaptive position sizing tasarımının teorik motivasyonu.

### [8] Guidolin & Timmermann (2007) — Multivariate regime switching
> M. Guidolin and A. Timmermann, "Asset allocation under multivariate regime switching," *J. Econ. Dyn. Control*, vol. 31, no. 11, pp. 3503-3544, 2007. DOI: 10.1016/j.jedc.2006.12.004.

Bayesian regime kararsızlığı: tail event'lerde posterior tek cluster'a çöküyor. Bizim 2025 boyunca P(Stress)=0.99 pathology'sinin literatürdeki açıklaması.

### [9] Tu (2010) — Bayesian regime switching with uncertainty
> J. Tu, "Is regime switching in stock returns important in portfolio decisions?," *Manag. Sci.*, vol. 56, no. 7, pp. 1198-1215, 2010. DOI: 10.1287/mnsc.1100.1181.

Klasik EM-fit GMM parametre belirsizliğini ihmal eder → yapay overconfidence. Bayesian shrinkage çözümü; biz bunun yerine rule-based threshold ile uncertainty bandını explicit yapıyoruz.

### [10] Nystrup, Kolm & Lindström (2021) — Statistical jump models
> P. Nystrup, P. N. Kolm, and E. Lindström, "Feature selection in jump models," *Expert Syst. Appl.*, vol. 184, art. 115558, 2021. DOI: 10.1016/j.eswa.2021.115558.

GMM'in modern alternatif: jump-penalty estimator persistence dayatır. **2018+ literatürün en çağdaş regime detection methodu**, ileride extension için referans.

---

## 3. Walk-Forward CV — Financial Time Series için Methodology

**Bizim problem:** 5-fold walk-forward expanding-window CV (Optuna), shuffle yok, train fold strict olarak validation fold'dan önce. Stage 1/2/3 OOF prediction'ları bu şemada üretildi. Test split %15 chronological (462 days BTC).

### [11] Bergmeir, Hyndman & Koo (2018) — Validity of CV for AR prediction
> C. Bergmeir, R. J. Hyndman, and B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," *Comput. Stat. Data Anal.*, vol. 120, pp. 70-83, 2018. DOI: 10.1016/j.csda.2017.11.003.

Saf AR modeller için K-fold geçerli, ama serial correlation varsa **walk-forward** zorunlu. Bizim crypto returns volatility clustering içerdiği için walk-forward'un teorik gerekçesi.

### [12] López de Prado (2018) — Advances in Financial Machine Learning
> M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ: Wiley, 2018, ch. 7.

**Purged K-fold** + **embargo** technique label leakage'a karşı. Bizim forward-return labels H bar boyunca uzandığı için bu teknik kritik. (Ayrıca Ch. 3 meta-labeling — bizim hiyerarşik tasarıma yakın.)

### [13] Bailey, Borwein, López de Prado & Zhu (2014) — Backtest overfitting
> D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "Pseudo-mathematics and financial charlatanism," *Notices Amer. Math. Soc.*, vol. 61, no. 5, pp. 458-471, 2014.

Yeterince hyperparameter trial ile *herhangi* bir backtest ihtişamlı görünebilir. Bizim Optuna 8-trial × 4-config × 7-model setupımıza karşı conservative report etme zorunluluğunun atfı.

### [14] Bailey & López de Prado (2014) — Deflated Sharpe Ratio
> D. H. Bailey and M. López de Prado, "The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality," *J. Portf. Manag.*, vol. 40, no. 5, pp. 94-107, 2014. DOI: 10.3905/jpm.2014.40.5.094.

Selection bias için Sharpe deflation. Bizim "Sharpe 1.68 vs B&H 0.75" iddiamızı istatistiksel olarak savunmak için DSR hesaplanabilir.

### [15] Cerqueira, Torgo & Mozetič (2020) — Empirical CV study
> V. Cerqueira, L. Torgo, and I. Mozetič, "Evaluating time series forecasting models: An empirical study on performance estimation methods," *Mach. Learn.*, vol. 109, no. 11, pp. 1997-2028, 2020. DOI: 10.1007/s10994-020-05910-7.

62 gerçek zaman serisinde benchmark: non-stationary veride **walk-forward en güvenilir**. Crypto highly non-stationary → bizim seçimimizin empirical destekçisi.

---

## 4. Trading Signal Classification + Cryptocurrency ML (Post-2017)

**Bizim problem:** 3-class signal prediction (Buy/Sell/Hold) BTC daily üzerinde, label = 5-day forward return ± 0.5×rolling_std(20). Best: ZZ-MLP Sharpe 1.68 / Return +89.5% / Win 72.2%.

### [16] Krauss, Do & Huck (2017) — Foundational ensemble for stat arb
> C. Krauss, X. A. Do, and N. Huck, "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500," *Eur. J. Oper. Res.*, vol. 259, no. 2, pp. 689-702, 2017. DOI: 10.1016/j.ejor.2016.10.031.

S&P 500 günlük üzerinde DNN/GBT/RAF ensemble — Sharpe 1.81. Bizim 4-classifier setupımızın direct precursor'ı; ensemble'ın tek-model'i geçtiğine dair empirical referans.

### [17] Sezer, Gudelek & Ozbayoglu (2020) — DL for financial time series survey
> O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019," *Appl. Soft Comput.*, vol. 90, art. 106181, 2020. DOI: 10.1016/j.asoc.2020.106181.

**Hiyerarşik multi-stage tasarımının literatürde nadir olduğu bulgusu** — bizim katkı boşluğunu işaret eden survey.

### [18] Henrique, Sobreiro & Kimura (2019) — ML for financial market prediction
> B. M. Henrique, V. A. Sobreiro, and H. Kimura, "Literature review: Machine learning techniques applied to financial market prediction," *Expert Syst. Appl.*, vol. 124, pp. 226-251, 2019. DOI: 10.1016/j.eswa.2019.01.012.

2,173 unique predictor variable katalogu (technical/macro/fundamental). Bizim feature engineering kategorilerinin geneolojisi.

### [19] Stawarz & Stasiak (2025) — **Direct competitor: BTC Buy/Sell/Hold**
> M. Stawarz and M. Stasiak, "Determining multi-class trading signals for Bitcoin: A comparative study of XGBoost, LightGBM, and Random Forest," in *Proc. ISD2025*, 2025. DOI: 10.62036/ISD.2025.42.

**En yakın paralel çalışma.** ±%1 fixed threshold (bizim ±0.5×rolling_std adaptive'inden zayıf), flat single-stage architecture (bizim hiyerarşik'inden basit). Doğrudan baseline.

### [20] Jaquart, Köpke & Weinhardt (2024) — DL for BTC + trading strategies
> P. Jaquart, S. Köpke, and C. Weinhardt, "Deep learning for Bitcoin price direction prediction: Models and trading strategies empirically compared," *Financial Innov.*, vol. 10, art. 1, 2024. DOI: 10.1186/s40854-024-00643-1.

CNN-LSTM/TCN/MLP/ARIMA empirical comparison. **MLP underperformance bulguları** — bizim ZZ-MLP Sharpe 1.68 başarısının non-trivial olduğunu vurgulamak için.

### [21] Goutte, Le, Liu & von Mettenheim (2023) — Crypto ensemble vs DL
> S. Goutte, H.-V. Le, F. Liu, and H.-J. von Mettenheim, "Cryptocurrency price forecasting — A comparative analysis of ensemble learning and deep learning methods," *Int. Rev. Financ. Anal.*, vol. 89, art. 102787, 2023. DOI: 10.1016/j.irfa.2023.102787.

XGBoost / RF tree ensemble crypto'da DL'i **yenebilir** bulgusu. Bizim XGBoost dominance'ımızın empirical desteği.

---

## 5. Feature Engineering for Financial ML

**Bizim problem:** V3 plan 32 feature setup — multi-period momentum (1/5/20/60/120d), multi-window vol (5/20/60d), drawdown + recovery, VIX dynamics, macro spreads, monthly FRED. 5 long-trend feature çıkarma kararının (B2) literatür referansı.

### [22] Peng, Long & Ding (2005) — mRMR Feature Selection (canonical)
> H. Peng, F. Long, and C. Ding, "Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 27, no. 8, pp. 1226-1238, 2005. DOI: 10.1109/TPAMI.2005.159.

Mutual Information ile redundancy minimization. **Bizim B2 5-feature drop kararının matematiksel zemini** — Stage 1 OOF posterior ile collinear feature'ları çıkar.

### [23] Moskowitz, Ooi & Pedersen (2012) — Time series momentum
> T. J. Moskowitz, Y. H. Ooi, and L. H. Pedersen, "Time series momentum," *J. Financ. Econ.*, vol. 104, no. 2, pp. 228-250, 2012. DOI: 10.1016/j.jfineco.2011.11.003.

1-12 ay return predictability (momentum), uzun-horizon reversal. Bizim multi-period log returns (5/20/60/120d) tasarımının ampirik temeli.

### [24] Whaley (2000) — VIX as investor fear gauge
> R. E. Whaley, "The investor fear gauge," *J. Portf. Manag.*, vol. 26, no. 3, pp. 12-17, 2000.

VIX'in temel atfı. Bizim VIX_level / VIX_change_5d / VIX_zscore_60 feature'larının dayanağı.

(Bekaert & Hoerova 2014 — VIX decomposition into variance & variance premium — ileri okuma için, J. Econometrics.)

---

## 6. Adaptive Position Sizing & Dynamic Allocation (V3 Plan)

**Bizim problem:** V3 plan'da binary long/no-position yerine fractional weight {High Risk: 0%, Medium: 50%, Low: 100%}. Önceki başarılı poster (Sharpe 1.41) Dynamic MVO with regime λ kullanmıştı.

### [25] Markowitz (1952) — Mean-Variance Optimization (foundational)
> H. Markowitz, "Portfolio Selection," *J. Finance*, vol. 7, no. 1, pp. 77-91, 1952. DOI: 10.1111/j.1540-6261.1952.tb01525.x.

MPT bedrock. Bizim regime-conditional weight'in efficient frontier üzerindeki noktalar olarak yorumlanması.

### [26] Moreira & Muir (2017) — Volatility-managed portfolios
> A. Moreira and T. Muir, "Volatility-managed portfolios," *J. Finance*, vol. 72, no. 4, pp. 1611-1644, 2017. DOI: 10.1111/jofi.12513.

**V3 plan için en kritik atıf.** Realized vol'a ters orantılı ölçeklendirme → büyük alpha + Sharpe iyileşmesi. Bizim regime-driven {0/50/100%} sizing'in kontinü σ-target ruler'ın ayrıklaştırılmış versiyonu.

### [27] Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert (2018) — Empirical vol-targeting
> C. R. Harvey et al., "The Impact of Volatility Targeting," *J. Portf. Manag.*, vol. 45, no. 1, pp. 14-33, 2018.

60 asset, 1926-2017. Risk asset'lerde Sharpe artışı + tail event severity azalması. Bizim adaptive sizing'in empirical desteği.

### [28] Ang & Bekaert (2002) — Regime-switching asset allocation (overlap with §2)
> Cf. [7]. Regime-aware allocation'ın 2-3 cents/dollar value'sunu kanıtlar.

### [29] MacLean, Thorp & Ziemba (2010) — Fractional Kelly
> L. C. MacLean, E. O. Thorp, and W. T. Ziemba, "Good and bad properties of the Kelly criterion," *Quant. Finance*, vol. 10, no. 7, pp. 681-687, 2010.

Half-Kelly volatilite ve drawdown'u proportional from'dan fazla azaltır. Bizim 50% Medium-Risk weight'in teorik altyapısı.

---

## 7. Class Imbalance in Financial ML — CM Sell-Bias Problemi

**Bizim problem:** XGBoost (B2 + ZZ-MLP) test setinde **%80+ Sell tahmin** ediyor, oysa actual dağılım Sell %34 / Hold %25 / Buy %40. Train set 2014-2024 birden fazla bear period içeriyor → Sell-heavy prior. Long-only stratejide "Sell prediction = no position" → Buy fırsatları kaçırma riski.

### [30] He & Garcia (2009) — Foundational imbalanced learning survey
> H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263-1284, 2009. DOI: 10.1109/TKDE.2008.239.

Sampling (SMOTE, ADASYN), cost-sensitive, threshold moving methodolojisi. Bizim CM bias'a karşı potansiyel post-hoc düzeltmelerin canonical atfı.

### [31] Chawla, Bowyer, Hall & Kegelmeyer (2002) — SMOTE
> N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321-357, 2002.

Sentetik minority interpolation. Caveat: López de Prado finansal time-series için kullanılmaması gerektiğini söylüyor. Bizim **deliberately rejected** çözümümüz.

### [32] Elkan (2001) — Foundations of Cost-Sensitive Learning
> C. Elkan, "The foundations of cost-sensitive learning," in *Proc. IJCAI*, 2001, pp. 973-978.

Optimal Bayes threshold p* = c(0,1)/(c(0,1)+c(1,0)). **Resampling ve threshold-shifting matematiksel olarak eşdeğer** — bizim threshold tuning post-hoc çözümünün theoretical basis'i.

### [33] López de Prado (2018, Ch. 3) — Meta-Labeling
> Cf. [12], Ch. 3.

İki-model architecture: primary high-recall + secondary meta-model sizing'i belirler. **Bizim hiyerarşik soft fusion ve V3 adaptive sizing tasarımının doğrudan parallelı.**

### [34] Lipton, Elkan & Naryanaswamy (2014) — Optimal F1 thresholding
> Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in *ECML PKDD 2014*, LNCS 8725, pp. 225-239.

Optimal F1 threshold s ≥ F*/2. Bizim post-hoc threshold sweep çözümümüzün algoritmic referansı.

### [35] Krawczyk (2016) — Modern imbalance learning challenges
> B. Krawczyk, "Learning from imbalanced data: open challenges and future directions," *Prog. Artif. Intell.*, vol. 5, no. 4, pp. 221-232, 2016.

He & Garcia 2009'un modern uzantısı. Future work bölümünde framing için.

---

## 8. Backtesting Methodology & Risk-Adjusted Metrics

**Bizim problem:** 7 metric (Total Return, Sharpe, MaxDD, Win Rate, Trades, Test Acc, F1, MCC). Best result: ZZ-MLP Sharpe 1.68 / Return +89.5% / MaxDD -11.9% (462g BTC test). B&H: 0.75 / +47.6% / -32.1%.

### [36] Sharpe (1994) — The Sharpe Ratio (revisited)
> W. F. Sharpe, "The Sharpe Ratio," *J. Portf. Manag.*, vol. 21, no. 1, pp. 49-58, 1994.

Canonical definition. Bizim Sharpe 1.68 vs 0.75 karşılaştırmasının dayanağı.

### [37] Lo (2002) — The Statistics of Sharpe Ratios
> A. W. Lo, "The statistics of Sharpe ratios," *Financ. Anal. J.*, vol. 58, no. 4, pp. 36-52, 2002. DOI: 10.2469/faj.v58.n4.2453.

Naive √T annualization serial correlation varsa **65%'e kadar Sharpe'ı şişiriyor**. Bizim daily crypto returns'da autocorrelation/vol clustering var → confidence interval rapor etmemiz gerek.

### [38] Magdon-Ismail & Atiya (2004) — Maximum Drawdown
> M. Magdon-Ismail and A. F. Atiya, "Maximum drawdown," *Risk*, vol. 17, no. 10, pp. 99-102, 2004.

MDD analytical scaling laws + Calmar ratio (Return/MDD). Bizim MaxDD -%11.9 vs B&H -%32.1 karşılaştırmasının methodology dayanağı.

### [39, 40] Bailey & López de Prado works (overlap with §3)
Cf. [13], [14] — backtest overfitting + Deflated Sharpe Ratio.

---

## Composite IEEE Reference List (40 unique, sıralı)

```
[1]  D. H. Wolpert, "Stacked generalization," Neural Netw., vol. 5, no. 2, pp. 241-259, 1992.
[2]  K. M. Ting and I. H. Witten, "Issues in stacked generalization," J. Artif. Intell. Res., vol. 10, pp. 271-289, 1999.
[3]  C. N. Silla and A. A. Freitas, "A survey of hierarchical classification across different application domains," Data Min. Knowl. Discov., vol. 22, no. 1-2, pp. 31-72, 2011.
[4]  O. Sagi and L. Rokach, "Ensemble learning: A survey," WIREs Data Min. Knowl. Discov., vol. 8, no. 4, p. e1249, 2018.
[5]  X. Dong et al., "A survey on ensemble learning," Front. Comput. Sci., vol. 14, no. 2, pp. 241-258, 2020.
[6]  J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," Econometrica, vol. 57, no. 2, pp. 357-384, 1989.
[7]  A. Ang and G. Bekaert, "International asset allocation with regime shifts," Rev. Financ. Stud., vol. 15, no. 4, pp. 1137-1187, 2002.
[8]  M. Guidolin and A. Timmermann, "Asset allocation under multivariate regime switching," J. Econ. Dyn. Control, vol. 31, no. 11, pp. 3503-3544, 2007.
[9]  J. Tu, "Is regime switching in stock returns important in portfolio decisions?," Manage. Sci., vol. 56, no. 7, pp. 1198-1215, 2010.
[10] P. Nystrup, P. N. Kolm, and E. Lindström, "Feature selection in jump models," Expert Syst. Appl., vol. 184, p. 115558, 2021.
[11] C. Bergmeir, R. J. Hyndman, and B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," Comput. Stat. Data Anal., vol. 120, pp. 70-83, 2018.
[12] M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ: Wiley, 2018.
[13] D. H. Bailey, J. M. Borwein, M. López de Prado, and Q. J. Zhu, "Pseudo-mathematics and financial charlatanism," Notices Amer. Math. Soc., vol. 61, no. 5, pp. 458-471, 2014.
[14] D. H. Bailey and M. López de Prado, "The deflated Sharpe ratio," J. Portf. Manag., vol. 40, no. 5, pp. 94-107, 2014.
[15] V. Cerqueira, L. Torgo, and I. Mozetič, "Evaluating time series forecasting models," Mach. Learn., vol. 109, no. 11, pp. 1997-2028, 2020.
[16] C. Krauss, X. A. Do, and N. Huck, "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500," Eur. J. Oper. Res., vol. 259, no. 2, pp. 689-702, 2017.
[17] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning," Appl. Soft Comput., vol. 90, p. 106181, 2020.
[18] B. M. Henrique, V. A. Sobreiro, and H. Kimura, "Literature review: Machine learning techniques applied to financial market prediction," Expert Syst. Appl., vol. 124, pp. 226-251, 2019.
[19] M. Stawarz and M. Stasiak, "Determining multi-class trading signals for Bitcoin," in Proc. ISD2025, 2025.
[20] P. Jaquart, S. Köpke, and C. Weinhardt, "Deep learning for Bitcoin price direction prediction," Financial Innov., vol. 10, art. 1, 2024.
[21] S. Goutte et al., "Cryptocurrency price forecasting — A comparative analysis of ensemble learning and deep learning methods," Int. Rev. Financ. Anal., vol. 89, art. 102787, 2023.
[22] H. Peng, F. Long, and C. Ding, "Feature selection based on mutual information," IEEE Trans. Pattern Anal. Mach. Intell., vol. 27, no. 8, pp. 1226-1238, 2005.
[23] T. J. Moskowitz, Y. H. Ooi, and L. H. Pedersen, "Time series momentum," J. Financ. Econ., vol. 104, no. 2, pp. 228-250, 2012.
[24] R. E. Whaley, "The investor fear gauge," J. Portf. Manag., vol. 26, no. 3, pp. 12-17, 2000.
[25] H. Markowitz, "Portfolio Selection," J. Finance, vol. 7, no. 1, pp. 77-91, 1952.
[26] A. Moreira and T. Muir, "Volatility-managed portfolios," J. Finance, vol. 72, no. 4, pp. 1611-1644, 2017.
[27] C. R. Harvey et al., "The Impact of Volatility Targeting," J. Portf. Manag., vol. 45, no. 1, pp. 14-33, 2018.
[28] L. C. MacLean, E. O. Thorp, and W. T. Ziemba, "Good and bad properties of the Kelly criterion," Quant. Finance, vol. 10, no. 7, pp. 681-687, 2010.
[29] H. He and E. A. Garcia, "Learning from imbalanced data," IEEE Trans. Knowl. Data Eng., vol. 21, no. 9, pp. 1263-1284, 2009.
[30] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321-357, 2002.
[31] C. Elkan, "The foundations of cost-sensitive learning," in Proc. IJCAI, 2001, pp. 973-978.
[32] Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in Proc. ECML PKDD, 2014, LNCS 8725, pp. 225-239.
[33] B. Krawczyk, "Learning from imbalanced data: open challenges and future directions," Prog. Artif. Intell., vol. 5, no. 4, pp. 221-232, 2016.
[34] W. F. Sharpe, "The Sharpe Ratio," J. Portf. Manag., vol. 21, no. 1, pp. 49-58, 1994.
[35] A. W. Lo, "The statistics of Sharpe ratios," Financ. Anal. J., vol. 58, no. 4, pp. 36-52, 2002.
[36] M. Magdon-Ismail and A. F. Atiya, "Maximum drawdown," Risk, vol. 17, no. 10, pp. 99-102, 2004.
[37] G. Bekaert and M. Hoerova, "The VIX, the variance premium and stock market volatility," J. Econometrics, vol. 183, no. 2, pp. 181-192, 2014.
```

---

## Sentez — Rapor Argümanlama Akışı

Bu literatür haritası, raporun **methodology savunmasının ana çatısını** kuruyor:

1. **Hiyerarşik soft fusion tasarımı** — Wolpert [1] + Ting & Witten [2] foundational; Sagi & Rokach [4] + Dong et al. [5] redundancy bulgumuzun teorik açıklaması.
2. **GMM stickiness limitation** — Hamilton [6] HMM ile alternatif, Tu [9] Bayesian uncertainty, Nystrup [10] modern jump models. V3'te rule-based regime detection için Ang & Bekaert [7].
3. **Walk-forward CV** — Bergmeir et al. [11] + López de Prado [12] + Cerqueira et al. [15] hepsi non-stationary financial data için walk-forward'u zorunlu kılıyor.
4. **Backtest overfitting savunması** — Bailey et al. [13] + DSR [14]: bizim Optuna multi-trial setupımız selection bias'a karşı dürüst rapor edilecek.
5. **Crypto ML positioning** — Stawarz & Stasiak [19] direct competitor (flat single-stage); biz hiyerarşik + adaptive ile farklılaşıyoruz. Jaquart et al. [20] MLP underperformance'ı, bizim ZZ-MLP başarısının non-trivial olduğunu gösteriyor.
6. **Feature engineering** — Peng et al. [22] mRMR teorik temeli; Moskowitz et al. [23] momentum horizon seçimi; Whaley [24] VIX rationale.
7. **Adaptive position sizing** — Moreira & Muir [26] + Harvey et al. [27] empirical vol-targeting; Markowitz [25] + MacLean et al. [28] fractional Kelly. V3 plan'ın akademik temeli.
8. **Class imbalance limitation** — Elkan [31] + Lipton et al. [32] threshold tuning teorisi; López de Prado meta-labeling [33] yapısal çözüm.

**Toplam: 37 unique atıf** (5 overlap), IEEE-style hazır.

---

_Bu doküman Research mode (8 paralel Agent) ile üretildi. Web search + manual synthesis. 2026-05-08._
