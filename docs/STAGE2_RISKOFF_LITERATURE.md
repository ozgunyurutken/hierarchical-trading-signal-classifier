# Stage 2 — Risk-Off Cluster Over-Specialization Problem & Literature

**Date:** 2026-05-09
**Trigger:** Decision Gate 2 review revealed Risk-Off cluster captures only COVID 2020 (180g, 2020-04-08 → 2020-12-15), missing 2008 GFC, 2002 dot-com, 2011 Eurozone, 2018 Şubat, 2022 Fed hike.
**Cause:** UNRATE_change_180d centroid +1.69 dominant; COVID UNRATE +311% change is an outlier extreme event.

---

## Literature Review (7 References, all NEW for our bibliography)

### [LR1] Witten & Tibshirani (2010) — Sparse K-Means / Feature Weighting
> D. M. Witten and R. Tibshirani, "A framework for feature selection in clustering," *J. Amer. Statist. Assoc.*, vol. 105, no. 490, pp. 713-726, 2010. DOI: 10.1198/jasa.2010.tm09415.

L1-penalized weighted between-cluster sum of squares. Otomatik per-feature weight öğrenir, dominant variable down-weight olur. **Direkt UNRATE dominance fix.**

### [LR2] Wagstaff, Cardie, Rogers & Schroedl (2001) — Constrained K-Means
> K. Wagstaff et al., "Constrained K-means clustering with background knowledge," ICML 2001, pp. 577-584.

Must-link / cannot-link pairwise constraints. Crisis date priors ile (Reinhart-Rogoff dates) Risk-Off cluster'ına force-assign edebiliriz.

### [LR3] Holló, Kremer & Lo Duca (2012) — CISS Composite Stress Index
> D. Holló, M. Kremer, M. Lo Duca, "CISS - A composite indicator of systemic stress in the financial system," ECB WP 1426, 2012. SSRN: 2018792.

15 stress measure × 5 segment, time-varying cross-correlation weighted. **Multi-segment co-movement** crisis severity'yi yakalar — UNRATE'in tek-feature dominance'ını kırar.

### [LR4] Liao, Hu, Hou & Wei (2025) — Equilibrium K-Means
> Y. Liao et al., "Semi-supervised equilibrium K-means for imbalanced data clustering," *Knowl.-Based Syst.*, vol. 305, art. 113428, 2025.

Centroid repulsion term. K-Means'in "uniform effect" pathology'sini açıkça kırar — rare cluster geometric identity korur.

### [LR5] Horvath, Issa & Muguruza (2023) — Wasserstein K-Means
> B. Horvath, Z. Issa, A. Muguruza, "Clustering market regimes using the Wasserstein distance," *J. Comput. Finance*, vol. 27, no. 1, 2023. arXiv:2110.11848.

Wasserstein barycenter centroidler, distributional shape match. **2010 Eurozone, 2011 US downgrade, 2015 China crash distinct cluster** — bizim case'de tam olarak istediğimiz.

### [LR6] Li, Geng, Wang, Yuan & Yang (2024) — GMM with Rare Events
> X. Li et al., "Gaussian mixture model with rare events," *J. Mach. Learn. Res.*, vol. 25, no. 215, 2024. arXiv:2405.16859.

Mixed-EM (MEM) algorithm, rare component (<5%) için partial labels ile EM convergence accelerate. Soft posterior: COVID 0.95, GFC 0.7, normal 0.0 sürekli değer.

### [LR7] Douzas, Bação & Last (2018) — K-Means SMOTE
> G. Douzas, F. Bação, F. Last, "Improving imbalanced learning through a heuristic oversampling method based on K-means and SMOTE," *Inf. Sci.*, vol. 465, pp. 1-20, 2018. DOI: 10.1016/j.ins.2018.06.056.

Cluster + SMOTE oversample. Stage 3 downstream training'de Risk-Off rare class up-sample edilir.

---

## V5 Pragmatic Decision Matrix

| Seçenek | Effort | Proposal Strict | Pro | Con |
|---|---|:---:|---|---|
| **A. UNRATE çıkar** (8 feat) | 5 dk | ✅ | En hızlı, root cause kes | COVID extreme'i Risk-Off'tan kaybedebilir |
| **B. Sparse K-Means** | 2-3h | ✅ | Akademik en güçlü, otomatik weighting | Custom impl gerekli |
| **C. CISS composite** | 1-2h | ⚠ feature change | ECB-level credibility, multi-crisis | Feature semantic değişir |
| **D. GMM + soft post.** | 1h | ⚠ K-Means sapma | Soft prob Stage 3'e doğal akar | "K-Means hybrid" proposal sözü kırılır |
| **E. Constrained K-Means** | 2-3h | ✅ | Domain knowledge injection | Crisis date seçimi subjektif |
| **F. Mevcut kabul** | 0 | ✅ | "extreme stress" tanımı, paper'da gap savun | Risk-Off practical değer azalır |

---

## Recommended Path

**Phase 1 (Quick test, 5 dk):** Option A — UNRATE çıkar, 8-feature K-Means deneyim. Risk-Off dağılımı nasıl değişiyor görelim.

**Phase 2 (eğer A yetersizse, 1h):** Option D — GMM + soft posterior. Proposal'dan minimum sapma, soft fusion proposal §3 zaten uyumlu (one-hot yerine probability vector → ama Ting & Witten 1999 [N2] soft fusion stack'in canonical referansı).

**Phase 3 (akademik en sağlam, 2-3h):** Option B — Sparse K-Means with L1 feature weighting. Final paper'da methodology contribution olarak savun.

**Avoid:** Option F (mevcut kabul) — "extreme stress detector" savunma akademik açıdan zayıf, jurinin kabul etmeyeceği muhtemel.

---

_Bu doküman Decision Gate 2 + Risk-Off karar süreci için literatür temelidir. LITERATURE_REVIEW_v2.md'e LR1-LR7 numaraları ile entegre edilecek._
