# CLAUDE.md - Project Instructions for AI Assistants

## Project Overview
**BBL514E Pattern Recognition Term Project**
Three-Stage Hierarchical ML Framework for Cryptocurrency Trading Signal Classification

## Team
- Ozgun Can Yurutken (707251003) — Model Development & Backend
- Cagatay Bilgin (504251013) — Data Pipeline & Frontend

## Key Rules

### Workflow Rules
- **User is the main decision-maker.** Never proceed to the next phase without explicit user approval.
- **Checkpoint-based development.** Complete each phase thoroughly, review results, get approval, then move on.
- **Notebooks for user interaction.** Data collection, label generation, and model training phases MUST produce notebook outputs for the user to manually review and approve.
- **Show results first, then wait.** After completing a step, present a brief summary/interpretation, then wait for the user to review and give feedback.
- **2-month project timeline.** No rushing. Quality over speed.
- **Report-writing gate (TWO required signals).** Final report yazımına başlamadan önce her ikisinin de gerçekleşmiş olması ŞART:
  1. **Kullanıcının açık onayı** — "içime sindi", "rapor yazabiliriz" tarzı net bir karar. "Sonuç iyi görünüyor"-tipi geçişken ifadeler YETMEZ.
  2. **Coursework requirements check (Claude tarafı)** — proje proposal'ında / ödev gereksinimlerinde listelenen TÜM isterlerin (data pipeline, three-stage architecture, ablation comparisons, walk-forward CV, soft fusion, FastAPI demo, BTC+ETH, vs.) karşılandığından eminim.
  - İki şart aynı anda sağlanmadıkça rapor yazımına geçilmez. Kullanıcı kararsızsa devam et, daha fazla deney/ölçüm öner. Coursework ister'i eksikse kullanıcı "yaz" dese bile önce eksiği tamamla, sonra yaz. Asla "hadi rapora başlayalım, kalan iş süreçte tamamlanır" yaklaşımı yok.
- **Report scope rule (paper odaklılığı).** Final paper SADECE en son başarılı sürüm (V3 veya hangisi onaylanırsa) hakkında yazılır. Geçmiş başarısız/öncül iterasyonlar (v1, v2/ablation, v2/bigger-dataset, v2/feature-selection B2, GMM stickiness, vb.) **paper'da bahsedilmez** — bunlar iç dokümantasyon (MEMORY, CHECKPOINT, V3_PLAN) ve Git history'de fallback referansı olarak kalır. IEEE 4-6 sayfa scope'unda her tarihçe detayı paper'ı uzatır ve okuyucuyu yorar; sadece final tasarım + ablation + sonuçlar yer alır. Geçmiş denemeler en fazla "preliminary work" diye tek bir cümle ile geçer, eğer gerekiyorsa.

### Code Rules
- All source code lives in `src/` with proper module structure.
- Config values come from `config.yaml` — never hardcode parameters.
- Every notebook should import from `src/` modules, not redefine logic.
- Time-series data: NEVER shuffle. All splits are chronological.
- Forward returns are ONLY for label creation, NEVER as input features.
- Use assertions to verify no data leakage.
- Tree models: no scaling. MLP/K-Means: StandardScaler (fit on train only).

### Git Rules
- Commit after each meaningful checkpoint.
- Push to remote after each phase completion.
- Commit messages in English, descriptive.

### File Structure
- `MEMORY.md` — Current project state, decisions made, progress log.
- `CLAUDE.md` — This file. Instructions for AI assistants.
- `config.yaml` — All hyperparameters and project configuration.
- `src/` — Production source code.
- `notebooks/` — Interactive analysis and experimentation.
- `app/` — FastAPI backend + web frontend.
- `docker/` — Containerization.

### Architecture
```
Stage 1 (Trend) ──► p̂(trend) ──┐
                                 ▼
Stage 2 (Macro) ──► p̂(macro) ──► Stage 3 (Signal) ──► Buy/Sell/Hold
                                 ▲
                           oscillator features
```

### Important Technical Decisions
- BTC and ETH modeled SEPARATELY (not pooled).
- Soft fusion: Stage 1 & 2 output probability vectors, not hard labels.
- Stage 3 training uses cross-validated out-of-fold (OOF) predictions.
- Walk-forward expanding-window validation (no random CV).
- 4 classifiers: XGBoost, LightGBM, Random Forest, MLP.
- 3 ablation configs: Flat baseline, 2-Stage, 3-Stage (full).

### Deadlines
- Proposal: 08.03.2026 ✅
- Final Report: 10.05.2026
- Final Presentation: Last 2 weeks of semester
