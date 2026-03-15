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
