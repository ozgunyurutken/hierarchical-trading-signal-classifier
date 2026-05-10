FROM python:3.11-slim

WORKDIR /project

# System dependencies for compiled wheels (xgboost, lightgbm, hmmlearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source
COPY src/ /project/src/
COPY app/ /project/app/
COPY config.yaml /project/config.yaml

# V5 demo bundle: pre-computed walk-forward OOF predictions + price + Stage 2 FSM
# Docker stays self-contained, no on-the-fly inference.
# Aligned OHLCV (price for /predict response + equity baseline)
COPY data/processed/btc_aligned_v5.csv /project/data/processed/
COPY data/processed/eth_aligned_v5.csv /project/data/processed/
# Stage 2 regime FSM (composite_macro_v5_v5.csv)
COPY data/processed/btc_regime_labels_composite_macro_v5_v5.csv /project/data/processed/
COPY data/processed/eth_regime_labels_composite_macro_v5_v5.csv /project/data/processed/
# Stage 1 RF tuned OOF (context for /predict response)
COPY data/processed/btc_stage1_oof_random_forest_v5_tuned.csv /project/data/processed/
COPY data/processed/eth_stage1_oof_random_forest_v5_tuned.csv /project/data/processed/
# Stage 3 architecture-ablation tuned OOF (4 arch x 4 model x 2 asset = 32 files)
COPY data/processed/btc_stage3_oof_xgboost_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_xgboost_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_xgboost_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_xgboost_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_lightgbm_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_lightgbm_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_lightgbm_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_lightgbm_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_random_forest_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_random_forest_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_random_forest_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_random_forest_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_mlp_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_mlp_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_mlp_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/btc_stage3_oof_mlp_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_xgboost_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_xgboost_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_xgboost_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_xgboost_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_lightgbm_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_lightgbm_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_lightgbm_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_lightgbm_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_random_forest_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_random_forest_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_random_forest_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_random_forest_v5_tuned_2stage_macro.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_mlp_v5_tuned_3stage_full.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_mlp_v5_tuned_flat.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_mlp_v5_tuned_2stage_trend.csv /project/data/processed/
COPY data/processed/eth_stage3_oof_mlp_v5_tuned_2stage_macro.csv /project/data/processed/
# Equity curves (Phase 5.1 ablation, used for /equity endpoint)
COPY reports/Phase5.1_arch_ablation/v5_p5_arch_equity_curves_btc.csv /project/reports/Phase5.1_arch_ablation/
COPY reports/Phase5.1_arch_ablation/v5_p5_arch_equity_curves_eth.csv /project/reports/Phase5.1_arch_ablation/

ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
