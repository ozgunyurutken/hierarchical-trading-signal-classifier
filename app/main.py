"""
FastAPI backend for the Crypto Signal Classifier (MVP build).

Endpoints
---------
GET  /                          serves the static frontend
GET  /health                    liveness + model loading status
GET  /test_dates/{symbol}       list of dates available in the test period (used by the dropdown)
POST /predict                   prediction by date (uses pre-computed features) or live (yfinance)
GET  /indicators/{symbol}       latest computed indicators for a symbol (live yfinance fetch)

Design notes
------------
- Stage 2 is a clustering artifact (`HierarchicalSoftPipeline`), not a supervised classifier.
- The default Stage 1 / Stage 3 path is **LDA** (classical PR baseline). MLP is loaded too and
  could be exposed via a `?model=mlp` query param later.
- Date-based predictions read from `data/processed/btc_features_*` so a live demo never depends
  on internet. The Live button hits yfinance and is best-effort.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Literal

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.features.macro_features import compute_derived_spreads, compute_macro_features
from src.features.technical_indicators import (
    compute_oscillator_indicators,
    compute_trend_indicators,
    compute_volatility_indicators,
    compute_volume_indicators,
)
from src.models.pipeline import HierarchicalSoftPipeline


app = FastAPI(title="Crypto Signal Classifier", version="1.0.0-mvp")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Global state populated on startup
state: dict = {
    "pipelines": {},          # {"lda": HierarchicalSoftPipeline, "mlp": HierarchicalSoftPipeline}
    "stage2_artifact": None,  # GMM cluster artifact dict
    "stage1_features": None,  # pre-computed BTC trend features
    "stage3_features": None,  # pre-computed BTC oscillator features
    "macro_features": None,   # pre-computed BTC macro features
    "aligned_btc": None,      # raw aligned BTC dataset (for Close prices)
    "test_dates": [],         # ISO date strings, dropdown source
}


# ---------- request / response models ----------

class PredictionRequest(BaseModel):
    symbol: str = "BTC"
    mode: Literal["date", "live"] = "date"
    date: str | None = None
    model: Literal["lda", "mlp"] = "lda"


class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    date: str
    price: float | None = None
    trend: dict[str, float]
    macro_regime: dict[str, float]
    signal_probs: dict[str, float]
    mode: str
    model: str


# ---------- startup ----------

@app.on_event("startup")
async def load_pipelines() -> None:
    models_dir = PROJECT_ROOT / "app" / "models"
    proc_dir = PROJECT_ROOT / "data" / "processed"

    # Pipelines (LDA + MLP) — both share the same Stage 1 LDA + Stage 2 GMM artifact
    for variant in ("lda", "mlp"):
        pipeline_dir = models_dir / f"pipeline_{variant}"
        if pipeline_dir.exists():
            state["pipelines"][variant] = HierarchicalSoftPipeline.load(pipeline_dir)
            print(f"[startup] Loaded pipeline_{variant} from {pipeline_dir}")
        else:
            print(f"[startup] WARNING: {pipeline_dir} missing — {variant} predictions unavailable")

    artifact_path = models_dir / "stage2_cluster_artifact.joblib"
    if artifact_path.exists():
        state["stage2_artifact"] = joblib.load(artifact_path)
        print(f"[startup] Loaded Stage 2 cluster artifact ({state['stage2_artifact']['method']}, "
              f"{state['stage2_artifact']['n_clusters']} clusters)")

    # Pre-computed features for date-based prediction
    try:
        state["stage1_features"] = pd.read_csv(
            proc_dir / "btc_features_stage1.csv", index_col=0, parse_dates=True,
        )
        state["stage3_features"] = pd.read_csv(
            proc_dir / "btc_features_stage3.csv", index_col=0, parse_dates=True,
        )
        state["macro_features"] = pd.read_csv(
            proc_dir / "btc_features_macro.csv", index_col=0, parse_dates=True,
        )
        state["aligned_btc"] = pd.read_csv(
            proc_dir / "btc_aligned.csv", index_col=0, parse_dates=True,
        )
        # Test dates = last 15% of aligned data, valid feature rows
        n = len(state["aligned_btc"])
        test_start = int(n * 0.85)
        candidate_dates = state["aligned_btc"].index[test_start:]
        valid_mask = state["stage1_features"].dropna().index.intersection(
            state["stage3_features"].dropna().index
        ).intersection(state["macro_features"].dropna().index)
        test_dates = [d for d in candidate_dates if d in valid_mask]
        state["test_dates"] = [d.strftime("%Y-%m-%d") for d in test_dates]
        print(f"[startup] Pre-computed features loaded; {len(state['test_dates'])} test dates available")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: could not load pre-computed features ({e}). Date mode will fail.")


# ---------- helpers ----------

def _build_stage_inputs_for_date(date_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Look up pre-computed features for a single date and return Stage 1/macro/Stage 3 frames."""
    if state["stage1_features"] is None:
        raise HTTPException(status_code=503, detail="Pre-computed features not loaded")

    try:
        ts = pd.Timestamp(date_str)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str!r}; expected YYYY-MM-DD")

    if ts not in state["stage1_features"].index:
        raise HTTPException(status_code=404, detail=f"No features for date {date_str}")

    artifact = state["stage2_artifact"]
    macro_cols = artifact["feature_names"] if artifact else []
    macro_subset = state["macro_features"].loc[[ts], macro_cols] if macro_cols else state["macro_features"].loc[[ts]]

    s1 = state["stage1_features"].loc[[ts]]
    s3 = state["stage3_features"].loc[[ts]]
    price = float(state["aligned_btc"].loc[ts, "Close"])
    return s1, macro_subset, s3, price


def _build_stage_inputs_live(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, str]:
    """Fetch live OHLCV + macro from yfinance and compute the latest feature row."""
    ticker_crypto = f"{symbol}-USD"
    ohlcv = yf.Ticker(ticker_crypto).history(period="2y", interval="1d")
    if ohlcv.empty:
        raise HTTPException(status_code=502, detail=f"yfinance returned no data for {ticker_crypto}")
    ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)

    trend = compute_trend_indicators(ohlcv)
    osc = compute_oscillator_indicators(ohlcv)
    vol = compute_volatility_indicators(ohlcv)
    volu = compute_volume_indicators(ohlcv)
    s1_live = trend.tail(1)
    s3_live = pd.concat([osc.tail(1), vol.tail(1), volu.tail(1)], axis=1)

    # Macro: assemble the same 8 features the Stage 2 artifact was trained on
    artifact = state["stage2_artifact"]
    if artifact is None:
        raise HTTPException(status_code=503, detail="Stage 2 artifact not loaded")
    macro_cols = artifact["feature_names"]

    # Reuse last 1 year of aligned macro from local CSV; only the live row would be missing
    # (good enough for a demo; replacing with live yfinance fetch for all 12 macros is slow)
    if state["macro_features"] is None:
        raise HTTPException(status_code=503, detail="Macro features cache not loaded; live mode unavailable")
    last_known = state["macro_features"][macro_cols].dropna().iloc[[-1]]

    last_price = float(ohlcv["Close"].iloc[-1])
    last_date = ohlcv.index[-1].strftime("%Y-%m-%d")
    return s1_live, last_known, s3_live, last_price, last_date


def _run_pipeline(model_key: str, s1: pd.DataFrame, macro: pd.DataFrame, s3: pd.DataFrame) -> dict:
    pipeline = state["pipelines"].get(model_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail=f"Pipeline {model_key!r} not loaded")

    proba = pipeline.predict_proba(s1, macro, s3)
    signal_probs = proba["signal_probs"][0]
    signal_idx = int(np.argmax(signal_probs))
    signal_label = str(pipeline.stage3.classes_[signal_idx])
    confidence = float(signal_probs[signal_idx])

    artifact = state["stage2_artifact"]
    semantic = artifact["semantic_map"] if artifact else {i: f"cluster_{i}" for i in range(len(proba["regime_probs"][0]))}

    return {
        "signal": signal_label,
        "confidence": confidence,
        "trend": {str(c): float(p) for c, p in zip(pipeline.stage1.classes_, proba["trend_probs"][0])},
        "macro_regime": {semantic.get(i, f"cluster_{i}"): float(p) for i, p in enumerate(proba["regime_probs"][0])},
        "signal_probs": {str(c): float(p) for c, p in zip(pipeline.stage3.classes_, proba["signal_probs"][0])},
    }


# ---------- endpoints ----------

@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "pipelines_loaded": list(state["pipelines"].keys()),
        "stage2_loaded": state["stage2_artifact"] is not None,
        "test_dates_count": len(state["test_dates"]),
        "ready": bool(state["pipelines"]) and state["stage2_artifact"] is not None,
    }


@app.get("/test_dates/{symbol}")
async def test_dates(symbol: str) -> dict:
    """Return the list of dates the demo dropdown can offer (precomputed features only)."""
    if symbol.upper() != "BTC":
        raise HTTPException(status_code=400, detail="MVP supports BTC only; ETH planned for second iteration")
    return {"symbol": symbol.upper(), "dates": state["test_dates"]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if request.symbol.upper() != "BTC":
        raise HTTPException(status_code=400, detail="MVP supports BTC only")
    if not state["pipelines"]:
        raise HTTPException(status_code=503, detail="No pipelines loaded; train models first")

    if request.mode == "date":
        if not request.date:
            raise HTTPException(status_code=400, detail="`date` field required for date mode")
        s1, macro, s3, price = _build_stage_inputs_for_date(request.date)
        result = _run_pipeline(request.model, s1, macro, s3)
        return PredictionResponse(
            **result,
            date=request.date,
            price=price,
            mode="date",
            model=request.model,
        )

    # live mode
    try:
        s1, macro, s3, price, last_date = _build_stage_inputs_live(request.symbol)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Live data fetch failed: {e}")

    result = _run_pipeline(request.model, s1, macro, s3)
    return PredictionResponse(
        **result,
        date=last_date,
        price=price,
        mode="live",
        model=request.model,
    )


@app.get("/indicators/{symbol}")
async def get_indicators(symbol: str) -> dict:
    """Convenience endpoint for the frontend to show a few key indicator values."""
    if symbol.upper() != "BTC":
        raise HTTPException(status_code=400, detail="MVP supports BTC only")
    try:
        ohlcv = yf.Ticker(f"{symbol}-USD").history(period="6mo", interval="1d")
        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)
        trend = compute_trend_indicators(ohlcv)
        osc = compute_oscillator_indicators(ohlcv)
        latest = pd.concat([trend.iloc[-1], osc.iloc[-1]])
        return {
            "symbol": symbol.upper(),
            "date": ohlcv.index[-1].strftime("%Y-%m-%d"),
            "price": float(ohlcv["Close"].iloc[-1]),
            "indicators": {k: round(float(v), 4) if pd.notna(v) else None for k, v in latest.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------- static frontend ----------

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Crypto Signal Classifier API. Visit /docs for API documentation."}
