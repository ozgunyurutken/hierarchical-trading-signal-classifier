"""
FastAPI backend for Crypto Signal Classifier.
Serves trained models and provides prediction API.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import StringIO

app = FastAPI(title="Crypto Signal Classifier", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}


class PredictionRequest(BaseModel):
    symbol: str = "BTC"
    date: str | None = None


class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    trend: dict[str, float]
    macro_regime: dict[str, float]
    signal_probs: dict[str, float]


@app.on_event("startup")
async def load_models():
    """Load trained models on startup."""
    model_dir = PROJECT_ROOT / "app" / "models"

    model_files = {
        "stage1": "stage1_model.joblib",
        "stage2": "stage2_model.joblib",
        "stage3": "stage3_model.joblib",
    }

    for name, filename in model_files.items():
        path = model_dir / filename
        if path.exists():
            models[name] = joblib.load(path)
            print(f"Loaded {name} from {path}")
        else:
            print(f"WARNING: {path} not found. Predictions will be unavailable.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    loaded = list(models.keys())
    return {
        "status": "ok",
        "models_loaded": loaded,
        "all_models_ready": len(loaded) == 3,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get trading signal prediction for a cryptocurrency.

    Returns signal (Buy/Sell/Hold) with confidence and stage-wise probabilities.
    """
    if len(models) < 3:
        raise HTTPException(
            status_code=503,
            detail=f"Models not fully loaded. Available: {list(models.keys())}",
        )

    symbol = request.symbol.upper()
    if symbol not in ["BTC", "ETH"]:
        raise HTTPException(status_code=400, detail="Symbol must be BTC or ETH")

    try:
        # Fetch recent price data
        ticker = f"{symbol}-USD"
        df = yf.Ticker(ticker).history(period="1y", interval="1d")
        if df.empty:
            raise HTTPException(status_code=500, detail="Could not fetch price data")

        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Compute features (simplified for serving)
        from src.features.technical_indicators import (
            compute_trend_indicators,
            compute_oscillator_indicators,
            compute_volatility_indicators,
            compute_volume_indicators,
        )

        trend_features = compute_trend_indicators(df)
        oscillator_features = compute_oscillator_indicators(df)
        volatility_features = compute_volatility_indicators(df)
        volume_features = compute_volume_indicators(df)

        # Get latest row
        latest_trend = trend_features.iloc[[-1]].fillna(0)
        latest_osc = oscillator_features.iloc[[-1]].fillna(0)

        # Stage 1: Trend prediction
        stage1 = models["stage1"]
        trend_proba = stage1.predict_proba(latest_trend)

        # Stage 2: Macro regime (use available macro data)
        stage2 = models["stage2"]
        # For demo, use approximate macro features from recent data
        macro_features = _get_latest_macro_features()
        regime_proba = stage2.predict_proba(macro_features)

        # Stage 3: Signal prediction
        trend_df = pd.DataFrame(
            trend_proba,
            columns=[f"trend_prob_{c}" for c in stage1.classes_],
            index=latest_osc.index,
        )
        regime_df = pd.DataFrame(
            regime_proba,
            columns=[f"regime_prob_{c}" for c in stage2.classes_],
            index=latest_osc.index,
        )
        X_stage3 = pd.concat([latest_osc, trend_df, regime_df], axis=1)

        stage3 = models["stage3"]
        signal_proba = stage3.predict_proba(X_stage3)
        signal_idx = np.argmax(signal_proba[0])
        signal_label = stage3.classes_[signal_idx]
        confidence = float(signal_proba[0][signal_idx])

        return PredictionResponse(
            signal=signal_label,
            confidence=confidence,
            trend={str(c): float(p) for c, p in zip(stage1.classes_, trend_proba[0])},
            macro_regime={str(c): float(p) for c, p in zip(stage2.classes_, regime_proba[0])},
            signal_probs={str(c): float(p) for c, p in zip(stage3.classes_, signal_proba[0])},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict from uploaded CSV file with OHLCV data.
    CSV must have columns: Date, Open, High, Low, Close, Volume
    """
    if len(models) < 3:
        raise HTTPException(status_code=503, detail="Models not fully loaded")

    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")), parse_dates=["Date"], index_col="Date")

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(df.columns)):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_cols}. Got: {set(df.columns)}",
            )

        # Compute features and predict
        from src.features.technical_indicators import (
            compute_trend_indicators,
            compute_oscillator_indicators,
        )

        trend_features = compute_trend_indicators(df)
        oscillator_features = compute_oscillator_indicators(df)

        latest_trend = trend_features.iloc[[-1]].fillna(0)
        latest_osc = oscillator_features.iloc[[-1]].fillna(0)

        stage1 = models["stage1"]
        trend_proba = stage1.predict_proba(latest_trend)

        stage2 = models["stage2"]
        macro_features = _get_latest_macro_features()
        regime_proba = stage2.predict_proba(macro_features)

        trend_df = pd.DataFrame(
            trend_proba, columns=[f"trend_prob_{c}" for c in stage1.classes_],
            index=latest_osc.index,
        )
        regime_df = pd.DataFrame(
            regime_proba, columns=[f"regime_prob_{c}" for c in stage2.classes_],
            index=latest_osc.index,
        )
        X_stage3 = pd.concat([latest_osc, trend_df, regime_df], axis=1)

        stage3 = models["stage3"]
        signal_proba = stage3.predict_proba(X_stage3)
        signal_idx = np.argmax(signal_proba[0])

        return {
            "signal": str(stage3.classes_[signal_idx]),
            "confidence": float(signal_proba[0][signal_idx]),
            "signal_probs": {str(c): float(p) for c, p in zip(stage3.classes_, signal_proba[0])},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Get latest computed indicators for visualization."""
    symbol = symbol.upper()
    if symbol not in ["BTC", "ETH"]:
        raise HTTPException(status_code=400, detail="Symbol must be BTC or ETH")

    try:
        ticker = f"{symbol}-USD"
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        df.index = pd.to_datetime(df.index).tz_localize(None)

        from src.features.technical_indicators import (
            compute_trend_indicators,
            compute_oscillator_indicators,
        )

        trend = compute_trend_indicators(df)
        osc = compute_oscillator_indicators(df)

        latest = pd.concat([trend.iloc[-1], osc.iloc[-1]])
        return {
            "symbol": symbol,
            "date": str(df.index[-1].date()),
            "price": float(df["Close"].iloc[-1]),
            "indicators": {k: round(float(v), 4) if pd.notna(v) else None for k, v in latest.items()},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_latest_macro_features() -> pd.DataFrame:
    """Fetch latest macro features for prediction."""
    try:
        macro_tickers = {"^GSPC": "S&P_500", "GC=F": "Gold_Futures", "DX-Y.NYB": "DXY_Dollar_Index", "^VIX": "VIX_Volatility"}
        data = {}
        for ticker, name in macro_tickers.items():
            hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
            if not hist.empty:
                data[f"macro_{name}"] = hist["Close"].iloc[-1]
                data[f"macro_{name}_roc_20"] = hist["Close"].pct_change(20).iloc[-1]

        return pd.DataFrame([data])
    except Exception:
        # Fallback: return zeros
        return pd.DataFrame([{f"macro_feat_{i}": 0 for i in range(10)}])


# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Crypto Signal Classifier API. Visit /docs for API documentation."}
