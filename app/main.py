"""V5 FastAPI demo for the Three-Stage Hierarchical Crypto Signal Classifier.

Endpoints
---------
GET  /                                serves the static frontend
GET  /health                          liveness + cache status
GET  /assets                          ["BTC", "ETH"]
GET  /test_dates/{asset}              chronological OOF dates for the asset
GET  /predict                         query params: asset, date, arch, rule
GET  /equity/{asset}                  equity curve data for best model (BTC: 3stage_full+xgb+stateful, ETH: flat+lgbm+prob_weighted)

Design
------
Reads pre-computed OOF predictions from data/processed/{asset}_stage3_oof_{model}_v5_tuned_{arch}.csv.
No on-the-fly inference — predictions are walk-forward OOF, deterministic reproducible.
This makes the demo Docker-stable: no model loading or sklearn version drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI(title="V5 Hierarchical Trading Signal Classifier", version="5.0")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")

ASSETS = ["BTC", "ETH"]
ARCHS = ["flat", "2stage_trend", "2stage_macro", "3stage_full"]
RULES = ["stateful", "defensive", "prob_weighted"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]

# Best per asset (Phase 5.1 ablation winners)
BEST_PER_ASSET = {
    "BTC": {"arch": "3stage_full", "model": "xgboost",  "rule": "stateful"},
    "ETH": {"arch": "flat",         "model": "lightgbm", "rule": "prob_weighted"},
}

state = {
    "oof_cache":      {},   # (asset, model, arch) -> DataFrame
    "price_cache":    {},   # asset -> Close Series
    "test_dates":     {},   # asset -> list[str]
    "equity_cache":   {},   # asset -> DataFrame (model_rule columns)
    "stage1_oof":     {},   # asset -> Stage 1 RF tuned OOF (for context)
    "stage2_regime":  {},   # asset -> regime FSM CSV
}


class PredictionResponse(BaseModel):
    asset:        str
    date:         str
    arch:         str
    model:        str
    rule:         str
    signal:       str
    confidence:   float
    probs:        dict[str, float]
    stage1_trend: dict[str, float] | None = None
    stage2_regime: str | None = None
    price:        float
    forward_return_5d: float | None = None


@app.on_event("startup")
async def warmup():
    proc = PROJECT_ROOT / "data" / "processed"
    for asset in ASSETS:
        # Aligned OHLCV
        ohlcv = pd.read_csv(proc / f"{asset.lower()}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        state["price_cache"][asset] = ohlcv["Close"]

        # Pre-load best per-asset OOF (the one shown by default)
        best = BEST_PER_ASSET[asset]
        oof_path = proc / (f"{asset.lower()}_stage3_oof_{best['model']}_"
                           f"v5_tuned_{best['arch']}.csv")
        if oof_path.exists():
            df = pd.read_csv(oof_path, index_col=0, parse_dates=True)
            state["oof_cache"][(asset, best["model"], best["arch"])] = df
            state["test_dates"][asset] = [d.strftime("%Y-%m-%d") for d in df.index]
            print(f"[startup] {asset} default OOF loaded: "
                  f"{best['arch']}/{best['model']}, {len(df)} dates")

        # Stage 1 RF tuned OOF (for context/transparency)
        s1_path = proc / f"{asset.lower()}_stage1_oof_random_forest_v5_tuned.csv"
        if s1_path.exists():
            state["stage1_oof"][asset] = pd.read_csv(s1_path, index_col=0, parse_dates=True)

        # Stage 2 FSM regime
        s2_path = proc / f"{asset.lower()}_regime_labels_composite_macro_v5_v5.csv"
        if s2_path.exists():
            state["stage2_regime"][asset] = pd.read_csv(s2_path, index_col=0, parse_dates=True)

        # Equity curves (Phase 5.1 has multi-arch; we use that)
        eq_path = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / f"v5_p5_arch_equity_curves_{asset.lower()}.csv"
        if eq_path.exists():
            state["equity_cache"][asset] = pd.read_csv(eq_path, index_col=0, parse_dates=True)
            print(f"[startup] {asset} equity curves loaded: {state['equity_cache'][asset].shape[1]} curves")
    print("[startup] V5 demo backend ready")


def _load_oof(asset: str, model: str, arch: str) -> pd.DataFrame:
    key = (asset, model, arch)
    if key in state["oof_cache"]:
        return state["oof_cache"][key]
    proc = PROJECT_ROOT / "data" / "processed"
    path = proc / f"{asset.lower()}_stage3_oof_{model}_v5_tuned_{arch}.csv"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"OOF file not found: {path.name}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    state["oof_cache"][key] = df
    return df


# ---------- endpoints ----------

@app.get("/health")
async def health() -> dict:
    return {
        "status":           "ok",
        "ready":            len(state["price_cache"]) > 0,
        "assets_loaded":    list(state["price_cache"].keys()),
        "default_oof_loaded": list(state["oof_cache"].keys()),
        "test_dates": {a: len(d) for a, d in state["test_dates"].items()},
    }


@app.get("/assets")
async def assets() -> dict:
    return {
        "assets": ASSETS,
        "archs":  ARCHS,
        "rules":  RULES,
        "models": MODELS,
        "best":   BEST_PER_ASSET,
    }


@app.get("/test_dates/{asset}")
async def test_dates(asset: str) -> dict:
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")
    if asset not in state["test_dates"]:
        raise HTTPException(status_code=503, detail="OOF dates not loaded yet")
    return {"asset": asset, "dates": state["test_dates"][asset]}


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    asset: str = Query(..., description="BTC or ETH"),
    date:  str = Query(..., description="YYYY-MM-DD, must be in test_dates"),
    arch:  str = Query("default", description="flat|2stage_trend|2stage_macro|3stage_full|default"),
    model: str = Query("default", description="xgboost|lightgbm|random_forest|mlp|default"),
    rule:  str = Query("default", description="stateful|defensive|prob_weighted|default"),
) -> PredictionResponse:
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")

    # Resolve "default" to BEST_PER_ASSET
    best = BEST_PER_ASSET[asset]
    if arch == "default":  arch  = best["arch"]
    if model == "default": model = best["model"]
    if rule == "default":  rule  = best["rule"]

    if arch not in ARCHS:   raise HTTPException(status_code=400, detail=f"Unknown arch: {arch}")
    if model not in MODELS: raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    if rule not in RULES:   raise HTTPException(status_code=400, detail=f"Unknown rule: {rule}")

    try:
        ts = pd.Timestamp(date)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}")

    oof = _load_oof(asset, model, arch)
    if ts not in oof.index:
        raise HTTPException(status_code=404,
                            detail=f"No OOF prediction for {asset} on {date}")

    row = oof.loc[ts]
    probs = {
        "Buy":  float(row["P_Buy"]),
        "Hold": float(row["P_Hold"]),
        "Sell": float(row["P_Sell"]),
    }
    pred_label = str(row["pred_label"])
    confidence = probs[pred_label]

    price = float(state["price_cache"][asset].loc[ts])

    # Forward 5-day return for context (educational; not a feature)
    px = state["price_cache"][asset]
    fwd = None
    if (loc := px.index.get_loc(ts)) + 5 < len(px):
        fwd = float((px.iloc[loc + 5] - px.iloc[loc]) / px.iloc[loc])

    # Stage 1 trend probabilities (RF tuned, context)
    s1_trend = None
    if asset in state["stage1_oof"]:
        s1 = state["stage1_oof"][asset]
        if ts in s1.index:
            s1_trend = {
                "downtrend": float(s1.loc[ts, "P_downtrend"]),
                "range":     float(s1.loc[ts, "P_range"]),
                "uptrend":   float(s1.loc[ts, "P_uptrend"]),
            }

    # Stage 2 regime (FSM)
    s2_regime = None
    if asset in state["stage2_regime"]:
        s2 = state["stage2_regime"][asset]
        if ts in s2.index:
            s2_regime = str(s2.loc[ts, "regime_label"])

    return PredictionResponse(
        asset=asset, date=date, arch=arch, model=model, rule=rule,
        signal=pred_label, confidence=confidence, probs=probs,
        stage1_trend=s1_trend, stage2_regime=s2_regime,
        price=price, forward_return_5d=fwd,
    )


@app.get("/equity/{asset}")
async def equity(asset: str) -> dict:
    asset = asset.upper()
    if asset not in state["equity_cache"]:
        raise HTTPException(status_code=503, detail="Equity cache not loaded")
    eq = state["equity_cache"][asset]
    best = BEST_PER_ASSET[asset]
    best_col = f"{best['arch']}_{best['model']}_{best['rule']}"

    if best_col not in eq.columns:
        raise HTTPException(status_code=500, detail=f"Best column missing: {best_col}")

    bh_col = "BUY_AND_HOLD"
    return {
        "asset":    asset,
        "best_label": f"{best['arch']}/{best['model']}/{best['rule']}",
        "dates":    [d.strftime("%Y-%m-%d") for d in eq.index],
        "best":     [None if pd.isna(v) else float(v) for v in eq[best_col]],
        "buy_hold": [None if pd.isna(v) else float(v) for v in eq[bh_col]] if bh_col in eq.columns else [],
    }


@app.get("/timeline")
async def timeline(
    asset: str = Query(..., description="BTC or ETH"),
    arch:  str = Query("default"),
    model: str = Query("default"),
    rule:  str = Query("default"),
) -> dict:
    """Full timeline for visualization: dates, prices, signals, positions, equity, B&H equity.

    Used by the front-end to draw trade markers on a price chart and a synced equity panel.
    """
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")

    best = BEST_PER_ASSET[asset]
    if arch  == "default": arch  = best["arch"]
    if model == "default": model = best["model"]
    if rule  == "default": rule  = best["rule"]

    oof = _load_oof(asset, model, arch)
    prices = state["price_cache"][asset].reindex(oof.index)
    pred = oof["pred_label"]

    # --- positions per rule ---
    if rule == "stateful":
        pos = _positions_stateful(pred)
    elif rule == "defensive":
        pos = (pred == "Buy").astype(float)
    elif rule == "prob_weighted":
        pos = (oof["P_Buy"] - oof["P_Sell"]).clip(lower=0.0, upper=1.0)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown rule: {rule}")

    # --- daily returns and equity ---
    px_ret = prices.pct_change().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    strat_ret = pos_lag * px_ret
    pos_change = pos.diff().abs().fillna(0.0)
    cost = pos_change * 0.001  # 0.1% TC
    strat_ret_net = strat_ret - cost
    equity = (1.0 + strat_ret_net).cumprod()
    bh_equity = (1.0 + px_ret).cumprod()

    # --- trade events (entry/exit dates with realized return per round-trip) ---
    trades = []
    in_trade = False
    entry_date, entry_eq, entry_price = None, None, None
    pos_arr = pos.values
    eq_arr = equity.values
    px_arr = prices.values
    dates_arr = list(oof.index)
    for i in range(len(pos_arr)):
        prev_pos = pos_arr[i - 1] if i > 0 else 0.0
        cur_pos = pos_arr[i]
        if cur_pos > 0 and prev_pos == 0:
            in_trade = True
            entry_date = dates_arr[i]
            entry_eq = eq_arr[i]
            entry_price = px_arr[i]
        elif cur_pos == 0 and prev_pos > 0 and in_trade:
            ret_pct = (eq_arr[i] / entry_eq) - 1.0
            trades.append({
                "entry_date":  entry_date.strftime("%Y-%m-%d"),
                "exit_date":   dates_arr[i].strftime("%Y-%m-%d"),
                "entry_price": float(entry_price),
                "exit_price":  float(px_arr[i]),
                "return_pct":  float(ret_pct),
                "won":         bool(ret_pct > 0),
            })
            in_trade = False
    if in_trade:
        ret_pct = (eq_arr[-1] / entry_eq) - 1.0
        trades.append({
            "entry_date":  entry_date.strftime("%Y-%m-%d"),
            "exit_date":   None,
            "entry_price": float(entry_price),
            "exit_price":  float(px_arr[-1]),
            "return_pct":  float(ret_pct),
            "won":         bool(ret_pct > 0),
        })

    n_trades = len(trades)
    win_rate = (sum(1 for t in trades if t["won"]) / n_trades) if n_trades else 0.0

    return {
        "asset":      asset,
        "arch":       arch,
        "model":      model,
        "rule":       rule,
        "label":      f"{arch}/{model}/{rule}",
        "dates":      [d.strftime("%Y-%m-%d") for d in oof.index],
        "prices":     [float(p) for p in prices.values],
        "signals":    [str(s) for s in pred.values],
        "positions":  [float(p) for p in pos.values],
        "equity":     [float(v) for v in equity.values],
        "bh_equity":  [float(v) for v in bh_equity.values],
        "trades":     trades,
        "n_trades":   n_trades,
        "win_rate":   win_rate,
        "final_return":     float(equity.values[-1] - 1.0),
        "bh_final_return":  float(bh_equity.values[-1] - 1.0),
    }


def _positions_stateful(pred: pd.Series) -> pd.Series:
    """Stateful long-only: Buy=enter, Sell=exit, Hold=keep."""
    pos = []
    state_local = 0
    for sig in pred.values:
        if sig == "Buy":   state_local = 1
        elif sig == "Sell": state_local = 0
        pos.append(float(state_local))
    return pd.Series(pos, index=pred.index)


@app.get("/")
async def root():
    return FileResponse(PROJECT_ROOT / "app" / "static" / "index.html")
