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

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI(title="V5 Hierarchical Trading Signal Classifier", version="5.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
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
    "ohlcv_cache":    {},   # asset -> OHLCV DataFrame
    "test_dates":     {},   # asset -> list[str]
    "equity_cache":   {},   # asset -> DataFrame (model_rule columns)
    "stage1_oof":     {},   # asset -> Stage 1 RF tuned OOF (for context)
    "stage2_regime":  {},   # asset -> regime FSM CSV
    "stage3_features": {},  # asset -> full 16-feature DataFrame (for /explain, /bundle, /predict_custom)
    "bt_summary":     None, # full backtest summary (Phase 5.1) for heatmap
    "whatif_models":  {},   # (asset, arch, model) -> {model, scaler, feature_cols, classes}
}


# Feature interpretations (paper-friendly explanations)
FEATURE_GROUPS = {
    "stage1_raw":      ["P1_down", "P1_range", "P1_up"],
    "stage1_smoothed": ["P1_down_smooth10", "P1_range_smooth10", "P1_up_smooth10"],
    "stage2_regime":   ["P2_Bull", "P2_Neutral", "P2_Bear", "regime_age_days"],
    "oscillator":      ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
                        "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"],
}

OSCILLATOR_INTERPRETATION = {
    "RSI_14":           ("Relative Strength Index (14d)",
                          [(0, 30, "Oversold (bullish reversal zone)"),
                           (30, 70, "Neutral"),
                           (70, 100, "Overbought (bearish reversal zone)")]),
    "MACD_signal_diff": ("MACD - Signal line",
                          [(-1e9, 0, "Bearish momentum (signal line above MACD)"),
                           (0, 1e9, "Bullish momentum (MACD above signal)")]),
    "Bollinger_pct_b":  ("Bollinger %B (position within 20d±2σ band)",
                          [(-1, 0, "Below lower band (oversold)"),
                           (0, 0.5, "Lower half of band"),
                           (0.5, 1.0, "Upper half of band"),
                           (1.0, 2.0, "Above upper band (overbought)")]),
    "Stochastic_K_14":  ("Stochastic %K (14d)",
                          [(0, 20, "Oversold"),
                           (20, 80, "Neutral"),
                           (80, 100, "Overbought")]),
    "volume_zscore_20": ("Volume z-score (20d)",
                          [(-10, -1, "Below-average volume"),
                           (-1, 1, "Average volume"),
                           (1, 10, "Above-average volume")]),
    "OBV_change_20d":   ("On-Balance Volume 20d change",
                          [(-10, -0.05, "Distribution (selling)"),
                           (-0.05, 0.05, "Flat"),
                           (0.05, 10, "Accumulation (buying)")]),
}


def _interpret_oscillator(name: str, value: float) -> str:
    spec = OSCILLATOR_INTERPRETATION.get(name)
    if not spec:
        return ""
    _, ranges = spec
    for lo, hi, label in ranges:
        if lo <= value < hi:
            return label
    return ""


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
        state["ohlcv_cache"][asset] = ohlcv
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

        # Full 16-feature Stage 3 dataset (used by /explain)
        feat_path = proc / f"{asset.lower()}_features_stage3_v5.csv"
        if feat_path.exists():
            state["stage3_features"][asset] = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            print(f"[startup] {asset} Stage 3 features: {state['stage3_features'][asset].shape}")

    # Phase 5.1 ablation backtest summary (used by /heatmap and /bundle stats)
    bt_path = (PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation"
               / "v5_p5_arch_backtest_summary.csv")
    if bt_path.exists():
        state["bt_summary"] = pd.read_csv(bt_path)
        print(f"[startup] Phase 5.1 backtest summary: {len(state['bt_summary'])} rows")

    # What-If final-fit model bundles (Phase F-prep output)
    wi_dir = PROJECT_ROOT / "app" / "models" / "v5"
    if wi_dir.exists():
        loaded = 0
        for asset in ASSETS:
            for arch in ARCHS:
                for model_name in MODELS:
                    p = wi_dir / f"{asset.lower()}_{arch}_{model_name}.joblib"
                    if p.exists():
                        try:
                            state["whatif_models"][(asset, arch, model_name)] = joblib.load(p)
                            loaded += 1
                        except Exception as e:
                            print(f"[startup] failed to load {p.name}: {e}")
        print(f"[startup] What-If model bundles loaded: {loaded}")

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


@app.get("/explain")
async def explain(
    asset: str = Query(...),
    date:  str = Query(...),
    arch:  str = Query("default"),
    model: str = Query("default"),
    rule:  str = Query("default"),
) -> dict:
    """Per-day decision explanation: feature breakdown + outcome verdict.

    For pedagogical demo: shows WHY the model made its decision
    (16 feature values, grouped + interpreted) and WHAT happened (5/10/30d returns,
    trade simulation if signal followed).
    """
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")
    best = BEST_PER_ASSET[asset]
    if arch == "default":  arch  = best["arch"]
    if model == "default": model = best["model"]
    if rule == "default":  rule  = best["rule"]

    try:
        ts = pd.Timestamp(date)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date: {date}")

    # OOF prediction
    oof = _load_oof(asset, model, arch)
    if ts not in oof.index:
        raise HTTPException(status_code=404, detail=f"No OOF for {asset} on {date}")
    pred_row = oof.loc[ts]
    probs = {
        "Buy":  float(pred_row["P_Buy"]),
        "Hold": float(pred_row["P_Hold"]),
        "Sell": float(pred_row["P_Sell"]),
    }
    pred_label = str(pred_row["pred_label"])
    confidence = probs[pred_label]
    second = sorted(probs.items(), key=lambda kv: -kv[1])[1]

    # Full Stage 3 feature row
    feats_df = state["stage3_features"].get(asset)
    if feats_df is None or ts not in feats_df.index:
        feature_row = None
    else:
        feature_row = feats_df.loc[ts]

    # Build feature breakdown
    feature_breakdown = {}
    if feature_row is not None:
        for group, cols in FEATURE_GROUPS.items():
            entries = []
            for c in cols:
                if c not in feature_row.index:
                    continue
                v = float(feature_row[c])
                interp = ""
                if group == "stage1_raw":
                    if c == "P1_up":     interp = "uptrend posterior"
                    if c == "P1_range":  interp = "range posterior"
                    if c == "P1_down":   interp = "downtrend posterior"
                elif group == "stage1_smoothed":
                    interp = f"10-day rolling mean of {c.replace('_smooth10','')}"
                elif group == "stage2_regime":
                    if c.startswith("P2_"):
                        interp = "FSM regime indicator (1.0 if active, 0 otherwise)"
                    else:
                        interp = "days since last regime transition"
                elif group == "oscillator":
                    interp = _interpret_oscillator(c, v)
                entries.append({"name": c, "value": v, "interpretation": interp})
            feature_breakdown[group] = entries

    # Forward outcomes (post-hoc, label-side info)
    px = state["price_cache"][asset]
    p0 = float(px.loc[ts])
    outcomes = {"5d": None, "10d": None, "30d": None}
    if ts in px.index:
        loc = px.index.get_loc(ts)
        for label, h in [("5d", 5), ("10d", 10), ("30d", 30)]:
            if loc + h < len(px):
                ph = float(px.iloc[loc + h])
                outcomes[label] = {
                    "price":  ph,
                    "return": (ph - p0) / p0,
                    "date":   px.index[loc + h].strftime("%Y-%m-%d"),
                }

    # Verdict: was the prediction "right"?
    verdict = None
    if outcomes["5d"] is not None:
        ret5 = outcomes["5d"]["return"]
        if pred_label == "Buy":
            verdict = "CORRECT" if ret5 > 0.005 else ("WRONG" if ret5 < -0.005 else "NEUTRAL")
        elif pred_label == "Sell":
            verdict = "CORRECT" if ret5 < -0.005 else ("WRONG" if ret5 > 0.005 else "NEUTRAL")
        else:  # Hold
            verdict = "CORRECT" if abs(ret5) < 0.015 else "MISSED_OPPORTUNITY"

    # Trade simulation: from this date forward, exit at next opposite signal
    trade_sim = None
    if pred_label != "Hold":
        # Find next opposite signal (or end of data)
        future = oof.loc[ts:]
        opposite = "Sell" if pred_label == "Buy" else "Buy"
        exit_idx = None
        for i, sig in enumerate(future["pred_label"].values[1:], start=1):
            if sig == opposite:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = len(future) - 1
        exit_ts = future.index[exit_idx]
        exit_price = float(px.loc[exit_ts])
        if pred_label == "Buy":
            pnl = (exit_price - p0) / p0
        else:  # Sell short — but we are long-only. Show "loss avoided"
            pnl = -(exit_price - p0) / p0
        trade_sim = {
            "action":    pred_label,
            "entry_date":  date,
            "entry_price": p0,
            "exit_date":   exit_ts.strftime("%Y-%m-%d"),
            "exit_price":  exit_price,
            "days_held":   int((exit_ts - ts).days),
            "pnl_pct":     pnl,
            "won":         bool(pnl > 0),
        }

    # Top-2 reasons (heuristic): which features stand out most
    top_reasons = []
    if feature_row is not None:
        # Stage 1 dominant class
        s1 = {"down": feature_row["P1_down"],
              "range": feature_row["P1_range"],
              "up": feature_row["P1_up"]}
        s1_dom = max(s1.items(), key=lambda kv: kv[1])
        if s1_dom[1] > 0.45:
            top_reasons.append(
                f"Stage 1 trend: {s1_dom[0]} dominant ({s1_dom[1]*100:.0f}% posterior)")
        # Stage 2 regime
        if "P2_Bull"     in feature_row and feature_row["P2_Bull"]     > 0.5:
            top_reasons.append("Stage 2 macro: Bull regime")
        elif "P2_Bear"   in feature_row and feature_row["P2_Bear"]     > 0.5:
            top_reasons.append("Stage 2 macro: Bear regime")
        elif "P2_Neutral" in feature_row and feature_row["P2_Neutral"] > 0.5:
            top_reasons.append("Stage 2 macro: Neutral regime")
        # Oscillator extremes
        rsi = feature_row.get("RSI_14")
        if rsi is not None:
            if rsi < 30:    top_reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:  top_reasons.append(f"RSI overbought ({rsi:.1f})")
        bbpct = feature_row.get("Bollinger_pct_b")
        if bbpct is not None:
            if bbpct < 0.0:  top_reasons.append(f"Below lower Bollinger band ({bbpct:.2f})")
            elif bbpct > 1.0: top_reasons.append(f"Above upper Bollinger band ({bbpct:.2f})")

    return {
        "asset":             asset,
        "date":              date,
        "arch":              arch,
        "model":             model,
        "rule":              rule,
        "config_label":      f"{arch}/{model}/{rule}",
        "prediction": {
            "signal":         pred_label,
            "confidence":     confidence,
            "probs":          probs,
            "second_choice":  second[0],
            "second_prob":    second[1],
            "margin":         confidence - second[1],
        },
        "feature_breakdown": feature_breakdown,
        "top_reasons":       top_reasons,
        "outcomes":          outcomes,
        "verdict":           verdict,
        "trade_simulation":  trade_sim,
        "current_price":     p0,
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


def _positions_for_rule(oof: pd.DataFrame, rule: str) -> pd.Series:
    pred = oof["pred_label"]
    if rule == "stateful":
        return _positions_stateful(pred)
    if rule == "defensive":
        return (pred == "Buy").astype(float)
    if rule == "prob_weighted":
        return (oof["P_Buy"] - oof["P_Sell"]).clip(lower=0.0, upper=1.0)
    raise HTTPException(status_code=400, detail=f"Unknown rule: {rule}")


def _resolve_defaults(asset: str, arch: str, model: str, rule: str) -> tuple[str, str, str]:
    best = BEST_PER_ASSET[asset]
    if arch  == "default": arch  = best["arch"]
    if model == "default": model = best["model"]
    if rule  == "default": rule  = best["rule"]
    return arch, model, rule


@app.get("/bundle")
async def bundle(
    asset: str = Query("BTC"),
    arch:  str = Query("default"),
    model: str = Query("default"),
    rule:  str = Query("default"),
) -> dict:
    """Mega payload: everything the frontend needs for one (asset, arch, model, rule) view.

    Time-aligned arrays — index i corresponds to dates[i] across every series.
    The frontend caches this once and slides through indices for the scrubber.
    """
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")
    arch, model, rule = _resolve_defaults(asset, arch, model, rule)

    # --- core OOF for selected (model, arch) ---
    oof = _load_oof(asset, model, arch)
    dates = oof.index
    n = len(dates)

    px = state["price_cache"][asset].reindex(dates)
    ohlcv = state["ohlcv_cache"][asset].reindex(dates)

    # --- positions, equity ---
    pos = _positions_for_rule(oof, rule)
    px_ret = px.pct_change().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    strat_ret = pos_lag * px_ret
    pos_change = pos.diff().abs().fillna(0.0)
    cost = pos_change * 0.001
    strat_ret_net = strat_ret - cost
    equity = (1.0 + strat_ret_net).cumprod()
    bh_equity = (1.0 + px_ret).cumprod()

    # --- regime (from FSM) ---
    s2 = state["stage2_regime"].get(asset)
    regime_label, p2_bull, p2_neut, p2_bear = [], [], [], []
    if s2 is not None:
        s2_r = s2.reindex(dates)
        regime_label = [None if pd.isna(x) else str(x) for x in s2_r["regime_label"].values]
        p2_bull = [None if pd.isna(x) else float(x) for x in s2_r["P_Bull"].values]
        p2_neut = [None if pd.isna(x) else float(x) for x in s2_r["P_Neutral"].values]
        p2_bear = [None if pd.isna(x) else float(x) for x in s2_r["P_Bear"].values]

    # --- regime ages + Stage 1 + oscillator features (full feature row) ---
    feats = state["stage3_features"].get(asset)
    regime_age = [None] * n
    s1_down = [None] * n; s1_range = [None] * n; s1_up = [None] * n
    osc = {c: [None] * n for c in ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
                                     "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"]}
    if feats is not None:
        f = feats.reindex(dates)
        regime_age = [None if pd.isna(x) else int(x) for x in f["regime_age_days"].values]
        s1_down  = [float(x) for x in f["P1_down"].fillna(0.0).values]
        s1_range = [float(x) for x in f["P1_range"].fillna(0.0).values]
        s1_up    = [float(x) for x in f["P1_up"].fillna(0.0).values]
        for c in osc.keys():
            if c in f.columns:
                osc[c] = [None if pd.isna(x) else float(x) for x in f[c].values]

    # --- 4-model votes (load all 4 models for the same arch, use selected one as 'active') ---
    votes = {}
    for m in MODELS:
        try:
            o_m = _load_oof(asset, m, arch)
            o_m = o_m.reindex(dates)
            votes[m] = {
                "P_Buy":  [float(x) for x in o_m["P_Buy"].fillna(0.0).values],
                "P_Hold": [float(x) for x in o_m["P_Hold"].fillna(0.0).values],
                "P_Sell": [float(x) for x in o_m["P_Sell"].fillna(0.0).values],
                "pred":   [None if pd.isna(x) else str(x) for x in o_m["pred_label"].values],
            }
        except HTTPException:
            votes[m] = None

    # --- trade events for marker overlay ---
    trades = []
    pos_arr = pos.values
    eq_arr  = equity.values
    px_arr  = px.values
    in_trade = False
    entry_idx = None
    for i in range(n):
        prev_pos = pos_arr[i - 1] if i > 0 else 0.0
        cur_pos  = pos_arr[i]
        if cur_pos > 0 and prev_pos == 0:
            in_trade = True; entry_idx = i
        elif cur_pos == 0 and prev_pos > 0 and in_trade:
            ret_pct = (eq_arr[i] / eq_arr[entry_idx]) - 1.0 if eq_arr[entry_idx] else 0.0
            trades.append({
                "entry_idx":  entry_idx,
                "exit_idx":   i,
                "entry_date": dates[entry_idx].strftime("%Y-%m-%d"),
                "exit_date":  dates[i].strftime("%Y-%m-%d"),
                "entry_price": float(px_arr[entry_idx]),
                "exit_price":  float(px_arr[i]),
                "return_pct":  float(ret_pct),
                "won":         bool(ret_pct > 0),
            })
            in_trade = False
    if in_trade and entry_idx is not None:
        ret_pct = (eq_arr[-1] / eq_arr[entry_idx]) - 1.0 if eq_arr[entry_idx] else 0.0
        trades.append({
            "entry_idx":  entry_idx, "exit_idx": n - 1,
            "entry_date": dates[entry_idx].strftime("%Y-%m-%d"),
            "exit_date":  None,
            "entry_price": float(px_arr[entry_idx]),
            "exit_price":  float(px_arr[-1]),
            "return_pct":  float(ret_pct),
            "won":         bool(ret_pct > 0),
        })

    # --- backtest stats from Phase 5.1 ---
    sharpe = total_ret = max_dd = bh_sharpe = bh_total = None
    if state["bt_summary"] is not None:
        bt = state["bt_summary"]
        m = bt[(bt["asset"] == asset.lower()) & (bt["arch"] == arch)
               & (bt["model"] == model) & (bt["rule"] == rule)]
        if len(m) > 0:
            r = m.iloc[0]
            sharpe = float(r["annualized_sharpe"]) if not pd.isna(r["annualized_sharpe"]) else None
            total_ret = float(r["total_return"]) if not pd.isna(r["total_return"]) else None
            max_dd = float(r["max_drawdown"]) if not pd.isna(r["max_drawdown"]) else None
        bh = bt[(bt["asset"] == asset.lower()) & (bt["arch"] == "buy_and_hold")]
        if len(bh) > 0:
            bh_sharpe = float(bh.iloc[0]["annualized_sharpe"]) if not pd.isna(bh.iloc[0]["annualized_sharpe"]) else None
            bh_total = float(bh.iloc[0]["total_return"]) if not pd.isna(bh.iloc[0]["total_return"]) else None

    return {
        "asset": asset, "arch": arch, "model": model, "rule": rule,
        "label": f"{arch}/{model}/{rule}",
        "n": n,
        "dates":  [d.strftime("%Y-%m-%d") for d in dates],
        "ohlc": {
            "open":   [float(x) for x in ohlcv["Open"].values],
            "high":   [float(x) for x in ohlcv["High"].values],
            "low":    [float(x) for x in ohlcv["Low"].values],
            "close":  [float(x) for x in ohlcv["Close"].values],
            "volume": [float(x) for x in ohlcv["Volume"].values],
        },
        "regime": {
            "label":     regime_label,
            "P_Bull":    p2_bull,
            "P_Neutral": p2_neut,
            "P_Bear":    p2_bear,
            "age_days":  regime_age,
        },
        "stage1": {
            "P_down":  s1_down,
            "P_range": s1_range,
            "P_up":    s1_up,
        },
        "osc": osc,
        "active_signals":  [str(s) for s in oof["pred_label"].values],
        "active_probs": {
            "Buy":  [float(x) for x in oof["P_Buy"].values],
            "Hold": [float(x) for x in oof["P_Hold"].values],
            "Sell": [float(x) for x in oof["P_Sell"].values],
        },
        "votes":   votes,                      # all 4 models per date
        "positions": [float(p) for p in pos.values],
        "equity":    [float(v) for v in equity.values],
        "bh_equity": [float(v) for v in bh_equity.values],
        "trades":    trades,
        "stats": {
            "sharpe":        sharpe,
            "total_return":  total_ret,
            "max_drawdown":  max_dd,
            "bh_sharpe":     bh_sharpe,
            "bh_total_return": bh_total,
            "n_trades":      len(trades),
            "win_rate":      (sum(1 for t in trades if t["won"]) / len(trades))
                              if trades else 0.0,
        },
    }


class CustomFeatures(BaseModel):
    asset: str
    arch:  str
    model: str
    features: dict[str, float]


@app.post("/predict_custom")
async def predict_custom(payload: CustomFeatures) -> dict:
    """What-If endpoint — accepts a 16-feature dict, returns class probabilities.

    Uses final-fit models (Phase F-prep) trained on all data. This is exclusively
    for interactive feature exploration; backtest curves elsewhere remain
    walk-forward OOF.
    """
    asset = payload.asset.upper()
    arch  = payload.arch
    model_name = payload.model
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Unknown asset: {asset}")
    if (asset, arch, model_name) not in state["whatif_models"]:
        raise HTTPException(status_code=404,
                            detail=f"What-If bundle not loaded: {asset}/{arch}/{model_name}")

    bundle = state["whatif_models"][(asset, arch, model_name)]
    feature_cols = bundle["feature_cols"]
    classes = bundle["classes"]
    fmap = payload.features

    # Validate all required features present
    missing = [c for c in feature_cols if c not in fmap]
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"Missing features: {missing}")

    x = np.array([[float(fmap[c]) for c in feature_cols]], dtype=float)
    if bundle.get("scaler") is not None:
        x = bundle["scaler"].transform(x)
    proba = bundle["model"].predict_proba(x)[0]

    # Map back to (Sell, Hold, Buy) order (LABEL_TO_IDX in trainer)
    out = {classes[i]: float(proba[i]) for i in range(len(classes))}
    pred = classes[int(np.argmax(proba))]
    return {
        "asset":      asset,
        "arch":       arch,
        "model":      model_name,
        "feature_cols": feature_cols,
        "probs":      out,
        "pred_label": pred,
        "confidence": float(out[pred]),
    }


@app.get("/heatmap")
async def heatmap() -> dict:
    """Phase 5.1 ablation grid: Sharpe per (asset, arch, model) — best rule per cell."""
    if state["bt_summary"] is None:
        raise HTTPException(status_code=503, detail="Backtest summary not loaded")
    bt = state["bt_summary"]
    bt = bt[bt["arch"] != "buy_and_hold"]

    cells = []
    for (asset, arch, model_name), grp in bt.groupby(["asset", "arch", "model"]):
        # best rule per (asset, arch, model)
        best = grp.loc[grp["annualized_sharpe"].idxmax()]
        cells.append({
            "asset":  str(asset).upper(),
            "arch":   str(arch),
            "model":  str(model_name),
            "rule":   str(best["rule"]),
            "sharpe": float(best["annualized_sharpe"]) if not pd.isna(best["annualized_sharpe"]) else None,
            "total_return": float(best["total_return"]) if not pd.isna(best["total_return"]) else None,
            "max_drawdown": float(best["max_drawdown"]) if not pd.isna(best["max_drawdown"]) else None,
        })

    bh = state["bt_summary"]
    bh = bh[bh["arch"] == "buy_and_hold"]
    bh_rows = [
        {"asset": str(r["asset"]).upper(),
         "sharpe": float(r["annualized_sharpe"]) if not pd.isna(r["annualized_sharpe"]) else None,
         "total_return": float(r["total_return"]) if not pd.isna(r["total_return"]) else None}
        for _, r in bh.iterrows()
    ]
    return {"cells": cells, "buy_hold": bh_rows}


@app.get("/")
async def root():
    return FileResponse(PROJECT_ROOT / "app" / "static" / "index.html")
