"""V5 Phase 5 — Trading rule backtester.

Three trading rules evaluated on Stage 3 OOF predictions:

  1. STATEFUL  long-only state machine
       state in {long, cash}
       Cash + Buy   -> long  (enter)
       Long + Sell  -> cash  (exit)
       Hold         -> stay  (no-op)
       -> few trades, bull-friendly

  2. DEFENSIVE reset every day
       Buy   -> long
       Hold  -> cash (exit)
       Sell  -> cash (exit)
       -> many trades, bear-friendly, transaction-cost-heavy

  3. PROB_WEIGHTED  position size = P(Buy) - P(Sell), clipped [0, 1] for long-only
       -> continuous exposure, Bayesian-style

Plus B&H benchmark (always long, single buy at first day).

Metrics: total return, annualized Sharpe (252 trading days), max drawdown,
n_trades, win rate, equity curve, daily returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRANSACTION_COST = 0.001   # 0.1% one-way (V5_PLAN spec)


@dataclass
class BacktestResult:
    rule: str
    asset: str
    model: str
    total_return: float
    annualized_sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    final_equity: float
    n_days: int
    span_start: pd.Timestamp
    span_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "rule":              self.rule,
            "asset":             self.asset,
            "model":             self.model,
            "total_return":      self.total_return,
            "annualized_sharpe": self.annualized_sharpe,
            "max_drawdown":      self.max_drawdown,
            "n_trades":          self.n_trades,
            "win_rate":          self.win_rate,
            "final_equity":      self.final_equity,
            "n_days":            self.n_days,
            "span_start":        self.span_start.date(),
            "span_end":          self.span_end.date(),
        }


# ---------- Trading rule -> position series ----------

def _stateful_positions(pred: pd.Series) -> pd.Series:
    """Stateful long-only: Buy=enter, Sell=exit, Hold=no-op."""
    pos = np.zeros(len(pred), dtype=float)
    state = 0  # cash
    for i, sig in enumerate(pred.values):
        if sig == "Buy":
            state = 1
        elif sig == "Sell":
            state = 0
        # Hold -> keep state
        pos[i] = state
    return pd.Series(pos, index=pred.index, name="position")


def _defensive_positions(pred: pd.Series) -> pd.Series:
    """Defensive: Buy=long, Hold/Sell=cash."""
    pos = (pred == "Buy").astype(float)
    return pos.rename("position")


def _prob_weighted_positions(oof: pd.DataFrame) -> pd.Series:
    """Position size = clip(P_Buy - P_Sell, 0, 1) for long-only."""
    raw = oof["P_Buy"] - oof["P_Sell"]
    return raw.clip(lower=0.0, upper=1.0).rename("position")


# ---------- Backtest engine ----------

def _backtest_position_series(positions: pd.Series, prices: pd.Series,
                              transaction_cost: float) -> tuple[pd.Series, list[float], int]:
    """Apply position series to price series, return:
       - daily strategy returns
       - per-trade returns (for win rate)
       - number of trades
    """
    common = positions.index.intersection(prices.index)
    pos = positions.loc[common]
    px = prices.loc[common]

    # Daily price return
    px_return = px.pct_change().fillna(0.0)

    # Strategy daily return = position[t-1] * px_return[t]
    # We hold position during day t after deciding at end of t-1.
    # For OOF prediction at date t (made at end of day t-1 effectively),
    # we apply it from t onward. Simpler approximation: pos[t] * px_return[t].
    strat_return = pos.shift(1).fillna(0.0) * px_return

    # Transaction cost: applied when position changes.
    pos_change = pos.diff().abs().fillna(0.0)
    cost_drag = pos_change * transaction_cost
    strat_return = strat_return - cost_drag

    # Trade tracking: count round trips, win rate
    trade_returns = []
    in_trade = False
    entry_eq = 0.0
    cum_eq = 1.0
    cum_eq_at_entry = 1.0
    pos_arr = pos.values
    sret_arr = strat_return.values
    for i in range(len(pos_arr)):
        cum_eq *= (1 + sret_arr[i])
        prev_pos = pos_arr[i - 1] if i > 0 else 0
        cur_pos = pos_arr[i]
        if cur_pos > 0 and prev_pos == 0:
            # Entered position
            in_trade = True
            cum_eq_at_entry = cum_eq
        elif cur_pos == 0 and prev_pos > 0 and in_trade:
            # Exited position
            trade_ret = (cum_eq / cum_eq_at_entry) - 1.0
            trade_returns.append(trade_ret)
            in_trade = False
    if in_trade:
        # Force-close at end
        trade_ret = (cum_eq / cum_eq_at_entry) - 1.0
        trade_returns.append(trade_ret)

    # n_trades = round trips (entry+exit pairs counted once)
    n_trades = len(trade_returns)
    return strat_return, trade_returns, n_trades


def _summary_from_returns(strat_return: pd.Series, trade_returns: list[float],
                          n_trades: int, prices: pd.Series,
                          rule: str, asset: str, model: str) -> BacktestResult:
    eq = (1 + strat_return).cumprod()
    final_equity = float(eq.iloc[-1])
    total_return = final_equity - 1.0
    daily_mean = strat_return.mean()
    daily_std = strat_return.std(ddof=1)
    sharpe = (daily_mean / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_std > 0 else 0.0)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = float(drawdown.min())
    win_rate = (sum(1 for r in trade_returns if r > 0) / max(n_trades, 1)
                if n_trades else 0.0)

    return BacktestResult(
        rule=rule, asset=asset, model=model,
        total_return=float(total_return),
        annualized_sharpe=float(sharpe),
        max_drawdown=max_dd,
        n_trades=int(n_trades),
        win_rate=float(win_rate),
        final_equity=final_equity,
        n_days=int(len(strat_return)),
        span_start=strat_return.index[0],
        span_end=strat_return.index[-1],
    )


# ---------- Public API ----------

def backtest_stateful(oof: pd.DataFrame, prices: pd.Series,
                      *, asset: str, model: str,
                      transaction_cost: float = DEFAULT_TRANSACTION_COST
                      ) -> tuple[BacktestResult, pd.Series]:
    pos = _stateful_positions(oof["pred_label"])
    sret, tret, nt = _backtest_position_series(pos, prices, transaction_cost)
    res = _summary_from_returns(sret, tret, nt, prices, "stateful", asset, model)
    eq = (1 + sret).cumprod()
    return res, eq


def backtest_defensive(oof: pd.DataFrame, prices: pd.Series,
                       *, asset: str, model: str,
                       transaction_cost: float = DEFAULT_TRANSACTION_COST
                       ) -> tuple[BacktestResult, pd.Series]:
    pos = _defensive_positions(oof["pred_label"])
    sret, tret, nt = _backtest_position_series(pos, prices, transaction_cost)
    res = _summary_from_returns(sret, tret, nt, prices, "defensive", asset, model)
    eq = (1 + sret).cumprod()
    return res, eq


def backtest_prob_weighted(oof: pd.DataFrame, prices: pd.Series,
                           *, asset: str, model: str,
                           transaction_cost: float = DEFAULT_TRANSACTION_COST
                           ) -> tuple[BacktestResult, pd.Series]:
    pos = _prob_weighted_positions(oof)
    sret, tret, nt = _backtest_position_series(pos, prices, transaction_cost)
    res = _summary_from_returns(sret, tret, nt, prices, "prob_weighted", asset, model)
    eq = (1 + sret).cumprod()
    return res, eq


def backtest_buy_and_hold(prices: pd.Series, span_start: pd.Timestamp,
                          span_end: pd.Timestamp, *, asset: str
                          ) -> tuple[BacktestResult, pd.Series]:
    """B&H over the OOF span. No transaction cost (one buy)."""
    p = prices.loc[span_start:span_end]
    daily_ret = p.pct_change().fillna(0.0)
    eq = (1 + daily_ret).cumprod()
    final_equity = float(eq.iloc[-1])
    total_return = final_equity - 1.0
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_ret.std(ddof=1) > 0 else 0.0)
    running_max = eq.cummax()
    max_dd = float((eq / running_max - 1.0).min())

    res = BacktestResult(
        rule="buy_and_hold", asset=asset, model="benchmark",
        total_return=float(total_return),
        annualized_sharpe=float(sharpe),
        max_drawdown=max_dd,
        n_trades=1,
        win_rate=1.0 if total_return > 0 else 0.0,
        final_equity=final_equity,
        n_days=int(len(daily_ret)),
        span_start=p.index[0],
        span_end=p.index[-1],
    )
    return res, eq
