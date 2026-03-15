"""
Backtesting module.
Simulates trading based on model predictions with transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class Backtester:
    """
    Simple long-only backtester for evaluating trading signals.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        transaction_fee: float | None = None,
        slippage: float | None = None,
    ):
        config = cfg()
        bt = config["backtesting"]
        self.initial_capital = initial_capital or bt["initial_capital"]
        self.transaction_fee = transaction_fee or bt["transaction_fee"]
        self.slippage = slippage or bt["slippage"]

    def run(
        self,
        signals: pd.Series,
        close_prices: pd.Series,
    ) -> dict:
        """
        Run backtest with given signals and prices.

        Parameters
        ----------
        signals : pd.Series
            Trading signals (Buy/Sell/Hold) indexed by date.
        close_prices : pd.Series
            Close prices indexed by date.

        Returns
        -------
        dict with backtest results.
        """
        common_idx = signals.index.intersection(close_prices.index)
        signals = signals.loc[common_idx]
        prices = close_prices.loc[common_idx]

        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long
        entry_price = 0
        trades = []
        equity_curve = [capital]
        dates = [prices.index[0]]

        for i in range(len(signals)):
            signal = signals.iloc[i]
            price = prices.iloc[i]

            if signal == "Buy" and position == 0:
                # Enter long position
                cost = self.transaction_fee + self.slippage
                entry_price = price * (1 + cost)
                position = 1
                shares = capital / entry_price
                trades.append({
                    "date": signals.index[i],
                    "type": "BUY",
                    "price": price,
                    "entry_price_adj": entry_price,
                })

            elif signal == "Sell" and position == 1:
                # Exit long position
                cost = self.transaction_fee + self.slippage
                exit_price = price * (1 - cost)
                pnl = (exit_price - entry_price) / entry_price
                capital = capital * (1 + pnl)
                position = 0
                trades.append({
                    "date": signals.index[i],
                    "type": "SELL",
                    "price": price,
                    "exit_price_adj": exit_price,
                    "pnl_pct": pnl * 100,
                })

            # Track equity
            if position == 1:
                unrealized = (price - entry_price) / entry_price
                equity = capital * (1 + unrealized)
            else:
                equity = capital

            equity_curve.append(equity)
            dates.append(signals.index[i])

        equity_series = pd.Series(equity_curve[1:], index=dates[1:], name="equity")

        results = self._compute_metrics(equity_series, trades, prices)
        results["equity_curve"] = equity_series
        results["trades"] = pd.DataFrame(trades) if trades else pd.DataFrame()

        return results

    def run_buy_and_hold(self, close_prices: pd.Series) -> pd.Series:
        """Compute buy-and-hold equity curve for benchmark."""
        returns = close_prices.pct_change().fillna(0)
        equity = self.initial_capital * (1 + returns).cumprod()
        equity.name = "buy_and_hold"
        return equity

    def _compute_metrics(
        self,
        equity: pd.Series,
        trades: list[dict],
        prices: pd.Series,
    ) -> dict:
        """Compute backtesting performance metrics."""
        returns = equity.pct_change().dropna()
        n_days = len(equity)
        n_years = n_days / 252

        # Total return
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Maximum drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        # Trade statistics
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        n_trades = len(sell_trades)
        if n_trades > 0:
            wins = [t for t in sell_trades if t.get("pnl_pct", 0) > 0]
            win_rate = len(wins) / n_trades

            gross_profit = sum(t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) > 0)
            gross_loss = abs(sum(t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) < 0))
            profit_factor = gross_profit / max(gross_loss, 0.01)
        else:
            win_rate = 0
            profit_factor = 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "transaction_fee": self.transaction_fee,
            "slippage": self.slippage,
        }

        logger.info(
            f"Backtest: return={total_return:.2%}, sharpe={sharpe:.2f}, "
            f"maxDD={max_drawdown:.2%}, trades={n_trades}, win_rate={win_rate:.2%}"
        )

        return metrics

    @staticmethod
    def plot_equity_curves(
        strategy_equity: pd.Series,
        benchmark_equity: pd.Series | None = None,
        title: str = "Equity Curve",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot equity curves for strategy vs benchmark."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(strategy_equity.index, strategy_equity.values, label="Strategy", linewidth=1.5)
        if benchmark_equity is not None:
            ax.plot(benchmark_equity.index, benchmark_equity.values,
                    label="Buy & Hold", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
