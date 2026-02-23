import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons

from .helper import alpha_beta
from .stock import Stock


class HistoricalPortfolioPlotMixin:
    """
    Plotting mixin for HistoricalPortfolio.

    Return:
    None
    """

    """
    Plots portfolio value over time.

    Parameters:
    None

    Return:
    None
    """
    def plot_portfolio_value(self):
        values = self._bt_port_value
        idx = self._bt_index

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        ax.plot(idx, values, linewidth=3, label="Portfolio Value", color="green")
        ax.set_ylabel("USD")
        ax.legend()
        ax.set_xlabel("Date")

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)


    """
    Plots portfolio returns over time against a benchmark.

    Parameters:
    bench (Stock) - Benchmark stock for performance comparison

    Return:
    None
    """
    def plot_performance_vs_bench(self, bench: Stock):
        if not hasattr(self, "_bt_index") or not hasattr(self, "_bt_prices"):
            raise ValueError("Run backtest() first so the backtest series are available.")

        idx = self._bt_index

        def normalize(series: pd.Series) -> pd.Series:
            s = series.astype(float).copy()
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.reindex(idx)
            s = s.dropna()
            if s.empty:
                return pd.Series(index=idx, dtype=float)
            base = float(s.iloc[0])
            if not np.isfinite(base) or np.isclose(base, 0.0):
                return pd.Series(index=idx, dtype=float)
            out = (s / base - 1) * 100
            return out.reindex(idx)
        

        portfolio_perf = normalize(self._bt_port_value)
        bench_perf = normalize(bench.prices)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        ax.plot(idx, portfolio_perf.values, linewidth=3, label="Portfolio", color="green")
        ax.plot(idx, bench_perf.values, linewidth=2, linestyle="--", label=f"{bench.ticker} Benchmark", color="orange")

        ax.set_title("Portfolio Performance vs Benchmark")
        ax.set_ylabel("Performance (%)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)



    """
    Plots portfolio performance (always visible) and allows selecting up to
    five comparison assets via a selector. All series are shown as % since start.
    Alpha, beta and R² are computed for the selected series versus the portfolio.
    The default selection shows no comparison series.

    Parameters:
    max_assets (int) - Maximum number of selectable comparison assets
    trading_days (int) - Trading days per year

    Return:
    None
    """
    def plot_performance(
        self,
        max_assets=5,
        trading_days=252
    ):
        if not hasattr(self, "_bt_index"):
            raise ValueError("Run backtest() first so the backtest series are available.")

        if not hasattr(self, "_bt_prices"):
            raise ValueError("Missing _bt_prices. Store prices_df as self._bt_prices in backtest().")

        idx = self._bt_index

        def normalize(series: pd.Series) -> pd.Series:
            s = series.astype(float).copy()
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.reindex(idx)
            s = s.dropna()
            if s.empty:
                return pd.Series(index=idx, dtype=float)
            base = float(s.iloc[0])
            if not np.isfinite(base) or np.isclose(base, 0.0):
                return pd.Series(index=idx, dtype=float)
            out = (s / base - 1) * 100
            return out.reindex(idx)

        portfolio_perf = normalize(self._bt_port_value)
        if portfolio_perf.isna().all():
            raise ValueError("Portfolio series is empty after normalization/alignment.")

        asset_perfs = {}
        for col in self._bt_prices.columns:
            perf = normalize(self._bt_prices[col])
            if not perf.isna().all():
                asset_perfs[col] = perf

        asset_names = list(asset_perfs.keys())[:max_assets]
        selectable = ["None"] + asset_names

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        port_line, = ax.plot(idx, portfolio_perf.values, linewidth=3, label="Portfolio")

        comp_line, = ax.plot(idx, portfolio_perf.values, linewidth=2, linestyle="--", label="", visible=False)

        info_text = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="#a8d6ff", alpha=0.85)
        )

        def update_legend():
            handles = [port_line]
            labels = ["Portfolio"]
            if comp_line.get_visible() and comp_line.get_label():
                handles.append(comp_line)
                labels.append(comp_line.get_label())
            ax.legend(handles, labels, loc="upper right")

        def autoscale_y():
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        def update_stats(name: str):
            if name == "None":
                info_text.set_text("")
                return

            s = asset_perfs[name].dropna()
            p = portfolio_perf.dropna()
            common = s.index.intersection(p.index)
            s = s.loc[common]
            p = p.loc[common]

            if len(common) < 5:
                info_text.set_text(f"{name} vs Portfolio\nNot enough overlapping data.")
                return

            stats = alpha_beta(
                ticker="Portfolio",
                benchmark_ticker=name,
                ticker_price=p,
                bench_price=s,
                trading_days=trading_days
            )

            info_text.set_text(
                f"Portfolio vs {name}\n"
                f"Alpha (ann): {stats['alpha_annual']:.2%}\n"
                f"Beta: {stats['beta']:.3f}\n"
                f"R²: {stats['r2']:.3f}"
            )

        rax = ax.inset_axes([0.78, 0.05, 0.2, 0.25])
        rax.set_facecolor("#a1d0f8")
        radio = RadioButtons(rax, selectable)

        def on_select(label: str):
            if label == "None":
                comp_line.set_visible(False)
                comp_line.set_label("")
                update_stats("None")
            else:
                series = asset_perfs[label]
                comp_line.set_ydata(series.values)
                comp_line.set_label(label)
                comp_line.set_visible(True)
                update_stats(label)

            ax.set_title("Portfolio Performance vs Selected Asset")
            ax.set_ylabel("Performance (%)")
            update_legend()
            autoscale_y()
            fig.canvas.draw_idle()

        radio.on_clicked(on_select)

        on_select("None")
        ax.set_xlabel("Date")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    """
    Plots the weights of the different assets inside the portfolio.

    Parameters:
    None

    Return:
    None
    """
    def plot_portfolio_weights(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")

        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        stack_labels = list(self.weights_df) + ["Cash"]
        stack_data = [self.weights_df[c].values for c in self.weights_df.columns] + [self.weights_cash.values]
        ax.stackplot(self._bt_index, *stack_data, labels=stack_labels, alpha=0.6)

        ax.set_title("Portfolio Weights (stocks + cash)")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", ncol=2)

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    """
    Plots the returns of the portfolio. Also the the returns of the underlying assets are shown

    Parameters:
    None

    Return:
    None
    """
    def plot_portfolio_returns(self):
        """
        Plots daily returns of the portfolio and optionally toggles the underlying stock returns.

        Parameters:
        None

        Return:
        None
        """
        if not hasattr(self, "_bt_index") or not hasattr(self, "port_returns") or not hasattr(self, "asset_returns_df"):
            raise ValueError("Run backtest() first.")

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#8ebff3")

        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        port_line, = ax.plot(
            self._bt_index,
            self.port_returns.values,
            label="Portfolio Return (%)",
            linewidth=2.2,
            color="green"
        )

        asset_lines = []
        for col in self.asset_returns_df.columns:
            ln, = ax.plot(
                self._bt_index,
                self.asset_returns_df[col].values,
                label=f"{col} Return (%)",
                linewidth=0.6,
                visible=False
            )
            asset_lines.append(ln)

        ax.axhline(0, color="white", alpha=0.35, linewidth=1)
        ax.set_title("Daily Returns (portfolio + each stock)")
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Date")

        rax = ax.inset_axes([0.77, 0.03, 0.2, 0.22])
        rax.set_facecolor("#a4d1f8")
        check = CheckButtons(rax, ["Show assets"], [False])

        state = {"show_assets": False}

        def update_legend():
            handles = [port_line]
            labels = [port_line.get_label()]
            if state["show_assets"]:
                for ln in asset_lines:
                    handles.append(ln)
                    labels.append(ln.get_label())
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.legend(handles, labels, loc="upper left", ncol=2)

        def toggle(label):
            if label == "Show assets":
                state["show_assets"] = not state["show_assets"]
                for ln in asset_lines:
                    ln.set_visible(state["show_assets"])
                update_legend()
                fig.canvas.draw_idle()

        check.on_clicked(toggle)

        update_legend()
        fig.autofmt_xdate(rotation=30)

        plt.show(block=True)
        plt.close(fig)
