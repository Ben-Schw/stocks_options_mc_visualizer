import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from .helper import alpha_beta


class StockPlotMixin:
    """
    Plotting mixin for Stock-like objects.

    Requirements:
    - self.ticker, self.start_date, self.end_date, self.price_field
    - self.prices (pd.Series), self.returns (pd.Series)
    - self._ensure_loaded()
    - self.fetch_prices(...)
    - self.compute_price_moving_averages(...)

    Return:
    None
    """

    """
    Plots price data and optionally compares it to a benchmark.

    Parameters:
    benchmark_ticker (str) - Benchmark symbol

    Return:
    None
    """
    def plot_prices(self, benchmark_ticker: str = "SPY"):
        self._ensure_loaded()

        benchmark_prices = None
        if benchmark_ticker:
            benchmark_prices = self.fetch_prices(
                benchmark_ticker,
                self.start_date,
                self.end_date,
                self.price_field
            )

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        state = {
            "percentage": False,
            "benchmark": False,
            "cumulative": False
        }

        plt.tight_layout(rect=[0, 0, 0.78, 1])

        info_text = ax.text(
            0.98, 0.98,
            "",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="#a8d6ff", alpha=0.85),
            visible=False
        )

        rax = ax.inset_axes([0.77, 0.03, 0.2, 0.22])
        rax.set_facecolor("#a1d0f8")
        labels = ["Percentage", "Benchmark", "Cumulative"]
        check = CheckButtons(rax, labels, [False, False, False])

        def recompute_plot():
            for line in list(ax.lines):
                line.remove()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            info_text.set_visible(False)
            info_text.set_text("")

            ax.set_facecolor("#0c89f7")
            ax.grid(alpha=0.4, color="white")
            ax.margins(x=0)

            if state["percentage"]:
                asset = (self.prices / self.prices.iloc[0] - 1) * 100
                ylabel = "Return (%)"
            else:
                asset = self.prices
                ylabel = "Price"

            ax.plot(asset.index, asset.values, color="blue", label=self.ticker)

            show_stats = False
            if state["benchmark"] and benchmark_prices is not None:
                if state["percentage"]:
                    bench = (benchmark_prices / benchmark_prices.iloc[0] - 1) * 100
                else:
                    bench = benchmark_prices
                ax.plot(bench.index, bench.values, color="orange", label=benchmark_ticker)
                show_stats = True

            if state["cumulative"]:
                rolling_max = self.prices.cummax()
                if state["percentage"]:
                    rolling_max = (rolling_max / rolling_max.iloc[0] - 1) * 100
                ax.plot(rolling_max.index, rolling_max.values, "--", color="blue", label="Cumulative Max")

            ax.set_title(f"{self.ticker} Prices ({self.start_date} to {self.end_date})")
            ax.set_xlabel("Date")
            ax.set_ylabel(ylabel)
            ax.legend()

            if show_stats:
                stats = alpha_beta(
                    self.ticker,
                    benchmark_ticker=benchmark_ticker,
                    ticker_price=self.prices,
                    bench_price=benchmark_prices
                )
                alpha_ann = stats["alpha_annual"]
                beta = stats["beta"]
                r2 = stats["r2"]

                info_text.set_text(
                    f"{self.ticker} vs {benchmark_ticker}\n"
                    f"Alpha (ann): {alpha_ann:.2%}\n"
                    f"Beta: {beta:.3f}\n"
                    f"R²: {r2:.3f}"
                )
                info_text.set_visible(True)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        def toggle(label):
            if label == "Percentage":
                state["percentage"] = not state["percentage"]
            elif label == "Benchmark":
                state["benchmark"] = not state["benchmark"]
            elif label == "Cumulative":
                state["cumulative"] = not state["cumulative"]

            recompute_plot()

        check.on_clicked(toggle)

        recompute_plot()
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)
        plt.show(block=True)
        plt.close(fig)

    """
    Plots the drawdown curve of the stock over time.

    Parameters:
    None

    Return:
    None
    """
    def plot_drawdown(self):
        self._ensure_loaded()

        rolling_max = self.prices.cummax()
        drawdown = (self.prices - rolling_max) / rolling_max

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.plot(drawdown.index, drawdown.values, color="red")
        ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)

        ax.set_title(f"{self.ticker} Drawdown ({self.start_date} to {self.end_date})")
        ax.set_xlim(drawdown.index.min(), drawdown.index.max())
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.4, color="white", linewidth=1)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    """
    Plots the daily returns of the stock (also with a benchmark):

    Parameters:
    benchmark_ticker (str) - Benchmark symbol

    Return:
    None
    """
    def plot_daily_returns(self, benchmark_ticker: str = "SPY"):
        self._ensure_loaded()

        benchmark_prices = None
        if benchmark_ticker:
            benchmark_prices = self.fetch_prices(
                benchmark_ticker,
                self.start_date,
                self.end_date,
                self.price_field
            )

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        state = {
            "benchmark": False,
            "cumulative": False,
            "hist": False
        }

        info_text = ax.text(
            0.98, 0.98,
            "",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="#8fc8fa", alpha=0.85),
            visible=False
        )

        def _daily_returns(series):
            return series.pct_change().dropna() * 100

        def _cumulative_from_returns(daily_ret_pct):
            growth = (1 + daily_ret_pct / 100.0).cumprod()
            return (growth - 1) * 100

        rax = ax.inset_axes([0.77, 0.03, 0.2, 0.22])
        rax.set_facecolor("#a4d1f8")
        labels = ["Benchmark", "Cumulative"]
        check = CheckButtons(rax, labels, [False, False])

        def recompute_plot():
            for line in list(ax.lines):
                line.remove()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            info_text.set_visible(False)
            info_text.set_text("")

            ax.set_facecolor("#0c89f7")
            ax.grid(alpha=0.4, color="white")
            ax.margins(x=0)

            asset_ret = _daily_returns(self.prices)

            ax.plot(asset_ret.index, asset_ret.values, color="blue", label=f"{self.ticker} Daily Returns")
            ax.axhline(0, color="white", alpha=0.35, linewidth=1)

            show_stats = False

            if state["benchmark"] and benchmark_prices is not None:
                bench_ret = _daily_returns(benchmark_prices)
                common = asset_ret.index.intersection(bench_ret.index)
                asset_ret_aligned = asset_ret.loc[common]
                bench_ret_aligned = bench_ret.loc[common]
                ax.plot(
                    bench_ret_aligned.index,
                    bench_ret_aligned.values,
                    color="orange",
                    label=f"{benchmark_ticker} Daily Returns"
                )
                show_stats = True

            if state["cumulative"]:
                asset_cum = _cumulative_from_returns(asset_ret)
                ax.plot(asset_cum.index, asset_cum.values, "--", color="blue", label=f"{self.ticker} Cumulative Return")

                if state["benchmark"] and benchmark_prices is not None:
                    bench_ret = _daily_returns(benchmark_prices)
                    bench_cum = _cumulative_from_returns(bench_ret)
                    ax.plot(bench_cum.index, bench_cum.values, "--", color="orange", label=f"{benchmark_ticker} Cumulative Return")

            ax.set_title(f"{self.ticker} Daily Returns ({self.start_date} to {self.end_date})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Daily Return (%)")
            ax.legend()

            if show_stats:
                stats = alpha_beta(
                    self.ticker,
                    benchmark_ticker=benchmark_ticker,
                    ticker_price=self.prices,
                    bench_price=benchmark_prices
                )
                alpha_ann = stats["alpha_annual"]
                beta = stats["beta"]
                r2 = stats["r2"]

                info_text.set_text(
                    f"{self.ticker} vs {benchmark_ticker}\n"
                    f"Alpha (ann): {alpha_ann:.2%}\n"
                    f"Beta: {beta:.3f}\n"
                    f"R²: {r2:.3f}"
                )
                info_text.set_visible(True)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        def toggle(label):
            if label == "Benchmark":
                state["benchmark"] = not state["benchmark"]
            elif label == "Cumulative":
                state["cumulative"] = not state["cumulative"]
            recompute_plot()

        check.on_clicked(toggle)
        recompute_plot()

        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)
        plt.show(block=True)
        plt.close(fig)

    """
    Plots the stock price together with multiple moving averages.

    Parameters:
    windows (list[int]) - Moving average window lengths in trading days
    ma_type (str) - Moving average type ("sma" or "ema")

    Return:
    None
    """
    def plot_price_moving_averages(
        self,
        windows=[5, 20, 50, 200],
        ma_type="sma"
    ):
        self._ensure_loaded()

        ma_df = self.compute_price_moving_averages(windows, ma_type)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")

        state = {col: False for col in ma_df.columns}

        price_line, = ax.plot(
            self.prices,
            linewidth=1.2,
            alpha=0.7,
            label=f"{self.ticker} Price",
            color="blue"
        )

        lines = {}
        for col in ma_df.columns:
            line, = ax.plot(
                ma_df.index,
                ma_df[col],
                linewidth=2.5,
                visible=False,
                label=col
            )
            lines[col] = line

        rax = ax.inset_axes([0.78, 0.03, 0.2, 0.25])
        rax.set_facecolor("#a1d0f8")
        labels = list(ma_df.columns)
        check = CheckButtons(rax, labels, [False] * len(labels))

        def update_legend():
            visible_lines = [price_line] + [l for l in lines.values() if l.get_visible()]
            labels_ = [l.get_label() for l in visible_lines]
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.legend(visible_lines, labels_)

        def toggle(label):
            ax.margins(x=0)
            state[label] = not state[label]
            lines[label].set_visible(state[label])
            update_legend()
            fig.canvas.draw_idle()

        check.on_clicked(toggle)

        ax.set_title(f"{self.ticker} Moving Averages")
        ax.set_ylabel("Price")
        ax.margins(x=0)
        update_legend()

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)
