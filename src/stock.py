import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, timedelta
from matplotlib.widgets import CheckButtons

class Stock:
    """
    Initializes the Stock object and loads historical price and return data.

    Parameters:
    ticker (str) - Stock symbol
    start_date (str) - Start date
    end_date (str | None) - End date
    price_field (str) - Price column to use
    risk_free_rate (float) - Risk-free rate

    Return:
    None
    """
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str | None = None,
        price_field: str = "Adj Close",
        risk_free_rate: float = 0.02
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date is not None else (date.today() - timedelta(days=1)).isoformat()
        self.price_field = price_field
        self.risk_free_rate = float(risk_free_rate)

        self.prices = self._fetch_prices_self()
        if self.prices.empty:
            raise ValueError("No price data loaded for the given ticker and date range.")
        self.returns = self.prices.pct_change().dropna()

    """
    Downloads historical price data and returns a cleaned price series.

    Parameters:
    ticker (str) - Stock symbol
    start (str) - Start date
    end (str) - End date
    price_field (str) - Price column to use

    Return:
    pd.Series - Cleaned price series
    """
    @staticmethod
    def fetch_prices(ticker: str, start: str, end: str, price_field: str = "Adj Close") -> pd.Series:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data downloaded for {ticker}.")

        if price_field in df.columns:
            s = df[price_field]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("Price field not found in downloaded data.")

        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 0:
                raise ValueError("Downloaded price data has 0 columns after selection.")
            s = s.iloc[:, 0]

        s = s.dropna().astype(float)
        if s.empty:
            raise ValueError("Price series is empty.")
        return s.sort_index()

    """
    Fetches and stores price and return data using the object configuration.

    Parameters:
    None

    Return:
    pd.Series - Loaded price series
    """
    def _fetch_prices_self(self) -> pd.Series:
        if not self.start_date or not self.end_date:
            raise ValueError("start_date and end_date must be set on the Stock object.")
        self.prices = self.fetch_prices(self.ticker, self.start_date, self.end_date, self.price_field)
        self.returns = self.prices.pct_change().dropna()
        return self.prices

    """
    Ensures that price and return data are loaded before accessing them.

    Parameters:
    None

    Return:
    None
    """
    def _ensure_loaded(self):
        if self.prices is None or self.returns is None:
            if not self.start_date or not self.end_date:
                raise ValueError("Prices not loaded. Provide start_date/end_date or call fetch_prices_self().")
            self._fetch_prices_self()

    """
    Returns the stock price for a given date.

    Parameters:
    date (str) - Target date
    method (str) - Lookup method

    Return:
    float - Stock price
    """
    def get_price_on_date(self, date: str, method: str = "exact") -> float:
        self._ensure_loaded()
        d = pd.to_datetime(date)

        if method == "exact":
            if d not in self.prices.index:
                raise ValueError(f"Date {date} not found in price data.")
            return float(self.prices.loc[d])

        idx = self.prices.index.get_indexer([d], method=method)[0]
        if idx == -1:
            raise ValueError(f"Could not locate date {date} with method='{method}'.")
        return float(self.prices.iloc[idx])

    """
    Returns the stock return for a given date.

    Parameters:
    date (str) - Target date
    method (str) - Lookup method

    Return:
    float - Daily return
    """
    def get_return_on_date(self, date: str, method: str = "exact") -> float:
        self._ensure_loaded()
        d = pd.to_datetime(date)

        if method == "exact":
            if d not in self.returns.index:
                raise ValueError(f"Date {date} not found in return data.")
            return float(self.returns.loc[d])

        idx = self.returns.index.get_indexer([d], method=method)[0]
        if idx == -1:
            raise ValueError(f"Could not locate date {date} with method='{method}'.")
        return float(self.returns.iloc[idx])

    """
    Calculates annualized mean return and annualized volatility.

    Parameters:
    None

    Return:
    tuple[float, float] - (Annual return, annual volatility)
    """
    def yearly_mu_sigma(self) -> tuple[float, float]:
        self._ensure_loaded()
        mu_daily = float(self.returns.mean())
        sigma_daily = float(self.returns.std())

        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)

        return mu_annual, sigma_annual

    """
    Computes alpha, beta, correlation, and R² versus a benchmark.

    Parameters:
    benchmark_ticker (str) - Benchmark symbol
    start_date (str | None) - Start date
    end_date (str | None) - End date
    trading_days (int) - Trading days per year

    Return:
    dict - Performance statistics
    """
    def alpha_beta(
        self,
        benchmark_ticker: str = "SPY",
        start_date: str | None = None,
        end_date: str | None = None,
        trading_days: int = 252,
    ) -> dict:
        start = start_date or self.start_date
        end = end_date or self.end_date
        if start is None or end is None:
            raise ValueError("start_date/end_date missing. Provide them or set them on the Stock object.")

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if end_dt <= start_dt:
            raise ValueError("end_date must be after start_date.")

        asset_prices = self.fetch_prices(self.ticker, str(start_dt.date()), str(end_dt.date()), self.price_field)
        bench_prices = self.fetch_prices(benchmark_ticker, str(start_dt.date()), str(end_dt.date()), self.price_field)

        if isinstance(asset_prices, pd.DataFrame):
            asset_prices = asset_prices.iloc[:, 0]
        if isinstance(bench_prices, pd.DataFrame):
            bench_prices = bench_prices.iloc[:, 0]

        asset_ret = asset_prices.pct_change()
        bench_ret = bench_prices.pct_change()

        df = pd.concat(
            [asset_ret.rename("asset"), bench_ret.rename("bench")],
            axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 5:
            raise ValueError(f"Not enough clean overlapping return data (n={len(df)}).")

        x = df["bench"].to_numpy()
        y = df["asset"].to_numpy()

        var_x = float(np.var(x, ddof=1))
        if np.isnan(var_x) or np.isclose(var_x, 0.0):
            raise ValueError("Benchmark returns variance is ~0 or NaN; beta undefined for this window.")

        beta, alpha_daily = np.polyfit(x, y, 1)
        alpha_annual = float(alpha_daily * trading_days)

        corr = float(np.corrcoef(y, x)[0, 1])
        r2 = float(corr**2) if np.isfinite(corr) else np.nan

        return {
            "ticker": self.ticker,
            "benchmark": benchmark_ticker,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "alpha_daily": float(alpha_daily),
            "alpha_annual": alpha_annual,
            "beta": float(beta),
            "correlation": corr,
            "r2": r2,
            "n_obs": int(len(df)),
        }

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
                stats = self.alpha_beta(benchmark_ticker, self.start_date, self.end_date)
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
        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.12
        )
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
                ax.plot(bench_ret_aligned.index, bench_ret_aligned.values, color="orange", label=f"{benchmark_ticker} Daily Returns")
                show_stats = True
            else:
                asset_ret_aligned = asset_ret
                bench_ret_aligned = None

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
                stats = self.alpha_beta(benchmark_ticker, self.start_date, self.end_date)
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

        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.12
        )
        plt.show(block=True)
        plt.close(fig)

    """
    Returns a string representation of the Stock object.

    Parameters:
    None

    Return:
    str - Object description
    """
    def __str__(self):
        s = f"Stock(ticker={self.ticker}, start_date={self.start_date}, end_date={self.end_date}, price_field={self.price_field}, risk_free_rate={self.risk_free_rate})"
        if self.prices is not None and len(self.prices) > 0:
            s += f"\n  Loaded prices: {len(self.prices)} rows ({self.prices.index[0].date()} → {self.prices.index[-1].date()})"
        return s
