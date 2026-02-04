import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

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
    benchmark_ticker (str | None) - Benchmark symbol
    percentage (bool) - Plot percentage change
    cummulative (bool) - Show cumulative max

    Return:
    None
    """
    def plot_prices(self, benchmark_ticker: str | None = None, percentage: bool = False, cummulative: bool = False):
        self._ensure_loaded()

        plt.figure(figsize=(10, 5))

        if percentage:
            asset_pct = (self.prices / self.prices.iloc[0] - 1) * 100
            plt.plot(asset_pct.index, asset_pct.values, label=self.ticker, color="blue")

            title = f"{self.ticker} Price Change (%) ({self.start_date} to {self.end_date})"

            if benchmark_ticker:
                benchmark_prices = self.fetch_prices(
                    benchmark_ticker,
                    self.start_date,
                    self.end_date,
                    self.price_field
                )
                bench_pct = (benchmark_prices / benchmark_prices.iloc[0] - 1) * 100

                plt.plot(
                    bench_pct.index,
                    bench_pct.values,
                    label=benchmark_ticker,
                    color="orange"
                )

                if cummulative:
                    rolling_max = self.prices.cummax()
                    plt.plot(asset_pct.index, ((rolling_max / rolling_max.iloc[0] - 1) * 100).values, label=f"{self.ticker} Cummulative Max", color="blue", linestyle="--")

                stats = self.alpha_beta(benchmark_ticker, self.start_date, self.end_date)

                alpha_ann = stats["alpha_annual"]
                beta = stats["beta"]
                r2 = stats["r2"]

                title = f"{self.ticker} vs {benchmark_ticker} Prices ({self.start_date} to {self.end_date})"

                textstr = (
                    f"Alpha (ann): {alpha_ann:.2%}\n"
                    f"Beta: {beta:.3f}\n"
                    f"R²: {r2:.3f}\n"
                )

                plt.gca().text(
                    0.02, 0.98,
                    textstr,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            plt.plot(self.prices.index, self.prices.values, label=self.ticker, color="blue")

            title = f"{self.ticker} Prices ({self.start_date} to {self.end_date})"

            if benchmark_ticker:
                benchmark_prices = self.fetch_prices(
                    benchmark_ticker,
                    self.start_date,
                    self.end_date,
                    self.price_field
                )

                plt.plot(
                    benchmark_prices.index,
                    benchmark_prices.values,
                    label=benchmark_ticker,
                    color="orange"
                )

                if cummulative:
                    rolling_max = self.prices.cummax()
                    plt.plot(self.prices.index, rolling_max.values, label=f"{self.ticker} Cummulative Max", color="blue", linestyle="--")

                stats = self.alpha_beta(benchmark_ticker, self.start_date, self.end_date)

                alpha_ann = stats["alpha_annual"]
                beta = stats["beta"]
                r2 = stats["r2"]

                title = f"{self.ticker} vs {benchmark_ticker} Prices ({self.start_date} to {self.end_date})"

                textstr = (
                    f"Alpha (ann): {alpha_ann:.2%}\n"
                    f"Beta: {beta:.3f}\n"
                    f"R²: {r2:.3f}\n"
                )

                plt.gca().text(
                    0.02, 0.98,
                    textstr,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )

            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

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
        plt.figure(figsize=(10, 4))
        plt.plot(drawdown.index, drawdown.values, color="red")
        plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        plt.title(f"{self.ticker} Drawdown ({self.start_date} to {self.end_date})")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

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
