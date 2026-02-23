import numpy as np
import pandas as pd
from datetime import date, timedelta


"""
Computes summary statistics (paths and returns) across simulated price paths.

Parameters:
S0 (float) - Initial price used to convert paths into returns
paths (np.ndarray) - Simulated price paths (time x simulations)

Return:
dict[str, np.ndarray] - Dictionary of summary arrays (per time step)
"""
def price_paths_macros(S0: float, paths: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mean_path": np.mean(paths, axis=1),
            "median_path": np.median(paths, axis=1),
            "std_dev_path": np.std(paths, axis=1),
            "mean_return": np.mean(paths, axis=1) / S0 - 1,
            "median_return": np.median(paths, axis=1) / S0 - 1,
            "std_dev_return": np.std(paths, axis=1) / S0,
            "25th_percentile": np.percentile(paths, 25, axis=1),
            "75th_percentile": np.percentile(paths, 75, axis=1),
            "5th_percentile": np.percentile(paths, 5, axis=1),
            "95th_percentile": np.percentile(paths, 95, axis=1)
        }


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
    ticker: str,
    benchmark_ticker: str,
    ticker_price: pd.Series,
    bench_price: pd.Series,
    trading_days = 252
) -> dict:
    asset_prices = ticker_price
    bench_prices = bench_price

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
        "ticker": ticker,
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

class TimeSeriesAsset:
    """
    Wraps any asset into aligned time series for value and cashflows.

    Parameters:
    name (str) - Identifier used for columns.
    value (pd.Series) - Time series of mark-to-market values.
    cashflow (pd.Series) - Time series of cashflows paid on dates (0 otherwise).
    """
    def __init__(self, name: str, value: pd.Series, cashflow: pd.Series | None = None):
        self.ticker = str(name)
        v = value.copy()
        v.index = pd.to_datetime(v.index)
        self.value = v.sort_index().astype(float)

        if cashflow is None:
            cf = pd.Series(0.0, index=self.value.index)
        else:
            cf = cashflow.copy()
            cf.index = pd.to_datetime(cf.index)
            cf = cf.sort_index().astype(float)

        self.cashflow = cf.reindex(self.value.index).fillna(0.0)

    """
    Returns the asset value series aligned to a given index.

    Parameters:
    idx (pd.DatetimeIndex) - Target index.

    Return:
    pd.Series - Aligned value series.
    """
    def value_on_index(self, idx: pd.DatetimeIndex) -> pd.Series:
        s = self.value.reindex(idx)
        return s.ffill()

    """
    Returns the asset cashflow series aligned to a given index.

    Parameters:
    idx (pd.DatetimeIndex) - Target index.

    Return:
    pd.Series - Aligned cashflow series.
    """
    def cashflow_on_index(self, idx: pd.DatetimeIndex) -> pd.Series:
        return self.cashflow.reindex(idx).fillna(0.0)