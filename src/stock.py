import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta


class StockCore:
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
    Computes moving averages for the stock price.

    Parameters:
    windows (list[int]) - Moving average window lengths in trading days
    ma_type (str) - Moving average type ("sma" or "ema")

    Return:
    pd.DataFrame - DataFrame containing moving averages for each window
    """
    def compute_price_moving_averages(
        self,
        windows=[5, 20, 50, 200],
        ma_type="sma"
    ) -> pd.DataFrame:
        self._ensure_loaded()

        if not windows:
            raise ValueError("windows must not be empty.")

        def MA(series, w):
            if ma_type == "ema":
                return series.ewm(span=w, adjust=False).mean()
            if ma_type == "sma":
                return series.rolling(w).mean()
            raise ValueError("ma_type must be 'sma' or 'ema'.")

        ma_df = pd.DataFrame(index=self.prices.index)

        for w in windows:
            ma_df[f"{ma_type.upper()} {w}d"] = MA(self.prices, w)

        return ma_df

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


class Stock(StockCore):
    """
    Convenience class for user imports.

    Return:
    None
    """
    pass
