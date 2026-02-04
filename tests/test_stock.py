import numpy as np
import pytest
from src.stock import Stock
import pandas as pd


@pytest.fixture
def fake_price_df():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame({
        "Adj Close": np.linspace(100, 109, len(idx)),
        "Close": np.linspace(100, 109, len(idx)),
        "Volume": np.arange(len(idx)) + 1000
    }, index=idx)


@pytest.fixture
def fake_download(monkeypatch, fake_price_df):
    import yfinance as yf

    def _fake_download(ticker, start=None, end=None, progress=False):
        return fake_price_df.copy()

    monkeypatch.setattr(yf, "download", _fake_download)
    return _fake_download


def test_fetch_prices_returns_series(fake_download):
    s = Stock.fetch_prices("AAPL", "2020-01-01", "2020-01-31", price_field="Adj Close")
    assert isinstance(s, pd.Series)
    assert s.dtype == float
    assert s.index.is_monotonic_increasing
    assert s.isna().sum() == 0
    assert len(s) > 0


def test_fetch_prices_fallback_to_close(monkeypatch, fake_price_df):
    import yfinance as yf
    df = fake_price_df.drop(columns=["Adj Close"])

    def _fake_download(*args, **kwargs):
        return df.copy()

    monkeypatch.setattr(yf, "download", _fake_download)

    s = Stock.fetch_prices("AAPL", "2020-01-01", "2020-01-31", price_field="Adj Close")
    assert "Adj Close" not in df.columns
    assert isinstance(s, pd.Series)
    assert len(s) == len(df)


def test_fetch_prices_raises_on_empty(monkeypatch):
    import yfinance as yf

    def _fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", _fake_download)

    with pytest.raises(ValueError, match="No data downloaded"):
        Stock.fetch_prices("AAPL", "2020-01-01", "2020-01-31")


def test_fetch_prices_raises_if_no_price_columns(monkeypatch):
    import yfinance as yf
    idx = pd.date_range("2020-01-01", periods=5, freq="B")

    def _fake_download(*args, **kwargs):
        return pd.DataFrame({"Open": [1,2,3,4,5]}, index=idx)

    monkeypatch.setattr(yf, "download", _fake_download)

    with pytest.raises(ValueError, match="Price field not found"):
        Stock.fetch_prices("AAPL", "2020-01-01", "2020-01-31")


def test_stock_init_loads_prices_and_returns(fake_download):
    s = Stock(ticker="AAPL", start_date="2020-01-01", end_date="2020-02-01")
    assert s.prices is not None and len(s.prices) > 0
    assert s.returns is not None and len(s.returns) == len(s.prices) - 1


def test_yearly_mu_sigma(fake_download):
    s = Stock(ticker="AAPL", start_date="2020-01-01", end_date="2020-02-01")

    mu_annual, sigma_annual = s.yearly_mu_sigma()
    assert isinstance(mu_annual, float)
    assert isinstance(sigma_annual, float)
    assert sigma_annual >= 0


def test_get_price_exact(fake_download):
    s = Stock("AAPL", "2020-01-01", "2020-02-01")
    d = str(s.prices.index[0].date())
    val = s.get_price_on_date(d, method="exact")
    assert isinstance(val, float)


def test_get_price_exact_missing_raises(fake_download):
    s = Stock("AAPL", "2020-01-01", "2020-02-01")
    with pytest.raises(ValueError):
        s.get_price_on_date("1999-01-01", method="exact")


def test_get_price_nearest(fake_download):
    s = Stock("AAPL", "2020-01-01", "2020-02-01")
    val = s.get_price_on_date("2020-01-04", method="nearest")
    assert isinstance(val, float)


def test_get_return_nearest(fake_download):
    s = Stock("AAPL", "2020-01-01", "2020-02-01")
    val = s.get_return_on_date("2020-01-04", method="nearest")
    assert isinstance(val, float)
