import numpy as np
import pandas as pd
from src.stock import Stock


"""
Tests whether fetch_prices returns a valid price series.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_fetch_prices_uses_adj_close(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    s = Stock.fetch_prices("TEST", "2020-01-01", "2020-03-01")
    assert isinstance(s, pd.Series)
    assert not s.empty
    assert np.isfinite(s.iloc[0])


"""
Tests Stock initialization including returns calculation.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_stock_initialization(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    st = Stock("TEST", "2020-01-01", "2020-03-01")

    assert st.prices is not None
    assert st.returns is not None
    assert len(st.returns) == len(st.prices) - 1


"""
Tests annualized mean return and volatility calculation.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_mu_sigma_sharpe(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    st = Stock("TEST", "2020-01-01", "2020-03-01")
    tot_mu, ann_mu, ann_sigma, sharpe = st.mu_sigma_sharpe()

    assert np.isfinite(tot_mu)
    assert np.isfinite(ann_mu)
    assert np.isfinite(ann_sigma)
    assert np.isfinite(sharpe)
    assert ann_sigma >= 0.0


"""
Tests moving average calculation.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_moving_average(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    st = Stock("TEST", "2020-01-01", "2020-03-01")
    ma = st.compute_price_moving_averages([5, 10])

    assert list(ma.columns) == ["SMA 5d", "SMA 10d"]
    assert len(ma) == len(st.prices)
