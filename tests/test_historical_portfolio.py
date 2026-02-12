from src.stock import Stock
from src.historical_portfolio import HistoricalPortfolio


"""
Tests portfolio initialization.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_portfolio_init(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    stock = Stock("TEST", "2020-01-01", "2020-03-01")
    pf = HistoricalPortfolio({stock: 1.0}, cash=100)

    assert pf.value > 100


"""
Tests backtest output creation.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_backtest(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    stock = Stock("TEST", "2020-01-01", "2020-03-01")
    pf = HistoricalPortfolio()

    pf.backtest("2020-01-01", "2020-03-01", trading_stocks=[stock])

    assert hasattr(pf, "_bt_port_value")
    assert hasattr(pf, "_bt_prices")
