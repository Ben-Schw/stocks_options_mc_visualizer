import numpy as np
from src.monte_carlo import MonteCarlo, MonteCarloOption, MonteCarloStock
from src.option import EuropeanOption
from src.stock import Stock


"""
Tests GBM simulation shape.

Parameters:
None

Return:
None
"""
def test_simulate_price_paths_shape():
    mc = MonteCarlo(S0=100, r=0.05, sigma=0.2, days=50, simulations=200)
    paths = mc.simulate_price_paths()

    assert paths.shape == (51, 200)
    assert np.isfinite(paths).all()


"""
Tests drift resolver modes.

Parameters:
None

Return:
None
"""
def test_resolve_drift_modes():
    mc = MonteCarlo(S0=100, r=0.07, sigma=0.2, days=10, simulations=10, risk_free_rate=0.02)

    assert mc.resolve_drift("historical") == 0.07
    assert mc.resolve_drift("risk_free") == 0.02
    assert mc.resolve_drift("zero") == 0.0


"""
Tests Monte Carlo option pricing produces positive estimate.

Parameters:
None

Return:
None
"""
def test_mc_option_price_positive():
    opt = EuropeanOption("call", 100, 100, 1, sigma=0.2)
    mc = MonteCarloOption(opt, simulations=2000)

    price = mc.estimate_option_price_mc(n_paths=5000)

    assert price > 0
    assert np.isfinite(price)


"""
Tests MonteCarloStock initialization.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_mc_stock_initialization(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    stock = Stock("TEST", "2020-01-01", "2020-03-01")
    mc_stock = MonteCarloStock(stock, horizon_days=20, simulations=100)

    assert mc_stock.S0 > 0
    assert mc_stock.steps == 20
    assert mc_stock.simulations == 100


"""
Tests simulated paths for MonteCarloStock.

Parameters:
monkeypatch
yf_df (fixture)

Return:
None
"""
def test_mc_stock_paths(monkeypatch, yf_df):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: yf_df.copy())

    stock = Stock("TEST", "2020-01-01", "2020-03-01")
    mc_stock = MonteCarloStock(stock, horizon_days=20, simulations=200)

    paths = mc_stock.simulate_price_paths()

    assert paths.shape == (21, 200)
    assert np.isfinite(paths).all()
