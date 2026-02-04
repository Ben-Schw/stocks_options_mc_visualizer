import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import MonteCarlo, MonteCarloStock, MonteCarloOption


@pytest.fixture
def mc_base():
    return MonteCarlo(
        S0=100.0,
        r=0.08,
        sigma=0.2,
        days=10,
        simulations=500,
        risk_free_rate=0.02,
        trading_days=252,
    )


def test_resolve_drift_historical(mc_base):
    assert mc_base.resolve_drift("historical") == pytest.approx(0.08)


def test_resolve_drift_risk_free(mc_base):
    assert mc_base.resolve_drift("risk_free") == pytest.approx(0.02)


def test_resolve_drift_zero(mc_base):
    assert mc_base.resolve_drift("zero") == pytest.approx(0.0)


def test_resolve_drift_invalid_raises(mc_base):
    with pytest.raises(ValueError):
        mc_base.resolve_drift("invalid_mode")


def test_simulate_price_paths_shape_and_finiteness(mc_base):
    paths = mc_base.simulate_price_paths(drift_mode="risk_free", seed=123)

    assert isinstance(paths, np.ndarray)
    assert paths.shape == (mc_base.steps + 1, mc_base.simulations)
    assert np.isfinite(paths).all()
    assert (paths > 0).all()


def test_simulate_price_paths_reproducible(mc_base):
    p1 = mc_base.simulate_price_paths(seed=42)
    p2 = mc_base.simulate_price_paths(seed=42)

    assert np.array_equal(p1, p2)


def test_simulate_price_paths_different_seeds_differ(mc_base):
    p1 = mc_base.simulate_price_paths(seed=1)
    p2 = mc_base.simulate_price_paths(seed=2)

    assert not np.array_equal(p1, p2)


@pytest.fixture
def fake_stock_df():
    idx = pd.date_range("2020-01-01", periods=15, freq="B")
    return pd.DataFrame(
        {
            "Adj Close": np.linspace(100, 114, len(idx)),
            "Close": np.linspace(100, 114, len(idx)),
            "Volume": np.arange(len(idx)) + 1000,
        },
        index=idx,
    )


@pytest.fixture
def fake_stock(monkeypatch, fake_stock_df):
    from src.stock import Stock

    def _fake_fetch_prices(ticker, start, end, price_field="Adj Close"):
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        s = fake_stock_df[price_field].copy()
        return s.loc[(s.index >= start_dt) & (s.index <= end_dt)]

    monkeypatch.setattr(Stock, "fetch_prices", staticmethod(_fake_fetch_prices))

    return Stock(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2020-01-31",
        price_field="Adj Close",
    )


def test_monte_carlo_stock_init_no_mid_date(fake_stock):
    mcs = MonteCarloStock(fake_stock, horizon_days=10, mid_date=None, simulations=200)

    assert isinstance(mcs, MonteCarloStock)
    assert mcs.simulations == 200
    assert mcs.steps == 10
    assert mcs.S0 == pytest.approx(float(fake_stock.prices.iloc[-1]))


def test_monte_carlo_stock_init_with_mid_date(fake_stock):
    mid = fake_stock.prices.index[5]
    mid_date = str(mid.date())

    mcs = MonteCarloStock(fake_stock, mid_date=mid_date, simulations=100)

    end = pd.to_datetime(fake_stock.end_date)
    n_true = fake_stock.prices.loc[mid:end].shape[0]

    assert mcs.steps + 1 == n_true
    assert mcs.S0 == pytest.approx(float(fake_stock.prices.loc[mid]))


@pytest.fixture
def fake_option():
    class DummyOption:
        def __init__(self):
            self.S = 100.0
            self.r = 0.02
            self.sigma = 0.25
            self.T = 1.0
            self.K = 100.0
            self.option_type = "call"
            self.premium = 10.0

    return DummyOption()


def test_monte_carlo_option_init(fake_option):
    mco = MonteCarloOption(fake_option, drift_mode="risk_free", simulations=300)

    assert isinstance(mco, MonteCarloOption)
    assert mco.S0 == pytest.approx(100.0)
    assert mco.simulations == 300
    assert mco.steps == int(fake_option.T * 252)


def test_estimate_option_price_mc_returns_float(fake_option):
    mco = MonteCarloOption(fake_option, simulations=200)

    price = mco.estimate_option_price_mc(n_paths=5000, steps=50, seed=123)

    assert isinstance(price, float)
    assert np.isfinite(price)


def test_plot_option_payoff_returns_dict(fake_option, monkeypatch):
    mco = MonteCarloOption(fake_option, simulations=200)

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

    out = mco.plot_option_payoff(n_paths=1000, steps=30, seed=123, discount=True)

    assert isinstance(out, dict)
    assert {"ST", "payoffs", "values", "cum_mean", "mc_estimate", "bs_price"} <= set(out.keys())
    assert len(out["ST"]) == 1000
    assert np.isfinite(out["mc_estimate"])
