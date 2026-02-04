import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.option import Option


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def test_init_rejects_invalid_option_type():
    with pytest.raises(ValueError, match="option_type must be"):
        Option(option_type="invalid", S=100, K=100, T=1.0, sigma=0.2)


def test_init_rejects_nonpositive_S_K_T():
    with pytest.raises(ValueError, match="S, K, T must be > 0"):
        Option(option_type="call", S=0, K=100, T=1.0, sigma=0.2)
    with pytest.raises(ValueError, match="S, K, T must be > 0"):
        Option(option_type="call", S=100, K=0, T=1.0, sigma=0.2)
    with pytest.raises(ValueError, match="S, K, T must be > 0"):
        Option(option_type="call", S=100, K=100, T=0, sigma=0.2)


def test_init_requires_sigma_or_premium():
    with pytest.raises(ValueError, match="Either sigma or premium must be provided"):
        Option(option_type="call", S=100, K=100, T=1.0)


def test_init_sigma_must_be_positive():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        Option(option_type="call", S=100, K=100, T=1.0, sigma=0.0)


def test_init_premium_must_be_positive_if_sigma_missing():
    with pytest.raises(ValueError, match="premium must be > 0"):
        Option(option_type="call", S=100, K=100, T=1.0, sigma=None, premium=0.0)


def test_price_call_put_parity_basic():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.25
    call = Option("call", S=S, K=K, T=T, r=r, sigma=sigma)
    put = Option("put", S=S, K=K, T=T, r=r, sigma=sigma)

    lhs = call.price() - put.price()
    rhs = S - K * np.exp(-r * T)
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-6, abs=1e-6)


def test_vega_positive_for_atm():
    opt = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=0.2)
    v = opt.vega()
    assert isinstance(v, float)
    assert v > 0.0


def test_delta_call_put_bounds():
    call = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=0.2)
    put = Option("put", S=100, K=100, T=1.0, r=0.02, sigma=0.2)

    dc = call.delta_hedge_ratio()
    dp = put.delta_hedge_ratio()

    assert 0.0 <= dc <= 1.0
    assert -1.0 <= dp <= 0.0


def test_implied_vol_recovers_sigma():
    true_sigma = 0.30
    opt = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=true_sigma)
    market_price = opt.price()

    opt2 = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=None, premium=market_price)
    assert opt2.sigma == pytest.approx(true_sigma, rel=1e-3, abs=1e-3)


def test_implied_vol_nonconvergence_raises():
    opt = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=0.2)
    with pytest.raises(ValueError, match="did not converge"):
        opt.implied_vol(market_price=1e9, initial_guess=0.2, tol=1e-12, max_iter=5)


def test_pnl_call_put_shapes_and_values():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.2
    call = Option("call", S=S, K=K, T=T, r=r, sigma=sigma)
    put = Option("put", S=S, K=K, T=T, r=r, sigma=sigma)

    ST = np.array([50, 100, 150], dtype=float)

    call_pnl = call.pnl(ST)
    put_pnl = put.pnl(ST)

    assert call_pnl.shape == ST.shape
    assert put_pnl.shape == ST.shape

    assert call_pnl[0] <= call_pnl[1] <= call_pnl[2]
    assert put_pnl[0] >= put_pnl[1] >= put_pnl[2]


def test_plot_pnl_runs(monkeypatch):
    opt = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=0.2)
    ST = np.linspace(50, 150, 10)
    opt.plot_pnl(ST)


def test_plot_heatmap_runs(monkeypatch):
    opt = Option("call", S=100, K=100, T=1.0, r=0.02, sigma=0.2)
    K_range = np.array([80, 100, 120], dtype=float)
    T_range = np.array([0.25, 0.5, 1.0], dtype=float)
    opt.plot_heatmap(K_range, T_range)


def test_str_contains_key_fields():
    opt = Option("call", S=100, K=110, T=0.5, r=0.01, sigma=0.25)
    s = str(opt)
    assert "option_type=" in s
    assert "stock_price=" in s
    assert "strike_price=" in s
    assert "time_to_maturity=" in s
    assert "risk_free_rate=" in s
    assert "volatility=" in s
