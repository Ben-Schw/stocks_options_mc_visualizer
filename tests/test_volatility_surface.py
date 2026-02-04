import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.option import Option
from src.volatility_surface import VolatilitySurface


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def opt_template():
    return Option(option_type="call", S=100.0, K=100.0, T=1.0, r=0.02, sigma=0.25)


def test_init_raises_on_bad_market_prices_shape(opt_template):
    K_range = np.array([90.0, 100.0, 110.0])
    T_range = np.array([0.5, 1.0])
    bad_prices = np.zeros((len(K_range), len(T_range) + 1))

    with pytest.raises(ValueError, match="market_prices must have shape"):
        VolatilitySurface(opt_template, K_range, T_range, bad_prices)


def test_compute_surface_recovers_known_sigma(opt_template):
    K_range = np.array([90.0, 100.0, 110.0], dtype=float)
    T_range = np.array([0.5, 1.0], dtype=float)

    true_sigma = 0.30
    market_prices = np.full((len(K_range), len(T_range)), np.nan, dtype=float)

    for i, K in enumerate(K_range):
        for j, T in enumerate(T_range):
            market_prices[i, j] = Option(
                option_type=opt_template.option_type,
                S=opt_template.S,
                K=float(K),
                T=float(T),
                r=opt_template.r,
                sigma=true_sigma,
            ).price()

    vs = VolatilitySurface(opt_template, K_range, T_range, market_prices)

    assert vs.surface.shape == (len(K_range), len(T_range))
    assert np.isfinite(vs.surface).all()
    assert np.nanmean(vs.surface) == pytest.approx(true_sigma, rel=1e-2, abs=1e-2)


def test_compute_surface_sets_nan_for_invalid_prices(opt_template):
    K_range = np.array([90.0, 100.0], dtype=float)
    T_range = np.array([0.5, 1.0], dtype=float)

    market_prices = np.array(
        [
            [10.0, np.nan],
            [0.0, -1.0],
        ],
        dtype=float,
    )

    vs = VolatilitySurface(opt_template, K_range, T_range, market_prices)

    assert vs.surface.shape == (2, 2)
    assert np.isnan(vs.surface[0, 1])
    assert np.isnan(vs.surface[1, 0])
    assert np.isnan(vs.surface[1, 1])


def test_compute_surface_handles_implied_vol_failure(opt_template, monkeypatch):
    K_range = np.array([100.0], dtype=float)
    T_range = np.array([1.0], dtype=float)
    market_prices = np.array([[10.0]], dtype=float)

    def _raise(*args, **kwargs):
        raise ValueError("fail")

    monkeypatch.setattr(Option, "implied_vol", _raise)

    vs = VolatilitySurface(opt_template, K_range, T_range, market_prices)
    assert np.isnan(vs.surface[0, 0])


def test_plot_surface_runs(opt_template):
    K_range = np.array([90.0, 100.0, 110.0], dtype=float)
    T_range = np.array([0.5, 1.0], dtype=float)

    market_prices = np.full((len(K_range), len(T_range)), np.nan, dtype=float)
    for i, K in enumerate(K_range):
        for j, T in enumerate(T_range):
            market_prices[i, j] = Option(
                option_type=opt_template.option_type,
                S=opt_template.S,
                K=float(K),
                T=float(T),
                r=opt_template.r,
                sigma=opt_template.sigma,
            ).price()

    vs = VolatilitySurface(opt_template, K_range, T_range, market_prices)
    vs.plot_surface()
