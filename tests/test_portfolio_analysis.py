import numpy as np
import pandas as pd
from src.portfolio_analysis import PortfolioAnalyzerRaw
import pytest


class DummyStock:
    """Minimal Stock stub for testing."""
    def __init__(self, ticker: str):
        self.ticker = ticker


class DummyHistoricalPortfolio:
    """
    Minimal HistoricalPortfolio-like object with the attributes
    required by PortfolioAnalyzerRaw.
    """
    def __init__(self):
        self._bt_index = pd.date_range("2020-01-01", periods=60, freq="B")
        n = len(self._bt_index)

        self.assets = ["AAPL", "MSFT"]

        aapl = 100 * (1.001) ** np.arange(n)
        msft = 200 * (1.0005) ** np.arange(n)

        self._bt_prices = pd.DataFrame(
            {"AAPL": aapl, "MSFT": msft},
            index=self._bt_index
        )

        self._bt_port_value = pd.Series(
            1000 * (1.0008) ** np.arange(n),
            index=self._bt_index,
            name="Portfolio"
        )

        self._bt_cash = pd.Series(
            100.0 + 0.0 * np.arange(n),
            index=self._bt_index,
            name="CashValue"
        )

        self._bt_asset_values = pd.DataFrame(
            {
                "AAPL": 500 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)),
                "MSFT": 400 + 8 * np.cos(np.linspace(0, 4 * np.pi, n)),
            },
            index=self._bt_index
        )

        w_aapl = 0.55 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, n))
        w_msft = 0.35 + 0.02 * np.cos(np.linspace(0, 2 * np.pi, n))
        weights_df = pd.DataFrame({"AAPL": w_aapl, "MSFT": w_msft}, index=self._bt_index)

        weights_cash = pd.Series(1.0 - (weights_df["AAPL"] + weights_df["MSFT"]), index=self._bt_index)
        self.weights_df = weights_df
        self.weights_cash = weights_cash

        self.asset_returns_df = self._bt_prices.pct_change().dropna()

        self.port_returns = self._bt_port_value.pct_change().dropna()


@pytest.fixture
def dummy_portfolio():
    return DummyHistoricalPortfolio()


@pytest.fixture
def analyzer(dummy_portfolio):
    return PortfolioAnalyzerRaw(dummy_portfolio)


def test_init_creates_weights_with_cash(analyzer):
    assert isinstance(analyzer.weights, pd.DataFrame)
    assert "Cash" in analyzer.weights.columns
    assert set(["AAPL", "MSFT"]).issubset(set(analyzer.weights.columns))


def test_return_sigma_sharpe_outputs_finite(analyzer):
    tot_ret, ann_ret, ann_vol, sharpe = analyzer.return_sigma_sharpe()

    assert np.isfinite(tot_ret)
    assert np.isfinite(ann_ret)
    assert np.isfinite(ann_vol)
    assert np.isfinite(sharpe)

    assert ann_vol >= 0.0


def test_pca_returns_expected_keys(analyzer):
    res = analyzer.pca(n_components=2)
    for k in ["weighted_returns", "loadings", "explained_variance_ratio", "factors_ts", "pca"]:
        assert k in res


def test_pca_shapes(analyzer):
    res = analyzer.pca(n_components=2)

    weighted_returns = res["weighted_returns"]
    loadings = res["loadings"]
    factors_ts = res["factors_ts"]
    ev = res["explained_variance_ratio"]

    assert list(weighted_returns.columns) == ["AAPL", "MSFT"]

    assert loadings.shape == (2, 2)

    assert factors_ts.shape[1] == 2
    assert factors_ts.index.equals(weighted_returns.index)

    assert len(ev) == 2


def test_pca_explained_variance_ratio_sums_to_one_approx(analyzer):
    res = analyzer.pca(n_components=2)
    ev = res["explained_variance_ratio"]

    assert np.isclose(float(ev.sum()), 1.0, atol=1e-8)


def test_alpha_beta_portfolio_smoke(monkeypatch, analyzer):
    """
    Smoke test: we don't validate alpha/beta numerically here because it depends
    on helper.alpha_beta. We just ensure the call works and returns a dict.

    If you want a strict test, patch alpha_beta to a deterministic function.
    """

    import src.portfolio_analysis as mod

    def fake_alpha_beta(*args, **kwargs):
        return {"alpha": 0.1, "beta": 1.2}

    monkeypatch.setattr(mod, "alpha_beta", fake_alpha_beta)

    bench = DummyStock("AAPL")
    out = analyzer.alpha_beta_portfolio(bench)

    assert isinstance(out, dict)
    assert "alpha" in out and "beta" in out
