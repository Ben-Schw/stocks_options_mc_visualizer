import numpy as np
import pandas as pd
from .historical_portfolio import HistoricalPortfolio
from .helper import alpha_beta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .stock import Stock


class PortfolioAnalyzerRaw:

    """
    Initialize the analyzer using a HistoricalPortfolio instance.

    Parameters:
    portfolio : HistoricalPortfolio
        Backtested portfolio object containing prices, returns,
        weights and asset values.
    """
    def __init__(self, portfolio: HistoricalPortfolio):
        self.values = portfolio._bt_port_value
        self.assets = portfolio._bt_prices
        self.asset_values_in_port = portfolio._bt_asset_values
        self.returns = portfolio.port_returns
        self.asset_returns = portfolio.asset_returns_df
        self.index = portfolio._bt_index
        self.cash = portfolio._bt_cash

        portfolio.weights_cash.name = "Cash"
        self.weights = pd.concat([portfolio.weights_df, portfolio.weights_cash], axis=1)


    """
    Compute portfolio alpha and beta relative to a benchmark stock.

    Parameters:
    bench : Stock
        Benchmark stock object.

    Returns:
    dict
        Dictionary containing alpha and beta statistics.
    """
    def alpha_beta_portfolio(self, bench: Stock) -> dict:
        return alpha_beta(
            ticker="Portfolio",
            benchmark_ticker=bench.ticker,
            ticker_price=self.values,
            bench_price=bench.prices
        )
    
    
    """
    Compute annualized return, volatility and Sharpe ratio.

    Returns:
    list
        [annual_return, annual_volatility, sharpe_ratio]
    """   
    def return_sigma_sharpe(self) -> list:
        sigma_daily = float(self.returns.std())

        total_return = self.values.iloc[-1] / self.values.iloc[0] - 1

        n_days = len(self.returns)
        n_years = n_days / 252

        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_vol = sigma_daily * np.sqrt(252)
        sharpe = annual_return / annual_vol

        return [total_return, annual_return, annual_vol, sharpe]
    

    """
    Perform a weighted PCA on asset returns.

    The PCA is computed on weighted returns:
        weighted_return = asset_return * portfolio_weight

    This means the PCA explains the variance of asset contributions
    to the portfolio rather than the raw asset returns.

    Parameters:
    n_components : int, default=2
        Number of principal components to compute.

    Returns:
    dict
        Dictionary containing:
        - weighted_returns : DataFrame
            Time series of asset return contributions.
        - loadings : DataFrame
            Asset loadings on each principal component.
        - explained_variance_ratio : Series
            Fraction of variance explained by each component.
        - factors_ts : DataFrame
            Time series of PCA factor scores.
        - pca : sklearn PCA object
            Fitted PCA model.
    """
    def pca(self, n_components: int = 4) -> dict:

        r = self.asset_returns.copy()
        w = self.weights.drop(columns=["Cash"])

        idx = r.index.intersection(w.index)
        r = r.loc[idx]
        w = w.loc[idx]

        common_assets = [c for c in r.columns if c in w.columns]
        r = r[common_assets].dropna()
        w = w.loc[r.index, common_assets].dropna()

        weighted_returns = r * w

        X = weighted_returns.values
        X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(X)

        pcs = [f"PC{i+1}" for i in range(pca.n_components_)]

        loadings = pd.DataFrame(
            pca.components_.T,
            index=weighted_returns.columns,
            columns=pcs
        )

        explained_var = pd.Series(
            pca.explained_variance_ratio_,
            index=pcs,
            name="explained_variance_ratio"
        )

        factors_ts = pd.DataFrame(
            factors,
            index=weighted_returns.index,
            columns=pcs
        )

        return {
            "weighted_returns": weighted_returns,
            "loadings": loadings,
            "explained_variance_ratio": explained_var,
            "factors_ts": factors_ts,
            "pca": pca
        }
