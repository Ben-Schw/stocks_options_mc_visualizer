import pytest
import pandas as pd
import numpy as np


"""
Creates deterministic fake Yahoo Finance dataframe.

Parameters:
None

Return:
pd.DataFrame
"""
@pytest.fixture
def yf_df():
    dates = pd.date_range("2020-01-01", periods=120, freq="B")

    prices = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, len(dates)))

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Adj Close": prices,
            "Volume": np.random.randint(1e6, 5e6, len(dates)),
        },
        index=dates,
    )

    return df
