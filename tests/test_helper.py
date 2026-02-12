import numpy as np
import pandas as pd
from src.helper import alpha_beta


"""
Tests alpha_beta output keys and finite values.

Parameters:
None

Return:
None
"""
def test_alpha_beta_basic():
    idx = pd.date_range("2020-01-01", periods=50)

    asset = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 50)), index=idx)
    bench = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 50)), index=idx)

    stats = alpha_beta("A", "B", asset, bench)

    assert "alpha_annual" in stats
    assert "beta" in stats
    assert "r2" in stats
    assert np.isfinite(stats["beta"])
