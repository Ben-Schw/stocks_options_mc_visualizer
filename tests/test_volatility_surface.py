import numpy as np
from src.option import EuropeanOption

from src.volatility_surface import VolatilitySurface


"""
Tests volatility surface dimensions.

Parameters:
None

Return:
None
"""
def test_surface_shape():
    opt = EuropeanOption("call", 100, 100, 1, sigma=0.2)

    K = np.array([90, 100, 110])
    T = np.array([0.5, 1.0])
    market = np.full((3, 2), opt.price())

    surf = VolatilitySurface(opt, K, T, market)
    assert surf.surface.shape == (3, 2)
