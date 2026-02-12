import numpy as np
from src.option import EuropeanOption, AmericanOption


"""
Tests Black-Scholes put-call parity.

Parameters:
None

Return:
None
"""
def test_put_call_parity_euro():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2

    call = EuropeanOption("call", S, K, T, r=r, sigma=sigma)
    put = EuropeanOption("put", S, K, T, r=r, sigma=sigma)

    C = call.price()
    P = put.price()

    rhs = S - K * np.exp(-r * T)
    assert abs((C - P) - rhs) < 1e-4


"""
Tests implied volatility recovery of EuropeanOption.

Parameters:
None

Return:
None
"""
def test_implied_vol():
    opt = EuropeanOption("call", 100, 100, 1, r=0.02, sigma=0.25)
    market_price = opt.price()

    opt2 = EuropeanOption("call", 100, 100, 1, r=0.02, premium=market_price)
    assert abs(opt2.sigma - 0.25) < 1e-3


"""
Tests option payoff shape of EuropeanOption.

Parameters:
None

Return:
None
"""
def test_pnl_shape():
    opt = EuropeanOption("call", 100, 100, 1, sigma=0.2)
    ST = np.linspace(50, 150, 100)

    pnl = opt.pnl(ST)
    assert pnl.shape == ST.shape


"""
Tests that an American call option is worth as much as a European, since q = 0.
However, the put option is worth more for American options

Parameters:
None

Return:
None
"""
def test_american_vs_european():
    S = 100
    K = 100
    T = 1
    r = 0.02
    sigma = 0.2
    q = 0

    am_call = AmericanOption("call", S, K, T, r=r, q=q, sigma=sigma)
    eu_call = EuropeanOption("call", S, K, T, r=r, q=q, sigma=sigma)

    am_put = AmericanOption("put", S, K, T, r=r, q=q, sigma=sigma)
    eu_put = EuropeanOption("put", S, K, T, r=r, q=q, sigma=sigma)

    am_call_price = am_call.price()
    eu_call_price = eu_call.price()

    am_put_price = am_put.price()
    eu_put_price = eu_put.price()

    assert abs(am_call_price - eu_call_price) < 5e-2
    assert am_put_price > eu_put_price



"""
Tests basic monotonicity: American call price should increase with S.

Parameters:
None

Return:
None
"""
def test_american_call_monotone_in_S():
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2
    q = 0.0

    S1 = 90.0
    S2 = 110.0

    am1 = AmericanOption("call", S1, K, T, r=r, sigma=sigma, q=q)
    am2 = AmericanOption("call", S2, K, T, r=r, sigma=sigma, q=q)

    assert am2.price() >= am1.price()




