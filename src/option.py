import numpy as np
import scipy.stats


class Option:
    """
    Base option class holding common parameters and utilities.

    Parameters:
    option_type (str) - "call" or "put"
    S (float) - Current underlying price
    K (float) - Strike price
    T (float) - Time to maturity in years
    r (float) - Risk-free rate
    q (float) - Continuous dividend yield (or convenience yield)
    premium (float | None) - Market/paid premium (optional)

    Return:
    None
    """
    def __init__(
        self,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float = 0.02,
        q: float = 0.0,
        premium: float | None = None,
    ):
        option_type = option_type.lower().strip()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        if S <= 0 or K <= 0 or T <= 0:
            raise ValueError("S, K, T must be > 0")

        self.option_type = option_type
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self.premium = None if premium is None else float(premium)

    """
    Sets the market premium on the object.

    Parameters:
    premium (float) - Premium to set

    Return:
    None
    """
    def set_premium(self, premium: float):
        premium = float(premium)
        if premium <= 0:
            raise ValueError("premium must be > 0")
        self.premium = premium

    """
    Computes intrinsic payoff for a given underlying price (vectorized).

    Parameters:
    ST (float | np.ndarray) - Underlying price(s) at exercise

    Return:
    np.ndarray - Payoff values
    """
    def payoff(self, ST):
        ST = np.asarray(ST, dtype=float)
        if self.option_type == "call":
            return np.maximum(ST - self.K, 0.0)
        return np.maximum(self.K - ST, 0.0)

    """
    Computes PnL at expiry across a range of terminal prices.

    Parameters:
    ST_range (np.ndarray) - Range of terminal underlying prices

    Return:
    np.ndarray - PnL values (payoff - premium)
    """
    def pnl(self, ST_range: np.ndarray) -> np.ndarray:
        if self.premium is None:
            raise ValueError("premium is not set. Provide premium to compute pnl.")
        return self.payoff(ST_range) - float(self.premium)

    """
    Prices the option.

    Parameters:
    None

    Return:
    float - Option price
    """
    def price(self, *args, **kwargs) -> float:
        raise NotImplementedError("price() must be implemented in derived classes.")

    """
    Computes implied volatility.

    Parameters:
    market_price (float) - Target option price

    Return:
    float - Implied volatility
    """
    def implied_vol(self, market_price: float, *args, **kwargs) -> float:
        raise NotImplementedError("implied_vol() is model-dependent and must be implemented where applicable.")

    """
    Returns a string representation of the Option object.

    Parameters:
    None

    Return:
    str - Object description
    """
    def __str__(self):
        return (
            f"Option(option_type={self.option_type}, S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, q={self.q}, premium={self.premium})"
        )


class EuropeanOption(Option):
    """
    European option pricer using the Black-Scholes model with continuous dividend yield q.

    Parameters:
    option_type (str) - "call" or "put"
    S (float) - Current underlying price
    K (float) - Strike price
    T (float) - Time to maturity in years
    r (float) - Risk-free rate
    q (float) - Continuous dividend yield
    sigma (float | None) - Volatility (if provided, premium is computed if missing)
    premium (float | None) - Market/paid premium (if provided without sigma, implied vol is computed)

    Return:
    None
    """
    def __init__(
        self,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float = 0.02,
        q: float = 0.0,
        sigma: float | None = None,
        premium: float | None = None,
    ):
        super().__init__(option_type=option_type, S=S, K=K, T=T, r=r, q=q, premium=premium)

        if sigma is None and premium is None:
            raise ValueError("Either sigma or premium must be provided")

        if sigma is not None:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError("sigma must be > 0")
            self.sigma = sigma
            if self.premium is None:
                self.premium = self.price()
        else:
            premium = float(premium)
            if premium <= 0:
                raise ValueError("premium must be > 0")
            self.sigma = self.implied_vol(premium)
            self.premium = premium

    """
    Computes the Black-Scholes d1 and d2 terms.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    q - Dividend yield override
    sigma - Volatility override

    Return:
    tuple - (d1, d2)
    """
    def _d1_d2(self, S=None, K=None, T=None, r=None, q=None, sigma=None):
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        q0 = self.q if q is None else float(q)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1 = (np.log(S0 / K0) + (r0 - q0 + 0.5 * sig0**2) * T0) / (sig0 * np.sqrt(T0))
        d2 = d1 - sig0 * np.sqrt(T0)
        return d1, d2

    """
    Computes the Black-Scholes price for a European option.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    q - Dividend yield override
    sigma - Volatility override

    Return:
    float - Option price
    """
    def price(self, S=None, K=None, T=None, r=None, q=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        q0 = self.q if q is None else float(q)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, d2 = self._d1_d2(S0, K0, T0, r0, q0, sig0)

        disc_r = np.exp(-r0 * T0)
        disc_q = np.exp(-q0 * T0)

        if self.option_type == "call":
            return float(S0 * disc_q * scipy.stats.norm.cdf(d1) - K0 * disc_r * scipy.stats.norm.cdf(d2))
        return float(K0 * disc_r * scipy.stats.norm.cdf(-d2) - S0 * disc_q * scipy.stats.norm.cdf(-d1))

    """
    Computes vega (derivative of option price with respect to volatility).

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    q - Dividend yield override
    sigma - Volatility override

    Return:
    float - Vega
    """
    def vega(self, S=None, K=None, T=None, r=None, q=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        q0 = self.q if q is None else float(q)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, _ = self._d1_d2(S0, K0, T0, r0, q0, sig0)
        return float(S0 * np.exp(-q0 * T0) * scipy.stats.norm.pdf(d1) * np.sqrt(T0))

    """
    Computes delta.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    q - Dividend yield override
    sigma - Volatility override

    Return:
    float - Delta
    """
    def delta(self, S=None, K=None, T=None, r=None, q=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        q0 = self.q if q is None else float(q)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, _ = self._d1_d2(S0, K0, T0, r0, q0, sig0)
        disc_q = np.exp(-q0 * T0)

        if self.option_type == "call":
            return float(disc_q * scipy.stats.norm.cdf(d1))
        return float(disc_q * (scipy.stats.norm.cdf(d1) - 1.0))

    """
    Computes gamma.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    q - Dividend yield override
    sigma - Volatility override

    Return:
    float - Gamma
    """
    def gamma(self, S=None, K=None, T=None, r=None, q=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        q0 = self.q if q is None else float(q)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, _ = self._d1_d2(S0, K0, T0, r0, q0, sig0)
        disc_q = np.exp(-q0 * T0)
        return float(disc_q * scipy.stats.norm.pdf(d1) / (S0 * sig0 * np.sqrt(T0)))

    """
    Computes implied volatility from a market price using Newton-Raphson iteration.

    Parameters:
    market_price (float) - Target option price
    initial_guess (float) - Starting volatility guess
    tol (float) - Convergence tolerance
    max_iter (int) - Maximum iterations

    Return:
    float - Implied volatility (sigma)
    """
    def implied_vol(self, market_price: float, initial_guess: float = 0.2, tol: float = 1e-6, max_iter: int = 100):
        market_price = float(market_price)
        if market_price <= 0:
            raise ValueError("market_price must be > 0")

        sigma = float(initial_guess)

        for _ in range(max_iter):
            price = self.price(sigma=sigma)
            v = self.vega(sigma=sigma)

            diff = market_price - price
            if abs(diff) < tol:
                self.sigma = sigma
                return sigma

            if np.isclose(v, 0.0):
                break

            sigma = sigma + diff / v
            if sigma <= 0:
                sigma = tol

        raise ValueError("Implied volatility did not converge")

    """
    Returns a string representation of the EuropeanOption object.

    Parameters:
    None

    Return:
    str - Object description
    """
    def __str__(self):
        return (
            f"EuropeanOption(option_type={self.option_type}, S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, q={self.q}, sigma={self.sigma}, premium={self.premium})"
        )


class AmericanOption(Option):
    """
    American option pricer using a Cox-Ross-Rubinstein binomial tree with continuous dividend yield q.

    Parameters:
    option_type (str) - "call" or "put"
    S (float) - Current underlying price
    K (float) - Strike price
    T (float) - Time to maturity in years
    r (float) - Risk-free rate
    q (float) - Continuous dividend yield
    sigma (float) - Volatility
    steps (int) - Number of binomial steps
    premium (float | None) - Market/paid premium (optional)

    Return:
    None
    """
    def __init__(
        self,
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float = 0.02,
        q: float = 0.0,
        sigma: float = 0.2,
        steps: int = 200,
        premium: float | None = None,
    ):
        super().__init__(option_type=option_type, S=S, K=K, T=T, r=r, q=q, premium=premium)

        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        steps = int(steps)
        if steps < 5:
            raise ValueError("steps must be >= 5")

        self.sigma = sigma
        self.steps = steps

        if self.premium is None:
            self.premium = self.price()

    """
    Computes the American option price via binomial tree with early exercise.

    Parameters:
    None

    Return:
    float - Option price
    """
    def price(self) -> float:
        S0 = self.S
        K = self.K
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma
        N = self.steps

        dt = T / N
        if dt <= 0:
            raise ValueError("Invalid dt computed from T/steps.")

        u = float(np.exp(sigma * np.sqrt(dt)))
        d = float(1.0 / u)

        disc = float(np.exp(-r * dt))
        p = float((np.exp((r - q) * dt) - d) / (u - d))

        if not (0.0 <= p <= 1.0):
            raise ValueError("Arbitrage condition violated in tree (check steps/sigma/T/q).")

        j = np.arange(N + 1)
        ST = S0 * (u ** j) * (d ** (N - j))

        if self.option_type == "call":
            values = np.maximum(ST - K, 0.0)
        else:
            values = np.maximum(K - ST, 0.0)

        for n in range(N - 1, -1, -1):
            values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

            j = np.arange(n + 1)
            ST = S0 * (u ** j) * (d ** (n - j))

            if self.option_type == "call":
                exercise = np.maximum(ST - K, 0.0)
            else:
                exercise = np.maximum(K - ST, 0.0)

            values = np.maximum(values, exercise)

        return float(values[0])

    """
    Approximates delta via a small bump finite difference (uses the same model).

    Parameters:
    bump (float) - Relative bump size (e.g. 0.01 means 1%)

    Return:
    float - Delta approximation
    """
    def delta(self, bump: float = 0.01) -> float:
        bump = float(bump)
        if bump <= 0:
            raise ValueError("bump must be > 0")

        S0 = float(self.S)
        Su = S0 * (1.0 + bump)
        Sd = S0 * (1.0 - bump)

        opt_u = AmericanOption(self.option_type, Su, self.K, self.T, self.r, self.q, self.sigma, self.steps).price()
        opt_d = AmericanOption(self.option_type, Sd, self.K, self.T, self.r, self.q, self.sigma, self.steps).price()

        denom = Su - Sd
        if np.isclose(denom, 0.0):
            return float("nan")
        return float((opt_u - opt_d) / denom)

    """
    Returns a string representation of the AmericanOption object.

    Parameters:
    None

    Return:
    str - Object description
    """
    def __str__(self):
        return (
            f"AmericanOption(option_type={self.option_type}, S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, q={self.q}, sigma={self.sigma}, steps={self.steps}, premium={self.premium})"
        )
