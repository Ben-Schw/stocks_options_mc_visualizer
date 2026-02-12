import numpy as np


class VolatilitySurface:
    """
    Initializes a volatility surface calculator from a template option and a grid of market prices.

    Parameters:
    option_template - Base option providing option_type, S, r (and optionally q)
    K_range (np.ndarray) - Strike grid
    T_range (np.ndarray) - Maturity grid (in years)
    market_prices (np.ndarray) - Market option prices with shape (len(K_range), len(T_range))

    Return:
    None
    """
    def __init__(self, option_template, K_range: np.ndarray, T_range: np.ndarray, market_prices: np.ndarray):
        self.opt = option_template
        self.K_range = np.asarray(K_range, dtype=float)
        self.T_range = np.asarray(T_range, dtype=float)

        market_prices = np.asarray(market_prices, dtype=float)
        if market_prices.shape != (len(self.K_range), len(self.T_range)):
            raise ValueError("market_prices must have shape (len(K_range), len(T_range))")

        self.market_prices = market_prices
        self.surface = self.compute_surface()

    """
    Computes the implied volatility surface by inverting model prices across the K/T grid.

    Parameters:
    None

    Return:
    np.ndarray - Implied volatility surface with shape (len(K_range), len(T_range))
    """
    def compute_surface(self) -> np.ndarray:
        if not hasattr(self.opt, "implied_vol") or not callable(getattr(self.opt, "implied_vol")):
            raise TypeError(
                "option_template must provide an implied_vol(market_price) method. "
                "Use a European option pricer (Black-Scholes) or implement implied_vol for American."
            )

        surface = np.full((len(self.K_range), len(self.T_range)), np.nan, dtype=float)

        opt_cls = self.opt.__class__

        q0 = float(getattr(self.opt, "q", 0.0))
        sigma0 = getattr(self.opt, "sigma", None)

        for i, K in enumerate(self.K_range):
            for j, T in enumerate(self.T_range):
                price_ij = float(self.market_prices[i, j])

                if not np.isfinite(price_ij) or price_ij <= 0 or T <= 0:
                    continue

                kwargs = dict(
                    option_type=self.opt.option_type,
                    S=float(self.opt.S),
                    K=float(K),
                    T=float(T),
                    r=float(self.opt.r),
                    q=q0,
                )

                if sigma0 is not None:
                    kwargs["sigma"] = float(sigma0)

                try:
                    opt_ij = opt_cls(**kwargs)
                except TypeError:
                    kwargs.pop("q", None)
                    opt_ij = opt_cls(**kwargs)

                try:
                    surface[i, j] = float(opt_ij.implied_vol(price_ij))
                except Exception:
                    surface[i, j] = np.nan

        return surface
