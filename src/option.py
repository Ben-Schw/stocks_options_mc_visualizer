import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Option:
    """
    Initializes a European option pricer using the Black-Scholes model.

    Parameters:
    option_type (str) - "call" or "put"
    S (float) - Current underlying price
    K (float) - Strike price
    T (float) - Time to maturity in years
    r (float) - Risk-free rate
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
        sigma: float | None = None,
        premium: float | None = None,
    ):
        option_type = option_type.lower().strip()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        if S <= 0 or K <= 0 or T <= 0:
            raise ValueError("S, K, T must be > 0")

        if sigma is None and premium is None:
            raise ValueError("Either sigma or premium must be provided")

        self.option_type = option_type
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)

        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma must be > 0")

            self.sigma = float(sigma)
            self.premium = float(premium) if premium is not None else self.price()
        else:
            if premium <= 0:
                raise ValueError("premium must be > 0")

            self.premium = float(premium)
            self.sigma = self.implied_vol(self.premium)

    """
    Computes the Black-Scholes d1 and d2 terms.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    sigma - Volatility override

    Return:
    tuple - (d1, d2)
    """
    def _d1_d2(self, S=None, K=None, T=None, r=None, sigma=None):
        S = self.S if S is None else float(S)
        K = self.K if K is None else float(K)
        T = self.T if T is None else float(T)
        r = self.r if r is None else float(r)
        sigma = self.sigma if sigma is None else float(sigma)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    """
    Computes the Black-Scholes price for a European option.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    sigma - Volatility override

    Return:
    float - Option price
    """
    def price(self, S=None, K=None, T=None, r=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, d2 = self._d1_d2(S0, K0, T0, r0, sig0)

        if self.option_type == "call":
            return float(S0 * scipy.stats.norm.cdf(d1) - K0 * np.exp(-r0 * T0) * scipy.stats.norm.cdf(d2))
        else:
            return float(K0 * np.exp(-r0 * T0) * scipy.stats.norm.cdf(-d2) - S0 * scipy.stats.norm.cdf(-d1))

    """
    Computes vega (derivative of option price with respect to volatility).

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    sigma - Volatility override

    Return:
    float - Vega
    """
    def vega(self, S=None, K=None, T=None, r=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, _ = self._d1_d2(S0, K0, T0, r0, sig0)
        return float(S0 * scipy.stats.norm.pdf(d1) * np.sqrt(T0))

    """
    Computes the delta hedge ratio for the option.

    Parameters:
    S - Underlying price override
    K - Strike price override
    T - Time to maturity override
    r - Risk-free rate override
    sigma - Volatility override

    Return:
    float - Delta
    """
    def delta_hedge_ratio(self, S=None, K=None, T=None, r=None, sigma=None) -> float:
        S0 = self.S if S is None else float(S)
        K0 = self.K if K is None else float(K)
        T0 = self.T if T is None else float(T)
        r0 = self.r if r is None else float(r)
        sig0 = self.sigma if sigma is None else float(sigma)

        d1, _ = self._d1_d2(S0, K0, T0, r0, sig0)

        if self.option_type == "call":
            return float(scipy.stats.norm.cdf(d1))
        else:
            return float(scipy.stats.norm.cdf(d1) - 1.0)

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
        sigma = float(initial_guess)

        for _ in range(max_iter):
            price = self.price(sigma=sigma)
            v = self.vega(sigma=sigma)

            diff = float(market_price) - price
            if abs(diff) < tol:
                self.sigma = sigma
                return sigma

            if v == 0:
                break

            sigma = sigma + diff / v

            if sigma <= 0:
                sigma = tol

        raise ValueError("Implied volatility did not converge")

    """
    Computes the payoff minus premium at expiry across a range of terminal prices.

    Parameters:
    ST_range (np.ndarray) - Range of terminal underlying prices

    Return:
    np.ndarray - Profit and loss values
    """
    def pnl(self, ST_range: np.ndarray) -> np.ndarray:
        ST = np.asarray(ST_range, dtype=float)

        if self.option_type == "call":
            payoff = np.maximum(ST - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - ST, 0.0)

        return payoff - float(self.premium)

    """
    Plots a heatmap of Black-Scholes option prices over strike and time-to-expiry grids,
    with sliders to adjust S, r, and sigma.

    Parameters:
    K_range (np.ndarray) - Strike grid values
    T_range (np.ndarray) - Time-to-maturity grid values

    Return:
    None
    """
    def plot_heatmap(self, K_range: np.ndarray, T_range: np.ndarray):
        K_range = np.asarray(K_range, dtype=float)
        T_range = np.asarray(T_range, dtype=float)

        T_grid, K_grid = np.meshgrid(T_range, K_range)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        plt.subplots_adjust(bottom=0.30)

        price_grid = self._price_grid(self.S, K_grid, T_grid, self.r, self.sigma)

        im = ax.imshow(
            price_grid,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            cmap="coolwarm"
        )
        ax.set_xticks(np.arange(len(T_range)))
        ax.set_yticks(np.arange(len(K_range)))
        ax.set_xticklabels([f"{t:.2f}" for t in T_range])
        ax.set_yticklabels([f"{k:.0f}" for k in K_range])
        ax.set_xlabel("Time to Expiration (T)")
        ax.set_ylabel("Strike Price (K)")
        title = ax.set_title(
            f"Black-Scholes {self.option_type.upper()} Prices\nS={self.S:.2f}, r={self.r:.3f}, σ={self.sigma:.2f}"
        )

        cb = fig.colorbar(im, ax=ax, label="Option Price")
        cb.ax.set_facecolor("#8ebff3")

        ax.grid(False)

        texts = []
        for i in range(price_grid.shape[0]):
            row = []
            for j in range(price_grid.shape[1]):
                row.append(ax.text(j, i, f"{price_grid[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white"))
            texts.append(row)

        ax_S = plt.axes([0.15, 0.20, 0.7, 0.03])
        ax_r = plt.axes([0.15, 0.15, 0.7, 0.03])
        ax_sigma = plt.axes([0.15, 0.10, 0.7, 0.03])

        ax_S.set_facecolor("#a4d1f8")
        ax_r.set_facecolor("#a4d1f8")
        ax_sigma.set_facecolor("#a4d1f8")

        slider_S = Slider(ax_S, "S", 0.7 * self.S, 1.3 * self.S, valinit=self.S, valstep=max(self.S * 0.01, 0.01))
        slider_r = Slider(ax_r, "r", -0.01, 0.10, valinit=self.r, valstep=0.001)
        slider_sigma = Slider(ax_sigma, "σ", 0.01, 1.0, valinit=self.sigma, valstep=0.01)

        title_box = dict(boxstyle="round", facecolor="#0c89f7", alpha=0.85)
        title.set_bbox(title_box)

        def update(_):
            S = float(slider_S.val)
            r = float(slider_r.val)
            sigma = float(slider_sigma.val)

            new_prices = self._price_grid(S, K_grid, T_grid, r, sigma)
            im.set_data(new_prices)
            im.set_clim(np.nanmin(new_prices), np.nanmax(new_prices))

            title.set_text(
                f"Black-Scholes {self.option_type.upper()} Prices\nS={S:.2f}, r={r:.3f}, σ={sigma:.2f}"
            )

            for i in range(new_prices.shape[0]):
                for j in range(new_prices.shape[1]):
                    texts[i][j].set_text(f"{new_prices[i, j]:.2f}")

            fig.canvas.draw_idle()

        slider_S.on_changed(update)
        slider_r.on_changed(update)
        slider_sigma.on_changed(update)

        plt.show(block=True)
        plt.close(fig)

    """
    Computes a vectorized grid of Black-Scholes prices for heatmap plotting.

    Parameters:
    S - Underlying price
    K_grid - Strike grid (mesh)
    T_grid - Time-to-expiry grid (mesh)
    r - Risk-free rate
    sigma - Volatility

    Return:
    np.ndarray - Price grid
    """
    def _price_grid(self, S, K_grid, T_grid, r, sigma):
        d1 = (np.log(S / K_grid) + (r + 0.5 * sigma**2) * T_grid) / (sigma * np.sqrt(T_grid))
        d2 = d1 - sigma * np.sqrt(T_grid)

        if self.option_type == "call":
            return S * scipy.stats.norm.cdf(d1) - K_grid * np.exp(-r * T_grid) * scipy.stats.norm.cdf(d2)
        else:
            return K_grid * np.exp(-r * T_grid) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)

    """
    Plots the profit and loss at expiry over a range of terminal underlying prices.

    Parameters:
    ST_range (np.ndarray) - Range of terminal underlying prices

    Return:
    None
    """
    def plot_pnl(self, ST_range: np.ndarray):
        ST = np.asarray(ST_range, dtype=float)
        pnl_values = self.pnl(ST) - self.premium

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.plot(ST, pnl_values, label="PnL", color="blue")
        ax.axhline(0, color="white", linestyle="--", alpha=0.8)

        ax.set_title(f"PnL at Expiry for {self.option_type.upper()} Option")
        ax.set_xlabel("Stock Price at Expiry (ST)")
        ax.set_ylabel("PnL")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        ax.legend()

        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.12
        )
        plt.show(block=True)
        plt.close(fig)

    """
    Returns a string representation of the Option object.

    Parameters:
    None

    Return:
    str - Object description
    """
    def __str__(self):
        return (
            f"Option(option_type={self.option_type}, stock_price={self.S}, strike_price={self.K}, "
            f"time_to_maturity={self.T}, risk_free_rate={self.r}, volatility={self.sigma})"
        )
