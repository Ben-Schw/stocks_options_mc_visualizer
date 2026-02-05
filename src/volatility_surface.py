import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .option import Option

class VolatilitySurface:
    """
    Initializes a volatility surface calculator from a template option and a grid of market prices.

    Parameters:
    option_template (Option) - Base option providing option_type, S, and r
    K_range (np.ndarray) - Strike grid
    T_range (np.ndarray) - Maturity grid (in years)
    market_prices (np.ndarray) - Market option prices with shape (len(K_range), len(T_range))

    Return:
    None
    """
    def __init__(self, option_template: Option, K_range: np.ndarray, T_range: np.ndarray, market_prices: np.ndarray):
        self.opt = option_template
        self.K_range = np.asarray(K_range, dtype=float)
        self.T_range = np.asarray(T_range, dtype=float)

        market_prices = np.asarray(market_prices, dtype=float)
        if market_prices.shape != (len(self.K_range), len(self.T_range)):
            raise ValueError("market_prices must have shape (len(K_range), len(T_range))")

        self.market_prices = market_prices
        self.surface = self.compute_surface()

    """
    Computes the implied volatility surface by inverting Black-Scholes prices across the K/T grid.

    Parameters:
    None

    Return:
    np.ndarray - Implied volatility surface with shape (len(K_range), len(T_range))
    """
    def compute_surface(self) -> np.ndarray:
        surface = np.full((len(self.K_range), len(self.T_range)), np.nan, dtype=float)

        for i, K in enumerate(self.K_range):
            for j, T in enumerate(self.T_range):
                price_ij = self.market_prices[i, j]
                if not np.isfinite(price_ij) or price_ij <= 0 or T <= 0:
                    continue

                opt_ij = Option(
                    option_type=self.opt.option_type,
                    S=self.opt.S,
                    K=float(K),
                    T=float(T),
                    r=self.opt.r,
                    sigma=self.opt.sigma,
                )

                try:
                    surface[i, j] = opt_ij.implied_vol(price_ij)
                except Exception:
                    surface[i, j] = np.nan

        return surface

    """
    Plots the implied volatility surface as a 3D plot.

    Parameters:
    None

    Return:
    None
    """
    def plot_surface(self):
        K_grid, T_grid = np.meshgrid(self.K_range, self.T_range, indexing="ij")

        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0c89f7")

        surf = ax.plot_surface(
            K_grid,
            T_grid,
            self.surface,
            cmap="coolwarm",
            edgecolor="none",
            antialiased=True,
            alpha=0.95
        )

        ax.set_xlabel("Strike (K)")
        ax.set_ylabel("Maturity (T)")
        ax.set_zlabel("Implied Vol")

        ax.set_title("Implied Volatility Surface")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")
        ax.title.set_color("white")

        ax.tick_params(colors="white")

        cbar = fig.colorbar(surf, shrink=0.65, aspect=18, pad=0.08)
        cbar.set_label("Implied Volatility")
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")
        cbar.ax.set_facecolor("#8ebff3")

        ax.view_init(elev=25, azim=-135)

        fig.subplots_adjust(
            left=0.02,
            right=0.95,
            top=0.92,
            bottom=0.05
        )

        plt.show(block=True)
        plt.close(fig)

