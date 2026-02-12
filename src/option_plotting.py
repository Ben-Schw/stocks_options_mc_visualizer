import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class OptionPlotMixin:
    """
    Plotting mixin for Option-like objects.

    Requirements:
    - self.option_type, self.S, self.K, self.T, self.r, self.q
    - self.price(...) is implemented
    - self.payoff(...) and self.pnl(...) are available

    Return:
    None
    """

    """
    Plots the profit and loss at expiry over a range of terminal underlying prices.

    Parameters:
    ST_range (np.ndarray) - Range of terminal underlying prices

    Return:
    None
    """
    def plot_pnl(self, ST_range: np.ndarray):
        ST = np.asarray(ST_range, dtype=float)
        pnl_values = self.pnl(ST)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.plot(ST, pnl_values, label="PnL", linewidth=2)
        ax.axhline(0, color="white", linestyle="--", alpha=0.8)

        ax.set_title(f"PnL at Expiry for {self.option_type.upper()} Option")
        ax.set_xlabel("Stock Price at Expiry (ST)")
        ax.set_ylabel("PnL")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        ax.legend()

        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)
        plt.show(block=True)
        plt.close(fig)

    """
    Plots option price versus underlying price for fixed parameters.

    Parameters:
    S_range (np.ndarray) - Range of underlying prices

    Return:
    None
    """
    def plot_price_vs_underlying(self, S_range: np.ndarray):
        S_range = np.asarray(S_range, dtype=float)

        old = (self.S,)
        prices = []
        for s in S_range:
            try:
                prices.append(float(self.price(S=float(s))))
            except TypeError:
                self.S = float(s)
                prices.append(float(self.price()))
        self.S = old[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.plot(S_range, prices, linewidth=2, label="Option Price")
        ax.set_title(f"{self.option_type.upper()} Price vs Underlying")
        ax.set_xlabel("Underlying Price (S)")
        ax.set_ylabel("Option Price")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        ax.legend()

        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)
        plt.show(block=True)
        plt.close(fig)

    """
    Plots a heatmap of option prices over strike and time-to-expiry grids with sliders for S, r, q and sigma (if present).

    Parameters:
    K_range (np.ndarray) - Strike grid values
    T_range (np.ndarray) - Time-to-maturity grid values

    Return:
    None
    """
    def plot_heatmap(self, K_range: np.ndarray, T_range: np.ndarray):
        K_range = np.asarray(K_range, dtype=float)
        T_range = np.asarray(T_range, dtype=float)

        if (K_range <= 0).any() or (T_range <= 0).any():
            raise ValueError("K_range and T_range must contain only positive values.")

        T_grid, K_grid = np.meshgrid(T_range, K_range)

        has_sigma = hasattr(self, "sigma")
        sigma0 = float(getattr(self, "sigma", 0.2)) if has_sigma else None

        S0 = float(self.S)
        r0 = float(self.r)
        q0 = float(getattr(self, "q", 0.0))

        def price_grid(S, r, q, sigma=None):
            out = np.empty_like(T_grid, dtype=float)

            old = (self.S, self.K, self.T, self.r, getattr(self, "q", 0.0), getattr(self, "sigma", None))
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    try:
                        out[i, j] = float(self.price(S=S, K=K_grid[i, j], T=T_grid[i, j], r=r, q=q, sigma=sigma))
                    except TypeError:
                        self.S = float(S)
                        self.K = float(K_grid[i, j])
                        self.T = float(T_grid[i, j])
                        self.r = float(r)
                        if hasattr(self, "q"):
                            self.q = float(q)
                        if sigma is not None and hasattr(self, "sigma"):
                            self.sigma = float(sigma)
                        out[i, j] = float(self.price())
            self.S, self.K, self.T, self.r = old[0], old[1], old[2], old[3]
            if hasattr(self, "q"):
                self.q = float(old[4])
            if hasattr(self, "sigma") and old[5] is not None:
                self.sigma = float(old[5])

            return out

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        plt.subplots_adjust(bottom=0.33)

        grid0 = price_grid(S0, r0, q0, sigma0)

        im = ax.imshow(
            grid0,
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
        title = ax.set_title("Option Price Heatmap")

        cb = fig.colorbar(im, ax=ax, label="Option Price")
        cb.ax.set_facecolor("#8ebff3")
        ax.grid(False)

        texts = []
        for i in range(grid0.shape[0]):
            row = []
            for j in range(grid0.shape[1]):
                row.append(ax.text(j, i, f"{grid0[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white"))
            texts.append(row)

        ax_S = plt.axes([0.15, 0.23, 0.7, 0.03])
        ax_r = plt.axes([0.15, 0.18, 0.7, 0.03])
        ax_q = plt.axes([0.15, 0.13, 0.7, 0.03])

        ax_S.set_facecolor("#a4d1f8")
        ax_r.set_facecolor("#a4d1f8")
        ax_q.set_facecolor("#a4d1f8")

        slider_S = Slider(ax_S, "S", 0.7 * S0, 1.3 * S0, valinit=S0, valstep=max(S0 * 0.01, 0.01))
        slider_r = Slider(ax_r, "r", -0.01, 0.10, valinit=r0, valstep=0.001)
        slider_q = Slider(ax_q, "q", 0.0, 0.20, valinit=q0, valstep=0.001)

        slider_sigma = None
        if has_sigma:
            ax_sigma = plt.axes([0.15, 0.08, 0.7, 0.03])
            ax_sigma.set_facecolor("#a4d1f8")
            slider_sigma = Slider(ax_sigma, "σ", 0.01, 1.0, valinit=float(sigma0), valstep=0.01)

        def update(_):
            S = float(slider_S.val)
            r = float(slider_r.val)
            q = float(slider_q.val)
            sigma = float(slider_sigma.val) if slider_sigma is not None else None

            new_grid = price_grid(S, r, q, sigma)
            im.set_data(new_grid)
            im.set_clim(np.nanmin(new_grid), np.nanmax(new_grid))

            if slider_sigma is not None:
                title.set_text(f"S={S:.2f}, r={r:.3f}, q={q:.3f}, σ={sigma:.2f}")
            else:
                title.set_text(f"S={S:.2f}, r={r:.3f}, q={q:.3f}")

            for i in range(new_grid.shape[0]):
                for j in range(new_grid.shape[1]):
                    texts[i][j].set_text(f"{new_grid[i, j]:.2f}")

            fig.canvas.draw_idle()

        slider_S.on_changed(update)
        slider_r.on_changed(update)
        slider_q.on_changed(update)
        if slider_sigma is not None:
            slider_sigma.on_changed(update)

        plt.show(block=True)
        plt.close(fig)
