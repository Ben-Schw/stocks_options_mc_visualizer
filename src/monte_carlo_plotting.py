import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

from .helper import price_paths_macros


class MonteCarloStockPlotMixin:
    """
    Plotting mixin for MonteCarloStock.

    Return:
    None
    """

    """
    Plots a subset of simulated paths and lets the user switch summary metrics via radio buttons.

    Parameters:
    paths (np.ndarray) - Simulated price paths (steps, simulations)
    num_paths (int) - Number of random paths to draw

    Return:
    None
    """
    def plot_simulated_paths(self, paths: np.ndarray, num_paths: int = 10):
        summary = price_paths_macros(self.S0, paths)

        steps, n_sims = paths.shape
        x_sim = np.arange(steps)

        num_paths = min(num_paths, n_sims)
        idx = np.random.choice(n_sims, size=num_paths, replace=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        show_date = self.stock.end_date

        true_line = None
        if getattr(self, "mid_date", None) is not None:
            mid = pd.to_datetime(self.mid_date)

            if mid not in self.stock.prices.index:
                raise ValueError(f"mid_date {self.mid_date} is not in stock.prices index.")

            prices = self.stock.fetch_prices(
                ticker=self.stock.ticker,
                start=self.mid_date,
                end=self.stock.end_date
            ).to_numpy()

            if len(prices) != steps:
                raise ValueError(f"Lengths are not compatible: true={len(prices)} vs paths={steps}")

            true_line, = ax.plot(x_sim, prices, linewidth=2, label="True price", color="red")
            show_date = self.mid_date

        for j in idx:
            ax.plot(x_sim, paths[:, j], lw=0.1, color="black")

        metric_keys = [
            "Mean",
            "Median",
            "25% Quantile",
            "75% Quantile",
            "5% Quantile",
            "95% Quantile"
        ]

        metric_map = {
            "Mean": "mean_path",
            "Median": "median_path",
            "25% Quantile": "25th_percentile",
            "75% Quantile": "75th_percentile",
            "5% Quantile": "5th_percentile",
            "95% Quantile": "95th_percentile"
        }

        current_key = "Mean"
        metric_line, = ax.plot(x_sim, summary[metric_map[current_key]], linewidth=2.5, label=current_key, color="green")

        ax.set_title(f"Monte-Carlo-Simulations (from {show_date})")
        ax.set_xlabel("days")
        ax.set_ylabel("price")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        ax.legend(loc="upper left")

        rax = ax.inset_axes([0.78, 0.65, 0.20, 0.30])
        rax.set_facecolor("#a4d1f8")
        radio = RadioButtons(rax, metric_keys, active=metric_keys.index(current_key))

        def on_click(label):
            metric_line.set_ydata(summary[metric_map[label]])
            metric_line.set_label(label)
            ax.legend(loc="upper left")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        radio.on_clicked(on_click)

        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.12
        )
        plt.show(block=True)
        plt.close(fig)


class MonteCarloOptionPlotMixin:
    """
    Plotting mixin for MonteCarloOption.

    Return:
    None
    """

    """
    Plots GBM simulated underlying paths and lets the user switch summary metrics via radio buttons.

    Parameters:
    paths (np.ndarray) - Simulated price paths (steps + 1, simulations)
    num_paths (int) - Number of random paths to draw
    S0 (float | None) - Initial price override
    mu (float | None) - Drift override
    sigma (float | None) - Volatility override
    T (float | None) - Time to maturity override

    Return:
    None
    """
    def plot_simulated_paths(
        self,
        paths: np.ndarray,
        num_paths: int = 10,
        S0: float | None = None,
        mu: float | None = None,
        sigma: float | None = None,
        T: float | None = None,
    ):
        S0 = self.S0 if S0 is None else float(S0)
        mu = self.r if mu is None else float(mu)
        sigma = self.sigma if sigma is None else float(sigma)
        T = self.option.T if T is None else float(T)

        summary = price_paths_macros(self.S0 if S0 is None else S0, paths)

        time_grid = np.linspace(0.0, T, self.steps + 1)
        idx = np.random.choice(self.simulations, size=num_paths, replace=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        for j in idx:
            ax.plot(time_grid, paths[:, j], lw=0.1, color="black")

        metric_keys = [
            "Mean",
            "Median",
            "25% Quantile",
            "75% Quantile",
            "5% Quantile",
            "95% Quantile"
        ]

        metric_map = {
            "Mean": "mean_path",
            "Median": "median_path",
            "25% Quantile": "25th_percentile",
            "75% Quantile": "75th_percentile",
            "5% Quantile": "5th_percentile",
            "95% Quantile": "95th_percentile"
        }

        current_key = "Mean"
        metric_line, = ax.plot(time_grid, summary[metric_map[current_key]], linewidth=2.5, label=current_key, color="green")

        ax.set_title("Monte-Carlo-Simulations (GBM)")
        ax.set_xlabel("days")
        ax.set_ylabel("price")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        ax.legend(loc="upper left")

        rax = ax.inset_axes([0.72, 0.67, 0.26, 0.30])
        rax.set_facecolor("#a4d1f8")
        radio = RadioButtons(rax, metric_keys, active=metric_keys.index(current_key))

        def on_click(label):
            metric_line.set_ydata(summary[metric_map[label]])
            metric_line.set_label(label)
            ax.legend(loc="upper left")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        radio.on_clicked(on_click)
        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.12
        )
        plt.show(block=True)
        plt.close(fig)

    """
    Plots per-path payoffs (optionally discounted) and the cumulative mean across paths.

    Parameters:
    S0 (float | None) - Initial price override
    mu (float | None) - Drift override
    sigma (float | None) - Volatility override
    T (float | None) - Time to maturity override
    steps (int) - Time steps used in simulation
    n_paths (int) - Number of Monte Carlo paths
    seed (int | None) - Random seed
    discount (bool) - Whether to discount payoffs

    Return:
    dict - Arrays and summary results (ST, payoffs, values, cum_mean, mc_estimate, bs_price)
    """
    def plot_option_payoff(
        self,
        S0: float | None = None,
        mu: float | None = None,
        sigma: float | None = None,
        T: float | None = None,
        steps: int = 252,
        n_paths: int = 1000,
        seed: int | None = 42,
        discount: bool = True
    ):
        S0 = self.S0 if S0 is None else float(S0)
        mu = self.r if mu is None else float(mu)
        sigma = self.sigma if sigma is None else float(sigma)
        T = self.option.T if T is None else float(T)

        rng = np.random.default_rng(seed)
        dt = T / steps

        Z = rng.standard_normal((n_paths, steps))
        log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        ST = S0 * np.exp(np.sum(log_inc, axis=1))

        if self.option.option_type == "call":
            payoffs = np.maximum(ST - self.option.K, 0.0)
        else:
            payoffs = np.maximum(self.option.K - ST, 0.0)

        values = payoffs if not discount else (np.exp(-self.r * T) * payoffs)

        idx = np.arange(1, n_paths + 1)
        cum_mean = np.cumsum(values) / idx

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.plot(idx, values, linewidth=0.5, alpha=0.6,
                label="Payoff per path" if not discount else "Discounted payoff per path", color="black")
        ax.plot(idx, cum_mean, linewidth=2.5, label="Cumulative mean", color="yellow")

        ax.axhline(self.option.premium, linestyle="--", linewidth=1.5, label="Black-Scholes price", color="red")

        ax.set_title("Payoffs and cumulative mean (Monte Carlo)")
        ax.set_xlabel("Number of simulation")
        ax.set_ylabel("Payoff")
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

        return {
            "ST": ST,
            "payoffs": payoffs,
            "values": values,
            "cum_mean": cum_mean,
            "mc_estimate": float(cum_mean[-1]),
            "bs_price": self.option.premium,
        }
