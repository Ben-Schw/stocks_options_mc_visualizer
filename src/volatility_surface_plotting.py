import numpy as np
import matplotlib.pyplot as plt


class VolatilitySurfacePlotMixin:
    """
    Plotting mixin for VolatilitySurface.

    Requirements:
    - self.K_range, self.T_range, self.surface

    Return:
    None
    """

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
