import matplotlib.pyplot as plt


class PortfolioAnalyzerPlotMixin:
    """
    Plotting mixin for PortfolioAnalyzer.

    This mixin assumes the parent class provides the method:
        pca_portfolio_simple(n_components)

    which must return a dictionary containing:
        - "explained_variance_ratio"
        - "loadings"
        - "factors_ts"
    """


    """
    Plot the explained variance ratio of the PCA components.

    Parameters:
    n_components : int
        Number of principal components used in the PCA.
    """
    def plot_pca_variance(self, n_components: int = 2):
        res = self.pca(n_components=n_components)
        ev = res["explained_variance_ratio"]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        ax.bar(ev.index, ev.values, color="orange")

        ax.set_title("PCA Explained Variance")
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.tick_params()
        ax.grid(alpha=0.4, color="white")

        plt.show()
        plt.close(fig)


    """
    Plot the PCA loadings (asset contributions to each component).

    Parameters:
    n_components : int
        Number of principal components used in the PCA.
    """
    def plot_pca_loadings(self, n_components: int = 2):

        res = self.pca(n_components=n_components)
        loadings = res["loadings"]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        loadings.plot(kind="bar", ax=ax)

        ax.set_title("PCA Asset Loadings")
        ax.set_ylabel("Loading")
        ax.tick_params()
        ax.grid(alpha=0.4, color="white")
        ax.axhline(0, color="white", linestyle="--", alpha=0.8)

        legend = ax.legend()
        legend.get_frame().set_facecolor("#8ebff3")
        for text in legend.get_texts():
            text.set_color("white")

        plt.show()
        plt.close(fig)


    """
    Plot the time series of the PCA factors.

    These are the projections of the weighted returns onto the
    principal component space.

    Parameters:
    n_components : int
        Number of principal components used in the PCA.
    """
    def plot_pca_factors(self, n_components: int = 2):
        res = self.pca(n_components=n_components)
        factors = res["factors_ts"]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")

        factors.plot(ax=ax)

        ax.set_title("PCA Factor Time Series")
        ax.tick_params()
        ax.margins(x=0)
        ax.grid(alpha=0.4, color="white")
        ax.axhline(0, color="white", linestyle="--", alpha=0.8)
        legend = ax.legend()
        legend.get_frame().set_facecolor("#8ebff3")
        for text in legend.get_texts():
            text.set_color("white")

        plt.show()
        plt.close(fig)
