import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .stock import Stock
from .helper import alpha_beta
from matplotlib.widgets import RadioButtons, CheckButtons


class HistoricalPortfolio:
    """
    Initializes the portfolio with a dictionary of positions and a cash balance.

    Parameters:
    positions (dict | None) - Dict with Stock objects as keys and share quantities as values
    cash (float) - Initial cash balance
    hist (bool) - True if the portfolio is a historic portfolio; False if today's value is examined

    Return:
    None
    """
    def __init__(self, positions: dict | None = None, cash: float = 0.0):
        self.positions = dict(positions) if positions else {}
        self.cash = float(cash)
        self.value = float(cash)

        if self.positions:
            for stock, quantity in self.positions.items():
                self.value += float(stock.prices.iloc[0]) * float(quantity)


    """
    Updates the portfolio value at a given time index without changing positions.

    Parameters:
    counter (int) - Index into each stock's price series representing the evaluation day

    Return:
    float - Updated portfolio value
    """
    def update_hold(self, counter: int):
        if not self.positions:
            self.value = float(self.cash)
            return self.value

        first_stock = next(iter(self.positions))
        if counter >= first_stock.prices.shape[0]:
            raise IndexError("Counter out of Bounds")

        self.value = float(self.cash)
        for stock, quantity in self.positions.items():
            self.value += float(stock.prices.iloc[counter]) * float(quantity)

        return self.value


    """
    Buys positions using a total cash amount and allocation parts per stock.

    Parameters:
    bought_positions (dict) - Dict with Stock keys and allocation parts as values (sum(parts) <= 1)
    price_buy (float) - Total cash amount to invest across bought_positions
    counter (int) - Index into each stock's price series representing the trade day

    Return:
    float - Updated portfolio value after the buy
    """
    def update_buy(self, bought_positions: dict, price_buy: float, counter: int):
        price_buy = float(price_buy)
        if price_buy > self.cash or (not self.positions and not bought_positions):
            return self.update_hold(counter)

        if sum(bought_positions.values()) > 1:
            raise ArithmeticError("Relational positions")

        first_stock = next(iter(bought_positions)) if bought_positions else next(iter(self.positions))
        if counter >= first_stock.prices.shape[0]:
            raise IndexError("Counter out of Bounds")

        for stock, part in bought_positions.items():
            price = float(stock.prices.iloc[counter])
            shares = (price_buy * float(part)) / price
            self.positions[stock] = self.positions.get(stock, 0.0) + shares

        self.cash -= price_buy
        return self.update_hold(counter)
    

    """
    Sells positions for a total cash amount and allocation parts per stock.

    Parameters:
    sold_positions (dict) - Dict with Stock keys and allocation parts as values (sum(parts) <= 1)
    price_sell (float) - Total cash amount to sell across sold_positions
    counter (int) - Index into each stock's price series representing the trade day

    Return:
    float - Updated portfolio value after the sell
    """
    def update_sell(self, sold_positions: dict, price_sell: float, counter: int):
        price_sell = float(price_sell)
        if price_sell > (self.value - self.cash) or not self.positions:
            return self.update_hold(counter)

        if sum(sold_positions.values()) > 1:
            raise ArithmeticError("Relational positions")

        first_stock = next(iter(self.positions))
        if counter >= first_stock.prices.shape[0]:
            raise IndexError("Counter out of Bounds")

        for stock, part in sold_positions.items():
            if stock not in self.positions:
                return self.update_hold(counter)

            price = float(stock.prices.iloc[counter])
            amount = price_sell * float(part)
            shares = amount / price

            self.positions[stock] -= shares
            if self.positions[stock] <= 0:
                del self.positions[stock]

        self.cash += price_sell
        return self.update_hold(counter)


    def backtest(self, start: str, end: str, cash: float = 100.0, trading_stocks: list[Stock] | None  = None):

        if trading_stocks is None:
            trading_stocks = [Stock(ticker="SPY", start_date=start, end_date=end)]
        self.cash += cash
        buy_pos = {}
        for stock in trading_stocks:
            buy_pos[stock] = 1 / len(trading_stocks)

        self.update_buy(buy_pos, price_buy=0.8*self.cash, counter=0)
        print(self.cash)
        print(self.positions)
        print(self.update_hold(0))
        all_stocks = list(self.positions.keys())
        if not all_stocks:
            raise ValueError("No stocks to backtest.")
        pos_name = [stock.ticker for stock in list(self.positions.keys())]
        print(pos_name)

        values = []
        cash_series = []
        counters = []
        shares_hist = {}

        ref_stock = all_stocks[0]
        T = min(len(s.prices) for s in all_stocks)

        for t in range(T):
            if t % 10 == 0:
                sell_dict = {all_stocks[0]: 1.0}
                invested = max(self.value - self.cash, 0.0)
                self.update_sell(sell_dict, price_sell=0.1 * invested, counter=t)

            if t % 5 == 0:
                buy_dict = {all_stocks[0]: 1.0}
                self.update_buy(buy_dict, price_buy=0.05 * self.cash, counter=t)

            values.append(self.update_hold(t))
            cash_series.append(self.cash)
            counters.append(ref_stock.prices.index[t])

            for s in all_stocks:
                shares_hist.setdefault(s, []).append(self.positions.get(s, 0.0))

        idx = pd.to_datetime(counters)

        port_value = pd.Series(values, index=idx, name="Portfolio")
        cash_s = pd.Series(cash_series, index=idx, name="Cash")

        stocks = sorted(all_stocks, key=lambda x: getattr(x, "ticker", str(x)))

        shares_df = pd.DataFrame(
            {s.ticker: pd.Series(shares_hist.get(s, [0.0]*len(idx)), index=idx) for s in stocks},
            index=idx
        )

        prices_df = pd.DataFrame(
            {s.ticker: s.prices.reindex(idx).astype(float) for s in stocks},
            index=idx
        )

        asset_values_df = shares_df * prices_df

        total_value = (asset_values_df.sum(axis=1) + cash_s).replace(0, np.nan)

        self.weights_df = asset_values_df.div(total_value, axis=0).fillna(0.0)
        self.weights_cash = (cash_s / total_value).fillna(0.0)

        self.asset_returns_df = prices_df.pct_change() * 100
        self.port_returns = port_value.pct_change() * 100

        self._bt_index = idx
        self._bt_port_value = port_value
        self._bt_cash = cash_s
        self._bt_asset_values = asset_values_df
        self._bt_prices = prices_df

        print("Final cash:", self._bt_cash)
        print("Final value:", self._bt_port_value)
        print("Positions:")
        for stock, qty in self.positions.items():
            print(stock.ticker, qty)


    """
    Plots portfolio value over time.

    Parameters: 
    None

    Return:
    None
    """
    def plot_portfolio_value(self):
        values = self._bt_port_value
        idx = self._bt_index

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        port_line, = ax.plot(idx, values, linewidth=3, label="Portfolio Value", color="green")
        ax.set_ylabel("USD")
        ax.legend()
        ax.set_xlabel("Date")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)


    """
    Plots portfolio performance (always visible) and allows selecting up to
    five comparison assets via a selector. All series are shown as % since start.
    Alpha, beta and R² are computed for the selected series versus the portfolio.
    The default selection shows no comparison series.

    Parameters:
    max_assets (int) - Maximum number of selectable comparison assets
    trading_days (int) - Trading days per year

    Return:
    None
    """ 
    def plot_performance(
        self,
        max_assets=5,
        trading_days=252
    ):
        if not hasattr(self, "_bt_index"):
            raise ValueError("Run backtest() first so the backtest series are available.")

        if not self.hist:
            raise ValueError("Only available for historical portfolio analysis")

        if not hasattr(self, "_bt_prices"):
            raise ValueError("Missing _bt_prices. Store prices_df as self._bt_prices in backtest().")

        idx = self._bt_index

        def normalize(series: pd.Series) -> pd.Series:
            s = series.astype(float).copy()
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.reindex(idx)
            s = s.dropna()
            if s.empty:
                return pd.Series(index=idx, dtype=float)
            base = float(s.iloc[0])
            if not np.isfinite(base) or np.isclose(base, 0.0):
                return pd.Series(index=idx, dtype=float)
            out = (s / base - 1) * 100
            return out.reindex(idx)

        portfolio_perf = normalize(self._bt_port_value)
        if portfolio_perf.isna().all():
            raise ValueError("Portfolio series is empty after normalization/alignment.")

        asset_perfs = {}
        for col in self._bt_prices.columns:
            perf = normalize(self._bt_prices[col])
            if not perf.isna().all():
                asset_perfs[col] = perf

        asset_names = list(asset_perfs.keys())[:max_assets]
        selectable = ["None"] + asset_names

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("#8ebff3")
        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        port_line, = ax.plot(idx, portfolio_perf.values, linewidth=3, label="Portfolio")

        comp_line, = ax.plot(idx, portfolio_perf.values, linewidth=2, linestyle="--", label="", visible=False)

        info_text = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="#a8d6ff", alpha=0.85)
        )

        def update_legend():
            handles = [port_line]
            labels = ["Portfolio"]
            if comp_line.get_visible() and comp_line.get_label():
                handles.append(comp_line)
                labels.append(comp_line.get_label())
            ax.legend(handles, labels, loc="upper right")

        def autoscale_y():
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        def update_stats(name: str):
            if name == "None":
                info_text.set_text("")
                return

            s = asset_perfs[name].dropna()
            p = portfolio_perf.dropna()
            common = s.index.intersection(p.index)
            s = s.loc[common]
            p = p.loc[common]

            if len(common) < 5:
                info_text.set_text(f"{name} vs Portfolio\nNot enough overlapping data.")
                return

            stats = alpha_beta(
                ticker="Portfolio",
                benchmark_ticker=name,
                ticker_price=p,
                bench_price=s,
                trading_days=trading_days
            )

            info_text.set_text(
                f"Portfolio vs {name}\n"
                f"Alpha (ann): {stats['alpha_annual']:.2%}\n"
                f"Beta: {stats['beta']:.3f}\n"
                f"R²: {stats['r2']:.3f}"
            )

        rax = ax.inset_axes([0.78, 0.05, 0.2, 0.25])
        rax.set_facecolor("#a1d0f8")
        radio = RadioButtons(rax, selectable)

        def on_select(label: str):
            if label == "None":
                comp_line.set_visible(False)
                comp_line.set_label("")
                update_stats("None")
            else:
                series = asset_perfs[label]
                comp_line.set_ydata(series.values)
                comp_line.set_label(label)
                comp_line.set_visible(True)
                update_stats(label)

            ax.set_title("Portfolio Performance vs Selected Asset")
            ax.set_ylabel("Performance (%)")
            update_legend()
            autoscale_y()
            fig.canvas.draw_idle()

        radio.on_clicked(on_select)

        on_select("None")
        ax.set_xlabel("Date")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)


    """
    Plots the weights of the different assets inside the portfolio.
    
    Parameters:
    None
    
    Return:
    None
    """
    def plot_portfolio_weights(self):
        fig, ax = plt.subplots( figsize=(10, 6))
        fig.patch.set_facecolor("#8ebff3")

        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)
        stack_labels = list(self.weights_df) + ["Cash"]
        stack_data = [self.weights_df[c].values for c in self.weights_df.columns] + [self.weights_cash.values]
        ax.stackplot(self._bt_index, *stack_data, labels=stack_labels, alpha=0.6)
        ax.set_title("Portfolio Weights (stocks + cash)")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", ncol=2)
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)


    """
    Plots the returns of the portfolio. Also the the returns of the underlying assets are shown
    
    Parameters:
    None
    
    Return:
    None
    """
    def plot_portfolio_returns(self):
        """
        Plots daily returns of the portfolio and optionally toggles the underlying stock returns.

        Parameters:
        None

        Return:
        None
        """
        if not hasattr(self, "_bt_index") or not hasattr(self, "port_returns") or not hasattr(self, "asset_returns_df"):
            raise ValueError("Run backtest() first.")

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#8ebff3")

        ax.set_facecolor("#0c89f7")
        ax.grid(alpha=0.4, color="white")
        ax.margins(x=0)

        port_line, = ax.plot(
            self._bt_index,
            self.port_returns.values,
            label="Portfolio Return (%)",
            linewidth=2.2,
            color="green"
        )

        asset_lines = []
        for col in self.asset_returns_df.columns:
            ln, = ax.plot(
                self._bt_index,
                self.asset_returns_df[col].values,
                label=f"{col} Return (%)",
                linewidth=0.6,
                visible=False
            )
            asset_lines.append(ln)

        ax.axhline(0, color="white", alpha=0.35, linewidth=1)
        ax.set_title("Daily Returns (portfolio + each stock)")
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Date")

        rax = ax.inset_axes([0.77, 0.03, 0.2, 0.22])
        rax.set_facecolor("#a4d1f8")
        check = CheckButtons(rax, ["Show assets"], [False])

        state = {"show_assets": False}

        def update_legend():
            handles = [port_line]
            labels = [port_line.get_label()]
            if state["show_assets"]:
                for ln in asset_lines:
                    handles.append(ln)
                    labels.append(ln.get_label())
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.legend(handles, labels, loc="upper left", ncol=2)

        def toggle(label):
            if label == "Show assets":
                state["show_assets"] = not state["show_assets"]
                for ln in asset_lines:
                    ln.set_visible(state["show_assets"])
                update_legend()
                fig.canvas.draw_idle()

        check.on_clicked(toggle)

        update_legend()
        fig.autofmt_xdate(rotation=30)

        plt.show(block=True)
        plt.close(fig)



