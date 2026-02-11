import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .stock import Stock


class Portfolio:
    """
    Initializes the portfolio with a dictionary of positions and a cash balance.

    Parameters:
    positions (dict | None) - Dict with Stock objects as keys and share quantities as values
    cash (float) - Initial cash balance
    hist (bool) - True if the portfolio is a historic portfolio; False if today's value is examined

    Return:
    None
    """
    def __init__(self, positions: dict | None = None, cash: float = 0.0, hist: bool = True):
        self.positions = dict(positions) if positions else {}
        self.cash = float(cash)
        self.value = float(cash)
        self.hist = hist

        if self.positions and hist:
            for stock, quantity in self.positions.items():
                self.value += float(stock.prices.iloc[0]) * float(quantity)
        elif self.positions and not hist:
            for stock, quantity in self.positions.items():
                self.value += float(stock.prices.iloc[-1]) * float(quantity)



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
        if not self.hist:
            raise ValueError("Must be historical portfolio to run backtests")

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

        weights_df = asset_values_df.div(total_value, axis=0).fillna(0.0)
        weights_cash = (cash_s / total_value).fillna(0.0)

        asset_returns_df = prices_df.pct_change() * 100
        port_returns = port_value.pct_change() * 100
        
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        fig.patch.set_facecolor("#8ebff3")

        for ax in axes:
            ax.set_facecolor("#0c89f7")
            ax.grid(alpha=0.4, color="white")
            ax.margins(x=0)

        axes[0].plot(idx, port_value.values, label="Portfolio Value", color="blue")
        axes[0].set_title("Portfolio Value")
        axes[0].set_ylabel("Value")
        axes[0].legend()

        axes[1].plot(idx, cash_s.values, label="Cash", color="cyan")
        axes[1].set_title("Cash")
        axes[1].set_ylabel("Cash")
        axes[1].legend()

        stack_labels = list(weights_df.columns) + ["Cash"]
        stack_data = [weights_df[c].values for c in weights_df.columns] + [weights_cash.values]
        axes[2].stackplot(idx, *stack_data, labels=stack_labels, alpha=0.6)
        axes[2].set_title("Portfolio Weights (each stock + cash)")
        axes[2].set_ylabel("Weight")
        axes[2].set_ylim(0, 1)
        axes[2].legend(loc="upper left", ncol=2)

        axes[3].plot(idx, port_returns.values, label="Portfolio Return (%)", color="blue")
        for col in asset_returns_df.columns:
            axes[3].plot(idx, asset_returns_df[col].values, label=f"{col} Return (%)", alpha=0.8)
        axes[3].axhline(0, color="white", alpha=0.35, linewidth=1)
        axes[3].set_title("Daily Returns (portfolio + each stock)")
        axes[3].set_ylabel("Return (%)")
        axes[3].legend(loc="upper left", ncol=2)

        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, hspace=0.30)

        plt.show(block=True)
        plt.close(fig)

        print("Final cash:", self.cash)
        print("Final value:", self.value)
        print("Positions:")
        for stock, qty in self.positions.items():
            print(stock.ticker, qty)