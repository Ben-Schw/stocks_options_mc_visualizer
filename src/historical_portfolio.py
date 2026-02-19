import pandas as pd
import numpy as np
from .stock import Stock


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
        start = list(self.positions.keys())[0].start_date
        end = list(self.positions.keys())[0].end_date

        if self.positions:
            for stock, quantity in self.positions.items():
                self.value += float(stock.prices.iloc[0]) * float(quantity)
            for stock in list(self.positions.keys())[1:]:
                if start is not stock.start_date or end is not stock.end_date:
                    raise ValueError("All stocks must be initalized over the same time frame.")

    """
    Updates the portfolio value at a given time index without changing positions.

    Parameters:
    counter (int) - Index into each stock's price series representing the evaluation day

    Return:
    float - Updated portfolio value
    """
    def _update_hold(self, counter: int):
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
    def _update_buy(self, bought_positions: dict, price_buy: float, counter: int):
        price_buy = float(price_buy)
        if price_buy > self.cash or (not self.positions and not bought_positions):
            return self._update_hold(counter)

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
        return self._update_hold(counter)
    

    """
    Sells positions for a total cash amount and allocation parts per stock.

    Parameters:
    sold_positions (dict) - Dict with Stock keys and allocation parts as values (sum(parts) <= 1)
    price_sell (float) - Total cash amount to sell across sold_positions
    counter (int) - Index into each stock's price series representing the trade day

    Return:
    float - Updated portfolio value after the sell
    """
    def _update_sell(self, sold_positions: dict, price_sell: float, counter: int):
        price_sell = float(price_sell)
        if price_sell > (self.value - self.cash) or not self.positions:
            return self._update_hold(counter)

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
        return self._update_hold(counter)
    

    """
    Method to backtest the portfolio based on own rules. Should be run before portfolio can be properly analysed.

    Parameters:
    start (str) - start date of the backtest (should be the same date used for initialisation of stock)
    end (str) - end date of the backtest (should be the same date used for initialisation of stock)
    cash (float) - amount of cash used for backtest (It is possible to start with an empty portfolio)
    trading_stocks (list) - lsit of stocks that will be traded

    Return:
    None 
    """
    def backtest(self, start: str | None = None, end: str | None = None, cash: float = 0.0, trading_stocks: list[Stock] | None = None):
        self.cash += cash
        # print(self.cash)
        # print(self.positions)
        # print(self._update_hold(0))
        all_stocks = list(self.positions.keys())
        for stock in list(self.positions.keys()):
            if start is not stock.start_date or end is not stock.end_date:
                raise ValueError("Backtest must run over the same time frame as stocks.")

        values = []
        cash_series = []
        counters = []
        shares_hist = {}

        ref_stock = all_stocks[0]
        T = min(len(s.prices) for s in all_stocks)

        for t in range(T):
            values.append(self._update_hold(t))
            cash_series.append(self.cash)
            counters.append(ref_stock.prices.index[t])

            for s in all_stocks:
                shares_hist.setdefault(s, []).append(self.positions.get(s, 0.0))

        idx = pd.to_datetime(counters)

        port_value = pd.Series(values, index=idx, name="Portfolio")
        cash_s = pd.Series(cash_series, index=idx, name="Cash")

        stocks = sorted(all_stocks, key=lambda x: getattr(x, "ticker", str(x)))

        shares_df = pd.DataFrame(
            {s.ticker: pd.Series(shares_hist.get(s, [0.0] * len(idx)), index=idx) for s in stocks},
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

        # print("Final cash:", self._bt_cash)
        # print("Final value:", self._bt_port_value)
        # print("Positions:")
        # for stock, qty in self.positions.items():
        #     print(stock.ticker, qty)
