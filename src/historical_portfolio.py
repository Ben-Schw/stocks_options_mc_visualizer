import pandas as pd
import numpy as np
from .helper import TimeSeriesAsset


class HistoricalPortfolio:
    """
    Initializes the portfolio with a dictionary of positions and a cash balance.

    Parameters:
    positions (dict | None) - Dict with asset objects as keys and share quantities as values
    cash (float) - Initial cash balance
    hist (bool) - True if the portfolio is a historic portfolio; False if today's value is examined

    Return:
    None
    """
    def __init__(self, positions: dict | None = None, cash: float = 0.0):
        self.positions = dict(positions) if positions else {}
        self.cash = float(cash)
        self.value = float(self.cash)

        if not self.positions:
            return

        assets = list(self.positions.keys())

        common_idx = None
        for a in assets:
            if not hasattr(a, "value"):
                raise ValueError("All assets must provide a .value pd.Series.")
            cur = pd.to_datetime(a.value.index).normalize()
            common_idx = cur if common_idx is None else common_idx.intersection(cur)

        common_idx = pd.DatetimeIndex(common_idx).sort_values().unique()
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between assets.")

        for a in assets:
            a.value.index = pd.to_datetime(a.value.index).normalize()
            a.value = a.value.reindex(common_idx).ffill()

            if hasattr(a, "cashflow"):
                a.cashflow.index = pd.to_datetime(a.cashflow.index).normalize()
                a.cashflow = a.cashflow.reindex(common_idx).fillna(0.0)

        for asset, quantity in self.positions.items():
            self.value += float(asset.value.iloc[0]) * float(quantity)

    """
    Updates the portfolio value at a given time index without changing positions.

    Parameters:
    counter (int) - Index into each asset's price series representing the evaluation day

    Return:
    float - Updated portfolio value
    """
    def _update_hold(self, counter: int):
        if not self.positions:
            self.value = float(self.cash)
            return self.value

        first_asset = next(iter(self.positions))
        if counter >= first_asset.value.shape[0]:
            raise IndexError("Counter out of Bounds")

        self.value = float(self.cash)
        for asset, quantity in self.positions.items():
            self.value += float(asset.value.iloc[counter]) * float(quantity)

        return self.value
    

    """
    Buys positions using a total cash amount and allocation parts per asset.

    Parameters:
    bought_positions (dict) - Dict with asset keys and allocation parts as values (sum(parts) <= 1)
    price_buy (float) - Total cash amount to invest across bought_positions
    counter (int) - Index into each asset's price series representing the trade day

    Return:
    float - Updated portfolio value after the buy
    """
    def _update_buy(self, bought_positions: dict, price_buy: float, counter: int):
        price_buy = float(price_buy)
        if price_buy > self.cash or (not self.positions and not bought_positions):
            return self._update_hold(counter)

        if sum(bought_positions.value()) > 1:
            raise ArithmeticError("Relational positions")

        first_asset = next(iter(bought_positions)) if bought_positions else next(iter(self.positions))
        if counter >= first_asset.value.shape[0]:
            raise IndexError("Counter out of Bounds")

        for asset, part in bought_positions.items():
            price = float(asset.value.iloc[counter])
            shares = (price_buy * float(part)) / price
            self.positions[asset] = self.positions.get(asset, 0.0) + shares

        self.cash -= price_buy
        return self._update_hold(counter)
    

    """
    Sells positions for a total cash amount and allocation parts per asset.

    Parameters:
    sold_positions (dict) - Dict with asset keys and allocation parts as values (sum(parts) <= 1)
    price_sell (float) - Total cash amount to sell across sold_positions
    counter (int) - Index into each asset's price series representing the trade day

    Return:
    float - Updated portfolio value after the sell
    """
    def _update_sell(self, sold_positions: dict, price_sell: float, counter: int):
        price_sell = float(price_sell)
        if price_sell > (self.value - self.cash) or not self.positions:
            return self._update_hold(counter)

        if sum(sold_positions.values()) > 1:
            raise ArithmeticError("Relational positions")

        first_asset = next(iter(self.positions))
        if counter >= first_asset.value.shape[0]:
            raise IndexError("Counter out of Bounds")

        for asset, part in sold_positions.items():
            if asset not in self.positions:
                return self._update_hold(counter)

            price = float(asset.value.iloc[counter])
            amount = price_sell * float(part)
            shares = amount / price

            self.positions[asset] -= shares
            if self.positions[asset] <= 0:
                del self.positions[asset]

        self.cash += price_sell
        return self._update_hold(counter)
    

    """
    Method to backtest the portfolio based on own rules. Should be run before portfolio can be properly analysed.

    Parameters:
    start (str) - start date of the backtest (should be the same date used for initialisation of asset)
    end (str) - end date of the backtest (should be the same date used for initialisation of asset)
    cash (float) - amount of cash used for backtest (It is possible to start with an empty portfolio)
    trading_assets (list) - lsit of assets that will be traded

    Return:
    None 
    """
    def backtest(self, start: str, end: str, cash: float = 0.0, trading_assets: list[TimeSeriesAsset] | None = None):
        values = []
        cash_series = []
        counters = []
        shares_hist = {}

        all_assets = list(self.positions.keys())
        if not all_assets:
            raise ValueError("Portfolio has no positions; provide positions or start with cash and trading assets.")

        ref = next(iter(all_assets))
        if hasattr(ref, "value"):
            idx = pd.to_datetime(ref.value.index)
        else:
            raise ValueError("Cannot infer timeline. Provide at least one asset or a TimeSeriesAsset with an index.")

        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]

        ts_assets = {}
        for a in all_assets:
            if hasattr(a, "value") and hasattr(a, "cashflow"):
                ts_assets[a] = a
            else:
                raise ValueError("Each position must be a asset or a TimeSeriesAsset.")

        value_series_map = {a: ts_assets[a].value_on_index(idx) for a in all_assets}
        cashflow_series_map = {a: ts_assets[a].cashflow_on_index(idx) for a in all_assets}

        for t in range(len(idx)):
            d = idx[t]

            for a, quantity in self.positions.items():
                cf = float(cashflow_series_map[a].iloc[t]) * float(quantity)
                self.cash += cf

            self.value = float(self.cash)
            for a, quantity in self.positions.items():
                self.value += float(value_series_map[a].iloc[t]) * float(quantity)

            values.append(self.value)
            cash_series.append(self.cash)
            counters.append(d)

            for a in all_assets:
                shares_hist.setdefault(a, []).append(self.positions.get(a, 0.0))

        port_value = pd.Series(values, index=idx, name="Portfolio")
        cash_s = pd.Series(cash_series, index=idx, name="Cash")

        assets_sorted = sorted(all_assets, key=lambda x: getattr(ts_assets[x], "ticker", str(x)))

        qty_df = pd.DataFrame(
            {ts_assets[a].ticker: pd.Series(shares_hist.get(a, [0.0] * len(idx)), index=idx) for a in assets_sorted},
            index=idx
        )

        prices_df = pd.DataFrame(
            {ts_assets[a].ticker: value_series_map[a].reindex(idx).astype(float) for a in assets_sorted},
            index=idx
        )

        asset_values_df = qty_df * prices_df
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
