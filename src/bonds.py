import pandas as pd
import numpy as np


class Bond:
    """
    Simple fixed-rate bond with constant market yield.

    Parameters:
    maturity_date (str) - Bond maturity in "YYYY-MM-DD".
    face_value (float) - Face value (notional).
    coupon_rate (float) - Annual coupon rate (e.g. 0.03 for 3%).
    market_yield (float) - Constant annual market yield.
    freq (int) - Coupon payments per year (1, 2, 4, 12).

    Notes:
    - Coupon schedule is generated backward from maturity in steps of 12/freq months.
    - Discounting uses ACT/365 year fraction.
    """
    def __init__(
        self,
        name: str,
        maturity_date: str,
        face_value: float,
        coupon_rate: float,
        market_yield: float,
        freq: int = 2
    ):
        self.name = str(name)
        self.maturity_date = pd.Timestamp(maturity_date).normalize()
        self.face_value = float(face_value)
        self.coupon_rate = float(coupon_rate)
        self.market_yield = float(market_yield)
        self.freq = int(freq)

        if self.freq <= 0:
            raise ValueError("freq must be a positive integer.")
        months = 12 / self.freq
        if abs(months - round(months)) > 1e-9:
            raise ValueError("freq must evenly divide 12 (e.g. 1, 2, 3, 4, 6, 12).")

        self._months_step = int(round(months))

    def _year_fraction_act365(self, d1: pd.Timestamp, d2: pd.Timestamp) -> float:
        return (d2 - d1).days / 365.0

    def _payment_schedule_from(self, start_date: pd.Timestamp) -> pd.DataFrame:
        if start_date >= self.maturity_date:
            return pd.DataFrame({"cashflow": []}, index=pd.DatetimeIndex([], name="date"))

        dates = []
        d = self.maturity_date
        while d >= start_date:
            dates.append(d)
            d = (d - pd.DateOffset(months=self._months_step)).normalize()

        dates = sorted(set(dates))
        coupon = self.face_value * self.coupon_rate / self.freq
        cfs = np.full(len(dates), coupon, dtype=float)
        cfs[-1] += self.face_value

        return pd.DataFrame({"cashflow": cfs}, index=pd.DatetimeIndex(dates, name="date"))

    """
    Returns present value of the bond on a given valuation date.

    Parameters:
    valuation_date (str) - Valuation date in "YYYY-MM-DD".

    Return:
    float - Present value based on remaining cashflows discounted at market_yield.
    """
    def value_on(self, valuation_date: str) -> float:
        vd = pd.Timestamp(valuation_date).normalize()
        if vd >= self.maturity_date:
            return 0.0

        sched = self._payment_schedule_from(vd)
        if sched.empty:
            return 0.0

        times = np.array([self._year_fraction_act365(vd, d) for d in sched.index], dtype=float)
        df = 1.0 / (1.0 + self.market_yield) ** times
        return float(np.sum(sched["cashflow"].to_numpy(dtype=float) * df))

    """
    Returns the sum of cashflows whose payment dates occur within (start_date, end_date].

    Parameters:
    start_date (str) - Start date in "YYYY-MM-DD" (exclusive).
    end_date (str) - End date in "YYYY-MM-DD" (inclusive).

    Return:
    float - Sum of due cashflows in the interval.
    """
    def cash_due_between(self, start_date: str, end_date: str) -> float:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        if e <= s:
            return 0.0

        sched = self._payment_schedule_from(s + pd.Timedelta(days=1))
        if sched.empty:
            return 0.0

        due = sched.loc[(sched.index > s) & (sched.index <= e)]
        return float(due["cashflow"].sum()) if not due.empty else 0.0

    """
    Returns a time series of bond values for all dates between start_date and end_date.

    Parameters:
    start_date (str) - Start date in "YYYY-MM-DD".
    end_date (str) - End date in "YYYY-MM-DD".
    calendar (str) - "D" for daily or "B" for business days.

    Return:
    pd.Series - Bond value series indexed by dates.
    """
    def value_series(self, start_date: str, end_date: str, calendar: str = "B") -> pd.Series:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        idx = pd.date_range(s, e, freq=calendar)

        vals = [self.value_on(d.strftime("%Y-%m-%d")) for d in idx]
        return pd.Series(vals, index=idx, name="bond_value")

    """
    Returns a time series of cashflows for all dates between start_date and end_date.

    Parameters:
    start_date (str) - Start date in "YYYY-MM-DD".
    end_date (str) - End date in "YYYY-MM-DD".
    calendar (str) - "D" for daily or "B" for business days.

    Return:
    pd.Series - Cashflow series indexed by dates (0 on non-payment dates).
    """
    def cashflow_series(self, start_date: str, end_date: str, calendar: str = "B") -> pd.Series:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        idx = pd.date_range(s, e, freq=calendar)

        sched = self._payment_schedule_from(s)
        cf = pd.Series(0.0, index=idx, name="cashflow")

        if not sched.empty:
            due = sched.loc[(sched.index >= s) & (sched.index <= e)]
            if not due.empty:
                due_series = due["cashflow"]
                due_series.index = due_series.index.normalize()
                cf = cf.add(due_series.reindex(cf.index, fill_value=0.0), fill_value=0.0)

        return cf

