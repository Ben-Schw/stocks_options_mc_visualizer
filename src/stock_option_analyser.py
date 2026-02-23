from .option import EuropeanOption, AmericanOption
from .option_plotting import OptionPlotMixin

class European(EuropeanOption, OptionPlotMixin):
    pass

class American(AmericanOption, OptionPlotMixin):
    pass


from .stock import StockCore
from .stock_plotting import StockPlotMixin

class Stock(StockCore, StockPlotMixin):
    pass

from .monte_carlo_plotting import MonteCarloStockPlotMixin, MonteCarloOptionPlotMixin
from .monte_carlo import MonteCarloStock, MonteCarloOption


class MonteCarloStock(MonteCarloStock, MonteCarloStockPlotMixin):
    pass


class MonteCarloOption(MonteCarloOption, MonteCarloOptionPlotMixin):
    pass

from .volatility_surface import VolatilitySurface
from .volatility_surface_plotting import VolatilitySurfacePlotMixin

class VolSurface(VolatilitySurface, VolatilitySurfacePlotMixin):
    pass

from .historical_portfolio import HistoricalPortfolio
from .historical_portfolio_plotting import HistoricalPortfolioPlotMixin

class HistPortfolio(HistoricalPortfolio, HistoricalPortfolioPlotMixin):
    pass

from .portfolio_analysis import PortfolioAnalyzerRaw
from .portfolio_analysis_plotting import PortfolioAnalyzerPlotMixin

class PortfolioAnalyzer(PortfolioAnalyzerRaw, PortfolioAnalyzerPlotMixin):
    pass

from .bonds import Bond
from .helper import TimeSeriesAsset