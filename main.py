import numpy as np
from src.stock import Stock     
from src.option import Option
from src.monte_carlo import MonteCarloOption, MonteCarloStock
from src.helper import price_paths_macros

if __name__ == "__main__":
    # Example usage of Stock and Option classes

    # Creating a Stock instance
    apple = Stock(ticker="AAPL", start_date="2022-01-01", end_date="2023-01-01")
    # mu, sigma = apple.yearly_mu_sigma()
    # print(mu, sigma)
    # prices = apple.prices.to_numpy()
    # overall_return = (prices[-1] - prices[0]) / prices[0]
    # print(overall_return)
    # print(apple.get_price_on_date("2021-04-05"), apple.get_return_on_date("2021-04-05"))
    apple.plot_prices(benchmark_ticker="SPY")
    apple.plot_daily_returns(benchmark_ticker="SPY")
    apple.plot_drawdown()
    
    # MC simulation of stock
    # mc_sim = MonteCarloStock(stock=apple, mid_date="2021-12-29")
    mc_sim_future = MonteCarloStock(stock=apple, mid_date="2022-08-03")

    # paths = mc_sim.simulate_price_paths(drift_mode="historical")
    paths_future = mc_sim_future.simulate_price_paths(drift_mode="historical")
    # print(price_paths_macros(apple.get_price_on_date("2021-12-31"), paths))
    # mc_sim.plot_simulated_paths(paths)
    mc_sim_future.plot_simulated_paths(paths_future, num_paths=100)

    # Creating an Option instance
    option = Option(option_type="call", S=100, K=120, T=1, sigma=0.2)
    print(option.premium)
    print(option.delta_hedge_ratio())
    t_range = np.arange(0.1, 1, 0.1)
    k_range = np.arange(90, 160, 10)
    st_range = np.arange(80, 160, 1)
    option.plot_heatmap(T_range=t_range, K_range=k_range)
    option.plot_pnl(ST_range=st_range)

    # MC simualtion of option
    op_mc_sim = MonteCarloOption(option=option)
    op_mc_sim.plot_option_payoff()
    paths = op_mc_sim.simulate_price_paths()
    op_mc_sim.plot_simulated_paths(paths=paths)


