from src.stock_option_analyser import *


if __name__ == "__main__":
    # Example usage of Stock and Option classes

    # Creating a Stock instance
    apple = Stock(ticker="AAPL", start_date="2020-01-01", end_date="2025-01-01")
    nvidia = Stock(ticker="NVDA", start_date="2020-01-01", end_date="2025-01-01")
    nike = Stock(ticker="NKE", start_date="2020-01-01", end_date="2025-01-01")
    pg = Stock(ticker="PG", start_date="2020-01-01", end_date="2025-01-01")
    pfizer = Stock(ticker="PFE", start_date="2020-01-01", end_date="2025-01-01")
    spy = Stock(ticker="SPY", start_date="2020-01-01", end_date="2025-01-01")


    # mu, sigma = apple.yearly_mu_sigma()
    # print(mu, sigma)
    # prices = apple.prices.to_numpy()
    # overall_return = (prices[-1] - prices[0]) / prices[0]
    # print(overall_return)
    # # print(apple.get_price_on_date("2021-04-05"), apple.get_return_on_date("2021-04-05"))
    # apple.plot_prices(benchmark_ticker="SPY")
    # apple.plot_daily_returns(benchmark_ticker="SPY")
    # apple.plot_drawdown()
    # apple.plot_price_moving_averages()
    
    # # MC simulation of stock
    # # mc_sim = MonteCarloStock(stock=apple, mid_date="2021-12-29")
    # mc_sim_future = MonteCarloStock(stock=apple, mid_date="2022-08-03")

    # # paths = mc_sim.simulate_price_paths(drift_mode="historical")
    # paths_future = mc_sim_future.simulate_price_paths(drift_mode="historical")
    # # print(price_paths_macros(apple.get_price_on_date("2021-12-31"), paths))
    # # mc_sim.plot_simulated_paths(paths)
    # mc_sim_future.plot_simulated_paths(paths_future, num_paths=100)

    # # Creating an Option instance
    # option = European(option_type="call", S=100, K=120, T=1, sigma=0.2)
    # print(option.premium)
    # print(option.delta())
    # t_range = np.arange(0.1, 1, 0.1)
    # k_range = np.arange(90, 160, 10)
    # st_range = np.arange(80, 160, 1)
    # option.plot_heatmap(T_range=t_range, K_range=k_range)
    # option.plot_pnl(ST_range=st_range)

    # # MC simualtion of option
    # op_mc_sim = MonteCarloOption(option=option)
    # op_mc_sim.plot_option_payoff()
    # paths = op_mc_sim.simulate_price_paths()
    # op_mc_sim.plot_simulated_paths(paths=paths)
    apple_ts = TimeSeriesAsset("AAPL", apple.prices, None)
    nvidia_ts = TimeSeriesAsset("NVDA", nvidia.prices, None)
    nike_ts = TimeSeriesAsset("NKE", nike.prices, None)
    pg_ts = TimeSeriesAsset("PG", pg.prices, None)
    pfizer_ts = TimeSeriesAsset("PFE", pfizer.prices, None)

    bond = Bond(
        name="BUND_2030",
        maturity_date="2030-01-01",
        face_value=1000,
        coupon_rate=0.02,
        market_yield=0.015,
        freq=2
    )
    bond_value = bond.value_series("2020-01-01", "2025-01-01", calendar="B")
    bond_cf = bond.cashflow_series("2020-01-01", "2025-01-01", calendar="B")
    bond_ts = TimeSeriesAsset("BUND_2030", bond_value, bond_cf)
    print(bond_value)
    opt = EuropeanOption(
        option_type="call",
        S=100,
        K=105,
        T=1,
        r=0.02,
        sigma=0.2
    )

    opt_df = opt.simulate(
        start_date="2020-01-01",
        end_date="2025-01-01",
        underlying=apple.prices,
        position="long"
    )
    print(opt_df["position_value"])

    opt_ts = TimeSeriesAsset("AAPL_CALL", opt_df["position_value"], opt_df["cashflow"])

    pf = HistPortfolio(positions={apple_ts: 10, bond_ts: 1, opt_ts: 40}, cash=0)
    pf.backtest(start="2020-01-01", end="2025-01-01")
    analyzer_pf = PortfolioAnalyzer(pf)
    print(pf._bt_cash)
    pf.plot_performance_vs_bench(bench=spy)
    pf.plot_portfolio_value()
    pf.plot_performance()
    pf.plot_portfolio_weights()
    print(analyzer_pf.alpha_beta_portfolio(bench=apple))
    analyzer_pf.plot_pca_variance(n_components=2)
    analyzer_pf.plot_pca_loadings(n_components=2)

    p = HistPortfolio(positions={apple_ts: 6, nvidia_ts: 30, nike_ts: 3, pg_ts: 5, pfizer_ts: 4}, cash=0.0)
    p.backtest(start="2020-01-01", end="2025-01-01", cash=0)
    p.plot_performance_vs_bench(bench=spy)
    p.plot_portfolio_value()
    p.plot_performance()
    p.plot_portfolio_weights()
    # p.plot_portfolio_returns()
    analyzer = PortfolioAnalyzer(p)
    print(apple.prices)
    print(analyzer.alpha_beta_portfolio(bench=apple))
    print(analyzer.return_sigma_sharpe())
    analyzer.plot_pca_variance(n_components=4)
    analyzer.plot_pca_loadings(n_components=4)
    analyzer.plot_pca_factors(n_components=4)





