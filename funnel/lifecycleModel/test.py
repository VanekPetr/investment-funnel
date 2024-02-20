from funnel.models.main import TradeBot


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(
        start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=5, plot=False
    )

    # RUN THE LIFECYCLE
    lifecycle = algo.scenario_analysis(
        subset_of_assets=mst_subset_of_assets,
        scenarios_type="MonteCarlo",
        n_simulations=1000,
        end_year=2050,
        risk_test="investmentFunnel",
        risk_class=[3, 4, 5, 6, 7],
    )
