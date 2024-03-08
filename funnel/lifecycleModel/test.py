from funnel.models.main import TradeBot

if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(
        start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=4, plot=False
    )
    """
    # RUN THE CLUSTERING METHOD
    _, clustering_subset_of_assets = algo.clustering(
        start_date="2000-01-01",
        end_date="2024-01-01",
        n_clusters=3,
        n_assets=15,
        plot=True,
    )
    """
    # RUN THE LIFECYCLE
    lifecycle = algo.scenario_analysis(
        subset_of_assets=mst_subset_of_assets,
        scenarios_type="MonteCarlo",  # ,  # Bootstrap,
        n_simulations=1000,
        end_year=2050,
        withdrawals=35000,
        initial_risk_appetite=0.16,
        initial_budget=1000000,
        rng_seed=19,
        test_split=0,
    )
