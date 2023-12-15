import numpy as np

# def test_monte_carlo_scenarios(mc_scenarios):
#     expected_scenarios = np.load("tests/scgen/scenarios_BASE.npz")["scenarios"]
#
#     # np.savez_compressed("scgen/scenarios_BASE.npz", scenarios=mc_scenarios)
#     np.testing.assert_array_equal(mc_scenarios, expected_scenarios)


def test_moments(moments, resource_dir):
    expected_sigmas = np.load(resource_dir / "scgen/sigma_list_BASE.npz")
    expected_sigma_list = list(expected_sigmas[k] for k in expected_sigmas)
    expected_mus = np.load(resource_dir / "scgen/mu_list_BASE.npz")
    expected_mu_list = list(expected_mus[k] for k in expected_mus)

    sigma_list, mu_list = moments
    # np.savez_compressed(
    #     "scgen/mu_list_BASE.npz",
    #     **dict(zip([f"mu_{i}" for i in range(len(mu_list))], mu_list))
    # )
    # np.savez_compressed(
    #     "scgen/sigma_list_BASE.npz",
    #     **dict(zip([f"sigma_{i}" for i in range(len(sigma_list))], sigma_list))
    # )

    # on different architectures you will not replicate results exactly
    np.testing.assert_array_almost_equal(mu_list, expected_mu_list)
    np.testing.assert_array_almost_equal(sigma_list, expected_sigma_list)
