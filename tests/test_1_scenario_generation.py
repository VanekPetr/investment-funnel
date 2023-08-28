import pytest
from pathlib import Path
import numpy as np


TEST_DIR = Path(__file__).parent


def test_monte_carlo_scenarios(mc_scenarios):
    expected_scenarios = np.load("tests/scgen/scenarios_BASE.npz")["scenarios"]

    #np.savez_compressed("tests/scgen/scenarios_ACTUAL.npz", scenarios=scenarios)
    np.testing.assert_array_equal(mc_scenarios, expected_scenarios)


def test_moments(moments):
    expected_sigmas = np.load("tests/scgen/sigma_list_BASE.npz")
    expected_sigma_list = list((expected_sigmas[k] for k in expected_sigmas))
    expected_mus = np.load("tests/scgen/mu_list_BASE.npz")
    expected_mu_list = list((expected_mus[k] for k in expected_mus))
    
    sigma_list, mu_list = moments
    #np.savez_compressed("tests/scgen/mu_list_ACTUAL.npz", **dict(zip([f"mu_{i}" for i in range(len(mu_list))], mu_list)))
    #np.savez_compressed("tests/scgen/sigma_list_ACTUAL.npz", **dict(zip([f"sigma_{i}" for i in range(len(sigma_list))], sigma_list)))
    np.testing.assert_array_equal(mu_list, expected_mu_list)
    np.testing.assert_array_equal(sigma_list, expected_sigma_list)





