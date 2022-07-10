import numpy as np
import math


# ----------------------------------------------------------------------
# Scenario Generation: THE MONTE CARLO METHOD
# ----------------------------------------------------------------------
def monte_carlo(data, n_simulations, n_test):
    """
    Monte Carlo simulations
    """

    n_test = n_test + 4  # +4 when no testing data but    

    n_iter = 4
    n_simulations = n_simulations  # 250 scenarios for each period
    n_indices = data.shape[1]

    sigma = np.cov(data, rowvar=False)  # The covariance matrix
    # RHO = np.corrcoef(ret_train, rowvar=False)    # The correlation matrix 
    mu = np.mean(data, axis=0)  # The mean array
    # sd = np.sqrt(np.diagonal(SIGMA))              # The standard deviation
    n_rolls = math.floor(n_test/n_iter)

    sim = np.zeros((n_test, n_simulations, n_indices), dtype=float)  # Match GAMS format

    print('-------Simulating Weekly Returns-------')
    for week in range(n_test):
        sim[week, :, :] = np.random.multivariate_normal(mean=mu, cov=sigma, size=n_simulations)

    monthly_sim = np.zeros((n_rolls, n_simulations, n_indices))

    print('-------Computing Monthly Returns-------')
    for roll in range(n_rolls):
        roll_mult = roll * n_iter
        for s in range(n_simulations):
            for index in range(n_indices):
                tmp_rets = 1 + sim[roll_mult:(roll_mult + 4), s, index]
                monthly_sim[roll, s, index] = np.prod(tmp_rets) - 1

    return monthly_sim


# ----------------------------------------------------------------------
# Scenario Generation: THE BOOTSTRAPPING METHOD
# ----------------------------------------------------------------------
def bootstrapping(data, n_simulations, n_test):
    n_iter = 4  # 4 weeks compounded in our scenario                                                         
    n_train_weeks = len(data.index) - n_test
    n_indices = data.shape[1]
    n_simulations = n_simulations
    n_rolls = math.floor(n_test / n_iter) + 1

    sim = np.zeros((int(n_rolls), n_simulations, n_indices, n_iter), dtype=float)
    monthly_sim = np.ones((int(n_rolls), n_simulations, n_indices,))
    for p in range(int(n_rolls)):
        for s in range(n_simulations):
            for w in range(n_iter):
                random_num = np.random.randint(4 * p, n_train_weeks + 4 * p)
                sim[p, s, :, w] = data.iloc[random_num, :]
                monthly_sim[p, s, :] *= (1 + sim[p, s, :, w])
            monthly_sim[p, s, :] += -1

    return monthly_sim
