import pandas as pd
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt


def log_diff_exp(x1: float, x2: float) -> float:
    if x1 > x2:
        a = x1
        b = x2
    else:
        a = x2
        b = x1
    value = a + np.log(1 - np.exp(b - a))
    return value


def main():
    n_live = 100
    n_iter = 100  # max no. of iterations to calculate the failure probability
    sigma_th = 3  # number of sigmas within likelihood mean we need a sample to not have a failure

    # parameters for the prior distribution
    mu_prior = 0
    sigma_prior = 4

    # parameters for the likelihood distribution
    sigma_like = 0.22
    mu_like_min = mu_prior
    mu_like_max = mu_prior + sigma_prior * 2 * sigma_th
    mu_like_step = 0.1

    mu_like_lab = "Likelihood Mean"
    p_fail_lab = "Failure Probability"
    df_p_fail = pd.DataFrame(columns=[mu_like_lab, p_fail_lab])
    df_p_fail.set_index(mu_like_lab, inplace=True)

    for mu_like in np.arange(mu_like_min, mu_like_max + mu_like_step, mu_like_step):
        failure_boundary = mu_like - sigma_like * sigma_th
        log_p_fail = n_live * norm.logcdf(failure_boundary, loc=mu_prior, scale=sigma_prior)
        log_sf = norm.logsf(failure_boundary, loc=mu_prior, scale=sigma_prior)
        for i in range(n_iter):
            log_expected_prior_volume = -(i + 1) / n_live
            log_p_fail += log_diff_exp(log_expected_prior_volume, log_sf) - log_expected_prior_volume
        p_fail = np.exp(log_p_fail)
        df_p_fail.loc[mu_like] = p_fail
        if p_fail >= 0.999:
            break
    df_p_fail.plot.line()
    plt.show()


if __name__ == "__main__":
    main()
