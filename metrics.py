"""
Metrics to measure distance between probability distributions (pdf, cdf)
"""

import numpy as np
import scipy.integrate as integrate
import scipy.stats as st



###################################################################################################
"""
Compute Kullback–Leibler divergence between two RVs
"""
def kl_divergence(rv_p, rv_q):
    def func(x):
        nonlocal rv_p, rv_q
        px = rv_p.pdf(x)
        qx = rv_q.pdf(x)

        return px * np.log(px / qx)

    result, error = integrate.quad(func, -10, 10)

    return result


###################################################################################################
"""
Perform one sample Kolmogorov–Smirnov test (KS test) and return the KS statistic (D) and p-value
One sample KS test is for comparing a data sample (generated from some estimated cdf/pdf) with a true cdf
Answers the question: did the data come from this cdf/ similar cdf? 
true_cdf must be a callable
"""
def one_sample_ks_test(samples, true_cdf):
    D, p = st.kstest(samples, true_cdf)
    return D, p


###################################################################################################
"""
Manually compute the KS statistic D = max(abs(ecdf - true_cdf))
Returns (x_val, D)
"""
def manual_ks_stat(x_vals, ecdf_vals, true_cdf):

    current_max = 0
    current_max_x = 0

    for x, y in zip(x_vals, ecdf_vals):
        diff = abs(true_cdf(x) - y)

        current_max = max(current_max, diff)

        if (current_max == diff):   # current_max was updated
            current_max_x = x

    # print ("ks_test; x={}, D={}".format(current_max_x, current_max))

    return (current_max_x, current_max)