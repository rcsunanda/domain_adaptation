"""
Metrics to measure distance between probability distributions (pdf, cdf)
"""

import numpy as np
import scipy.integrate as integrate



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