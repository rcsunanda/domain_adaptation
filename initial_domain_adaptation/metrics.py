"""
Metrics to measure distance between probability distributions (pdf, cdf)
"""

import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import sklearn.metrics as skmetrics
import pyemd


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
Compute approximate Kullback–Leibler divergence between a given true pdf and an estimated pdf
true_pdf must be callable
"""

def approx_kl_divergence(x_vals, estimated_pdf_vals, true_pdf):

    kl_func_vals = []   # The function to integrate
    for x, qx in zip(x_vals, estimated_pdf_vals):
        px = true_pdf(x)
        func_val = px * np.log(px / qx)
        kl_func_vals.append(func_val)

    result = np.trapz(kl_func_vals, x_vals)

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
Return (x_val, D)
true_cdf must be callable
"""

def manual_ks_stat(x_vals, estimated_ecdf_vals, true_cdf):
    assert (len(x_vals) == len(estimated_ecdf_vals))

    current_max = 0
    current_max_x = 0

    for x, y in zip(x_vals, estimated_ecdf_vals):
        diff = abs(true_cdf(x) - y)

        current_max = max(current_max, diff)

        if (current_max == diff):   # current_max was updated
            current_max_x = x

    # print ("ks_test; x={}, D={}".format(current_max_x, current_max))

    return (current_max_x, current_max)



###################################################################################################
"""
Return the mean squared error between an estimate and a true function
x_vals and estimated_func_vals are arrays corresponding to the estimate
true_func must be callable
"""

def mean_squared_error(x_vals, estimated_func_vals, true_func):
    assert (len(x_vals) == len(estimated_func_vals))

    true_func_vals = []
    for x in x_vals:
        true_func_vals.append(true_func(x))

    mse = skmetrics.mean_squared_error(estimated_func_vals, true_func_vals)

    return mse



###################################################################################################
"""
Return the Earth Mover's Distance (EMD) or the 1st Wasserstein distance between two functions (pdfs)
Taken from pyemd (https://github.com/wmayner/pyemd) - cite the papers in this page
true_func must be callable
"""

def emd(x_vals, estimated_func_vals, true_func):
    assert (len(x_vals) == len(estimated_func_vals))

    n = len(x_vals)

    true_func_vals = np.zeros(n)
    for i, x in enumerate(x_vals):
        true_func_vals[i] = true_func(x)

    distance_matrix = np.zeros((n,n))

    for i in range(0, n):
        for j in range (0, n):
            distance_matrix[i][j] = abs(x_vals[i] - x_vals[j])

    # print("distance_matrix=\n{}".format(distance_matrix))

    estimated_func_vals = np.array(estimated_func_vals) # Convert list to np.array type

    emd = pyemd.emd(estimated_func_vals, true_func_vals, distance_matrix)   # Expensive computation

    return emd