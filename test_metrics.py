"""
Tests for statistical distance metric functions
"""

import time_varying_gmm as tvgmm
import metrics
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import distribution_estimation as est



###################################################################################################
"""
Test the KL Divergence function
"""
def test_kl_divergence():
    component_params = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmm_1 = tvgmm.GaussianMixtureModel(component_params)

    kl_div_1 = metrics.kl_divergence(gmm_1, gmm_1)
    print("kl_div_1", end=' = ')
    print(kl_div_1)

    component_params = [(1 / 3, 1, 0.5), (1 / 3, -2, 1), (1 / 3, 4, 1)]
    gmm_2 = tvgmm.GaussianMixtureModel(component_params)

    kl_div_2 = metrics.kl_divergence(gmm_1, gmm_2)
    print("kl_div_2", end=' = ')
    print(kl_div_2)



###################################################################################################
"""
Test the one sample Kolmogorov–Smirnov test function on a Gaussian
"""
def test_one_sample_ks_test_on_gaussian():

    # Create a Gaussian RV, get some samples and run KS test

    rv1 = st.norm(loc=0, scale=1)
    num_samples = 1000

    samples = rv1.rvs(num_samples)

    d1, p1 = metrics.one_sample_ks_test(samples, rv1.cdf)

    print("test_one_sample_ks_test_on_gaussian; one_sample_ks_test; rv1; d1={:.4f}, p1={:.4f}".format(d1, p1))

    # Estimate ecdf from the samples, and compute KS statistic manually

    sorted, ecdf = est.estimate_ecdf(samples)

    x1, D1 = metrics.manual_ks_stat(sorted, ecdf, rv1.cdf)

    # Plot true cdf, ecdf, and D

    sorted_index = np.where(sorted == x1)

    print("test_one_sample_ks_test_on_gaussian; manual_ks_stat; rv1; D1={:.4f}, true-cdf-point=({:.4f},{:.4f}); ecdf-point=({:.4f},{:.4f})"
          .format(D1, x1, rv1.cdf(x1), x1, ecdf[sorted_index][0]))

    x = np.linspace(-4, 4, 1000)
    plt.plot(x, rv1.cdf(x), label='True cdf')
    plt.plot(sorted, ecdf, label='ecdf')

    plt.plot(x1, rv1.cdf(x1), "o")
    plt.plot(x1, ecdf[sorted_index], "o")
    plt.plot([x1, x1], [rv1.cdf(x1), ecdf[sorted_index]], 'k-', label="D")

    plt.legend(loc='upper right')
    plt.xlabel('x')


    # Same sample, compare with different Gaussian (rv2)

    print("############################")

    plt.figure()

    rv2 = st.norm(loc=3, scale=1)

    d2, p2 = metrics.one_sample_ks_test(samples, rv2.cdf)

    print("test_one_sample_ks_test_on_gaussian; one_sample_ks_test; rv2; d1={:.4f}, p1={:.4f}".format(d2, p2))

    x2, D2 = metrics.manual_ks_stat(sorted, ecdf, rv2.cdf)

    sorted_index = np.where(sorted == x2)

    print("test_one_sample_ks_test_on_gaussian; manual_ks_stat; rv2; D2={:.4f}, true-cdf-point=({:.4f},{:.4f}); ecdf-point=({:.4f},{:.4f})"
          .format(D2, x2, rv2.cdf(x2), x2, ecdf[sorted_index][0]))

    x = np.linspace(-1, 7, 1000)
    plt.plot(x, rv2.cdf(x), label='True cdf')
    plt.plot(sorted, ecdf, label='ecdf')

    plt.plot(x2, rv2.cdf(x2), "o")
    plt.plot(x2, ecdf[sorted_index], "o")
    plt.plot([x2, x2], [rv2.cdf(x2), ecdf[sorted_index]], 'k-', label="D")


    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()



###################################################################################################
"""
Test the one sample Kolmogorov–Smirnov test function on a GMM
"""
def test_one_sample_ks_test_on_gmm():

    # Create a GMM RV, get some samples and run KS test

    component_params = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmm_1 = tvgmm.GaussianMixtureModel(component_params)

    num_samples = 1000
    samples = gmm_1.rvs(size=num_samples)

    d1, p1 = metrics.one_sample_ks_test(samples, gmm_1.cdf)
    print("test_one_sample_ks_test_on_gmm; one_sample_ks_test; d1={}, p1={}".format(d1, p1))


    # Estimate ecdf from the samples, and compute KS statistic manually

    sorted, ecdf = est.estimate_ecdf(samples)

    x1, D1 = metrics.manual_ks_stat(sorted, ecdf, gmm_1.cdf)

    # Plot true cdf, ecdf, and D

    sorted_index = np.where(sorted == x1)

    print("test_one_sample_ks_test_on_gmm; manual_ks_stat; true-cdf-point=({},{}); ecdf-point=({},{})"
          .format(x1, gmm_1.cdf(x1), x1, ecdf[sorted_index]))

    x = np.linspace(-6, 6, 1000)
    plt.plot(x, gmm_1.cdf(x), label='True cdf')
    plt.plot(sorted, ecdf, label='ecdf')

    plt.plot(x1, gmm_1.cdf(x1), "o")
    plt.plot(x1, ecdf[sorted_index], "o")
    plt.plot([x1, x1], [gmm_1.cdf(x1), ecdf[sorted_index]], 'k-', label="D")

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()


###################################################################################################
"""
Test the mean_squared_error() function on a Gaussian
"""
def test_mean_squared_error_on_gaussian():

    # Create a Gaussian RV, get some samples, estimate ecdf

    rv1 = st.norm(loc=0, scale=1)
    num_samples = 1000

    samples = rv1.rvs(num_samples)

    sorted, ecdf = est.estimate_ecdf(samples)

    mse1 = metrics.mean_squared_error(sorted, ecdf, rv1.cdf)

    print("test_mean_squared_error_on_gaussian; rv1; mse1={:.6f}".format(mse1))


    # Same sample (ecdf), compare with different Gaussian (rv2)

    print("############################")

    plt.figure()

    rv2 = st.norm(loc=3, scale=1)

    mse2 = metrics.mean_squared_error(sorted, ecdf, rv2.cdf)

    print("test_mean_squared_error_on_gaussian; rv2; mse2={:.6f}".format(mse2))



###################################################################################################

# Call test functions

# test_kl_divergence()
# test_one_sample_ks_test_on_gaussian()
# test_one_sample_ks_test_on_gmm()
test_mean_squared_error_on_gaussian()