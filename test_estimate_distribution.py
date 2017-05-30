"""
Tests for estimate_ecdf(), estimate_pdf()
"""


import distribution_estimation as est
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt



###################################################################################################
"""
Create a 1-D GMM, generate some samples, and estimate ecdf and pdf
Visually verify the similarity of true pdf vs. estimated pdf
"""

def test_estimate_gmm_ecdf_pdf():
    fig, ax1 = plt.subplots(1, 1, sharey=True)

    print("Creating GMM RV and plotting pdf")

    component_param = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmm_rv = tvgmm.GaussianMixtureModel(component_param)

    num_samples = 50000

    x = np.linspace(-7, 7, num_samples)
    ax1.plot(x, gmm_rv.pdf(x), label='True pdf')
    #gmm_cdf = gmm_rv.cdf(x)
    #ax1.plot(x, gmm_cdf, label='GMM cdf')

    ############

    samples = gmm_rv.rvs(size=num_samples)

    print("Plotting histogram")
    bin_width = 0.2
    bins = np.arange(min(samples), max(samples) + bin_width, bin_width)
    ax1.hist(samples, normed=True, histtype='stepfilled', bins=bins, alpha=0.2, label='Histogram')

    #############

    sorted, ecdf = est.estimate_ecdf(samples)
    ax1.plot(sorted, ecdf, label="estimated-ecdf")

    xnew, derivatives = est.estimate_pdf(sorted, ecdf)
    ax1.plot(xnew, derivatives, label="estimated-pdf")

    #############

    ax1.legend(loc='upper right')
    #ax2.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()

    print("Done")



###################################################################################################

# Call test functions

test_estimate_gmm_ecdf_pdf()