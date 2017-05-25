import estimate_distribution as est
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt


def test_estimate_gmm_ecdf_pdf():
    fig, ax1 = plt.subplots(1, 1, sharey=True)

    print("Creating GMM RV and plotting pdf")

    componentParamList = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmmRV = tvgmm.GaussianMixtureModel(componentParamList)

    numSamples = 50000

    x = np.linspace(-7, 7, numSamples)
    ax1.plot(x, gmmRV.pdf(x), label='True pdf')
    #gmmCDF = gmmRV.cdf(x)
    #ax1.plot(x, gmmCDF, label='GMM cdf')

    ############

    samples = gmmRV.rvs(size=numSamples)

    print("Plotting histogram")
    binwidth = 0.2
    bins = np.arange(min(samples), max(samples) + binwidth, binwidth)
    histPlot = ax1.hist(samples, normed=True, histtype='stepfilled', bins=bins, alpha=0.2, label='Histogram')

    #############

    sorted, ecdf = est.estimate_ecdf(samples)
    ax1.plot(sorted, ecdf, label="estimated-ecdf")    # this plots correct ecdf

    xnew, derivatives = est.estimate_pdf(sorted, ecdf)
    ax1.plot(xnew, derivatives, label="estimated-pdf")

    #############

    ax1.legend(loc='upper right')
    #ax2.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()

    print("Done")


test_estimate_gmm_ecdf_pdf()