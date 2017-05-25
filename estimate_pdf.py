import time_varying_gmm as tvgmm
import matplotlib.pyplot as plt
import numpy as np
import time as time_module
import sklearn.neighbors as neighb
import scipy.interpolate as interp
import scipy.misc as misc

import csv

def estimate_gmm_pdf():
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


    print("Creating GMM RV and plotting pdf")

    componentParamList = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmmRV = tvgmm.GaussianMixtureModel(componentParamList)

    # newParamList = [(1/3, 4, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
    # gmmRV.setModelParams(newParamList)	# To test setModelParams()

    numSamples = 10000

    print("Plotting pdf and cdf of GMM")

    x = np.linspace(-7, 7, numSamples)
    pdfPlot = ax1.plot(x, gmmRV.pdf(x), label='GMM pdf')
    #gmmCDF = gmmRV.cdf(x)
    #cdfPlot = ax1.plot(x, gmmCDF, label='GMM cdf')

    ############

    print("Generating samples")

    t0 = time_module.time()

    samples = gmmRV.rvs(size=numSamples)

    t1 = time_module.time()
    timeElapsed = t1 - t0

    print("Time taken to generate samples = %.2f secs" % (timeElapsed))

    ############

    # print("Plotting samples")
    # plt.scatter(samples, np.zeros(numSamples))

    print("Plotting histogram")
    binwidth = 0.2
    histPlot = ax1.hist(samples, normed=True, histtype='stepfilled',
                        bins=np.arange(min(samples), max(samples) + binwidth, binwidth), alpha=0.2, label='Histogram')

    #############

    # print("Estimating pdf with Kernedl Density Estimation - KDE")

    # kde = neighb.KernelDensity(kernel='gaussian', bandwidth=0.5)
    # kde_pdf = kde.fit(samples)
    # log_dens = kde_pdf.score_samples(x[:numSamples])
    # print(log_dens)
    # plt.plot(x[:numSamples], np.exp(log_dens), '-', label="KDE estimation")

    #############

    print("Estimating pdf with Empirical Cumulative Distribution Function - ECDF")

    sorted = np.sort(samples)
    ecdf = np.arange(len(sorted)) / float(len(sorted))
    ax1.plot(sorted, ecdf, label="sorted-vs-ecdf")    # this plots correct ecdf
    ax1.plot(x, ecdf, label="x-vs-ecdf")    # this plots a straight line

    ecdfInterpObj = interp.interp1d(sorted, ecdf, kind='quadratic')
    xnew=np.linspace(sorted[0], sorted[-1], numSamples)
    ax1.plot(xnew, ecdfInterpObj(xnew), label="x-vs-ecdfInterpObj")

    # Rearrange ecdf to correct order
    #newECDF = [ecdf[sampleVal] for sampleVal in sorted]

#    ax1.plot(x, newECDF, label="x-vs-newECDF")    # this plots a straight line

    # for i, sampleVal in sorted:
    #     newECDF.append(ecdf[sampleVal])

    #dx = x[1] - x[0]
    # deriv_pdf_of_actual_cdf = np.diff(gmmRV.cdf(x))/dx
    # plt.plot(x[1:], deriv_pdf_of_actual_cdf, label="deriv_pdf_of_actual_cdf")

    manual_difference = []
    for i in range(len(samples) - 1):
        dx = sorted[i + 1] - sorted[i]
        diff = (ecdf[i + 1] - ecdf[i]) / dx
        #diff = (gmmCDF[i + 1] - gmmCDF[i]) / dx # gives correct pdf approx
        manual_difference.append(diff)
    # print("diff={}".format(ecdf[i+1] - ecdf[i]))
    manual_difference.append(manual_difference[-1])    # Append element equal to one before that

    # ax2.plot(sorted, manual_difference, label="difference")

    derivatives = []
    xnewForDerivation = xnew[1000:len(xnew)-1000]   # Must take some subset of xnew, as misc.derivative() will call ecdfInterpObj() outside the given values
    for xp in xnewForDerivation:
        deriv = misc.derivative(func=ecdfInterpObj, x0=xp)
        derivatives.append(deriv)

    ax2.plot(xnewForDerivation, derivatives, label="xnew-vs-derivatives")


    # deriv_pdf = np.diff(ecdf)/dx
    # ax2.plot(sorted[0:len(samples)-1], deriv_pdf, label="deriv_pdf")	#put sorted here for x axies !!!!!!!!!!!!!!!!!!!!! --> didn't work --> try with an interpolated continues version of ecdf

    # deriv_pdf = np.gradient(gmmRV.cdf(x), dx)
    # ax2.plot(x, deriv_pdf, label="deriv_pdf")

    #############

    #
    # with open("data.csv", 'w') as resultFile:
    #     wr = csv.writer(resultFile, dialect='excel')
    #     wr.writerow(gmmCDF)
    #     wr.writerow(ecdf)
    #     wr.writerow(manual_difference)


    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()

    print("Done")


def test_differentiation():
    x = np.linspace(-5, 5, 10000)
    y = x**3

    theoratical_diff = 3*x**2

    dx = x[1] - x[0]
    manual_difference = []
    for i in range(len(x) - 1):
        diff = (y[i + 1] - y[i]) / dx
        manual_difference.append(diff)
    manual_difference.append(manual_difference[-1])    # Append element equal to one before that

    np_difference = np.diff(y) / dx
    np_difference = list(np_difference)
    np_difference.append(np_difference[-1])    # Append element equal to one before that

    plt.plot(x, y, label="y")
    plt.plot(x, theoratical_diff, linestyle=':', label="theoratical_diff")
    plt.plot(x, manual_difference, linestyle='--', label="manual_difference")
    plt.plot(x, np_difference, linestyle='steps', label="np_difference")

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()

# Call functions

estimate_gmm_pdf()
#test_differentiation()
