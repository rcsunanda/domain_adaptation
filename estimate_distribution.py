import time_varying_gmm as tvgmm
import matplotlib.pyplot as plt
import numpy as np
import time as time_module
import sklearn.neighbors as neighb
import scipy.interpolate as interp
import scipy.misc as misc


def estimate_ecdf(samples):
    print("estimate_ecdf; Estimating Empirical Cumulative Distribution Function - ECDF")

    sorted = np.sort(samples)
    ecdf = np.arange(len(sorted)) / float(len(sorted))

    return (sorted, ecdf)


def estimate_pdf(x, ecdf):
    print("estimate_pdf; Estimating PDF from ECDF")

    ecdfInterpObj = interp.interp1d(x, ecdf)

    xnew = np.linspace(x[0], x[-1], len(x))

    #ax1.plot(xnew, ecdfInterpObj(xnew), label="x-vs-ecdfInterpObj")

    derivatives = []
    start = len(x) * 10 // 100
    end = len(x) * 90 // 100
    xnewForDerivation = xnew[start:end]  # Must take some subset of xnew, as misc.derivative() will call ecdfInterpObj() outside the given values
    for xp in xnewForDerivation:
        deriv = misc.derivative(func=ecdfInterpObj, x0=xp)
        derivatives.append(deriv)

    #ax2.plot(xnewForDerivation, derivatives, label="xnew-vs-derivatives")

    return (xnewForDerivation, derivatives)


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

#test_differentiation()
