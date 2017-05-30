"""
Functions for estimating ecdf and pdf
"""

import numpy as np
import scipy.interpolate as interp
import scipy.misc as misc


###################################################################################################
"""
Given a set of samples, computes ecdf
Returns (x,y) pairs of the ecdf
"""

def estimate_ecdf(samples):
    print("estimate_ecdf; Estimating Empirical Cumulative Distribution Function - ECDF")

    sorted = np.sort(samples)
    ecdf = np.arange(len(sorted)) / float(len(sorted))

    return (sorted, ecdf)



###################################################################################################
"""
Given the ecdf (as x,y), estimates pdf by interpolating the ecdf and differentiating it
Returns (x,y) pairs of the pdf
"""

def estimate_pdf(x, ecdf):
    print("estimate_pdf; Estimating PDF from ECDF")

    ecdf_interp_obj = interp.interp1d(x, ecdf)

    new_x = np.linspace(x[0], x[-1], len(x))

    #ax1.plot(new_x, ecdf_interp_obj(new_x), label="x-vs-ecdf_interp_obj")


    # Must take some subset of new_x, as misc.derivative() will call ecdf_interp_obj() outside the given values
    # 10% trimming sometimes gives bound errors - may need to use another method to estimate pdf
    start = len(x) * 10 // 100
    end = len(x) * 90 // 100
    new_x_for_derivation = new_x[start:end]

    derivatives = []
    for xp in new_x_for_derivation:
        deriv = misc.derivative(func=ecdf_interp_obj, x0=xp)
        derivatives.append(deriv)

    #ax2.plot(new_x_for_derivation, derivatives, label="new_x-vs-derivatives")

    return (new_x_for_derivation, derivatives)


