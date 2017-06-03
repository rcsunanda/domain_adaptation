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



###################################################################################################
"""
Generate sample_count samples from a given discrete cdf
See https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
"""

def generate_samples_from_cdf(x_values, cdf_values, sample_count):
    assert len(x_values) == len(cdf_values)

    samples = []
    for i in range(0, sample_count):
        threshold = np.random.uniform(cdf_values[0], cdf_values[-1])    # Start and end of cdf (ideally should be 0 and 1)

        value_found = False
        for x, y in zip(x_values, cdf_values):
            if y >= threshold:
                samples.append(x)
                value_found = True
                break

        assert value_found, "A sample value MUST be found in each iteration of outer loop"

    return samples