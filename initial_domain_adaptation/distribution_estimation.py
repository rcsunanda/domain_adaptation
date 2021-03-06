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
    # print("estimate_ecdf; Estimating Empirical Cumulative Distribution Function - ECDF")

    sorted = np.sort(samples)
    ecdf = np.arange(len(sorted)) / float(len(sorted))

    return (sorted, ecdf)



###################################################################################################
"""
Given the ecdf (as x,y), estimates pdf by interpolating the ecdf and differentiating it
Returns (x,y) pairs of the pdf
Implement in this manner - https://scicomp.stackexchange.com/questions/480/how-can-i-numerically-differentiate-an-unevenly-sampled-function
"""

def estimate_pdf(x_vals, ecdf):
    ecdf_interp_obj = interp.interp1d(x_vals, ecdf)

    new_x = np.linspace(x_vals[0], x_vals[-1], len(x_vals))
    new_y = ecdf_interp_obj(new_x)

    dx = new_x[1] - new_x[0]
    np_difference = np.gradient(new_y) / dx

    # Moving average filter (twice) (the difference function is very noisy)

    N = len(x_vals) * 10 // 100
    np_difference = np.convolve(np_difference, np.ones((N,)) / N, mode='same')

    N = len(x_vals) * 2 // 100
    np_difference = np.convolve(np_difference, np.ones((N,)) / N, mode='same')

    return (new_x, np_difference)



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