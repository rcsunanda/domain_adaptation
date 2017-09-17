"""
Methods estimate pdf and compute Total Variation Distance between pdfs
"""

import sklearn.neighbors.kde as kde
import numpy as np
import scipy.integrate as integrate
import random

import itertools
import math

###################################################################################################
"""
Monte Carlo integration from mcint: https://pypi.python.org/pypi/mcint/
Could be better to use a more solid implementation: https://pypi.python.org/pypi/scikit-monaco
"""

def mc_integrate(integrand, sampler, measure=1.0, n=100):
    # Sum elements and elements squared
    total = 0.0
    total_sq = 0.0
    for x in itertools.islice(sampler, n):
        f = integrand(x)
        total += f
        total_sq += (f**2)
    # Return answer
    sample_mean = total/n
    sample_var = (total_sq - ((total/n)**2)/n)/(n-1.0)
    return (measure*sample_mean, measure*math.sqrt(sample_var/n))


###################################################################################################
"""
Estimate pdf using Kernel Density Estimation (KDE)
"""

def estimate_pdf_kde(samples):

    kde_estimator = kde.KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde_estimator.fit(samples)
    return kde_estimator


###################################################################################################
"""
Estimate Total Variation Distance (TVD) between two distributions (given by a KDE estimator)
Uses a Monte Carlo integration to integrate the difference function over a given region (bounds)
"""

def total_variation_distance(kde1, kde2, bounds):

    def diff_func(*args):   # Difference function to integrate
        args = np.reshape(args, (1, -1))
        log_vals1 = kde1.score_samples(args)
        log_vals2 = kde2.score_samples(args)

        diff = np.exp(log_vals1) - np.exp(log_vals2)

        # diff = diff * diff
        diff = np.absolute(diff)

        return diff


    def sampler():  # Sample generator for Monte Carlo integration
        while True:
            sample_vec = []
            for bound in bounds:
                feature_value = random.uniform(bound[0], bound[1])  #
                sample_vec.append(feature_value)

            yield (sample_vec)


    # options = {'epsabs':1.49e-04, 'epsrel':1.49e-04}
    # quad_integral = integrate.nquad(diff_func, [[-3, +3], [-3, +3]], opts=options)
    # quad_integral = integrate.nquad(diff_func, [[-3, +7], [-3, +7]], opts=options)
    # quad_integral = integrate.nquad(diff_func, [[-3, +3], [-3, +3]])
    # quad_integral = integrate.nquad(diff_func, [[-3, +3]])
    # print("quad_integral={}".format(quad_integral))

    mc_integral = mc_integrate(diff_func, sampler(), n=1000)
    mc_integral = mc_integral[0][0]/2

    return mc_integral
