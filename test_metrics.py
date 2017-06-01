"""
Tests for statistical distance metric functions
"""

import time_varying_gmm as tvgmm
import metrics




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

# Call test functions

test_kl_divergence();
