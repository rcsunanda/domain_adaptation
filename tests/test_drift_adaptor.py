"""
Tests for DriftAdaptor
"""

import domain_adaptation.drift_adaptor as da
import domain_adaptation.process as prc
import domain_adaptation.data_point as dp

import matplotlib.pyplot as plt


###################################################################################################
"""
Test that given a window of samples bounds are correctly output
"""

def test_compute_window_bounds():
    window_1 = [0,-20], [1,1], [2,2], [3,3], [10,15]

    bounds = da.compute_window_bounds(window_1)

    print(bounds)
    assert bounds == [(0,10), (-20,15)]


###################################################################################################
"""
Test that given two sets of bounds, overall bounds are correctly cimputed
"""

def test_compute_overall_bounds():
    bounds_1 = [(0,-20), (1,1), (2,30)]   # 3 dimensions
    bounds_2 = [(-10,10), (5,10), (10,20)]

    expected_overall_bounds = [(-10,10), (1,10), (2,30)]

    bounds = da.compute_overall_bounds(bounds_1, bounds_2)

    print(bounds)
    assert bounds == expected_overall_bounds


###################################################################################################

# Call test functions

# test_compute_window_bounds()
test_compute_overall_bounds()

