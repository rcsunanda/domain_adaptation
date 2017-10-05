"""
Tests for DriftAdaptor
"""

import domain_adaptation.drift_adaptor as da
import domain_adaptation.model_ensemble as ens
import domain_adaptation.process as prc
import domain_adaptation.data_point as dp
import domain_adaptation.ann_submodel as ann_sm

import matplotlib.pyplot as plt


###################################################################################################
"""
Use given model to predict on test_data and print the results
"""

def predict_and_print_results(model, test_data, print_str):
    model.predict(test_data)

    test_data_count = len(test_data)
    num_errors = 0

    for point in test_data:
        if (point.predicted_y != point.true_y):
            num_errors += 1

    error_percent = num_errors * 100 / test_data_count

    print("{}: num_errors={}, error_percent={}".format(print_str, num_errors, error_percent))


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
"""
Call DriftAdaptor.adapt_ensemble for 2 new DataPoint windows (from 2 processes) and verify the following
(1) For first adapt_ensemble() call, ensemble weight = 1
(2) For second adapt_ensemble() call, ensemble weight = 1/diff (where diff = difference between pdfs of the two windows)

"""

def test_adaptation():
    ensemble = ens.ModelEnsmeble()
    adaptor = da.DriftAdaptor(ensemble, "ANN_Submodel", "Artifcial")

    print(adaptor)


    # Setup 2 processes

    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]
    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params_1 = []
    gauss_params_1.append((mean_1, cov_1))
    gauss_params_1.append((mean_2, cov_2))
    process_1 = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params_1)

    # class 1 Gaussian distribution params
    mean_1 = [0, 5]
    cov_1 = [[1, 0], [0, 1]]
    # class 2 Gaussian distribution params
    mean_2 = [5, 0]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params_2 = []
    gauss_params_2.append((mean_1, cov_1))
    gauss_params_2.append((mean_2, cov_2))
    process_2 = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params_2)

    print(process_1)
    print(process_2)


    # Generate 2 training datasets from the 2 processes
    training_data_1 = process_1.generate_data_points_from_all_labels(total_count=1000)
    training_data_2 = process_2.generate_data_points_from_all_labels(total_count=1000)

    # Generate a test data from process_2
    test_data = process_2.generate_data_points_from_all_labels(total_count=500)


    # Call adapt_ensemble() on first dataset
    adaptor.adapt_ensemble(training_data_1)
    print("adaptor after first adapt_ensemble() call = {}".format(adaptor))

    adaptor.adapt_ensemble(training_data_2)
    print("adaptor after second adapt_ensemble() call = {}".format(adaptor))

    predict_and_print_results(adaptor.ensemble, test_data, "test_adaptation: ensemble")


    # For comparision train 2 ANN_Submodels on the 2 datasets and check results

    ann_submodel_1 = ann_sm.ANN_Submodel(weight=1, pdf=None)
    ann_submodel_2 = ann_sm.ANN_Submodel(weight=1, pdf=None)

    ann_submodel_1.train(training_data_1)
    ann_submodel_2.train(training_data_2)

    predict_and_print_results(ann_submodel_1, test_data, "test_adaptation: ann_submodel_1")
    predict_and_print_results(ann_submodel_2, test_data, "test_adaptation: ann_submodel_2")



###################################################################################################

# Call test functions

# test_compute_window_bounds()
# test_compute_overall_bounds()
test_adaptation()
