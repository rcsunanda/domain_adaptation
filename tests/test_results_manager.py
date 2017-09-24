"""
Tests for ResultsManager
"""

import domain_adaptation.results_manager as rman
import domain_adaptation.process as prc

import matplotlib.pyplot as plt


###################################################################################################
"""
Generate a sequence of DataPoints from a process, set true and predicted y values that results in noticeable errors
Add these to a ResultsManager and print results
"""

def test_results_manager():
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]
    process = prc.Process(num_dimensions=2, num_classes=1, class_distribution_parameters=[(mean_1, cov_1)])

    data_points = process.generate_data_points(label=0, count=100)

    results_manager = rman.ResultsManager(avg_error_window_size=10)

    batch_size = 5
    batch = []

    diff_sum = 0
    for index, data_point in enumerate(data_points):
        batch.append(data_point)
        diff = 1/(index+1)
        diff_sum += diff
        is_drift_detected = False

        # Set predicted y values
        data_point.predicted_y = data_point.true_y
        if (index % 3 == 0):
            data_point.predicted_y = -100

        if (index % 15 == 0):
            is_drift_detected = True

        if (index % batch_size == 0):
            results_manager.add_prediction_result(index, batch)
            results_manager.add_detection_info(index, diff, diff_sum, is_drift_detected)
            batch = []

        if (index % 20 == 0):
            results_manager.print_results()

    results_manager.add_special_marker(73, "marker_1")
    results_manager.add_special_marker(83, "marker_2")
    results_manager.plot_results()



###################################################################################################

# Call test functions

test_results_manager()