"""
Tests for DriftDetector
"""

import domain_adaptation.drift_detector as dd
import domain_adaptation.process as prc


###################################################################################################
"""
Test the windowing and difference calculations in DriftDetector
Generate a sequence of data from Process, add it to DriftDetector
Check that data was windowed properly, and correct difference metrics were calculated 
"""

def test_difference_calculation():

    drift_detector = dd.DriftDetector(10)
    print(drift_detector)


    # Generate some data from a 2 class stochastic process

    # Setup process

    gauss_params = []
    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]
    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    process = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data
    data_points_label_0 = process.generate_data_points(label=0, count=50)
    data_points_label_1 = process.generate_data_points(label=1, count=50)
    data_points = data_points_label_0 + data_points_label_1

    # Emulate a data point sequence
    for point in data_points:
        drift_detector.add_data_points([point])
        drift_detector.run_detection()



###################################################################################################

# Call test functions

test_difference_calculation()

