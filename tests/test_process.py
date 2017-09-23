"""
Tests for Process
"""

import domain_adaptation.process as prc

import matplotlib.pyplot as plt


###################################################################################################
"""
Test generation of data points from each class of the Process
Process is a 2 dimensional data set with 2 classes
Class 0 has mean [0,0], Class 1 has mean [3,3]
"""

def test_generate_data_points():
    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1,0], [0,1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    process = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data from class 1 and plot
    data_points_label_0 = process.generate_data_points(label=0, count=500)

    x = [point.X[0] for point in data_points_label_0]
    y = [point.X[1] for point in data_points_label_0]
    plt.scatter(x, y)


    # Generate data from class 2 and plot
    data_points_label_1 = process.generate_data_points(label=1, count=500)

    x = [point.X[0] for point in data_points_label_1]
    y = [point.X[1] for point in data_points_label_1]
    plt.scatter(x, y)

    plt.show()



###################################################################################################
"""
Test generation of data points from two classes of the Process
"""

def test_generate_data_points_from_all_labels():
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

    # Generate data from all labels and plot
    data_points_all_labels = process.generate_data_points_from_all_labels(total_count=1000)

    x = [point.X[0] for point in data_points_all_labels]
    y = [point.X[1] for point in data_points_all_labels]
    plt.scatter(x, y)

    plt.show()



###################################################################################################

# Call test functions

test_generate_data_points()
test_generate_data_points_from_all_labels()