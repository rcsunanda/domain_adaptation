"""
Tests for methods in distribution_difference
"""

import domain_adaptation.distribution_difference as ddif
import domain_adaptation.drift_detector as dd
import domain_adaptation.process as prc


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import timeit


###################################################################################################
"""
Test estimate_pdf_kde by generating some data (1-dimensional) and estimating its pdf
"""

def test_estimate_1d_pdf_kde():

    # Generate some data from a class

    gauss_params = []
    mean_1 = [0]
    cov_1 = [[1]]

    gauss_params.append((mean_1, cov_1))

    process = prc.Process(num_dimensions=1, num_classes=1, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data
    data_points = process.generate_data_points(label=0, count=1000)
    X_dataset = [point.X for point in data_points]

    # Estimate pdf
    kde_estimator = ddif.estimate_pdf_kde(X_dataset)

    # Plot pdf
    log_values = kde_estimator.score_samples(X_dataset)
    pdf_values = np.exp(log_values)

    X_dataset, pdf_values = (list(x) for x in zip(*sorted(zip(X_dataset, pdf_values))))

    # Plot
    plt.scatter(X_dataset, pdf_values)
    plt.plot(X_dataset, pdf_values)
    plt.show()


###################################################################################################
"""
Test estimate_pdf_kde by generating some data (2-dimensional) and estimating its pdf
"""

def test_estimate_2d_pdf_kde():

    # Generate some data from a class

    gauss_params = []
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))

    process = prc.Process(num_dimensions=2, num_classes=1, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data
    data_points = process.generate_data_points(label=0, count=1000)
    X_dataset = [point.X for point in data_points]

    # Estimate pdf
    kde_estimator = ddif.estimate_pdf_kde(X_dataset)

    # Plot pdf
    log_values = kde_estimator.score_samples(X_dataset)
    pdf_values = np.exp(log_values)

    # X_dataset, pdf_values = (list(x) for x in zip(*sorted(zip(X_dataset, pdf_values))))

    x = [sample[0] for sample in X_dataset]
    y = [sample[1] for sample in X_dataset]

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, pdf_values)
    plt.show()


###################################################################################################
"""
Generate data from different distributions, estimate their pdfs, and compute differences 
"""

def test_total_variation_distance():
    # Generate some data from a class

    gauss_params = []
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]

    mean_2 = [4, 4]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    process = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data
    data_points1 = process.generate_data_points(label=0, count=500)
    data_points2 = process.generate_data_points(label=0, count=500)
    data_points3 = process.generate_data_points(label=1, count=500) # From different label

    X_dataset1 = [point.X for point in data_points1]
    X_dataset2 = [point.X for point in data_points2]
    X_dataset3 = [point.X for point in data_points3]

    # Estimate probability distributions of datasets
    kde_estimator1 = ddif.estimate_pdf_kde(X_dataset1)
    kde_estimator2 = ddif.estimate_pdf_kde(X_dataset2)
    kde_estimator3 = ddif.estimate_pdf_kde(X_dataset3)

    (w1, w2, bounds1) = dd.prepare_sample_windows(data_points1, data_points2)
    (w1, w2, bounds2) = dd.prepare_sample_windows(data_points1, data_points3)

    # Compute difference between distributions (multiple times)

    diff_list_1 = []
    diff_list_2 = []

    for i in range (50):
        diff1 = ddif.total_variation_distance(kde_estimator1, kde_estimator2, bounds1)
        diff2 = ddif.total_variation_distance(kde_estimator1, kde_estimator3, bounds2)

        diff_list_1.append(diff1)
        diff_list_2.append(diff2)

        if (i % 5 == 0):
            print("index={}".format(i))

    # Plot
    plt.plot(diff_list_1, label='diff_1_2 - same label')
    plt.plot(diff_list_2, label='diff_1_3 - different labels')
    plt.legend(loc='upper right')
    plt.ylabel('diff')
    plt.show()



###################################################################################################

# Call test functions

# test_estimate_1d_pdf_kde()
# test_estimate_2d_pdf_kde()

time = timeit.timeit(test_total_variation_distance, number=1)
# print ("time={}".format(time))
# test_total_variation_distance()

