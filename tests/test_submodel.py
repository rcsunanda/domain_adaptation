"""
Tests for Submodel, ANN_Submodel and other derived classes
"""

import domain_adaptation.ann_submodel as ann_sm
import domain_adaptation.process as prc

import matplotlib.pyplot as plt


###################################################################################################
"""
Test the training and predicting of an ANN_Submodel
Train the ANN_Submodel on some training data generated from Process
Then predict on some test data generated from Process, and report error percent
"""

def test_ann_submodel_training():

    ann_submodel = ann_sm.ANN_Submodel()
    print(ann_submodel)


    # Generate some data from a 2 class stochastic process (for training and testing)

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

    # Generate training and data
    data_points_label_0 = process.generate_data_points(label=0, count=500)
    data_points_label_1 = process.generate_data_points(label=1, count=500)
    training_data = data_points_label_0 + data_points_label_1

    # Generate test and data
    test_data_count = 500
    data_points_label_0 = process.generate_data_points(label=0, count=int(test_data_count/2))
    data_points_label_1 = process.generate_data_points(label=1, count=int(test_data_count/2))
    test_data = data_points_label_0 + data_points_label_1


    # Train ANN_Submodel
    ann_submodel.train(training_data)

    # Test ANN_Submodel and report results

    ann_submodel.predict(test_data)

    num_errors = 0
    for point in test_data:
        if (point.predicted_y != point.true_y):
            num_errors += 1

    error_percent = num_errors*100/test_data_count

    print("test_ann_submodel_training: num_errors={}, error_percent={}".format(num_errors, error_percent))


###################################################################################################

# Call test functions

test_ann_submodel_training()