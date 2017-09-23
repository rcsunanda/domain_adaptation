"""
Tests for Submodel, ANN_Submodel and other derived classes
"""

import domain_adaptation.ann_submodel as ann_sm
import domain_adaptation.model_ensemble as ens
import domain_adaptation.process as prc

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
Test the training and predicting of an ANN_Submodel
Train the ANN_Submodel on some training data generated from Process
Then predict on some test data generated from Process, and report error percent
"""

def test_ann_submodel_training():

    ann_submodel = ann_sm.ANN_Submodel(weight=1, pdf=None)
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

    # Generate training and test data
    training_data = process.generate_data_points_from_all_labels(total_count=1000)
    test_data = process.generate_data_points_from_all_labels(total_count=1000)

    # Train ANN_Submodel
    ann_submodel.train(training_data)

    # Test ANN_Submodel and report results
    predict_and_print_results(ann_submodel, test_data, "test_ann_submodel_training")




###################################################################################################
"""
Train 2 submodels on 2 different datasets and add them to ensemble
Then test prediction accuracy of ensemble on second dataset
"""

def test_ensemble_prediction():

    ann_submodel_1 = ann_sm.ANN_Submodel(weight=1, pdf=None)
    ann_submodel_2 = ann_sm.ANN_Submodel(weight=1, pdf=None)
    ensemble = ens.ModelEnsmeble()
    print("ann_submodel_1={}".format(ann_submodel_1))
    print("ann_submodel_2={}".format(ann_submodel_2))
    print("ensemble={}".format(ensemble))


    # Setup 2 processes (class distributions are flipped)

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

    gauss_params_2 = []
    gauss_params_2.append((mean_2, cov_2))
    gauss_params_2.append((mean_1, cov_1))
    process_2 = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params_2)

    print(process_1)
    print(process_2)

    # Generate 2 training datasets from the 2 processes
    training_data_1 = process_1.generate_data_points_from_all_labels(total_count=1000)
    training_data_2 = process_2.generate_data_points_from_all_labels(total_count=1000)

    # Generate a test data from process_2
    test_data = process_2.generate_data_points_from_all_labels(total_count=1000)

    # Train ANN_Submodels and add to ensemble
    ann_submodel_1.train(training_data_1)
    ann_submodel_2.train(training_data_2)

    ensemble.add_submodel(ann_submodel_1)
    ensemble.add_submodel(ann_submodel_2)

    # Test ANN_Submodels and the ensemble seperately and report results
    predict_and_print_results(ann_submodel_1, test_data, "test_ensemble_prediction: ann_submodel_1")
    predict_and_print_results(ann_submodel_2, test_data, "test_ensemble_prediction: ann_submodel_2")
    predict_and_print_results(ensemble, test_data, "test_ensemble_prediction: ensemble")



###################################################################################################

# Call test functions

# test_ann_submodel_training()
test_ensemble_prediction()