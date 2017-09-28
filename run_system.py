"""
Main script (entry point) of the system
"""

import domain_adaptation.system_coordinator as sys_coord


###################################################################################################
"""
Set system parameters and run it with Abrupt_Drift scenario
"""

def run_abrupt_drift():

    print("Setting system parameters...")
    sys_parameters = sys_coord.SystemParameters()

    # Process parameters

    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    sys_parameters.process_num_dimensions = 2
    sys_parameters.process_num_classes = 2
    sys_parameters.process_class_distribution_parameters = gauss_params

    # Second set of process paramers (to set after abrupt drift)

    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 3]
    cov_1 = [[1, 0], [0, 1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 0]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    sys_parameters.process_class_distribution_parameters_2 = gauss_params


    # Other parameters

    sys_parameters.adaptor_submodel_type = "ANN_Submodel"

    sys_parameters.detector_window_size = 500
    sys_parameters.detector_diff_threshold_to_sum = 0.005
    sys_parameters.detector_diff_sum_threshold_to_detect = 0.05

    sys_parameters.results_manager_avg_error_window_size = 50

    sys_parameters.system_coordinator_initial_dataset_size = 1000
    sys_parameters.system_coordinator_total_sequence_size = 4000
    sys_parameters.system_coordinator_batch_size = 10

    sys_parameters.system_coordinator_drift_scenario = "Abrupt_Drift"

    print("System parameters are set: \n{}".format(sys_parameters))


    # Create and run SystemCoordinator

    sys_coordinator = sys_coord.SystemCoordinator(sys_parameters)

    print("Starting system... \n{}".format(sys_coordinator))

    sys_coordinator.run()

    print("Run finished. Exiting system...")




###################################################################################################
"""
Set system parameters and run it with Gradual_Drift scenario
"""

def run_gradual_drift():

    print("Setting system parameters...")
    sys_parameters = sys_coord.SystemParameters()

    # Process parameters

    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    sys_parameters.process_num_dimensions = 2
    sys_parameters.process_num_classes = 2
    sys_parameters.process_class_distribution_parameters = gauss_params


    # Other parameters

    sys_parameters.adaptor_submodel_type = "ANN_Submodel"

    sys_parameters.detector_window_size = 500
    sys_parameters.detector_diff_threshold_to_sum = 0.0035
    sys_parameters.detector_diff_sum_threshold_to_detect = 0.04

    sys_parameters.results_manager_avg_error_window_size = 50

    sys_parameters.system_coordinator_initial_dataset_size = 1000
    sys_parameters.system_coordinator_total_sequence_size = 5000
    sys_parameters.system_coordinator_batch_size = 10

    sys_parameters.system_coordinator_drift_scenario = "Gradual_Drift"

    sys_parameters.system_coordinator_drift_period_size = 2000
    sys_parameters.system_coordinator_mean_dim1_shift = 3

    print("System parameters are set: \n{}".format(sys_parameters))


    # Create and run SystemCoordinator

    sys_coordinator = sys_coord.SystemCoordinator(sys_parameters)

    print("Starting system... \n{}".format(sys_coordinator))

    sys_coordinator.run()

    print("Run finished. Exiting system...")



###################################################################################################
"""
Set system parameters and run it with Abrupt_Drift scenario
"""

def run_recurring_context():

    print("Setting system parameters...")
    sys_parameters = sys_coord.SystemParameters()

    # Process parameters

    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 3]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    sys_parameters.process_num_dimensions = 2
    sys_parameters.process_num_classes = 2
    sys_parameters.process_class_distribution_parameters = gauss_params

    # Second set of process paramers (to set after abrupt drift)

    gauss_params = []

    # class 1 Gaussian distribution params
    mean_1 = [0, 3]
    cov_1 = [[1, 0], [0, 1]]

    # class 2 Gaussian distribution params
    mean_2 = [3, 0]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    sys_parameters.process_class_distribution_parameters_2 = gauss_params


    # Other parameters

    sys_parameters.adaptor_submodel_type = "ANN_Submodel"

    sys_parameters.detector_window_size = 500
    sys_parameters.detector_diff_threshold_to_sum = 0.005
    sys_parameters.detector_diff_sum_threshold_to_detect = 0.05

    sys_parameters.results_manager_avg_error_window_size = 50

    sys_parameters.system_coordinator_initial_dataset_size = 1000
    sys_parameters.system_coordinator_total_sequence_size = 7000
    sys_parameters.system_coordinator_batch_size = 10

    sys_parameters.system_coordinator_drift_scenario = "Recurring_Context"
    sys_parameters.system_coordinator_recurrence_count = 3

    print("System parameters are set: \n{}".format(sys_parameters))


    # Create and run SystemCoordinator

    sys_coordinator = sys_coord.SystemCoordinator(sys_parameters)

    print("Starting system... \n{}".format(sys_coordinator))

    sys_coordinator.run()

    print("Run finished. Exiting system...")



###################################################################################################

# Call main functions

run_abrupt_drift()
# run_gradual_drift()
# run_recurring_context()