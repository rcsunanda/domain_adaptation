"""
Main script (entry point) of the system
"""

import domain_adaptation.system_coordinator as sys_coord


###################################################################################################
"""
Set system parameters and run it
"""

def main():

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

    sys_parameters.results_manager_avg_error_window_size = 50

    sys_parameters.system_coordinator_initial_dataset_size = 1000

    print("System parameters are set: \n{}".format(sys_parameters))


    # Create and run SystemCoordinator

    sys_coordinator = sys_coord.SystemCoordinator(sys_parameters)

    print("Starting system... \n{}".format(sys_coordinator))

    sys_coordinator.run()

    print("Run finished. Exiting system...")




###################################################################################################

# Call main function

main()