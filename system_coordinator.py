"""
SystemParameters and SystemCoordinator classes
"""

import domain_adaptation.data_point as dp
import domain_adaptation.ann_submodel as ann_sm

import domain_adaptation.process as prc
import domain_adaptation.model_ensemble as ens
import domain_adaptation.drift_detector as dd
import domain_adaptation.drift_adaptor as da
import domain_adaptation.results_manager as rman


###################################################################################################
"""
SystemParameters contains the parameters to be passed when initializing SystemCoordinator
"""

class SystemParameters:
    def __init__(self):
        self.process_num_dimensions = None
        self.process_num_classes = None
        self.process_class_distribution_parameters = None
        self.process_class_distribution_parameters_2 = None

        self.detector_window_size = None

        self.adaptor_submodel_type = None

        self.results_manager_avg_error_window_size = None

        self.system_coordinator_initial_dataset_size = None
        self.system_coordinator_total_sequence_size = None
        self.system_coordinator_drift_scenario = None


    def __repr__(self):
        return "SystemParameters(\n\tprocess_num_dimensions={} \n\tprocess_num_classes={} \n\t" \
               "process_class_distribution_parameters={} \n\t process_class_distribution_parameters_2={} \n\t detector_window_size={} \n\t" \
               "results_manager_avg_error_window_size={} \n\tsystem_coordinator_initial_dataset_size={} \n\ttotal_sequence_size={} \n\t" \
               "system_coordinator_drift_scenario={} \n)"\
            .format(self.process_num_dimensions, self.process_num_classes,
                    self.process_class_distribution_parameters, self.process_class_distribution_parameters_2, self.detector_window_size,
                    self.results_manager_avg_error_window_size, self.system_coordinator_initial_dataset_size,
                    self.system_coordinator_initial_dataset_size, self.system_coordinator_drift_scenario)



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
SystemCoordinator holds objects of important classes of the system, emulates a non-stationary process,
 ,runs our detection+adaptation algorithm on the process, and reports results  
"""

class SystemCoordinator:
    def __init__(self, sys_params):

        self.process = prc.Process(
            sys_params.process_num_dimensions,
            sys_params.process_num_classes,
            sys_params.process_class_distribution_parameters)


        self.ensemble = ens.ModelEnsmeble()

        self.detector = dd.DriftDetector(sys_params.detector_window_size)
        self.adaptor = da.DriftAdaptor(self.ensemble, sys_params.adaptor_submodel_type)

        self.results_manager = rman.ResultsManager(sys_params.results_manager_avg_error_window_size)

        self.initial_dataset_size = sys_params.system_coordinator_initial_dataset_size
        self.total_sequence_size = sys_params.system_coordinator_total_sequence_size
        self.submodel_type = sys_params.adaptor_submodel_type
        self.drift_scenario = sys_params.system_coordinator_drift_scenario

        # For "Abrupt_Drift" drift scenario
        self.process_class_distribution_parameters_2 = sys_params.process_class_distribution_parameters_2
        self.was_drift_occured = False  # To ensure we set the second set of process parameters only once

        # For "Gradual_Drift" drift scenario
        self.is_left_printed = False  # To print drift start and end only once
        self.is_right_printed = False


    def __repr__(self):
        return "SystemCoordinator(\n\tprocess={} \n\tensemble={} \n\tdetector={} \n\tadaptor={} \n\tresults_manager={} \n)"\
            .format(self.process, self.ensemble, self.detector, self.adaptor, self.results_manager)


    # Given the current seq no, set time varying process parameters
    def set_process_parameters(self, seq):
        if (self.drift_scenario == "Abrupt_Drift"):
            midpoint = (self.total_sequence_size + self.initial_dataset_size)/2
            if (self.was_drift_occured == False and seq >= midpoint):
                print("Abrupt_Drift scenario: changing process parameters to second set")
                self.process.set_class_distribution_params(self.process_class_distribution_parameters_2)
                print(self.process)
                self.was_drift_occured = True

        elif (self.drift_scenario == "Gradual_Drift"):
            midpoint = (self.total_sequence_size + self.initial_dataset_size)/2

            drift_period_size = 2000
            left = midpoint - drift_period_size/2
            right = midpoint + drift_period_size/2

            batch_size = 10 # MUST BE PARAMETERIZED (same as used in run())
            increment = 3*batch_size/drift_period_size
            if (seq >= left and seq <= right):
                mean_1 = self.process.class_distribution_params[0][0]
                mean_2 = self.process.class_distribution_params[1][0]
                mean_1[0] += increment
                mean_2[0] += increment
                self.process.set_class_distribution_params(self.process.class_distribution_params)

            if (seq >= left and self.is_left_printed == False):
                self.results_manager.add_special_marker(seq, "drift_start")
                self.is_left_printed = True
                print("Gradual drift started: seq={}, process={}".format(seq, self.process))

            if (seq >= right and self.is_right_printed == False):
                self.results_manager.add_special_marker(seq, "drift_end")
                self.is_right_printed = True
                print("Gradual drift finished: seq={}, process={}".format(seq, self.process))

        elif (self.drift_scenario == "Recurring_Context"):
            assert False
        else:
            assert False


    # Generate an initial training dataset, train a submodel, and add it to the ensemble
    def train_initial_model(self):

        initial_training_dataset = self.process.generate_data_points_from_all_labels(self.initial_dataset_size)

        self.adaptor.adapt_ensemble(initial_training_dataset)

        # initial_submodel = da.create_submodel(self.submodel_type)
        # initial_submodel.train(initial_training_dataset)
        #
        # self.ensemble.add_submodel(initial_submodel)


        # Generate some test data and check initial_model results

        test_data = self.process.generate_data_points_from_all_labels(self.initial_dataset_size)

        self.ensemble.predict(test_data)

        # predict_and_print_results(self.ensemble, test_data, "initial model")

        # Add to results manager

        self.results_manager.add_prediction_result(len(test_data), test_data)   # First set of results added to results manager
        self.results_manager.print_results()
        # self.results_manager.plot_results()


    def run(self):
        self.train_initial_model()

        # parameterize
        batch_size = 10
        info_print_interval = 2000
        progress_print_interval = 100
        detection_batch_size = 10


        total_samples = 0
        res_manager_seq_num = self.initial_dataset_size - 1 # Because a result set was added in train_initial_model

        info_print_counter = 0
        progress_print_counter = 0
        detection_counter = 0

        while(True):
            batch = self.process.generate_data_points_from_all_labels(total_count=batch_size)
            total_samples += batch_size
            res_manager_seq_num += batch_size

            info_print_counter += batch_size
            progress_print_counter += batch_size
            detection_counter += batch_size

            if (total_samples > self.total_sequence_size):
                break

            self.ensemble.predict(batch)

            self.results_manager.add_prediction_result(res_manager_seq_num, batch)

            if (detection_counter > detection_batch_size):  # Run detection after a batch of samples has been added
                detection_counter = 0
                self.detector.add_data_points(batch)
                (is_drift_detected, diff, diff_sum) = self.detector.run_detection()
                self.results_manager.add_detection_info(res_manager_seq_num, diff, diff_sum, is_drift_detected)

                if (is_drift_detected == True):
                    latest_window = self.detector.get_latest_window()
                    self.adaptor.adapt_ensemble(latest_window)
                    # print("Drift detected: adapted ensemble: submodel_count={} \nensemble={}"
                    #       .format(len(self.ensemble.submodels), self.ensemble))

            self.set_process_parameters(res_manager_seq_num)

            if (info_print_counter > info_print_interval):
                info_print_counter = 0
                self.results_manager.print_results()

            if (progress_print_counter > progress_print_interval):
                progress_print_counter = 0
                print("res_manager_seq_num={}".format(res_manager_seq_num))

        print(self.detector)
        self.results_manager.plot_results()


