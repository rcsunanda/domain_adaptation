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
        self.process_class_distribution_parameters_2 = None # For Abrupt_Drift and Recurring_Context drift scenarios

        self.detector_window_size = None
        self.detector_diff_threshold_to_sum = None
        self.detector_diff_sum_threshold_to_detect = None

        self.adaptor_submodel_type = None

        self.results_manager_avg_error_window_size = None

        self.system_coordinator_initial_dataset_size = None
        self.system_coordinator_total_sequence_size = None
        self.system_coordinator_drift_scenario = None

        # For "Gradual_Drift" drift scenario
        self.system_coordinator_drift_period_size = None   # How long the gradual drifting takes
        self.system_coordinator_mean_dim1_shift = None  # How much the mean of each class distribution will be shifted along dimension 1

        # For "Recurring_Context" drift scenario
        self.system_coordinator_recurrence_count = None # How many times to switch contexts


    def __repr__(self):
        return "SystemParameters(\n\tprocess_num_dimensions={} \n\tprocess_num_classes={} \n\t" \
               "process_class_distribution_parameters={} \n\tprocess_class_distribution_parameters_2={} \n\tdetector_window_size={} \n\t" \
               "detector_diff_threshold_to_sum={} \n\tdetector_diff_sum_threshold_to_detect={}"\
               "results_manager_avg_error_window_size={} \n\tsystem_coordinator_initial_dataset_size={} \n\ttotal_sequence_size={} \n\t" \
               "system_coordinator_drift_scenario={} \n)"\
            .format(self.process_num_dimensions, self.process_num_classes,
                    self.process_class_distribution_parameters, self.process_class_distribution_parameters_2, self.detector_window_size,
                    self.detector_diff_threshold_to_sum, self.detector_diff_sum_threshold_to_detect,
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

        self.detector = dd.DriftDetector(
            sys_params.detector_window_size,
            sys_params.detector_diff_threshold_to_sum,
            sys_params.detector_diff_sum_threshold_to_detect)

        self.adaptor = da.DriftAdaptor(self.ensemble, sys_params.adaptor_submodel_type)

        self.results_manager = rman.ResultsManager(sys_params.results_manager_avg_error_window_size)

        self.initial_dataset_size = sys_params.system_coordinator_initial_dataset_size
        self.total_sequence_size = sys_params.system_coordinator_total_sequence_size
        self.batch_size = sys_params.system_coordinator_batch_size  # How many DataPoints are generated per main loop iteration
        self.submodel_type = sys_params.adaptor_submodel_type
        self.drift_scenario = sys_params.system_coordinator_drift_scenario

        # Parameters specific to scenarios

        if (self.drift_scenario == "Abrupt_Drift"):
            self.process_class_distribution_parameters_2 = sys_params.process_class_distribution_parameters_2
            self.was_drift_occured = False  # To ensure we set the second set of process parameters only once
            self.midpoint = (self.total_sequence_size + self.initial_dataset_size) / 2

            self.set_process_parameters = self.set_abrupt_drift_process_params

        elif (self.drift_scenario == "Gradual_Drift"):
            self.midpoint = (self.total_sequence_size + self.initial_dataset_size) / 2

            drift_period_size = sys_params.system_coordinator_drift_period_size
            self.drift_start_seq = self.midpoint - drift_period_size / 2
            self.drift_end_seq = self.midpoint + drift_period_size / 2

            self.increment = sys_params.system_coordinator_mean_dim1_shift * self.batch_size / drift_period_size

            self.is_left_printed = False  # To print drift start and end only once
            self.is_right_printed = False

            self.set_process_parameters = self.set_gradual_drift_process_params


        elif (self.drift_scenario == "Recurring_Context"):
            self.process_class_distribution_parameters = sys_params.process_class_distribution_parameters
            self.process_class_distribution_parameters_2 = sys_params.process_class_distribution_parameters_2

            self.recurrence_count = sys_params.system_coordinator_recurrence_count

            self.between_switch_size = (self.total_sequence_size + self.initial_dataset_size) / (self.recurrence_count + 1)
            assert self.between_switch_size >= 4 * self.detector.window_size  # To have enough diff-stable periods between switches
            self.next_switch_point = self.between_switch_size

            self.set_process_parameters = self.set_recurring_context_process_params

        else:
            assert False


    def __repr__(self):
        return "SystemCoordinator(\n\tprocess={} \n\tensemble={} \n\tdetector={} \n\tadaptor={} \n\tresults_manager={} \n)"\
            .format(self.process, self.ensemble, self.detector, self.adaptor, self.results_manager)


    # Following 3 functions set time varying process parameters given the current seq no

    def set_abrupt_drift_process_params(self, seq):
        if (self.was_drift_occured == False and seq >= self.midpoint):
            print("Abrupt_Drift scenario: changing process parameters to second set: seq={}".format(seq))
            self.process.set_class_distribution_params(self.process_class_distribution_parameters_2)
            print(self.process)
            self.results_manager.add_special_marker(seq, "drift")
            self.was_drift_occured = True


    def set_gradual_drift_process_params(self, seq):
        if (seq >= self.drift_start_seq and seq <= self.drift_end_seq):
            for class_distribution_params in self.process.class_distribution_params:
                mean = class_distribution_params[0]
                mean[0] += self.increment  # Increment mean along dimension 1

            self.process.set_class_distribution_params(self.process.class_distribution_params)

        if (seq >= self.drift_start_seq and self.is_left_printed == False):
            self.results_manager.add_special_marker(seq, "drift_start")
            self.is_left_printed = True
            print("Gradual drift started: seq={}, process={}".format(seq, self.process))

        if (seq >= self.drift_end_seq and self.is_right_printed == False):
            self.results_manager.add_special_marker(seq, "drift_end")
            self.is_right_printed = True
            print("Gradual drift finished: seq={}, process={}".format(seq, self.process))


    def set_recurring_context_process_params(self, seq):
        if (seq >= self.next_switch_point):
            print("Recurring_Context scenario: switching context: seq={}".format(seq))
            self.results_manager.add_special_marker(seq, "recurrent_drift")

            if (self.process.class_distribution_params == self.process_class_distribution_parameters):
                self.process.set_class_distribution_params(self.process_class_distribution_parameters_2)
            elif (self.process.class_distribution_params == self.process_class_distribution_parameters_2):
                self.process.set_class_distribution_params(self.process_class_distribution_parameters)
            else:
                assert False

            print(self.process)
            self.next_switch_point += self.between_switch_size


    # Generate an initial training dataset, train a submodel, and add it to the ensemble
    def train_initial_model(self):

        initial_training_dataset = self.process.generate_data_points_from_all_labels(self.initial_dataset_size)

        self.adaptor.adapt_ensemble(initial_training_dataset)

        # Generate some test data and check initial_model results (generate in small batches, otherwise results in class imbalance)

        test_data = []
        num_batches = int(self.initial_dataset_size / self.batch_size)
        for i in range(num_batches):
            batch = self.process.generate_data_points_from_all_labels(self.batch_size)
            test_data.extend(batch)

        self.ensemble.predict(test_data)
        self.detector.add_data_points(test_data)    # So the detector can use this data for diff calculation

        # predict_and_print_results(self.ensemble, test_data, "initial model")

        self.results_manager.add_prediction_result(len(test_data), test_data)   # First set of results added to results manager
        self.results_manager.print_results()


    def run(self):
        self.train_initial_model()

        # parameterize
        info_print_interval = 2000
        progress_print_interval = 100
        detection_batch_size = 20

        seq_num = len(self.detector.data_point_sequence)    # Because a result set was added in train_initial_model

        info_print_counter = 0
        progress_print_counter = 0
        detection_counter = 0

        while(True):
            batch = self.process.generate_data_points_from_all_labels(total_count=self.batch_size)
            seq_num += self.batch_size

            info_print_counter += self.batch_size
            progress_print_counter += self.batch_size
            detection_counter += self.batch_size

            if (seq_num > (self.total_sequence_size + self.initial_dataset_size)):
                break

            self.ensemble.predict(batch)

            self.results_manager.add_prediction_result(seq_num, batch)
            self.detector.add_data_points(batch)

            if (detection_counter >= detection_batch_size):  # Run detection after a certain no. of samples has been added
                detection_counter = 0
                (is_drift_detected, diff, diff_sum) = self.detector.run_detection(seq_num)
                self.results_manager.add_detection_info(seq_num, diff, diff_sum, is_drift_detected)

                if (is_drift_detected == True):
                    latest_window = self.detector.get_latest_window()
                    self.adaptor.adapt_ensemble(latest_window)
                    # print("Drift detected: adapted ensemble: submodel_count={} \nensemble={}"
                    #       .format(len(self.ensemble.submodels), self.ensemble))

            self.set_process_parameters(seq_num)

            if (info_print_counter >= info_print_interval):
                info_print_counter = 0
                self.results_manager.print_results()

            if (progress_print_counter >= progress_print_interval):
                progress_print_counter = 0
                print("seq_num={}".format(seq_num))

        print(self.detector)
        self.results_manager.plot_results()


