"""
SystemParameters and SystemCoordinator classes
"""

import numpy as np

import domain_adaptation.process as prc
import domain_adaptation.model_ensemble as ens
import domain_adaptation.drift_detector as dd
import domain_adaptation.drift_adaptor as da
import domain_adaptation.results_manager as rman
import domain_adaptation.data_point as data_point
import domain_adaptation.ann_submodel as ann_sm


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

        self.ensemble = ens.ModelEnsmeble()

        self.detector = dd.DriftDetector(
            sys_params.detector_window_size,
            sys_params.detector_diff_threshold_to_sum,
            sys_params.detector_diff_sum_threshold_to_detect)

        self.results_manager = rman.ResultsManager(
            sys_params.results_manager_avg_error_window_size,
            sys_params.system_coordinator_drift_scenario)

        self.results_manager.init_baseline(baseline_name="no_adaptation", baseline_num=0)
        self.results_manager.init_baseline(baseline_name="all_data", baseline_num=1)
        self.results_manager.init_baseline(baseline_name="latest_window", baseline_num=2)

        self.initial_dataset_size = sys_params.system_coordinator_initial_dataset_size
        self.total_sequence_size = sys_params.system_coordinator_total_sequence_size
        self.batch_size = sys_params.system_coordinator_batch_size  # How many DataPoints are generated per main loop iteration
        self.submodel_type = sys_params.adaptor_submodel_type
        self.drift_scenario = sys_params.system_coordinator_drift_scenario

        # Parameters specific to scenarios

        create_process = True   # For artificial datasets
        classfier_type = "Artificial"   # For artificial datasets

        if (self.drift_scenario == "Abrupt_Drift"):
            self.process_class_distribution_parameters_2 = sys_params.process_class_distribution_parameters_2
            self.was_drift_occured = False  # To ensure we set the second set of process parameters only once
            self.midpoint = (self.total_sequence_size + self.initial_dataset_size) / 2

            self.set_process_parameters = self.set_abrupt_drift_process_params
            self.generate_data_batch = self.generate_artificial_data_batch

        elif (self.drift_scenario == "Gradual_Drift"):
            self.midpoint = (self.total_sequence_size + self.initial_dataset_size) / 2

            drift_period_size = sys_params.system_coordinator_drift_period_size
            self.drift_start_seq = self.midpoint - drift_period_size / 2
            self.drift_end_seq = self.midpoint + drift_period_size / 2

            self.increment = sys_params.system_coordinator_mean_dim1_shift * self.batch_size / drift_period_size

            self.is_left_printed = False  # To print drift start and end only once
            self.is_right_printed = False

            self.set_process_parameters = self.set_gradual_drift_process_params
            self.generate_data_batch = self.generate_artificial_data_batch

        elif (self.drift_scenario == "Recurring_Context"):
            self.process_class_distribution_parameters = sys_params.process_class_distribution_parameters
            self.process_class_distribution_parameters_2 = sys_params.process_class_distribution_parameters_2

            self.recurrence_count = sys_params.system_coordinator_recurrence_count

            self.between_switch_size = (self.total_sequence_size + self.initial_dataset_size) / (self.recurrence_count + 1)
            assert self.between_switch_size >= 4 * self.detector.window_size  # To have enough diff-stable periods between switches
            self.next_switch_point = self.between_switch_size

            self.set_process_parameters = self.set_recurring_context_process_params
            self.generate_data_batch = self.generate_artificial_data_batch

        elif (self.drift_scenario == "Real_World_Dataset"):
            self.real_data_points = []
            self.curr_real_dataset_pos = 0
            self.load_real_dataset(sys_params.system_coordinator_real_dataset_filename)

            self.set_process_parameters = self.set_real_dataset_params
            self.generate_data_batch = self.generate_real_data_batch

            create_process = False
            classfier_type = "Real"

        else:
            assert False

        self.process = None
        if (create_process == True):
            self.process = prc.Process(
                sys_params.process_num_dimensions,
                sys_params.process_num_classes,
                sys_params.process_class_distribution_parameters)

        # Baseline models
        self.original_model = da.create_submodel(sys_params.adaptor_submodel_type, classfier_type)
        self.all_data_model = da.create_submodel(sys_params.adaptor_submodel_type, classfier_type)
        self.latest_window_model = da.create_submodel(sys_params.adaptor_submodel_type, classfier_type)
        self.all_data = []  # Store all the data points, to be used by "all_data" baseline

        self.adaptor = da.DriftAdaptor(self.ensemble, sys_params.adaptor_submodel_type, classfier_type)



    def __repr__(self):
        return "SystemCoordinator(\n\tprocess={} \n\tensemble={} \n\tdetector={} \n\tadaptor={} \n\tresults_manager={} \n)"\
            .format(self.process, self.ensemble, self.detector, self.adaptor, self.results_manager)


    def load_real_dataset(self, filename):
        dataset = np.genfromtxt(filename, delimiter=',', dtype=None)

        for example in dataset:
            X = [example[i] for i in (0,1,4,6,7)]   # Only these 5 features are important
            X[1] = X[1]/7   # Normalize day of week feature to [0,1]
            label = example[8]
            if (label == b'UP'):
                y = 0
            elif (label == b'DOWN'):
                y = 1
            else:
                assert False

            point = data_point.DataPoint(X, y, -1)
            self.real_data_points.append(point)

        pass

        # for i in range(10):
        #     print("data_point={}".format(self.data_points[i]))


    # Following 4 functions set time varying process parameters given the current seq no

    def set_abrupt_drift_process_params(self, seq):
        if (self.was_drift_occured == False and seq >= self.midpoint):
            print("Abrupt_Drift scenario: changing process parameters to second set: seq={}".format(seq))
            self.process.set_class_distribution_params(self.process_class_distribution_parameters_2)
            print(self.process)
            self.results_manager.add_special_marker(seq, "actual_drift")
            self.was_drift_occured = True


    def set_gradual_drift_process_params(self, seq):
        if (seq >= self.drift_start_seq and seq <= self.drift_end_seq):
            for class_distribution_params in self.process.class_distribution_params:
                mean = class_distribution_params[0]
                mean[0] += self.increment  # Increment mean along dimension 1

            self.process.set_class_distribution_params(self.process.class_distribution_params)

        if (seq >= self.drift_start_seq and self.is_left_printed == False):
            self.results_manager.add_special_marker(seq, "actual_drift_start")
            self.is_left_printed = True
            print("Gradual drift started: seq={}, process={}".format(seq, self.process))

        if (seq >= self.drift_end_seq and self.is_right_printed == False):
            self.results_manager.add_special_marker(seq, "actual_drift_end")
            self.is_right_printed = True
            print("Gradual drift finished: seq={}, process={}".format(seq, self.process))


    def set_recurring_context_process_params(self, seq):
        if (seq >= self.next_switch_point):
            print("Recurring_Context scenario: switching context: seq={}".format(seq))
            self.results_manager.add_special_marker(seq, "actual_drift")

            if (self.process.class_distribution_params == self.process_class_distribution_parameters):
                self.process.set_class_distribution_params(self.process_class_distribution_parameters_2)
            elif (self.process.class_distribution_params == self.process_class_distribution_parameters_2):
                self.process.set_class_distribution_params(self.process_class_distribution_parameters)
            else:
                assert False

            print(self.process)
            self.next_switch_point += self.between_switch_size


    def set_real_dataset_params(self, seq): # No parameters to set
        pass


    # Following 2 functions are for generating data (for artificial and real world datasets)

    def generate_artificial_data_batch(self, count):
        return self.process.generate_data_points_from_all_labels(total_count=count)

    def generate_real_data_batch(self, count):
        start = self.curr_real_dataset_pos
        end = start + count
        self.curr_real_dataset_pos = end
        return self.real_data_points[start: end]


    # Generate an initial training dataset, train a submodel, and add it to the ensemble
    def train_initial_model(self):

        initial_training_dataset = self.generate_data_batch(self.initial_dataset_size)
        self.all_data.extend(initial_training_dataset)

        # Baselines
        self.original_model.train(initial_training_dataset)
        self.all_data_model.train(initial_training_dataset)
        self.latest_window_model.train(initial_training_dataset)

        self.adaptor.adapt_ensemble(initial_training_dataset)

        # Generate some test data and check initial_model results

        if (self.drift_scenario == "Real_World_Dataset"):
            test_data = self.generate_data_batch(self.initial_dataset_size)

        else:   # Artificial dataset
            # Generate in small batches, otherwise results in class imbalance
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

        # Predict using original_model and add results
        self.original_model.predict(test_data)
        self.results_manager.add_baseline_prediction_result(0, len(test_data), test_data)

        self.all_data_model.predict(test_data)
        self.results_manager.add_baseline_prediction_result(1, len(test_data), test_data)

        self.latest_window_model.predict(test_data)
        self.results_manager.add_baseline_prediction_result(2, len(test_data), test_data)


    def run(self):
        self.train_initial_model()

        # parameterize
        info_print_interval = 2000
        progress_print_interval = 100
        detection_batch_size = 20

        total_count_todo = self.total_sequence_size + self.initial_dataset_size

        seq_num = len(self.detector.data_point_sequence)    # Because a result set was added in train_initial_model

        info_print_counter = 0
        progress_print_counter = 0
        detection_counter = 0

        while(True):
            batch = self.generate_data_batch(self.batch_size)
            seq_num += self.batch_size

            info_print_counter += self.batch_size
            progress_print_counter += self.batch_size
            detection_counter += self.batch_size

            if (seq_num >= total_count_todo):
                break

            self.detector.add_data_points(batch)


            # Predict with adapted ensemble and add results

            self.ensemble.predict(batch)
            self.results_manager.add_prediction_result(seq_num, batch)

            # Predict using baselines and add results

            self.original_model.predict(batch)
            self.results_manager.add_baseline_prediction_result(0, seq_num, batch)

            self.all_data_model.train(self.all_data)
            self.all_data_model.predict(batch)
            self.results_manager.add_baseline_prediction_result(1, seq_num, batch)

            latest_window = self.all_data[-500:]
            self.latest_window_model.train(latest_window)
            self.latest_window_model.predict(batch)
            self.results_manager.add_baseline_prediction_result(2, seq_num, batch)

            self.all_data.extend(batch)


            if (detection_counter >= detection_batch_size):  # Run detection after a certain no. of samples has been added
                detection_counter = 0
                (is_drift_detected, diff, diff_sum) = self.detector.run_detection(seq_num)
                self.results_manager.add_detection_info(seq_num, diff, diff_sum, is_drift_detected)

                if (is_drift_detected == True):
                    latest_window = self.detector.get_latest_window()
                    self.adaptor.adapt_ensemble(latest_window)
                    print("Drift detected: adapted ensemble: submodel_count={}".format(len(self.ensemble.submodels)))

            self.set_process_parameters(seq_num)

            if (info_print_counter >= info_print_interval):
                info_print_counter = 0
                self.results_manager.print_results()

            if (progress_print_counter >= progress_print_interval):
                progress_print_counter = 0
                print("seq_num={}/{}".format(seq_num, total_count_todo))

        print(self.detector)
        print("--------------------------------------")
        self.results_manager.print_results()
        self.results_manager.plot_results()


