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

        self.detector_window_size = None

        self.adaptor_submodel_type = None

        self.results_manager_avg_error_window_size = None

        self.system_coordinator_initial_dataset_size = None


    def __repr__(self):
        return "SystemParameters(\n\tprocess_num_dimensions={} \n\tprocess_num_classes={} \n\t" \
               "process_class_distribution_parameters={} \n\tdetector_window_size={} \n\t" \
               "results_manager_avg_error_window_size={} \n\tsystem_coordinator_initial_dataset_size={} \n)"\
            .format(self.process_num_dimensions, self.process_num_classes,
                    self.process_class_distribution_parameters, self.detector_window_size,
                    self.results_manager_avg_error_window_size, self.system_coordinator_initial_dataset_size)



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
        self.submodel_type = sys_params.adaptor_submodel_type

    def __repr__(self):
        return "SystemCoordinator(\n\tprocess={} \n\tensemble={} \n\tdetector={} \n\tadaptor={} \n\tresults_manager={} \n)"\
            .format(self.process, self.ensemble, self.detector, self.adaptor, self.results_manager)


    # Generate an initial training dataset, train a submodel, and add it to the ensemble
    def train_initial_model(self):

        per_class_esample_count =  int(self.initial_dataset_size/ self.process.num_classes)

        initial_training_dataset = []
        for label in range(self.process.num_classes):
            data_points = self.process.generate_data_points(label=label, count=per_class_esample_count)
            initial_training_dataset.extend(data_points)

        initial_submodel = da.create_submodel(self.submodel_type)
        initial_submodel.train(initial_training_dataset)

        self.ensemble.add_submodel(initial_submodel)


        # Generate some test data and check initial_model results

        test_data = []
        for label in range(self.process.num_classes):
            data_points = self.process.generate_data_points(label=label, count=per_class_esample_count)
            test_data.extend(data_points)

        self.ensemble.predict(test_data)

        # predict_and_print_results(self.ensemble, test_data, "initial model")

        # Add to results manager

        self.results_manager.add_prediction_result(len(test_data), test_data)

        self.results_manager.print_results()
        self.results_manager.plot_results()
        # self.results_manager.add_prediction_result(index, batch)


    def run(self):
        self.train_initial_model()