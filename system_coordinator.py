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

        self.results_manager_avg_error_window_size = None

    def __repr__(self):
        return "SystemParameters(\n\tprocess_num_dimensions={} \n\tprocess_num_classes={} \n\t" \
               "process_class_distribution_parameters={} \n\tdetector_window_size={} \n\t" \
               "results_manager_avg_error_window_size={} \n)"\
            .format(self.process_num_dimensions, self.process_num_classes,
                    self.process_class_distribution_parameters, self.detector_window_size,
                    self.results_manager_avg_error_window_size)




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
        self.adaptor = da.DriftAdaptor(self.ensemble)

        self.results_manager = rman.ResultsManager(sys_params.results_manager_avg_error_window_size)

    def __repr__(self):
        return "SystemCoordinator(\n\tprocess={} \n\tensemble={} \n\tdetector={} \n\tadaptor={} \n\tresults_manager={} \n)"\
            .format(self.process, self.ensemble, self.detector, self.adaptor, self.results_manager)


    def train_initial_model(self):
        pass


    def run(self):
        pass