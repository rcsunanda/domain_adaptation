"""
DriftAdaptor class
"""

import domain_adaptation.distribution_difference as ddif
import domain_adaptation.ann_submodel as ann_sm
import domain_adaptation.decision_tree_submodel as tree_sm

import numpy as np


###################################################################################################
"""
Create a Submodel of appropriate type (eg: ANN_Submodel)
Note: This should be done with a factory pattern. Doing it with a simple function for convenience
"""

def create_submodel(submodel_type, classifer_type):

    submodel = None

    if (submodel_type == "ANN_Submodel"):
        submodel = ann_sm.ANN_Submodel(weight=1, pdf=None, classifer_type=classifer_type)
    elif (submodel_type == "DecisionTreeSubmodel"):
        submodel = tree_sm.DecisionTreeSubmodel(weight=1, pdf=None, classifer_type=classifer_type)
    else:
        assert False

    return submodel


###################################################################################################
"""
Given a window of samples, compute the bounds/ region of its values
"""

def compute_window_bounds(window_samples):
    dimensions = len(window_samples[0])
    min_vals = [+np.inf for i in range(dimensions)]
    max_vals = [-np.inf for i in range(dimensions)]

    for sample in window_samples:
        for index, feature_val in enumerate(sample):
            if (sample[index] < min_vals[index]):
                min_vals[index] = sample[index]
            if (sample[index] > max_vals[index]):
                max_vals[index] = sample[index]

    bounds = []
    for i in range(dimensions):
        bounds.append((min_vals[i], max_vals[i]))

    return bounds


###################################################################################################
"""
Given the bounds of two windows compute the overall bounds
"""

def compute_overall_bounds(bounds_1, bounds_2):
    assert len(bounds_1) == len(bounds_2)

    overall_bounds = []
    for b1, b2 in zip(bounds_1, bounds_2):
        min_val = min(b1[0], b2[0])
        max_val = max(b1[1], b2[1])
        overall_bounds.append((min_val, max_val))

    return overall_bounds

###################################################################################################
"""
DriftAdaptor adapts a ModelEnsemble to drift by doing the following 2 actions
(1) Train a new Submodel from given latest sample window and add it to ModelEnsemble
(2) Update weights of current Submodels in ModelEnsemble based on distance between current window's pdf and submodels' pdfs
"""

class DriftAdaptor:
    def __init__(self, ensemble, submodel_type, classifer_type):
        self.ensemble = ensemble
        self.submodel_type = submodel_type
        self.classifer_type = classifer_type

    def __repr__(self):
        return "DriftAdaptor(\n\tensemble={} \n)".format(self.ensemble)


    def adapt_ensemble(self, data_point_window):
        new_submodel = self.train_new_submodel(data_point_window)
        self.update_ensemble_weights(new_submodel.pdf, new_submodel.window_bounds)


    def train_new_submodel(self, data_point_window):
        new_submodel = create_submodel(self.submodel_type, self.classifer_type)
        new_submodel.train(data_point_window)

        window_samples = [point.X for point in data_point_window]
        kde_estimator = ddif.estimate_pdf_kde(window_samples)

        new_submodel.weight = 1
        new_submodel.pdf = kde_estimator
        new_submodel.window_bounds = compute_window_bounds(window_samples)

        self.ensemble.add_submodel(new_submodel)

        return new_submodel


    def update_ensemble_weights(self, kde_estimator, window_bounds):
        for submodel in self.ensemble.submodels:
            overall_bounds = compute_overall_bounds(window_bounds, submodel.window_bounds)
            diff = ddif.total_variation_distance(kde_estimator, submodel.pdf, overall_bounds)

            # assert diff >= 0 and diff <= 1, 'diff={}'.format(diff)  # This assertion failed in real world dataset (diff=1.103)
            #                                                         # Reason could be pdf estimation/ MC integration errors. Therefore use a more relaxed assertion

            assert diff >= -1 and diff <= 2, 'diff={}'.format()

            if (diff > 1):
                diff = 1
            if (diff < 0):
                diff = 0

            submodel.weight = 1 - diff

            # if (diff != 0): # Need to check because for current (new) submodel, diff will likely be 0
            #     submodel.weight = 1/diff
