"""
Submodel class
"""

# import domain_adaptation.data_point as data_point
#
# import scipy.stats as st


###################################################################################################
"""
Submodel is one of the many models/ learners in the ModelEnsemble
Also known as base learner in some works
This is an abstract parent class. Any specific submodel must be derived from this (eg: ANN_Submodel, SVM_Submodel)
"""

class Submodel:
    def __init__(self, weight, pdf):
        self.weight = weight    # may not make sense to have weight in initializer
        self.pdf = pdf

    def __repr__(self):
        return "Submodel(\n\tweight={} \n\tpdf={} \n)".format(self.weight, self.pdf)


    def get_weight(self):
         return self.weight

    def set_weight(self, weight):
        self.weight = weight


    def train(self, data_points):
        assert False  # Not expected to be called in parent abstract class

    def predict(self, data_points):
        assert False  # Not expected to be called in parent abstract class
