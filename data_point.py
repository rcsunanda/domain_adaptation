"""
DataPoint class
"""


###################################################################################################
"""
Holds (X, true_y, predicted_y) for each sample or data point
"""

class DataPoint:
    def __init__(self, X, true_y, predicted_y):
        self.X = X
        self.true_y = true_y
        self.predicted_y = predicted_y
        self.predicted_class_probs = None  # array of predicted probabilities for each class
        self.weighted_predicted_class_probs = None  # array of weighted predicted probabilities for each class

    def __repr__(self):
        return "DataPoint(\n\tX={} \n\ttrue_y={} \n\tpredicted_y={} \n\t predicted_class_probs={}\n)".\
            format(self.X, self.true_y, self.predicted_y, self.predicted_class_probs)
