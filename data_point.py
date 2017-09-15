"""
DataPoint class
"""


###################################################################################################
"""
Holds (X, true_y, predicted_y) 
"""

class DataPoint:
    def __init__(self, X, true_y, predicted_y):
        self.X = X
        self.true_y = true_y
        self.predicted_y = predicted_y

    def __repr__(self):
        return "DataPoint(\n\tX={} \n\ttrue_y={} \n\tpredicted_y={} \n)".format(self.X, self.true_y, self.predicted_y)
