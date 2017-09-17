"""
ModelEnsmeble class
"""


###################################################################################################
"""
ModelEnsmeble is the collection of submodels (or base learners)
Prediction is done by weighted invoking of submodels
"""

class ModelEnsmeble:
    def __init__(self):
        self.submodels = []

    def __repr__(self):
        return "ModelEnsmeble(\n\tsubmodels={} \n)".format(self.submodels)


    def add_submodel(self, submodel):
        self.submodels.append(submodel)


    def predict(self, data_points):

        weight_sum = 0
        weighted_probs = 0
        for submodel in self.submodels:
            submodel.predict(data_points)
            weight = submodel.get_weight()

            for point in data_points:
                weighted_probs = [weight * prob for prob in point.predicted_class_probs]

                if point.weighted_predicted_class_probs is None:
                    point.weighted_predicted_class_probs = weighted_probs
                else:
                    point.weighted_predicted_class_probs = [sum(x) for x in zip(point.weighted_predicted_class_probs, weighted_probs)]

            weight_sum += weight

        for point in data_points:
            point.weighted_predicted_class_probs = [prob/weight_sum for prob in point.predicted_class_probs]
            point.predicted_y = point.weighted_predicted_class_probs.index(max(point.weighted_predicted_class_probs))