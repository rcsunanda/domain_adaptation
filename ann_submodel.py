"""
ANN_Submodel class
"""

import domain_adaptation.submodel as sm

import sklearn.neural_network as ann

###################################################################################################
"""
ANN_Submodel is a concrete submodel derived from above Submodel class (an artificial neural network model)
"""

class ANN_Submodel(sm.Submodel):
    def __init__(self, weight, pdf):
        # solver=lbfgs is good for small datasets
        # solver=adam is good for large datasets

        sm.Submodel.__init__(self, weight, pdf)

        self.classfier = ann.MLPClassifier(solver='lbfgs', alpha=1e-5,
                                           hidden_layer_sizes=(5, 2), random_state=1)

    def __repr__(self):
        base_class_str = sm.Submodel.__repr__(self)
        child_class_str = "ANN_Submodel(\n\tclassfier={} \n)".format(self.classfier)
        return base_class_str + child_class_str


    def train(self, data_points):
        X_train = []
        y_train = []

        for point in data_points:
            X_train.append(point.X)
            y_train.append(point.true_y)

        self.classfier.fit(X_train, y_train)


    def predict(self, data_points):
        X_test = [point.X for point in data_points]
        predicted_y = self.classfier.predict(X_test)
        predicted_class_probs = self.classfier.predict_proba(X_test)

        for index, point in enumerate(data_points):
            point.predicted_y = predicted_y[index]
            point.predicted_class_probs = predicted_class_probs[index]
