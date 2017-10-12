"""
DecisionTreeSubmodel class
"""

import domain_adaptation.submodel as sm

import sklearn.tree as dec_tree

###################################################################################################
"""
DecisionTreeSubmodel is a concrete submodel derived from above Submodel class (decision tree model)
"""

class DecisionTreeSubmodel(sm.Submodel):
    def __init__(self, weight, pdf, classifer_type):
        # solver=lbfgs is good for small datasets
        # solver=adam is good for large datasets

        sm.Submodel.__init__(self, weight, pdf)

        if (classifer_type == "Artificial"):
            # Following works well for artificial data ~ 2 - 4 % error
            self.classfier = dec_tree.DecisionTreeClassifier(random_state=1)
        elif (classifer_type == "Real"):
            # assert False

            # # Following is experimentation for real world dataset ~ 36 - 40 % error
            self.classfier = dec_tree.DecisionTreeClassifier(random_state=1)
        else:
            assert False


    def __repr__(self):
        base_class_str = sm.Submodel.__repr__(self)
        child_class_str = "DecisionTreeSubmodel(\n\tclassfier={} \n)".format(self.classfier)
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
