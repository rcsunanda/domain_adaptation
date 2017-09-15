"""
Process class
"""

import domain_adaptation.data_point as data_point

import scipy.stats as st


###################################################################################################
"""
Process models a stochastic process that generates a multivariate data points (X, y)
X is d-dimensional attributes, and y is a target (class label in classification or real value in regression)
P(X|y) are Gaussian distributions
"""

class Process:
    def __init__(self, num_dimensions, num_classes, class_distribution_parameters):
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.distributions = [] # class conditional data distributions P(X|y)

        assert (len(class_distribution_parameters) == num_classes)

        # class_distribution_parameters is a list of (mean, cov) tuples for Gaussian data distribution P(X|y) of each class
        for param_set in class_distribution_parameters:
            mean = param_set[0]
            cov = param_set[1]

            assert (len(mean) == num_dimensions)
            assert (len(cov) == num_dimensions)
            assert (len(cov[0]) == num_dimensions)

            self.distributions.append(st.multivariate_normal(mean=mean, cov=cov))


    def __repr__(self):
        return "Process(\n\tnum_dimensions={} \n\tnum_classes={} \n\tclass_distributions={} \n)".format(self.num_dimensions, self.num_classes, self.distributions)


    def generate_data_points(self, label, count):
        # labels are 0-based

        # Sample from relevant P(X|label)
        dist = self.distributions[label]
        samples = dist.rvs(size=count)

        data_points = [data_point.DataPoint(X, label, -1) for X in samples]

        return data_points
