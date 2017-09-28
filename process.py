"""
Process class
"""

import domain_adaptation.data_point as data_point

import scipy.stats as st
import numpy as np
import math


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
        self.distributions = [] # Class conditional data distributions P(X|y)
        self.class_distribution_params = class_distribution_parameters  # Store for later retrieving

        assert (len(class_distribution_parameters) == num_classes)

        self.set_class_distribution_params(class_distribution_parameters)



    def __repr__(self):
        return "Process(\n\tnum_dimensions={} \n\tnum_classes={} \n\t class_distribution_params={} \n\tclass_distributions={} \n)"\
            .format(self.num_dimensions, self.num_classes, self.class_distribution_params, self.distributions)


    def set_class_distribution_params(self, class_distribution_parameters):
        # class_distribution_parameters is a list of (mean, cov) tuples for Gaussian data distribution P(X|y) of each class

        self.class_distribution_params = class_distribution_parameters  # Store for later retrieving
        self.distributions = []

        for param_set in class_distribution_parameters:
            mean = param_set[0]
            cov = param_set[1]

            assert (len(mean) == self.num_dimensions)
            assert (len(cov) == self.num_dimensions)
            assert (len(cov[0]) == self.num_dimensions)

            self.distributions.append(st.multivariate_normal(mean=mean, cov=cov))


    # Generate given count of DataPoints from given label
    def generate_data_points(self, label, count):
        # Labels are 0-based

        # Sample from relevant P(X|label)
        dist = self.distributions[label]
        samples = dist.rvs(size=count)

        samples = np.reshape(samples, (count, self.num_dimensions)) # Actually only needed in the self.num_dimensions = 1 case

        data_points = [data_point.DataPoint(X, label, -1) for X in samples]

        return data_points


    # Generate given count of DataPoints from all labels (evenly distributed)
    def generate_data_points_from_all_labels(self, total_count):
        per_class_esample_count = math.floor(total_count / self.num_classes)

        ret_data_points = []
        for label in range(self.num_classes):
            data_points = self.generate_data_points(label=label, count=per_class_esample_count)
            ret_data_points.extend(data_points)

        # If there some remaining count to complete total_count, generate some more samples from the last label
        remaning = total_count - per_class_esample_count * self.num_classes
        if (remaning > 0):
            data_points = self.generate_data_points(label=self.num_classes-1, count=remaning)
            ret_data_points.extend(data_points)

        return ret_data_points


