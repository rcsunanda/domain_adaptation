"""
DriftDetector class and required functions
"""

import domain_adaptation.distribution_difference as ddif

import numpy as np


###################################################################################################
"""
Given two windows of DataPoints, convert them to windows of samples (vectors)
Also, find the [min, max] bounds for each dimension in the two windows
Return tuple (window_1_samples, window_2_samples, bounds)
"""

def prepare_sample_windows(window_1, window_2):

    dimensions = len(window_1[0].X)
    min_vals = [+np.inf for i in range(dimensions)]
    max_vals = [-np.inf for i in range(dimensions)]

    def set_min_max_vals(sample):
        for index, feature_val in enumerate(sample):
            if (sample[index] < min_vals[index]):
                min_vals[index] = sample[index]
            if (sample[index] > max_vals[index]):
                max_vals[index] = sample[index]


    window_1_samples = []
    for point in window_1:
        sample = point.X
        window_1_samples.append(sample)
        set_min_max_vals(sample)

    window_2_samples = []
    for point in window_2:
        sample = point.X
        window_2_samples.append(sample)
        set_min_max_vals(sample)

    bounds = []
    for i in range(dimensions):
        bounds.append((min_vals[i], max_vals[i]))

    return (window_1_samples, window_2_samples, bounds)



###################################################################################################
"""
DriftDetector saves a sequence of data points and continuously runs a drift detection algorithm on them
The algorithm calculates the difference between pdfs of some two windows occurring before current time point
This difference is summed, and when it increases above a threshold, it is considered as a drift occurrence
"""

class DriftDetector:
    def __init__(self, window_size):
        self.data_point_sequence = []

        self.diff_sequence = []  # Temporary; for debugging
        self.diff_sum_sequence = []  # Temporary; for debugging
        self.drift_detected_seq_nums = []  # Temporary; for debugging

        self.window_size = window_size
        self.diff_threshold_to_sum = 0.005  # Parameterize
        self.diff_sum_threshold_to_detect = 0.05
        #self.current_diff = 0   # may not need to be a memeber
        self.diff_sum = 0

    def __repr__(self):
        return "DriftDetector(\n\twindow_size={} \n\tdiff_sum={} \n\tdrift_detected_seq_nums={} \n)"\
            .format(self.window_size, self.diff_sum, self.drift_detected_seq_nums)

    def add_data_points(self, data_points):
        # data_points must be an iterable
        self.data_point_sequence.extend(data_points)


    # Return tuple (is_drift_detected, current_diff, diff_sum)
    def run_detection(self, seq):
        sequence_size = len(self.data_point_sequence)

        # print("seq={}, sequence_size={}".format(seq, sequence_size))

        # Not enough data points for two windows
        if (sequence_size < 2 * self.window_size):
            diff = 0
            self.diff_sequence.append(diff)
            self.diff_sum_sequence.append(self.diff_sum)
            return (False, 0, 0)

        window_2_left_bound = sequence_size - self.window_size
        window_1_left_bound = sequence_size - 2 * self.window_size

        window_1 = self.data_point_sequence[window_1_left_bound: window_2_left_bound]
        window_2 = self.data_point_sequence[window_2_left_bound: ]

        (window_1_samples, window_2_samples, bounds) = prepare_sample_windows(window_1, window_2)

        kde_estimator_1 = ddif.estimate_pdf_kde(window_1_samples)
        kde_estimator_2 = ddif.estimate_pdf_kde(window_2_samples)

        diff = ddif.total_variation_distance(kde_estimator_1, kde_estimator_2, bounds)

        if (diff > self.diff_threshold_to_sum):
            self.diff_sum += diff

        self.diff_sequence.append(diff)
        self.diff_sum_sequence.append(self.diff_sum)

        is_drift_detected = False
        if (self.diff_sum >= self.diff_sum_threshold_to_detect):
            self.drift_detected_seq_nums.append(sequence_size-1)
            self.diff_sum = 0
            is_drift_detected = True

        return (is_drift_detected, diff, self.diff_sum)


    def get_latest_window(self):
        left_bound = len(self.data_point_sequence) - self.window_size
        latest_window = self.data_point_sequence[left_bound: ]
        return latest_window
