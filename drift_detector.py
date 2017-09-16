"""
DriftDetector class
"""

# import domain_adaptation.data_point as data_point
#
# import scipy.stats as st


###################################################################################################
"""
DriftDetector saves a sequence of data points and continuously runs a drift detection algorithm on them
The algorithm calculates the difference between pdfs of some two windows occurring before current time point
This difference is summed, and when it increases above a threshold, it is considered as a drift occurrence
"""

class DriftDetector:
    def __init__(self, window_size):
        self.data_point_sequence = []

        self.window_size = window_size
        #self.current_diff = 0   # may not need to be a memeber
        self.diff_sum = 0

    def __repr__(self):
        return "DriftDetector(\n\twindow_size={} \n\tdiff_sum={} \n\tdata_point_sequence={} \n)"\
            .format(self.window_size, self.diff_sum, self.data_point_sequence)

    def add_data_points(self, data_points):
        # data_points must be an iterable
        self.data_point_sequence.extend(data_points)


    # Return tuple (is_drift_detected, current_diff, diff_sum)
    def run_detection(self):
        sequence_size = len(self.data_point_sequence)

        # Not enough data points for two windows
        if (sequence_size < 2 * self.window_size):
            return (False, 0, 0)

        window_2_left_bound = sequence_size - self.window_size
        window_1_left_bound = sequence_size - 2 * self.window_size

        window_1 = self.data_point_sequence[window_1_left_bound: window_2_left_bound]
        window_2 = self.data_point_sequence[window_2_left_bound: ]

        return