"""
ResultsManager class
"""

import matplotlib.pyplot as plt
import numpy as np


###################################################################################################
"""
Moving average filter
"""
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


###################################################################################################
"""
Combine multiple legend entries to one
"""
def combine_duplicate_legends():
    handles, labels = plt.gca().get_legend_handles_labels()
    i = 1
    while i < len(labels):
        if labels[i] in labels[:i]:
            del (labels[i])
            del (handles[i])
        else:
            i += 1

    plt.legend(handles, labels, loc='upper left')


###################################################################################################
"""
Class to hold error information (for a baseline)
"""

class ErrorInfo:
    def __init__(self, baseline_name):
        self.baseline_name = baseline_name
        self.total_error_count = 0
        self.errors_in_window = 0
        self.window_avg_error_seq = []
        self.window_sample_count = 0

###################################################################################################
"""
ResultsManager collects results-related information as the data sequence and algorithms progress
It can then be called to print/ plot the collected results
"""

class ResultsManager:
    def __init__(self, avg_error_window_size, title_suffix):
        self.title_suffix = title_suffix

        self.window_error_count_seq = []
        self.window_avg_error_seq = []
        self.detection_points_seq = []  # Sequence numbers when a drift was detected (is a 1/0 seq more suitable for this)
        self.diff_seq = []
        self.diff_sum_seq = []
        self.special_marker_seq = []    # List of (seq, "marker-name") pairs

        self.current_seq = -1
        self.total_sample_count = 0
        self.total_error_count = 0

        self.sample_count_since_last_detection = 0
        self.error_count_since_last_detection = 0

        self.window_size = avg_error_window_size    # Size of window to compute running average error
        self.window_sample_count = 0    # No. of samples in current windows
        self.errors_in_window = 0

        self.baseline_error_info = []   # List of ErrorInfo

    def __repr__(self):
        return "ResultsManager()"

    def init_baseline(self, baseline_name, baseline_num):
        assert baseline_num == len(self.baseline_error_info)    # Baselines can only be added in order 0, 1 ...
        self.baseline_error_info.append(ErrorInfo(baseline_name))

    def add_prediction_result(self, seq_num, data_point_batch):

        self.current_seq = seq_num
        batch_start_seq = seq_num - len(data_point_batch) + 1

        for index, data_point in enumerate(data_point_batch):

            seq = batch_start_seq + index

            self.total_sample_count += 1
            self.window_sample_count += 1
            self.sample_count_since_last_detection += 1

            if (data_point.predicted_y != data_point.true_y):
                self.errors_in_window += 1
                self.total_error_count += 1
                self.error_count_since_last_detection += 1


            if (self.window_sample_count == self.window_size):
                self.window_error_count_seq.append((seq, self.errors_in_window))

                avg_error = self.errors_in_window / self.window_size
                self.window_avg_error_seq.append((seq, avg_error))

                self.window_sample_count = 0
                self.errors_in_window = 0


    def add_baseline_prediction_result(self, baseline_num, seq_num, data_point_batch):

        batch_start_seq = seq_num - len(data_point_batch) + 1
        error_info = self.baseline_error_info[baseline_num]

        for index, data_point in enumerate(data_point_batch):

            seq = batch_start_seq + index

            error_info.window_sample_count += 1

            if (data_point.predicted_y != data_point.true_y):
                error_info.errors_in_window += 1
                error_info.total_error_count += 1

            if (error_info.window_sample_count == self.window_size):
                avg_error = error_info.errors_in_window / self.window_size
                error_info.window_avg_error_seq.append((seq, avg_error))

                error_info.window_sample_count = 0
                error_info.errors_in_window = 0


    def add_detection_info(self, seq_num, diff, diff_sum, is_drift_detected):
        self.diff_seq.append((seq_num, diff))
        self.diff_sum_seq.append((seq_num, diff_sum))

        if (is_drift_detected == True):
            self.detection_points_seq.append(seq_num)
            self.sample_count_since_last_detection = 0
            self.error_count_since_last_detection = 0


    def add_special_marker(self, seq_num, marker_name):
        self.special_marker_seq.append((seq_num, marker_name))

    def add_adaptation_info(self, seq_num):
        assert False


    def print_results(self):
        print("\n-----------------------------------------------------------------------\n")

        last_window_error_count = 'NA'
        last_window_avg_error = 'NA'

        if (len(self.window_error_count_seq) > 0):
            last_window_error_count = self.window_error_count_seq[-1][1]
            last_window_avg_error = self.window_avg_error_seq[-1][1]

        avg_error_since_last_detection = 'NA'
        if (self.sample_count_since_last_detection > 0):
            avg_error_since_last_detection = self.error_count_since_last_detection / self.sample_count_since_last_detection

        total_avg_error = 0
        if (self.total_sample_count > 0):
         total_avg_error = self.total_error_count / self.total_sample_count

        print("current_seq={}, last_window_error_count={}, error_count_since_last_detection={}, total_error_count={}".
                format(self.current_seq, last_window_error_count, self.error_count_since_last_detection, self.total_error_count))


        print("current_seq={}, last_window_avg_error={}, avg_error_since_last_detection={}, total_avg_error={}".
              format(self.current_seq, last_window_avg_error, avg_error_since_last_detection, total_avg_error))

        # print("window_error_count_seq={}".format(self.window_error_count_seq))
        # print("window_avg_error_seq={}".format(self.window_avg_error_seq))
        print("detection_points_seq={}".format(self.detection_points_seq))

        for error_info in self.baseline_error_info:
            total_baseline_avg_error = error_info.total_error_count / self.total_sample_count
            print("baseline={}, total_error_count={}, total_avg_error={}".
                  format(error_info.baseline_name, error_info.total_error_count, total_baseline_avg_error))


    def plot_results(self):

        # Plot 1

        plt.figure(1)

        # Concept drift map (total variation drift)
        x = [pair[0] for pair in self.diff_seq]
        y = [pair[1] for pair in self.diff_seq]
        plt.plot(x, y, label='total_variation_distance')

        # # Smoothed diff seq
        # N = 7
        # x = [pair[0] for pair in self.diff_seq]
        # x = x[N-1:]
        # y = [pair[1] for pair in self.diff_seq]
        # y = running_mean(y, N)
        # plt.plot(x, y, label='smoothed_diff_seq')

        # # Diff sum seq
        # x = [pair[0] for pair in self.diff_sum_seq]
        # y = [pair[1] for pair in self.diff_sum_seq]
        # plt.plot(x, y, label='diff_sum_seq')

        # Drift detection points vertical lines
        for x in self.detection_points_seq:
            plt.axvline(x, label='drift_detected', color='k', linestyle='--', linewidth=0.5)

        # Special marker vertical lines
        for pair in self.special_marker_seq:
            plt.axvline(pair[0], color='r', linewidth=2, label=pair[1])

        combine_duplicate_legends()
        plt.title("Concept drift map: " + self.title_suffix)
        plt.xlabel("no_of_samples")
        plt.ylabel("difference_measure")


        # Plot 2

        plt.figure(2)

        # Online error rate of our adaptation algorithm
        x = [pair[0] for pair in self.window_avg_error_seq]
        y = [pair[1] for pair in self.window_avg_error_seq]
        plt.plot(x, y, label='with_adaptation')

        # Plots for online error rate of baselines
        for error_info in self.baseline_error_info:
            if (len(error_info.window_avg_error_seq) > 0):
                x = [pair[0] for pair in error_info.window_avg_error_seq]
                y = [pair[1] for pair in error_info.window_avg_error_seq]
                plt.plot(x, y, label=error_info.baseline_name, linestyle='--')

        # Average error horizontal line
        # if (self.total_sample_count > 0):
        #     total_avg_error = self.total_error_count / self.total_sample_count
        #     plt.axhline(total_avg_error, label='total_avg_error', color='k', linestyle='--', linewidth=0.5)

        # Drift detection points vertical lines
        for x in self.detection_points_seq:
            plt.axvline(x, label='drift_detected', color='k', linestyle='--', linewidth=0.5)

        # Special marker vertical lines
        for pair in self.special_marker_seq:
            plt.axvline(pair[0], color='#550000ff', linewidth=2, label=pair[1])

        combine_duplicate_legends()
        plt.title("Error rate: " + self.title_suffix)
        plt.xlabel("no_of_samples")
        plt.ylabel("error_rate")

        plt.show()

