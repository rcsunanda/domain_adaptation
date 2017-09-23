"""
ResultsManager class
"""

import matplotlib.pyplot as plt



###################################################################################################
"""
ResultsManager collects results-related information as the data sequence and algorithms progress
It can then be called to print/ plot the collected results
"""

class ResultsManager:
    def __init__(self, avg_error_window_size):
        self.window_error_count_seq = []
        self.window_avg_error_seq = []
        self.detection_points_seq = []  # Sequence numbers when a drift was detected (is a 1/0 seq more suitable for this)
        self.diff_seq = []
        self.diff_sum_seq = []

        self.current_seq = -1
        self.total_sample_count = 0
        self.total_error_count = 0

        self.sample_count_since_last_detection = 0
        self.error_count_since_last_detection = 0

        self.window_size = avg_error_window_size    # Size of window to compute running average error
        self.window_sample_count = 0    # No. of samples in current windows
        self.errors_in_window = 0

    def __repr__(self):
        return "ResultsManager()"


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



    def add_detection_info(self, seq_num, diff, diff_sum, is_drift_detected):
        self.diff_seq.append((seq_num, diff))
        self.diff_sum_seq.append((seq_num, diff_sum))

        if (is_drift_detected == True):
            self.detection_points_seq.append(seq_num)
            self.sample_count_since_last_detection = 0
            self.error_count_since_last_detection = 0


    def add_adaptation_info(self, seq_num):
        assert False


    def print_results(self):
        print("\n-----------------------------------------------------------------------\n")

        last_window_error_count = 'NA'
        last_window_avg_error = 'NA'

        if (len(self.window_error_count_seq) > 0):
            last_window_error_count = self.window_error_count_seq[-1]
            last_window_avg_error = self.window_avg_error_seq[-1][1]*100

        avg_error_since_last_detection = 'NA'
        if (self.sample_count_since_last_detection > 0):
            avg_error_since_last_detection = self.error_count_since_last_detection*100 / self.sample_count_since_last_detection

        total_avg_error = 0
        if (self.total_sample_count > 0):
         total_avg_error = self.total_error_count*100 / self.total_sample_count

        print("current_seq={}, last_window_error_count={}, error_count_since_last_detection={}, total_error_count={}".
                format(self.current_seq, last_window_error_count, self.error_count_since_last_detection, self.total_error_count))


        print("current_seq={}, last_window_avg_error={}%, avg_error_since_last_detection={}%, total_avg_error={}%".
              format(self.current_seq, last_window_avg_error, avg_error_since_last_detection, total_avg_error))

        # print("window_error_count_seq={}".format(self.window_error_count_seq))
        # print("window_avg_error_seq={}".format(self.window_avg_error_seq))
        print("detection_points_seq={}".format(self.detection_points_seq))


    def plot_results(self):
        plt.figure(1)

        x = [pair[0] for pair in self.diff_seq]
        y = [pair[1] for pair in self.diff_seq]
        plt.plot(x, y, label='diff_seq')

        x = [pair[0] for pair in self.diff_sum_seq]
        y = [pair[1] for pair in self.diff_sum_seq]
        plt.plot(x, y, label='diff_sum_seq')

        for x in self.detection_points_seq:
            plt.axvline(x, color='c', linestyle='--', linewidth=0.5)

        plt.legend(loc='upper right')

        plt.figure(2)

        x = [pair[0] for pair in self.window_avg_error_seq]
        y = [pair[1] for pair in self.window_avg_error_seq]
        plt.plot(x, y, label='window_avg_error_seq')

        total_avg_error = 0
        if (self.total_sample_count > 0):
            total_avg_error = self.total_error_count / self.total_sample_count

        plt.axhline(total_avg_error, label='total_avg_error', color='k', linestyle='--', linewidth=0.5)

        for x in self.detection_points_seq:
            plt.axvline(x, color='c', linestyle='--', linewidth=0.5)

        plt.legend(loc='upper right')
        plt.show()

