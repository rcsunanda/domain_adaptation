"""
Tests for DriftDetector
"""

import domain_adaptation.drift_detector as dd
import domain_adaptation.process as prc
import domain_adaptation.data_point as dp

import matplotlib.pyplot as plt


###################################################################################################
"""
Test that given 2 windows of DataPoints, sample windows are output, and also bounds are correctly set
"""

def test_prepare_sample_windows():

    # samples of window_1 = [0,0], [1,1], [2,2], [3,3], [4,4]
    window_1 = [dp.DataPoint([x, x], -1, -1) for x in range(0,5)]

    # samples of window_1 = [1,2], [2,3], [3,4], [4,5], [5,6]
    window_2 = [dp.DataPoint([x+1, x+2], -1, -1) for x in range(0,5)]

    (window_1_samples, window_2_samples, bounds) = dd.prepare_sample_windows(window_1, window_2)

    print(bounds)
    assert (bounds == [(0,5), (0,6)])


###################################################################################################
"""

Generate data points from two classes of a Process as a single sequence and run detection
The boundary between the two classes in the sequence marks abrupt drift
Compare actual drift with expected drift  
"""

def test_abrupt_drift_detection():

    window_size=500
    drift_detector = dd.DriftDetector(window_size)
    print(drift_detector)

    # Generate some data from a 2 class stochastic process

    # Setup process

    gauss_params = []
    # class 1 Gaussian distribution params
    mean_1 = [0, 0]
    cov_1 = [[1, 0], [0, 1]]
    # class 2 Gaussian distribution params
    mean_2 = [4, 4]
    cov_2 = [[1, 0], [0, 1]]

    gauss_params.append((mean_1, cov_1))
    gauss_params.append((mean_2, cov_2))

    process = prc.Process(num_dimensions=2, num_classes=2, class_distribution_parameters=gauss_params)
    print(process)

    # Generate data
    count=2000  # How many samples from one label
    data_points_label_0 = process.generate_data_points(label=0, count=count)
    data_points_label_1 = process.generate_data_points(label=1, count=count)
    data_points = data_points_label_0 + data_points_label_1

    # Emulate a data point sequence
    seq_no = []
    expected_drift_seq = []
    detection_batch_size = 50   # 10 is a good value
    for index, point in enumerate(data_points):
        drift_detector.add_data_points([point])

        if (index % detection_batch_size == 0):   # Run detection after a batch of samples has been added
            drift_detector.run_detection()
            seq_no.append(index)

            # Expected drift
            if (index > count and index < count + 2*window_size):   # Part of the two windows fall on either side of 'count'
                to_right = index - count
                to_left = 2*window_size - to_right
                expected_drift = min(to_left, to_right) / window_size
            else:
                expected_drift = 0

            expected_drift_seq.append(expected_drift)

            print("index={}".format(index))

    # Scale expected drift values to compare with actual drift values
    peak_actual_drift = max(drift_detector.diff_sequence)
    expected_drift_seq = [val * peak_actual_drift for val in expected_drift_seq]

    plt.plot(seq_no, drift_detector.diff_sequence, label='actual_drift')
    plt.plot(seq_no, expected_drift_seq, label='expected_drift')
    plt.legend(loc='upper right')
    plt.ylabel('diff')
    plt.show()



###################################################################################################

# Call test functions

# test_prepare_sample_windows()
test_abrupt_drift_detection()

