import estimate_distribution as est
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt


def update_moving_buffer(moving_buffer, elements_to_add):
    new_list_size = len(elements_to_add)
    moving_buffer[0: len(moving_buffer) - new_list_size] = moving_buffer[new_list_size:]  # shift
    moving_buffer[len(moving_buffer) - new_list_size:] = elements_to_add

    return moving_buffer


def test_update_moving_buffer():
    moving_buffer = np.array(range(0,10))
    print("moving_buffer_initial={}".format(moving_buffer))

    for i in range(1,5):  # do the move 4 times for testing
        new_list_size = 3   # 3 elements to add to moving list
        new_list = list(range(10*i, 10*i + new_list_size))
        moving_buffer = update_moving_buffer(moving_buffer, new_list)
        print("iter={}, moving_buffer={}".format(i, moving_buffer))


def estimate_time_varying_gmm():
    fig = plt.figure()

    ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))

    x = np.linspace(-10, 10, 1000)

    numTimePoints = 100  # 3000 is a good number without the KL divergence call, 100 is good with it
    timePoints = np.linspace(-1, 1, numTimePoints)

    initialCompParamList = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    componentTimeParamList = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

    tvGMM = tvgmm.TimeVaryingGMM(initialCompParamList, componentTimeParamList)

    initialGMM = tvgmm.TimeVaryingGMM(initialCompParamList, componentTimeParamList)

    initial_y = initialGMM.getCurrentPDFCurve(x)
    plt.plot(x, initial_y, label='Reference pdf')

    plt.ion()

    buffer_size = 5000   # no of initial samples
    moving_buffer = tvGMM.gmm.rvs(size=buffer_size)

    for frame in range(numTimePoints):
        time = timePoints[frame]

        tvGMM.updateModel(time)

        kl_div = -1
        kl_div = tvgmm.gmm_kl_divergence(tvGMM.gmm, initialGMM.gmm)

        print("\t frame=%d, time=%.3f, kl_div=%.3f" % (frame, time, kl_div))

        # print(tvGMM)

        y = tvGMM.getCurrentPDFCurve(x)

        pdfPlot, = plt.plot(x, y, color='red', label='Current pdf')

        ####

        print("\t Generate some samples for this frame and add it to the moving buffer")

        num_new_samples = 500
        new_samples = tvGMM.gmm.rvs(size=num_new_samples)
        moving_buffer = update_moving_buffer(moving_buffer, new_samples)

        binwidth = 0.2
        bins = np.arange(min(moving_buffer), max(moving_buffer) + binwidth, binwidth)
        counts, bins_s, bars = plt.hist(moving_buffer, normed=True, histtype='stepfilled', bins=bins, alpha=0.2, label='Moving buffer histogram')

        ####

        plt.legend(loc='upper right')
        plt.xlabel('x')
        plt.pause(0.05)

        input("Press Enter to continue...")

        pdfPlot.remove()
        _ = [b.remove() for b in bars]  # remove histogram

    while True:
        plt.pause(0.05)





# Call functions

estimate_time_varying_gmm()
#test_update_moving_buffer()
