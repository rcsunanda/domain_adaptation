"""
Domain adaptation technique by maintaining a moving buffer of samples generated by a time varying GMM
"""

import distribution_estimation as est
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt



###################################################################################################
"""
Shift buffer to left (oldest elements are removed), and append given elements to end of buffer 
"""

def update_moving_buffer(moving_buffer, elements_to_add):
    new_list_size = len(elements_to_add)
    moving_buffer[0: len(moving_buffer) - new_list_size] = moving_buffer[new_list_size:]  # shift
    moving_buffer[len(moving_buffer) - new_list_size:] = elements_to_add

    return moving_buffer



###################################################################################################
"""
Test for update_moving_buffer()
"""

def test_update_moving_buffer():
    moving_buffer = np.array(range(0,10))
    print("moving_buffer_initial={}".format(moving_buffer))

    for i in range(1,5):  # do the move 4 times for testing
        new_list_size = 3   # 3 elements to add to moving list
        new_list = list(range(10*i, 10*i + new_list_size))
        moving_buffer = update_moving_buffer(moving_buffer, new_list)
        print("iter={}, moving_buffer={}".format(i, moving_buffer))



###################################################################################################
"""
Create a time varying GMM, generate samples at each time point, and estimate ecdf and pdf using those samples
Plot to visualize the true pdf, generated samples, and estimated ecdf and pdf
"""

def estimate_time_varying_gmm():
    #fig = plt.figure()

    #ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))

    x = np.linspace(-10, 10, 1000)

    num_time_points = 100  # 3000 is a good number without the KL divergence call, 100 is good with it
    time_points = np.linspace(-1, 1, num_time_points)

    initial_time = time_points[0]
    comp_weight_list = [1 / 3, 1 / 3, 1 / 3]
    component_time_params = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

    tv_gmm = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)

    initial_gmm = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)

    initial_y = initial_gmm.get_current_pdf_curve(x)
    plt.plot(x, initial_y, label='Reference pdf')

    plt.ion()

    buffer_size = 5000   # no of initial samples
    moving_buffer = tv_gmm.gmm.rvs(size=buffer_size)

    # Iterate through time points

    for frame in range(num_time_points):

        # Update GMM (time varying)

        time = time_points[frame]
        tv_gmm.update_model(time)

        kl_div = -1
        kl_div = tvgmm.kl_divergence(tv_gmm.gmm, initial_gmm.gmm)

        print("\t frame=%d, time=%.3f, kl_div=%.3f" % (frame, time, kl_div))

        # print(tv_gmm)

        y = tv_gmm.get_current_pdf_curve(x)

        pdf_plot, = plt.plot(x, y, color='red', label='Current pdf')


        # Generate samples and update moving buffer

        print("\t Generate some samples for this frame and add it to the moving buffer")

        num_new_samples = 500
        new_samples = tv_gmm.gmm.rvs(size=num_new_samples)
        moving_buffer = update_moving_buffer(moving_buffer, new_samples)

        bin_width = 0.2
        bins = np.arange(min(moving_buffer), max(moving_buffer) + bin_width, bin_width)
        counts, bins_s, bars = plt.hist(moving_buffer, normed=True, histtype='stepfilled', bins=bins, alpha=0.2, color='orange', label='Moving buffer histogram')


        # Estimate ecdf and pdf

        sorted, ecdf = est.estimate_ecdf(moving_buffer)
        ecdf_plot, = plt.plot(sorted, ecdf, color='purple', label="estimated-ecdf")

        new_x, derivatives = est.estimate_pdf(sorted, ecdf)
        derivatives_plot, = plt.plot(new_x, derivatives, color='green', label="estimated-pdf")

        #######

        plt.legend(loc='upper right')
        plt.xlabel('x')
        plt.pause(0.05)

        input("Press Enter to continue...")

        # Remove plots of current time point
        pdf_plot.remove()
        ecdf_plot.remove()
        derivatives_plot.remove()
        _ = [b.remove() for b in bars]  # remove histogram



###################################################################################################

# Call functions

# test_update_moving_buffer()
estimate_time_varying_gmm()
