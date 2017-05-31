
"""
Tests for GaussianMixtureModel, TimeVaryingGMM, and kl_divergence()
"""

# Workaround for Ctrl-C bug in scipy
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['FOR_IGNORE_EXCEPTIONS'] = '1'


import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt
import time as time_module
import matplotlib.animation as anim



###################################################################################################
"""
Create a 1-D GMM and plot its pdf and histogram of some generated samples
"""

def test_gmm():
    plt.figure()

    print("Creating GMM RV and plotting pdf")

    component_params = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmm_rv = tvgmm.GaussianMixtureModel(component_params)

    # new_params = [(1/3, 4, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
    # gmm_rv.setModelParams(new_params)	# To test setModelParams()

    x = np.arange(-7, 7, .1)
    pdf_plot = plt.plot(x, gmm_rv.pdf(x), label='GMM pdf')

    ############

    print("Generating samples")

    t0 = time_module.time()

    num_samples = 1000000
    samples = gmm_rv.rvs(size=num_samples)

    t1 = time_module.time()
    time_elapsed = t1 - t0

    print("Time taken to generate samples = %.2f secs" % (time_elapsed))

    ############

    # print("Plotting samples")
    # plt.scatter(samples, np.zeros(num_samples))

    print("Plotting histogram")
    bin_width = 0.2
    bins = np.arange(min(samples), max(samples) + bin_width, bin_width)
    plt.hist(samples, normed=True, histtype='stepfilled', bins=bins, alpha=0.2, label='Histogram')

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()

    print("Done")



###################################################################################################
"""
Test the KL Divergence function
"""

def test_kl_divergence():
    component_params = [(1 / 3, 0, 0.5), (1 / 3, -3, 1), (1 / 3, 3, 1)]
    gmm_1 = tvgmm.GaussianMixtureModel(component_params)

    kl_div_1 = tvgmm.kl_divergence(gmm_1, gmm_1)
    print("kl_div_1", end=' = ')
    print(kl_div_1)

    component_params = [(1 / 3, 1, 0.5), (1 / 3, -2, 1), (1 / 3, 4, 1)]
    gmm_2 = tvgmm.GaussianMixtureModel(component_params)

    kl_div_2 = tvgmm.kl_divergence(gmm_1, gmm_2)
    print("kl_div_2", end=' = ')
    print(kl_div_2)



###################################################################################################
"""
Create a time varying 1-D GMM and plot its pdf (animated)
Visually verify that the pdf is varying
"""

def test_time_varying_gmm_animated():
    fig = plt.figure()

    ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))
    curve, = ax.plot([], [], lw=2)

    x = np.linspace(-10, 10, 1000)

    num_time_points = 100  # 3000 is a good number without the KL divergence call, 100 is good with it
    time_points = np.linspace(-1, 1, num_time_points)

    initial_time = time_points[0]
    comp_weight_list = [1/3, 1/3, 1/3]
    component_time_params = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

    tv_gmm = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)

    initial_gmm = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)   # Keep a copy to compute distance metrics

    initial_y = initial_gmm.get_current_pdf_curve(x)
    plt.plot(x, initial_y, label='Reference pdf')

    def update_and_get_current_curve(frame):
        nonlocal time_points, tv_gmm, x, curve

        time = time_points[frame]

        tv_gmm.update_model(time)

        kl_div = -1
        kl_div = tvgmm.kl_divergence(tv_gmm.gmm, initial_gmm.gmm)

        print("\t frame=%d, time=%.3f, kl_div=%.3f" % (frame, time, kl_div))

        # print(tv_gmm)

        y = tv_gmm.get_current_pdf_curve(x)

        curve.set_data(x, y)

        return curve,

    anim_obj = anim.FuncAnimation(fig=fig, func=update_and_get_current_curve, frames=num_time_points, interval=10, blit=True)

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()



###################################################################################################
"""
Create a time varying 1-D GMM and plot its pdf (loop)
Visually verify that the pdf is varying
"""


def test_time_varying_gmm_loop():
    fig = plt.figure()

    ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))

    x = np.linspace(-10, 10, 1000)

    num_time_points = 100  # 3000 is a good number without the KL divergence call, 100 is good with it
    time_points = np.linspace(-1, 1, num_time_points)

    initial_time = time_points[0]
    comp_weight_list = [1 / 3, 1 / 3, 1 / 3]
    component_time_params = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

    tvGMM = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)

    initialGMM = tvgmm.TimeVaryingGMM(initial_time, comp_weight_list, component_time_params)   # Keep a copy to compute distance metrics

    initial_y = initialGMM.get_current_pdf_curve(x)
    plt.plot(x, initial_y, label='Reference pdf')

    plt.ion()

    for frame in range(num_time_points):
        time = time_points[frame]

        tvGMM.update_model(time)

        kl_div = -1
        kl_div = tvgmm.kl_divergence(tvGMM.gmm, initialGMM.gmm)

        print("\t frame=%d, time=%.3f, kl_div=%.3f" % (frame, time, kl_div))

        # print(tvGMM)

        y = tvGMM.get_current_pdf_curve(x)

        pdf_plot, = plt.plot(x, y, color='red', label='Current pdf')
        plt.pause(0.05)

        pdf_plot.remove()

    while True:
        plt.pause(0.05)



###################################################################################################

# Call test functions

# test_gmm();
# test_kl_divergence();
# test_time_varying_gmm_animated();
test_time_varying_gmm_loop();
