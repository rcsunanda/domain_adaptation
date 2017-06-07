"""
Misc tests for various Python functionality
"""

import numpy as np
import math
import matplotlib.pyplot as plt



###################################################################################################
"""
Test to check whether the first order difference is approximately equal to the first order derivative
Result is that they are approximately equal
"""

def test_differentiation():
    x = np.linspace(-5, 5, 10000)
    y = x**3

    theoratical_diff = 3*x**2

    dx = x[1] - x[0]
    manual_difference = []
    for i in range(len(x) - 1):
        diff = (y[i + 1] - y[i]) / dx
        manual_difference.append(diff)
    manual_difference.append(manual_difference[-1])    # Append element equal to one before that

    np_difference = np.diff(y) / dx
    np_difference = list(np_difference)
    np_difference.append(np_difference[-1])    # Append element equal to one before that

    plt.plot(x, y, label="y")
    plt.plot(x, theoratical_diff, linestyle=':', label="theoratical_diff")
    plt.plot(x, manual_difference, linestyle='--', label="manual_difference")
    plt.plot(x, np_difference, linestyle='steps', label="np_difference")

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.show()



###################################################################################################
"""
Try different relationships between memory factor (k), and buffer size (y)
"""

def test_k_vs_buffer_size():
    k = np.linspace(0, 1, 10000)

    y1 = 1000 + 10000*k

    y2 = (200) * np.exp(4*k + 1.2)

    plt.plot(k, y1, label="y1 - linear")
    plt.plot(k, y2, label="y2 - exponential")

    plt.legend(loc='upper right')
    plt.xlabel('k')
    plt.ylabel('buffer_size')
    plt.show()



###################################################################################################
"""
Subplot example
"""

metric_fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

def compute_metrics():
    x = np.linspace(-3, 3, 100)
    y1 = -2*x + 4
    y2 = 3*x
    y3 = x + 3
    y4 = x**2

    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax3.plot(x, y3)
    ax4.plot(x, y4)

    metric_fig.show()



###################################################################################################

# Call functions

# test_differentiation()
test_k_vs_buffer_size()