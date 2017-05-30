"""
Misc tests for various Python functionality
"""

import numpy as np
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

# Call functions

# test_differentiation()
