# also check the following
# http://stackoverflow.com/questions/11874767/real-time-plotting-in-while-loop-with-matplotlib
# http://stackoverflow.com/questions/24783530/python-realtime-plotting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


fig = plt.figure()

ax = plt.axes(xlim=(0, 2), ylim=(-10, 10))
curve, = ax.plot([], [], lw=2)

a_vals = np.linspace(0, 100, 1000)
b = 0


def getCurrentCurve(t):
	#x = np.linspace(0, 2, 1000)
	#y = np.sin(2 * np.pi * (x - 0.01 * t))
	
	x = np.linspace(0, 2, 1000)
	y = a_vals[t]*x + b

	#print(t)
	#plt.plot(x, y)
	curve.set_data(x, y)
	return curve,



print ("1 -----")
anim = animation.FuncAnimation(fig=fig, func=getCurrentCurve, frames=200, interval=20, blit=True)
print ("2 -----")


plt.show()