"""

Create a 1-D pdf that is a mixure of 3 Gaussians, generate some samples and plot the pdf and sample histogram
Visually verify that the pdf and histogram are correct

"""

import time_varying_pdf as tvp
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib import animation


############

#fig = plt.figure()

#print("Creating GMM RV pdf and plotting")		

#componentParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
#gmmRV = tvp.GaussianMixtureModel(componentParamList)

##newParamList = [(1/3, 4, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
##gmmRV.setModelParams(newParamList)

#x = np.arange(-7, 7, .1)
#pdfPlot = plt.plot(x, gmmRV.pdf(x), label='GMM pdf')


#############

#print("Generating samples")

#t0 = time.time()

#numSamples = 1050
#samples = gmmRV.rvs(size=numSamples)

#t1 = time.time()
#timeElapsed = t1-t0

#print("Time taken to generate samples = %.2f secs" % (timeElapsed))


#############



##print("Plotting samples")
##plt.scatter(samples, np.zeros(numSamples))


#print("Plotting histogram")
#histPlot = plt.hist(samples, normed=True, histtype='stepfilled', alpha=0.2, label='Histogram')


#plt.legend(loc='upper right')
#plt.xlabel('x')
#plt.show()

###########


#print("Done")


##########

# time varying

fig = plt.figure()

ax = plt.axes(xlim=(-10, 10), ylim=(0, 0.5))
curve, = ax.plot([], [], lw=2)

x = np.linspace(-10, 10, 1000)


componentParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
tvGMM = tvp.TimeVaryingGMM(componentParamList)


def updateAndGetCurrentCurve(t):
	print("\t t=%d" % t)
	tvGMM.updateNextTimestep()
	print(tvGMM)

	y = tvGMM.getCurrentPDFCurve(x)
	
	curve.set_data(x, y)

	return curve,

anim = animation.FuncAnimation(fig=fig, func=updateAndGetCurrentCurve, frames=100, interval=100, blit=True)

plt.show()