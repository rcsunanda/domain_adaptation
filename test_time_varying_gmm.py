
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as anim



"""
Create a 1-D pdf that is a mixure of 3 Gaussians, generate some samples and plot the pdf and sample histogram
Visually verify that the pdf and histogram are correct
"""

def test_gmm():
	fig = plt.figure()

	print("Creating GMM RV and plotting pdf")		

	componentParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
	gmmRV = tvgmm.GaussianMixtureModel(componentParamList)

	#newParamList = [(1/3, 4, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
	#gmmRV.setModelParams(newParamList)	# To test setModelParams()

	x = np.arange(-7, 7, .1)
	pdfPlot = plt.plot(x, gmmRV.pdf(x), label='GMM pdf')

	############

	print("Generating samples")

	t0 = time.time()

	numSamples = 1050
	samples = gmmRV.rvs(size=numSamples)

	t1 = time.time()
	timeElapsed = t1-t0

	print("Time taken to generate samples = %.2f secs" % (timeElapsed))


	############

	#print("Plotting samples")
	#plt.scatter(samples, np.zeros(numSamples))


	print("Plotting histogram")
	histPlot = plt.hist(samples, normed=True, histtype='stepfilled', alpha=0.2, label='Histogram')


	plt.legend(loc='upper right')
	plt.xlabel('x')
	plt.show()

	print("Done")



"""
Create a time varying 1-D GMM and plot its pdf (animated)
Visually verify that the pdf is varying
"""

def test_time_varying_gmm():
	fig = plt.figure()

	ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))
	curve, = ax.plot([], [], lw=2)

	x = np.linspace(-10, 10, 1000)

	numTimePoints = 3000
	timePoints = np.linspace(-1, 1, numTimePoints)

	initialCompParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
	componentTimeParamList = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

	tvGMM = tvgmm.TimeVaryingGMM(initialCompParamList, componentTimeParamList)

	
	def updateAndGetCurrentCurve(frame):
		nonlocal timePoints, tvGMM, x, curve

		time = timePoints[frame]
		print("\t frame=%d, time=%.3f" % (frame, time))

		tvGMM.updateModel(time)
	
		#print(tvGMM)

		y = tvGMM.getCurrentPDFCurve(x)
	
		curve.set_data(x, y)

		return curve,

	anim_obj = anim.FuncAnimation(fig=fig, func=updateAndGetCurrentCurve, frames=numTimePoints, interval=10, blit=True)

	plt.show()




# Call test functions

#test_gmm();
test_time_varying_gmm();
