
import time_varying_gmm as tvgmm
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as anim
import copy


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
Test the KL Divergence function
"""

def test_kl_divergence():
	componentParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
	gmm_1 = tvgmm.GaussianMixtureModel(componentParamList)

	kl_div_1 = tvgmm.gmm_kl_divergence(gmm_1, gmm_1)
	print("kl_div_1", end=' = ')
	print(kl_div_1)

	componentParamList = [(1/3, 1, 0.5), (1/3, -2, 1), (1/3, 4, 1)]
	gmm_2 = tvgmm.GaussianMixtureModel(componentParamList)

	kl_div_2 = tvgmm.gmm_kl_divergence(gmm_1, gmm_2)
	print("kl_div_2", end=' = ')
	print(kl_div_2)


"""
Create a time varying 1-D GMM and plot its pdf (animated)
Visually verify that the pdf is varying
"""

def test_time_varying_gmm():
	fig = plt.figure()

	ax = plt.axes(xlim=(-10, 10), ylim=(0, 1))
	curve, = ax.plot([], [], lw=2)

	x = np.linspace(-10, 10, 1000)

	numTimePoints = 100	# 3000 is a good number without the KL divergence call, 100 is good with it
	timePoints = np.linspace(-1, 1, numTimePoints)

	initialCompParamList = [(1/3, 0, 0.5), (1/3, -3, 1), (1/3, 3, 1)]
	componentTimeParamList = [(5, 5, 1, 0.5), (4, 1.5, 0.5, 1), (3, 2, 1.5, 2)]

	tvGMM = tvgmm.TimeVaryingGMM(initialCompParamList, componentTimeParamList)

	initialGMM = tvgmm.TimeVaryingGMM(initialCompParamList, componentTimeParamList)

	initial_y = initialGMM.getCurrentPDFCurve(x)
	plt.plot(x, initial_y, label='Reference pdf')
	
	def updateAndGetCurrentCurve(frame):
		nonlocal timePoints, tvGMM, x, curve

		time = timePoints[frame]

		tvGMM.updateModel(time)

		kl_div = -1
		kl_div = tvgmm.gmm_kl_divergence(tvGMM.gmm, initialGMM.gmm)
		
		print("\t frame=%d, time=%.3f, kl_div=%.3f" % (frame, time, kl_div))

		#print(tvGMM)

		y = tvGMM.getCurrentPDFCurve(x)

	
		curve.set_data(x, y)

		return curve,

	anim_obj = anim.FuncAnimation(fig=fig, func=updateAndGetCurrentCurve, frames=numTimePoints, interval=10, blit=True)

	plt.legend(loc='upper right')
	plt.xlabel('x')
	plt.show()




# Call test functions

#test_gmm();
test_kl_divergence();
test_time_varying_gmm();
