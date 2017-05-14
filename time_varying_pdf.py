"""

Create a 1-D pdf that is a mixure of Gaussians, parameterized by the number of Gaussians (N), and mean and variance of each Gaussian
Make this pdf time varying by making the above parameters time varying (keep N constant) in a deterministic fashion

Plot the time varying pdf

"""

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import time
import math

np.random.seed()

class mixureDensity(st.rv_continuous):

	# Gaussian components and weights of the mixure model
	
	m_RV1 = st.norm(loc = 0, scale = 0.5)
	w1 = 1/3
	
	m_RV2 = st.norm(loc = -3, scale = 1.0)
	w2 = 1/3
	
	m_RV3 = st.norm(loc = 3, scale = 1.0)
	w3 = 1/3
		
	def _pdf(self, x):
		C = mixureDensity	# Alias class name
		return C.w1 * C.m_RV1.pdf(x) + C.w2 * C.m_RV2.pdf(x) + C.w3 * C.m_RV3.pdf(x)
	
	# Implement the _rvs() function as the default sampling method is very slow
	# http://stackoverflow.com/questions/42552117/subclassing-of-scipy-stats-rv-continuous
	def _rvs(self):
		C = mixureDensity	# Alias class name
		
		assert len(self._size) == 1	# Currently we support only a simple "no. of samples" for rvs(size)
		sampleCount = self._size[0];
		
		# Sample from components in batches
		
		numBatches = 100
		numBatches = min(numBatches, sampleCount)
		batchSize = math.floor(sampleCount/numBatches)
		
		returnSamples = []	# Empty list
		
		print("mixureDensity::_rvs; numBatches=%d; sampleCount=%d; batchSize=%d" % (numBatches, sampleCount, batchSize))
		
		for i in range(numBatches):
			# Select a component according to the distribution given by weights
			component = np.random.choice(np.arange(1, 4), p=[C.w1, C.w2, C.w3])
			
			#print("\tmixureDensity::_rvs; iter=%d component=%d" % (i, component))
			
			# Sample from the Gaussian of the selected component
			
			if component == 1:
				returnSamples.extend(C.m_RV1.rvs(size=batchSize))
			elif component == 2:
				returnSamples.extend(C.m_RV2.rvs(size=batchSize))
			elif component == 3:
				returnSamples.extend(C.m_RV3.rvs(size=batchSize))
			else:
				assert False
		
		# Check if required sample count was generated
		# If not, take remaining count from first component
		currSampleCount = len(returnSamples)
		if currSampleCount < sampleCount:
			remainingSampleCount = sampleCount - currSampleCount
			print("mixureDensity::_rvs; Some more to be sampled; currSampleCount=%d; remainingSampleCount=%d" % (currSampleCount, remainingSampleCount))
			returnSamples.extend(C.m_RV1.rvs(size=remainingSampleCount))
		
		print("mixureDensity::_rvs; totalSampleCount=%d" % (len(returnSamples)))
		#print(returnSamples)
		
		return returnSamples
		

		
		
############

print("Starting")		

myRV = mixureDensity(name='mixureDensity')


############

print("Plotting pdf")

x = np.arange(-7, 7, .1)
pdfPlot = plt.plot(x, myRV.pdf(x), label='pdf')


############

print("Generating samples")

t0 = time.time()

numSamples = 1050
samples = myRV.rvs(size=numSamples)	# This call is very slow compared to sampling from a scipy.stats Gaussian
									# may have to override _rvs according to https://stats.stackexchange.com/questions/226834/sampling-from-a-mixture-of-two-gamma-distributions/226837#226837
#samples = mixureDensity.m_RV1.rvs(size=numSamples)

t1 = time.time()
timeElapsed = t1-t0

print("Time taken to generate samples = %.2f secs" % (timeElapsed))

############

#print("Plotting samples")
#plt.scatter(samples, np.zeros(numSamples))


############

print("Plotting histogram")
histPlot = plt.hist(samples, normed=True, histtype='stepfilled', alpha=0.2, label='histogram')


############

plt.legend(loc='upper right')
plt.xlabel('x')
plt.show()

##########


print("Done")