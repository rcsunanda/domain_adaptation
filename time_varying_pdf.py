"""

Create a 1-D pdf that is a mixure of Gaussians, parameterized by the number of Gaussians (N), and mean and variance of each Gaussian
Make this pdf time varying by making the above parameters time varying (keep N constant) in a deterministic fashion

Plot the time varying pdf

"""

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class mixureDensity(st.rv_continuous):
	m_RV1 = st.norm(loc = 0, scale = 0.5)
	m_RV2 = st.norm(loc = -3, scale = 1.0)
	m_RV3 = st.norm(loc = 3, scale = 1.0)
	weight = 1/3
		
	def _pdf(self,x):
		return (mixureDensity.m_RV1.pdf(x) + mixureDensity.m_RV2.pdf(x) + mixureDensity.m_RV3.pdf(x)) / 3

		
############

print("Starting")		

myRV = mixureDensity(name='mixureDensity')


############

print("Plotting pdf")

x = np.arange(-7, 7, .1)
pdfPlot = plt.plot(x, myRV.pdf(x), label='pdf')


############

print("Generating samples")

numSamples = 10
samples = myRV.rvs(size=numSamples)	# This call is very slow
									# may have to override _rvs according to https://stats.stackexchange.com/questions/226834/sampling-from-a-mixture-of-two-gamma-distributions/226837#226837
#samples = mixureDensity.m_RV.rvs(size=numSamples)


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