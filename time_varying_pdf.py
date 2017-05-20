"""

Create a 1-D pdf that is a mixure of Gaussians, parameterized by the number of Gaussians (N), and mean and variance of each Gaussian
Make this pdf time varying by making the above parameters time varying (keep N constant) in a deterministic fashion

Plot the time varying pdf

"""

import scipy.stats as st
import numpy as np
import math

np.random.seed()

class GaussianMixtureModel(st.rv_continuous):

	# Internal class to hold the parameters and RV of each Gaussian component
	class Component:
		def __init__(self, weight, mean, std):
			self.weight = weight
			self.mean = mean
			self.std = std
			
			self.RV = st.norm(loc=mean, scale=std)

		def setParams(self, weight, mean, std):
			self.weight = weight
			self.mean = mean
			self.std = std

			self.RV = st.norm(loc=mean, scale=std)	# for now create new Gaussian, but this should be changed to proper setters

		def __repr__(self):
			return "Component(weight=%.2f, mean=%.2f, std=%.2f)" % (self.weight, self.mean, self.std)
	
	
	# componentParamList must be a list of tuples with 3 elements (weight, mean, std) - std is standard deviation
	def __init__(self, componentParamList):
		st.rv_continuous.__init__(self, name='GaussianMixtureModel')
		
		Comp = GaussianMixtureModel.Component
		self.componentList = [Comp(t[0], t[1], t[2]) for t in componentParamList]
		print(self.componentList)

	def __repr__(self):
		return "GMM --> %s" % self.componentList


	def setModelParams(self, componentParamList):
		for i in range(len(self.componentList)):
			params = componentParamList[i]
			self.componentList[i].setParams(params[0], params[1], params[2])
	
		#print(self.componentList)
	
	def _pdf(self, x):
		pdfVal = 0
		
		for comp in self.componentList:
			pdfVal += comp.weight * comp.RV.pdf(x)
		
		return pdfVal
	
	
	# Implement the _rvs() function as the default sampling method is very slow
	# http://stackoverflow.com/questions/42552117/subclassing-of-scipy-stats-rv-continuous
	def _rvs(self):
		#C = GaussianMixtureModel	# Alias class name
		
		assert len(self._size) == 1	# Currently we support only a simple "no. of samples" for rvs(size)
		sampleCount = self._size[0];
		
		# Sample from components in batches
		
		numBatches = 100
		numBatches = min(numBatches, sampleCount)
		batchSize = math.floor(sampleCount/numBatches)
		
		returnSamples = []	# Empty list
		
		print("GaussianMixtureModel::_rvs; numBatches=%d; sampleCount=%d; batchSize=%d" % (numBatches, sampleCount, batchSize))
		
		for i in range(numBatches):
			# Select a component according to the distribution given by weights
			componentNum = np.random.choice(np.arange(0, len(self.componentList)), p=[comp.weight for comp in self.componentList])
			
			#print("\tGaussianMixtureModel::_rvs; iter=%d componentNum=%d" % (i, componentNum))
			
			# Sample from the Gaussian of the selected component
			
			component = self.componentList[componentNum]
			
			returnSamples.extend(component.RV.rvs(size=batchSize))
			
		# Check if required sample count was generated
		# If not, take remaining count from first component
		currSampleCount = len(returnSamples)
		if currSampleCount < sampleCount:
			remainingSampleCount = sampleCount - currSampleCount
			print("GaussianMixtureModel::_rvs; Some more to be sampled; currSampleCount=%d; remainingSampleCount=%d" % (currSampleCount, remainingSampleCount))
			returnSamples.extend(self.componentList[0].RV.rvs(size=remainingSampleCount))
		
		print("GaussianMixtureModel::_rvs; totalSampleCount=%d" % (len(returnSamples)))
		#print(returnSamples)
		
		return returnSamples
		

class TimeVaryingGMM:

	class ComponentTimeParams:
		def __init__(self, mean_amp, mean_theta, std_amp, std_theta):
			self.mean_amp = mean_amp
			self.mean_theta = mean_theta
			self.std_amp = std_amp
			self.std_theta = std_theta

		def getParams(self, t):
			mean_t = self.mean_amp * np.sin (self.mean_theta * t)
			std_t = self.std_amp * np.sin (self.std_theta * t)
			std_t = abs(std_t)
			return (mean_t, std_t)

		def __repr__(self):
			return "ComponentTimeParams --> mean_amp=%.2f, mean_theta=%.2f, std_amp=%.2f, std_theta=%.2f" % (self.mean_amp, self.mean_theta, self.std_amp, self.std_theta)
	

	# componentTimeParamList must be a list of tuples with 4 elements (mean_amp, mean_theta, std_amp, std_theta)
	def __init__(self, initialCompParamList, componentTimeParamList):
		assert(len(initialCompParamList) == len(componentTimeParamList))

		Comp = TimeVaryingGMM.ComponentTimeParams
		
		self.gmm = GaussianMixtureModel(initialCompParamList)
		self.componentTimeParamList = [Comp(t[0], t[1], t[2], t[3]) for t in componentTimeParamList]

		print(self.componentTimeParamList)

	
	def __repr__(self):
		return "TimeVaryingGMM --> %s" % self.gmm
	
	
	def updateModel(self, t):
		componentParamList = []
		
		i = 0
		for timePrams in self.componentTimeParamList:
			mean_t, std_t = timePrams.getParams(t)
			weight = self.gmm.componentList[i].weight

			componentParamList.append((weight, mean_t, std_t))
			i += 1

		self.gmm.setModelParams(componentParamList)
	
	
	def getCurrentPDFCurve(self, x):
		return self.gmm.pdf(x)
