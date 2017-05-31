"""
GaussianMixtureModel and TimeVaryingGMM classes
"""

import scipy.stats as st
import scipy.integrate as integrate
import numpy as np
import math

np.random.seed()



###################################################################################################
"""
1-D GMM parametrized by a set of Gaussian components, each having (weight, mean, variance)
"""

class GaussianMixtureModel(st.rv_continuous):

    """
    Internal class to hold the parameters and RV of each Gaussian component
    """
    class Component:
        def __init__(self, weight, mean, std):
            self.weight = weight
            self.mean = mean
            self.std = std

            self.RV = st.norm(loc=mean, scale=std)

        def set_params(self, weight, mean, std):
            self.weight = weight
            self.mean = mean
            self.std = std

            # For now create new Gaussian, but this should be changed to proper setters
            self.RV = st.norm(loc=mean, scale=std)

        def __repr__(self):
            return "Component(weight=%.2f, mean=%.2f, std=%.2f)" % (self.weight, self.mean, self.std)


    # component_param_list must be a list of tuples with 3 elements (weight, mean, std) - std is standard deviation
    def __init__(self, component_param_list):
        st.rv_continuous.__init__(self, name='GaussianMixtureModel')

        Comp = GaussianMixtureModel.Component
        self.component_list = [Comp(t[0], t[1], t[2]) for t in component_param_list]
        print(self.component_list)


    def __repr__(self):
        return "GMM --> %s" % self.component_list


    def set_model_params(self, component_param_list):
        for i in range(len(self.component_list)):
            params = component_param_list[i]
            self.component_list[i].set_params(params[0], params[1], params[2])

        # print(self.component_list)


    def _pdf(self, x):
        pdf_val = 0

        for comp in self.component_list:
            pdf_val += comp.weight * comp.RV.pdf(x)

        return pdf_val


    # Implement the _rvs() function as the default sampling method is very slow
    # http://stackoverflow.com/questions/42552117/subclassing-of-scipy-stats-rv-continuous
    def _rvs(self):
        assert len(self._size) == 1  # Currently we support only a simple "no. of samples" for rvs(size)
        sample_count = self._size[0];

        # Sample from components in batches

        num_batches = 100
        num_batches = min(num_batches, sample_count)
        batch_size = math.floor(sample_count / num_batches)

        return_samples_list = []

        print("GaussianMixtureModel::_rvs; num_batches={}; sample_count={}; batch_size={}"
                .format(num_batches, sample_count, batch_size))

        for i in range(num_batches):
            # Select a component according to the distribution given by weights
            component_index = np.random.choice(np.arange(0, len(self.component_list)),
                                            p=[comp.weight for comp in self.component_list])

            # print("\tGaussianMixtureModel::_rvs; iter=%d component_index=%d" % (i, component_index))

            # Sample from the Gaussian of the selected component

            component = self.component_list[component_index]

            return_samples_list.extend(component.RV.rvs(size=batch_size))

        # Check if required sample count was generated
        # If not, take remaining count from first component
        curr_sample_count = len(return_samples_list)
        if curr_sample_count < sample_count:
            remaining_sample_count = sample_count - curr_sample_count
            print("GaussianMixtureModel::_rvs; Some more to be sampled; curr_sample_count={}; remaining_sample_count={}"
                  .format(curr_sample_count, remaining_sample_count))
            return_samples_list.extend(self.component_list[0].RV.rvs(size=remaining_sample_count))

        print("GaussianMixtureModel::_rvs; totalSampleCount={}".format(len(return_samples_list)))
        # print(return_samples_list)

        return return_samples_list



###################################################################################################
"""
1-D GMM that can be updated given a time parameter
Updating changes the mean and variance of each component following a sinusoidal, y(t) = A * sin(theta * t)
"""

class TimeVaryingGMM:

    """
    Internal class to hold the parameters of the sinusoidal for each Gaussian component
    """
    class ComponentTimeParams:
        def __init__(self, mean_amp, mean_theta, std_amp, std_theta):
            self.mean_amp = mean_amp
            self.mean_theta = mean_theta
            self.std_amp = std_amp
            self.std_theta = std_theta

        def get_params(self, t):
            mean_t = self.mean_amp * np.sin(self.mean_theta * t)
            std_t = self.std_amp * np.sin(self.std_theta * t)
            std_t = abs(std_t)
            return (mean_t, std_t)

        def __repr__(self):
            return "ComponentTimeParams --> mean_amp=%.2f, mean_theta=%.2f, std_amp=%.2f, std_theta=%.2f" % (
                        self.mean_amp, self.mean_theta, self.std_amp, self.std_theta)


    # comp_weight_list must be a list of weights of each Gaussian component
    # initial_time = t0, the time point to which the GMM is initialized (mean and std)
    # component_time_param_list must be a list of tuples with 4 elements (mean_amp, mean_theta, std_amp, std_theta),
    # where each tuple in the list gives time variation parameters of a Gaussian component
    def __init__(self, initial_time, comp_weight_list, component_time_param_list):
        assert (len(comp_weight_list) == len(component_time_param_list))

        # Create list of ComponentTimeParams
        Comp = TimeVaryingGMM.ComponentTimeParams
        self.component_time_param_list = [Comp(e[0], e[1], e[2], e[3]) for e in component_time_param_list]

        # Setup GMM by computing mean, std at initial_time
        initial_gmm_comp_param_list = []
        for weight, comp_time_params in zip(comp_weight_list, self.component_time_param_list):
            init_mean, init_std = comp_time_params.get_params(initial_time)
            initial_gmm_comp_param_list.append((weight, init_mean, init_std))

        self.gmm = GaussianMixtureModel(initial_gmm_comp_param_list)

        print(self.component_time_param_list)


    def __repr__(self):
        return "TimeVaryingGMM --> %s" % self.gmm


    def update_model(self, t):
        component_param_list = []

        for i, time_prams in enumerate(self.component_time_param_list):
            mean_t, std_t = time_prams.get_params(t)
            weight = self.gmm.component_list[i].weight

            component_param_list.append((weight, mean_t, std_t))

        self.gmm.set_model_params(component_param_list)


    def get_current_pdf_curve(self, x):
        return self.gmm.pdf(x)



###################################################################################################
"""
Compute Kullback–Leibler divergence between two RVs
"""
def kl_divergence(rv_p, rv_q):
    def func(x):
        nonlocal rv_p, rv_q
        px = rv_p.pdf(x)
        qx = rv_q.pdf(x)

        return px * np.log(px / qx)

    result, error = integrate.quad(func, -10, 10)

    return result
