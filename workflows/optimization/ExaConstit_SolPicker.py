import numpy as np

class BestSol:

    # Except for ASF, all the below ways use as a utopian point the (0,0) and a minimization strategy

    def __init__(self, pop_fit, weights=None, normalize=False):
        
        if not type(pop_fit).__module__ == np.__name__:
            pop_fit = np.array(pop_fit)

        # If weights are not given
        if weights == None:
            self.weights = np.array([1] * len(pop_fit[0]))
        else:
            self.weights = np.array(weights)
        
        # If normalize true
        if normalize == True:
            approx_ideal = pop_fit.min(axis=0)
            approx_nadir = pop_fit.max(axis=0)
            self.fit = (pop_fit - approx_ideal)/(approx_nadir - approx_ideal)
        else:
            self.fit = pop_fit


    def ASF(self):

        # ASF Decomposition Method
        # Multiply by weighs the fit_values and then pick the max for each row. Then pick the min
        # This way gives best solution based on the least max error in the solution's obj functions for the population
        asf = ((self.fit - 0) * self.weights).max(axis=1)
        best_idx = np.argmin(asf)

        return best_idx
        

    def EUDIST(self, p=2):

        # Calcualte Weighted Distance (When p=2 then Euclidean distance from utopian point (0,0) or the origin)
        dist = (np.sum(self.weights * self.fit**p, axis=1))**(1/p)
        best_idx = np.argmin(dist)

        return best_idx