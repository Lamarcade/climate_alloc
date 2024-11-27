# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:43:21 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd
from collections import defaultdict
from scipy.stats import multivariate_normal

# Scenar : index 1 to k
# mus[scenar] = mu_{Delta_t}
#(di)i = intensities


class MC():   
    def __init__(self, initial_law, mus, T):
        self.rng = np.random.default_rng()
        self.pi = initial_law
        self.mus = mus
        self.history_count = 0
        self.history_marginal = np.ones(T)
        self.history_pi = initial_law * np.ones((T, len(self.pi)))
    
    def intensity(self, eps, eta, relative = True):
        if relative:
            return eps * (1 + eta)
        else:
            return eps + eta
            
    def simulate_macro(self,previous, intensities, scenar, loc, scale, g):
        chi = self.rng.normal(loc,scale)
        return(g(previous, intensities, scenar, chi))
    
    def toy_g(previous, intensities, scenar, chi):
        return previous + intensities + chi
    
    def toy_f(previous):
        return 1/previous
    
    def one_density(self, intensities, previous_intensity, scenar, beta, nus, sigmas, central_var):
        cov = central_var + beta * (previous_intensity - self.mus[scenar])**2
        matrix = cov * np.ones((len(sigmas), len(sigmas)))
        matrix += np.diag(np.array([cov]) + sigmas)
        mn = multivariate_normal(self.mus[scenar] + nus, matrix)
        return(mn.pdf(intensities))
    
    def full_density(self, intensities, previous_intensity, beta, nus, sigmas, central_var):
        density = np.zeros(len(self.mus))
        for j in range(len(self.mus)):
            # Scenario j
            density[j] = self.one_density(intensities, j, beta, nus, sigmas, central_var)
        self.density = density
        return(self.density_val)
    
    def update_law(self, intensities, previous_intensity, beta, nus, sigmas, central_var):

        density_val = self.full_density(intensities, previous_intensity, beta, nus, sigmas, central_var)
        num = np.multiply(self.pi, density_val)
        marginal = np.ones(self.pi.shape).dot(num)
        self.history_pi[self.history_count] = self.pi
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal
        self.pi = num/marginal
        
    def log_lk(self):
        return np.sum(np.log(self.history_marginal)) #first term is 0
    
        
    
    