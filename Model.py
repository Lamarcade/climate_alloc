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
from scipy.optimize import minimize

# Scenar : index 1 to k
# mus[scenar,t] = mu_{Delta_t},t
#(d_{i,t})_i = intensities
# (d_{i,1},..., d_{i,T})_i = full_intensities
# previous_intensity = d_t (mean of previous intensities)
# pi : Initial law
# probas : Filter probabilities
# full_intensities 

class MC():  
    
#%% Initialization
    def __init__(self, T0, Time, initial_law = [0.25, 0.25, 0.25, 0.25], data_file = "Data/Stoxx_data.xlsx"):
        self.rng = np.random.default_rng()
        self.T0 = T0
        self.Time = Time 
        self.pi = initial_law
        self.probas = self.pi
        self.history_count = 0
        self.history_marginal = np.ones(Time)
        self.history_pi = initial_law * np.ones((Time, len(self.pi)))
        
        # Carbon and revenue data per company
        df = pd.read_excel(data_file)
        df.drop(df.columns[0], axis = 1)
        self.df = df
        
        indicators = df[["Instrument","GICS Sector Name"]].copy()
        for i in range(13, -1, -1):
            # Actual year / Former year from Y-13 to Y-0
            indicators["Rate Y-{i}".format(i = i)] = 100 * df["Total Y-{i}".format(i = i)] / df["Total Y-{j}".format(j = i+1)]
            
        #for i in range(13, -1, -1):
        #    # Total emissions / Revenue 
        #    indicators["Intensity Y-{i}".format(i = i)] = df["Total Y-{i}".format(i = i)] / df["Revenue Y-{i}".format(i = i)]
        self.indicators = indicators
        
        self.mus = np.zeros((len(initial_law),Time)) # here compute the historical mu
        
    def initialize_parameters(self, central_var, beta, nus, sigmas):
        self.theta = (central_var, beta, nus, sigmas)
        # 0 : Central var
        # 1 : Beta
        # 2 : Nus
        # 3 : Sigmas


#%% Evaluation functions    
    def intensity(self, eps, eta, relative = True):
        if relative:
            return eps * (1 + eta)
        else:
            return eps + eta
            
    def simulate_macro(self,previous, intensities, scenar, loc, scale, g):
        # Evolution of macroeconomics
        chi = self.rng.normal(loc,scale)
        return(g(previous, intensities, scenar, chi))
    
    def toy_g(previous, intensities, scenar, chi):
        return previous + intensities + chi
    
    def toy_f(previous):
        return 1/previous
    
    #def one_density(self, intensities, previous_intensity, scenar, beta, nus, sigmas, central_var):
        # f_t((d_i)i | Delta = j, F_{t-1})
       # cov = central_var + beta * (previous_intensity - self.mus[scenar])**2
        # matrix = cov * np.ones((len(sigmas), len(sigmas)))
       # matrix += np.diag(np.array([cov]) + sigmas)
       # mn = multivariate_normal(self.mus[scenar] + nus, matrix)
        # return(mn.pdf(intensities))
    
    def one_density(self, theta, intensities, previous_intensity, scenar, t):
        # f_t((d_i)i | Delta = j, F_{t-1})
        cov = theta[0] + theta[1] * (previous_intensity - self.mus[scenar, t])**2
        matrix = cov * np.ones((len(theta[3]), len(theta[3])))
        matrix += np.diag(np.array([cov]) + theta[3])
        mn = multivariate_normal(self.mus[scenar] + theta[2], matrix)
        return(mn.pdf(intensities))
 
    def explicit_density(self, theta, intensities, previous_intensity, scenar, t, is_log = False):
        
        cov = theta[0] + theta[1] * (previous_intensity - self.mus[scenar, t])**2
        n = len(theta[3])
        
        # cov_inverse
        diago = np.diag( np.array([cov]) / theta[3])
        #denom = (1 + np.ones(n).dot(diago).dot(np.ones(n)))
        # Simplified formula
        denom = 1 + cov * np.sum(1/theta[3])
            
        inverse = 1/ (cov) * (diago - diago.dot(np.ones((n, n))).dot(diago) / denom)
    
        # determinant
        # cov**n * det(D_{j,t}) * denom
        det = np.prod(theta[3]) * denom
        
        # Computing the density
        coeff = 1/ np.srqt( (2* np.pi) **n * det) 
        
        # mu is a single value
        vector = (intensities - (self.mus[scenar, t] + theta[2]))
        inside = -1/2 * vector.dot(inverse).dot(vector)
        if is_log:
            return(np.log(coeff) + inside)
        else:
            return(coeff * np.exp(inside))
 
# intensities = self.df.loc[:,self.history_count + 2]
# previous_intensity = self.df.loc[:,self.history_count + 1].mean(axis = 0)
# If history count is 0 need to add historical data
 
    def full_density(self, theta, intensities, previous_intensity, t):
        # \bold(f)_{t|t-1}
        density = np.zeros(len(self.mus))
        for j in range(len(self.mus)):
            # Scenario j
            density[j] = self.one_density(theta, intensities, previous_intensity, j, t)
        self.density = density
        return(self.density_val)

#%% Filter
    
    def filter_step(self, intensities, previous_intensity):
        # 2.12
        density_val = self.full_density(self.theta, intensities, previous_intensity, self.history_count + 2)
        num = np.multiply(self.pi, density_val)
        marginal = np.ones(self.pi.shape).dot(num)
        self.history_pi[self.history_count] = self.probas
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal
        self.probas = num/marginal
 
        
#%% EM functions
    def hist_log_lk(self):
        # Log likelihood with historical parameters
        return np.sum(np.log(self.history_marginal)) # first term is 0
    
    def q1(self, full_intensities):
        q1 = 0
        for t in range(full_intensities.shape[1] -2):
            # Parameters theta[0:3] are used in the density here
            
            # At t = 0 need to use historical
            q1 += - np.dot(np.log(self.full_density(full_intensities[:,t+2], full_intensities[:,t+1].mean(axis = 0)), t), self.pi)
        return q1
    
    def q2(self):
        return np.sum(np.log(self.probas), self.pi)
    
    def log_lk(self, full_intensities):
        return(self.q1(full_intensities) + self.q2())
        
    def M_step(self, full_intensities):
        result = minimize(self.q1, self.theta, args = (full_intensities), method = 'SLSQP')
        if result.success:
            self.theta = result.x
        
    def EM(self, full_intensities, n_iter):
        for l in range(n_iter):
            # E step
            self.history_count = 0
            for t in range(full_intensities.shape[1] - 2):
                self.filter_step(full_intensities[:,t+2], full_intensities[:,t+1].mean(axis = 0))
            
            # M step
            self.M_step(full_intensities)
    
    
    