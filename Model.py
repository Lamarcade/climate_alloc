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
# mus[scenar,t] = mu_{Delta_t},t
#(d_{i,t})_i = intensities
# (d_{i,1},..., d_{i,T})_i = full_intensities
# previous_intensity = d_t (mean of previous intensities)
# pi : Initial law
# probas : Filter probabilities
# full_intensities 

class Model():  
    
#%% Initialization
    def __init__(self, Time, initial_law = [0.25, 0.25, 0.25, 0.25], data_file = "Data/Stoxx_data.xlsx"):
        '''
        Initialize an instance of the model computing the filter and performing the EM algorithm

        Parameters
        ----------
        Time : Int
            Number of years for the whole scenarios.
        initial_law : Float list, optional
            Initial probabilities of the scenarios. The default is [0.25, 0.25, 0.25, 0.25].
        data_file : String, optional
            Path to the file with the carbon and revenue data. The default is "Data/Stoxx_data.xlsx".

        Returns
        -------
        None.

        '''
        self.rng = np.random.default_rng()
        self.Time = Time 
        
        # Initial probabilities
        self.pi = initial_law
        
        # Probabilities updated for the filter
        self.probas = self.pi
        
        # Keep track of the current time and all the probabilities
        self.history_count = 0
        self.history_marginal = np.ones(Time)
        self.history_pi = initial_law * np.ones((len(self.pi), Time))
        
        # Carbon and revenue data per company
        df = pd.read_excel(data_file)
        df = df.drop(df.columns[0], axis = 1)
        self.df = df
        
        indicators = df[["Instrument","GICS Sector Name"]].copy()
        for i in range(13, -1, -1):
            # Actual year / Former year from Y-13 to Y-0
            indicators["Rate Y-{i}".format(i = i)] = 100 * df["Total Y-{i}".format(i = i)] / df["Total Y-{j}".format(j = i+1)]
        
        indicators.replace([np.inf, -np.inf], np.nan, inplace = True)
        #for i in range(13, -1, -1):
        #    # Total emissions / Revenue 
        #    indicators["Intensity Y-{i}".format(i = i)] = df["Total Y-{i}".format(i = i)] / df["Revenue Y-{i}".format(i = i)]
        self.indicators = indicators
        
        # Historical central carbon rate
        self.mus = np.zeros((len(initial_law),Time))
        
        mu = 0
        
        # Duration of the historical data
        self.T0 = indicators.shape[1] - 2
        
        for t in range(2, self.T0 + 2):
            mu += indicators.loc[:, indicators.columns[t]].mean(skipna = True)
        mu /= self.T0
        for t in range(self.T0):
            self.mus[:, t] = mu * np.ones(len(initial_law))   
    
        
    def initialize_parameters(self, central_std, beta, nus, sigmas):
        '''
        Initial guess for the EM parameters

        Parameters
        ----------
        central_std : Float
            Systemic standard deviation of the carbon rate.
        beta : Float
            Amplification factor of the historical standard deviation for the systemic carbon rate.
        nus : Float list
            Carbon rate spread mean for all companies.
        sigmas : Float list
            Standard deviation of the relative carbon rate spread.

        Returns
        -------
        None.

        '''
        self.theta = (central_std, beta, nus, sigmas)
        assert len(nus) == len(sigmas), "Mismatch in dimensions of nus and sigmas"
        # 0 : Central std
        # 1 : Beta
        # 2 : Nus
        # 3 : Sigmas


#%% Evaluation functions    
    def intensity(self, eps, eta, relative = True):
        '''
        Get a total carbon rate according to systemic and intrinsic carbon rates

        Parameters
        ----------
        eps : Float
            Systemic carbon rate.
        eta : Float
            Carbon rate spread.
        relative : Bool, optional
            Indicates whether we use the relative or net adjustment. The default is True.

        Returns
        -------
        Float
            Total carbon rate.

        '''
        if relative:
            return eps * (1 + eta)
        else:
            return eps + eta
            
    def simulate_macro(self,previous, intensities, scenar, loc, scale, g):
        # Evolution of macroeconomics
        chi = self.rng.normal(loc,scale)
        return(g(previous, intensities, scenar, chi))
    
    def toy_g(self, previous, intensities, scenar, chi):
        return previous + intensities + chi
    
    def toy_f(self, previous):
        return 1/previous
    
    #def one_density(self, intensities, previous_intensity, scenar, beta, nus, sigmas, central_var):
        # f_t((d_i)i | Delta = j, F_{t-1})
       # cov = central_var + beta * (previous_intensity - self.mus[scenar])**2
        # matrix = cov * np.ones((len(sigmas), len(sigmas)))
       # matrix += np.diag(np.array([cov]) + sigmas)
       # mn = multivariate_normal(self.mus[scenar] + nus, matrix)
        # return(mn.pdf(intensities))
    
    def one_density(self, theta, intensities, previous_intensity, scenar, t):
        '''
        Compute the density of given carbon rates knowing the scenario

        Parameters
        ----------
        theta : Tuple
            Parameters of the density.
        intensities : Series
            Carbon rates at year t for the companies.
        previous_intensity : Float
            Mean carbon rate for the previous year.
        scenar : Int
            Index of the known scenario.
        t : Int
            Current year.

        Returns
        -------
        Float
            The value of the density.

        '''
        # f_t((d_i)i | Delta = j, F_{t-1})
        cov = theta[0] + theta[1] * (previous_intensity - self.mus[scenar, t])**2
        matrix = cov * np.ones((len(theta[3]), len(theta[3])))
        matrix += np.diag(np.array([cov]) + theta[3])
        mn = multivariate_normal(self.mus[scenar, t] + theta[2], matrix)
        return(mn.pdf(intensities))
 
    def explicit_density(self, theta, intensities, previous_intensity, scenar, t, is_log = False):
        '''
        Computes the density of given carbon rates knowing the scenario using the explicit formula instead of numerical approximations

        Parameters
        ----------
        theta : Tuple
            Parameters of the density.
        intensities : Series
            Carbon rates at year t for the companies.
        previous_intensity : Float
            Mean carbon rate for the previous year.
        scenar : Int
            Index of the known scenario.
        t : Int
            Current year.
        is_log : Bool, optional
            If True, return the log-density. The default is False.

        Returns
        -------
        Float
            The value of the density.

        '''
        
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
        coeff = 1/ np.sqrt( (2* np.pi) **n * det) 
        
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
        '''
        Return a vector with the density values for each scenario

        Parameters
        ----------
        theta : Tuple
            Parameters of the density.
        intensities : Series
            Carbon rates at year t for the companies.
        previous_intensity : Float
            Mean carbon rate for the previous year.
        t : Int
            Current year.

        Returns
        -------
        Float list
            The vector with the density values.

        '''
        # \bold(f)_{t|t-1}
        density = np.zeros(len(self.mus))
        for j in range(len(self.mus)):
            # Scenario j
            density[j] = self.one_density(theta, intensities, previous_intensity, j, t)
        self.density = density
        return(self.density)

#%% Filter
    
    def filter_step(self, intensities, previous_intensity):
        '''
        Perform a step of the Hamilton filter to evaluate the new conditional probabilities

        Parameters
        ----------
        intensities : Series
            Carbon rates at year t for the companies.
        previous_intensity : Float
            Mean carbon rate for the previous year.

        Returns
        -------
        None.

        '''
        # 2.12
        density_val = self.full_density(self.theta, intensities, previous_intensity, self.history_count + 2)
        num = np.multiply(self.pi, density_val)
        marginal = np.ones(self.pi.shape).dot(num)
        self.history_pi[:, self.history_count] = self.probas
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal
        self.probas = num/marginal
 
        
#%% EM functions
    def hist_log_lk(self):
        # Log likelihood with historical parameters
        return np.sum(np.log(self.history_marginal)) # first term is 0
    
    def q1(self, theta, full_intensities):
        '''
        Computes the part of the log-likelihood depending on the density parameters

        Parameters
        ----------
        theta : Tuple
            Parameters of the density.
        full_intensities : DataFrame
            All the carbon rates for different dates up to Time.

        Returns
        -------
        q1 : Float
            First term of the log-likelihood.

        '''
        q1 = 0
        for t in range(full_intensities.shape[1] -2):
            # Parameters theta[0:3] are used in the density here
            
            # At t = 0 need to use historical
            q1 += - np.dot(np.log(self.full_density(theta, full_intensities[:,t+2], full_intensities[:,t+1].mean(axis = 0)), t), self.pi)
        return q1
    
    def q2(self):
        '''
        Computes the part of the log-likelihood depending on the conditional probabilities, which have to be computed first      

        Returns
        -------
        Float
            Second term of the log-likelihood.

        '''
        return np.sum(np.log(self.probas), self.pi)
    
    def log_lk(self, theta, full_intensities):
        '''
        Computes the log-likelihood to maximize in the EM

        Parameters
        ----------
        theta : Tuple
            Parameters of the density.
        full_intensities : DataFrame
            All the carbon rates for different dates up to Time.

        Returns
        -------
        Float
            Value of the log-likelihood.

        '''
        return(self.q1(theta, full_intensities) + self.q2())
        
    def M_step(self, full_intensities):
        '''
        Perform a maximization step in the EM algorithm

        Parameters
        ----------
        full_intensities : DataFrame
            All the carbon rates for different dates up to Time.

        Returns
        -------
        None.

        '''
        result = minimize(self.q1, self.theta, args = (full_intensities), method = 'SLSQP')
        if result.success:
            self.theta = result.x
        
    def EM(self, full_intensities, n_iter):
        '''
        Perform the EM algorithm to find better estimates for the density parameters

        Parameters
        ----------
        full_intensities : DataFrame
            All the carbon rates for different dates up to Time.
        n_iter : Int
            Number of iterations of the algorithm.

        Returns
        -------
        None.

        '''
        for l in range(n_iter):
            # E step
            self.history_count = 0
            for t in range(full_intensities.shape[1] - 2):
                self.filter_step(full_intensities[:,t+2], full_intensities[:,t+1].mean(axis = 0))
            
            # M step
            self.M_step(full_intensities)
    
    
    