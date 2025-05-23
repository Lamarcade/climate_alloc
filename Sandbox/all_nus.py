# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:25:08 2025

@author: LoïcMARCADET
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
import sys
import numdifftools as nd

# Scenar : index 1 to k
# mus[scenar,t] = mu_{Delta_t},t
#(d_{i,t})_i = intensities
# (d_{i,1},..., d_{i,T})_i = full_intensities
# previous_intensity = d_t (mean of previous intensities)
# pi : Initial law
# probas : Filter probabilities
# full_intensities 

class Modelnu():  
    
#%% Initialization
    def __init__(self, Time, initial_law = np.array([0.35, 0.15, 0.25, 0.25]), data_file = "Data/Stoxx_data.xlsx"):
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
        
        # Format to a column matrix for storage of probabilities
        initial = initial_law[:, np.newaxis]
        self.pi = initial
        
        # Probabilities updated for the filter
        self.probas = self.pi
        
        # Keep track of the current time and all the probabilities
        self.history_count = 0
        self.history_marginal = np.ones(Time)
        self.history_pi = initial * np.ones((len(self.pi), Time))
        
        # Carbon and revenue data per company
        df = pd.read_excel(data_file)
        df = df.drop(df.columns[0], axis = 1)
        self.df = df
        
        indicators = df[["Instrument","GICS Sector Name"]].copy()
        for i in range(13, -1, -1):
            # Actual year / Former year from Y-13 to Y-0
            #indicators["Rate Y-{i}".format(i = i)] = 100 * df["Total Y-{i}".format(i = i)] / df["Total Y-{j}".format(j = i+1)]
            
            # Keep as a percentage, with absolute percentage increase to account for negative values
            indicators["Rate Y-{i}".format(i = i)] = df["Total Y-{i}".format(i = i)] / abs(df["Total Y-{j}".format(j = i+1)])
        
        # Reduce number of rates and drop NaN
        indicators.replace([np.inf, -np.inf], np.nan, inplace = True)
        ind = indicators.dropna()
        ind = ind.iloc[:6]
        
        # Use sector decarbonation rates
        sectors = indicators.copy()
        sectors.drop(sectors.columns[0], axis = 1, inplace = True)
        sectors = sectors.groupby(by= "GICS Sector Name").mean()
        
        # Real Estate decarbonation rates have many outliers
        sectors.drop("Real Estate", inplace = True)
        
        self.indicators = sectors
        #print(sectors.columns)
        
        # Central carbon rates
        self.mus = pd.DataFrame(np.zeros((len(initial_law),Time)))
        
        # Historical central carbon rate
        mu = 0
        
        # Duration of the historical data
        self.T0 = self.indicators.shape[1] 
        
        for t in range(self.T0):
            mu += self.indicators.mean(axis = None, skipna = True)
        mu /= self.T0
        for t in range(self.T0):
            self.mus.iloc[:, t] = mu * np.ones(len(initial_law))   
    
        
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
        
        self.theta = np.concatenate([np.array([central_std, beta]), nus, sigmas])
        self.theta = self.theta.flatten()
        assert len(nus) == len(sigmas), "Mismatch in dimensions of nus and sigmas"
        # 0 : Central std
        # 1 : Beta
        # 2:2+n (excluded) : Nus
        # 2+n:2+2n : Sigmas

    def rename_rates(self):
        start_year = 2010
        
        for i in range(0, 14):
            self.indicators.rename(columns = {f"Rate Y-{i}": 2023-i}, inplace = True)
        #self.mus.columns = [str(start_year + i) for i in range(self.mus.shape[1])]
        self.mus.columns = [start_year + i for i in range(self.mus.shape[1])]

    def get_scenario_data(self, path = "Data/scenarios.xlsx"):
        rates = pd.read_excel(path)
        
        scenar_dict = {i: index for i, index in enumerate(rates["Scenario"])}
        
        # Begin at 2021
        self.mus.loc[:, rates.columns[2:]] = rates 
        return scenar_dict
    
    def get_simul_data(self, path = "Data/simul.xlsx", sheet = 0):
        simul = pd.read_excel("Data/simul.xlsx", sheet_name = sheet)

        # First columns are an index and NA values
        simul = simul[simul.columns[2:]]
        simul_sorted = simul.sort_values(by=2021, ascending=False).reset_index(drop=True)
        
        # The sector with the highest rate in 2021 gets the highest rate 
        # from the simulation, and so on
        new_year = self.indicators.columns[-1]
        histo_means = self.indicators.mean(axis = 1)
        histo_order = histo_means.sort_values(ascending=False).index
        simul_sorted.index = histo_order
        
        # Start at future data keeping rates up to 2023
        num_dates = simul.columns[-1] - new_year
        for i in range(new_year +1, new_year +1 + num_dates):
            self.indicators[i] = simul_sorted[i]

#%% Evaluation functions    
 
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
        
        cov = theta[0] + theta[1] * (previous_intensity - self.mus.iloc[scenar, t])**2
        
        #print("cov", cov)
        n = len(intensities)
        
        # cov_inverse
        diago = np.diag( np.array([cov]) / theta[2+n:])
        #denom = (1 + np.ones(n).dot(diago).dot(np.ones(n)))
        # Simplified formula
        denom = 1 + cov * np.sum(1/theta[2+n:])
            
        inverse = 1/ (cov) * (diago - diago.dot(np.ones((n, n))).dot(diago) / denom)
    
        # determinant
        # cov**n * det(D_{j,t}) * denom
        det = np.prod(theta[2+n:]) * denom
        
        # Computing the density
        coeff = 1/ np.sqrt( (2* np.pi) **n * det) 
        #print("denom", denom)
        #print("det", det)
        #print("Coeff", coeff)
        
        # self.mus.iloc[scenar, t+1] =  mu is a single value
        # Add back nu_n as the opposite of the sum of the (nu_i)i
        vector = (intensities - (self.mus.iloc[scenar, t+1] + theta[2:2+n]))

        inside = -1/2 * vector.dot(inverse).dot(vector)
        
        if t==41:
            #print("PI", previous_intensity)
            print("Scenar", scenar)
            print("Mu", self.mus.iloc[scenar,t])
            #print("coeff", coeff)
            #print("inside", inside)
            print("intensities", intensities)
            print("nus", mm.theta[2:2+n])
            print()
            print("vector", vector)
            #print("inside", inside)
            wait = input("Enter")
        #print("Mahalanobis", inside)
        #print()
       # wait = input("Press Enter")
        if is_log:
            #print("Logcoeff", np.log(coeff))
            return(np.log(coeff) + inside)
        else:
            return(coeff * np.exp(inside))
 
# intensities = self.df.loc[:,self.history_count + 1]
# previous_intensity = self.df.loc[:,self.history_count].mean(axis = 0)
# If history count is 0 need to add historical data
 
    def full_density(self, theta, intensities, previous_intensity, t, is_log = False):
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
            density[j] = self.explicit_density(theta, intensities, previous_intensity, j, t, is_log)
        self.density = density
# =============================================================================
#         if t == 38:
#             print("densities at time 38", density)
#             
#             print("intensities", intensities)
#             print("mus", self.mus.iloc[:, t])
#             print()
#             wait = input("Enter")
# =============================================================================
        
        # Put the densities as a column matrix
        return(self.density[:, np.newaxis])

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
        #self.probas = self.pi
        density_val = self.full_density(self.theta, intensities, previous_intensity, self.history_count)
        #print("Density val", density_val)
        #print()
        num = np.multiply(self.probas, density_val)
        
        marginal = np.ones(self.probas.shape).T.dot(num).item() # since it is a matrix
        
        #print("marginal", marginal)
        
        
        self.history_pi[:, self.history_count] = self.probas.flatten()
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal

        self.probas = num/marginal
        #print(self.probas)
        #wait = input("Enter")
 
        
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
        for t in range(full_intensities.shape[1] -1):
            # Parameters central std, betas, nus are used in the density here
            
            # At t = 0 need to use historical
            q1 -= np.dot(self.full_density(theta, full_intensities.iloc[:,t+1], full_intensities.iloc[:,t].mean(axis = 0), t, is_log = True).T, self.probas).item()
        return q1
    
    def q2(self, pi):
        '''
        Computes the part of the log-likelihood depending on the initial law    

        Returns
        -------
        Float
            Second term of the log-likelihood.

        '''
        return - np.sum(self.probas * np.log(pi + sys.float_info.min))
    
    def log_lk(self, theta, pi, full_intensities):
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
        print("Q1:", self.q1(theta, full_intensities))
        print("Q2:", self.q2(pi))
        print()
        return(self.q1(theta, full_intensities) + self.q2(pi))
    
    def constraint_eq(self, theta,n):
        '''Constraint: theta[n+1] == -sum(theta[2:n+1]). aka sum(nu_i) = 0'''
        return theta[n+1] + np.sum(theta[2:n+1])
    
    def constraint_eq_pi(self, pi):
        '''Constraint: sum(pi) == 1'''
        return 1-np.sum(pi)

    def constraint_ineq(self, theta, n):
        '''Constraint: theta[2+n : 2+2n] > 0. aka (sigma_i)i > 0'''
        return theta[2+n : 2+2*n]
    
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
        n = len(full_intensities)
        
        epsilon = 1e-6
        bounds = [
        (epsilon, None),                 # theta[0] > 0
        (epsilon, 1 - epsilon),    # 0 < theta[1] < 1
    ] + [(None, None)] * (n) + [(epsilon, None)] * n  # Positivité pour theta[2+n : 2+2n]
        
        constraints = [
            {'type': 'eq', 'fun': self.constraint_eq, 'args': (n,)},
            #{'type': 'ineq', 'fun': self.constraint_ineq, 'args': (n,)}
            ]

# =============================================================================
#         fun = lambda x: self.q1(x, full_intensities)
#         grad_func = nd.Gradient(fun, step=1e-6, order = 6)
#         nn = 0.2 * np.ones(p)
#         nn[p//2:] -= 0.4 * np.ones(p - p//2)
#         
#         mm = 100 * np.ones(p)
#         mm[p//2:] -= 200 * np.ones(p - p//2)
#         te = np.concatenate([np.array([1., 0.5]), nn, 0.1*np.ones(len(nn))])
#         te2 = np.concatenate([np.array([1., 0.5]), mm, 0.1*np.ones(len(nn))])
#         
#         #print("Func", fun(te))
#         #print("Func 2", fun(te2))
#         print("Func theta", fun(self.theta))
#         print("GRAD", grad_func(self.theta))
#         wait = input("enter")
# =============================================================================

        def check_bounds(theta, bounds):
            for i, (value, (lower, upper)) in enumerate(zip(theta, bounds)):
                if lower is not None and value < lower:
                    print(f"theta[{i}] = {value} is lower than {lower}")
                    return False
                if upper is not None and value > upper:
                    print(f"theta[{i}] = {value} is higher than {upper}")
                    return False
            return True
        
        if not check_bounds(self.theta, bounds):
            raise ValueError("Initial theta values are not in the bounds")
        
        result = minimize(self.q1, self.theta, args = (full_intensities), 
                          bounds = bounds, method = 'SLSQP', options={'disp': True})
        if result.success:
            self.theta = result.x
            print("Theta", self.theta)
            #wait = input("Press Enter")
        else:
            print("Failure optimization Q1")
            
        bounds_law = [(0, 1)] * self.pi.size
        constraints_pi = [{'type': 'eq', 'fun': self.constraint_eq_pi}]    
        # Flatten probabilities to optimize
        print(self.probas)
        
# =============================================================================
#         result_law = minimize(self.q2, self.pi.flatten(), bounds= bounds_law, 
#                           constraints = constraints_pi, method = 'SLSQP', options={'disp': True})
#         if result_law.success:
#             self.pi = result_law.x.reshape(len(self.pi), 1)
#             print("Pi", self.pi)
#             wait = input("Press Enter")
#         else:
#             print("Failure optimization Q2")
# =============================================================================
        self.pi = self.probas/sum(self.probas)

      
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
        expected_loglk = []
        loglk = [self.hist_log_lk()]
        for l in range(n_iter):
            # E step
            expected_loglk.append(self.log_lk(self.theta, self.pi, full_intensities))
            print("Q1 + Q2 =", expected_loglk[l])
            
            # Reset time, which is updated at every filter step 
            self.history_count = 0
            for t in range(full_intensities.shape[1] - 1):
                # Update probabilities thanks to the filter
                self.filter_step(full_intensities.iloc[:,t+1], full_intensities.iloc[:,t].mean(axis = 0))
            
            # M step: Update theta and pi
            self.M_step(full_intensities)
            self.probas = self.pi
            loglk.append(self.hist_log_lk())

        print("Final Q1+Q2 =",self.log_lk(self.theta, self.pi, full_intensities))
        print("Q1 + Q2 history :", expected_loglk)
        return expected_loglk, loglk
            
    
# Below 2, CurPo, Delayed, Fragmented, Low Dem, NDC, NZ
mm = Modelnu(41, initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]))
mm.rename_rates()
fi = mm.indicators
p,q = fi.shape

nn = 0.2 * np.ones(p)
nn[p//2:] -= 0.4 * np.ones(p - p//2)
np.sum(nn)
mm.initialize_parameters(1, 0.5, nn, 0.1 * np.ones(p))
mm.EM(mm.indicators, n_iter = 5)    
dicti = mm.get_scenario_data()
mm.get_simul_data(sheet = 1)

elk, lk = mm.EM(mm.indicators, n_iter = 5)   