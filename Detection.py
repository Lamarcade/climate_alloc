# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:29:04 2025

@author: LoïcMARCADET
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt

# Scenar : index 1 to k
# mus[scenar,t] = mu_{Delta_t},t
#(d_{i,t})_i = intensities
# (d_{i,1},..., d_{i,T})_i = full_intensities
# previous_intensity = d_t (mean of previous intensities)
# pi : Initial law
# probas : Filter probabilities
# full_intensities 

class Detection():  
    def __init__(self, initial_law = np.ones(7)/7, data_file = "Data/stoxx_1311.xlsx", scenarios_path = "Data/scenarios.xlsx", 
                 date_max = 2051, has_history = True):
        self.rng = np.random.default_rng()
        self.start_year = 2012
        self.years = range(self.start_year, date_max)
        self.Time = len(self.years)
        
        # Initial probabilities
        
        # Format to a column matrix for storage of probabilities
        initial = initial_law[:, np.newaxis]
        self.pi = initial
        self.K = len(initial_law)
        
        # Probabilities updated for the filter
        self.probas = self.pi
        
        # Keep track of the current time and all the probabilities
        self.history_count = 0
        self.history_marginal = pd.DataFrame(np.ones(self.Time))
        self.history_pi = pd.DataFrame(initial * np.ones((len(self.pi), self.Time)))
        
        # Carbon and revenue data per company
        df = pd.read_excel(data_file, index_col = 0)
        self.df = df
        

        # CO2eq emissions history, starting from 2012
        history = pd.read_excel(data_file, index_col = 0)
        
        df_valid = history[history["Financial Period Absolute"].notna()]
        
        instruments = df_valid["Instrument"].unique()
        
        history = history[history["Instrument"].isin(instruments)]
        
        history['Year'] = history['Financial Period Absolute'].str.extract(r'(\d+)').astype(str)
        
        history["NACE Classification"] = history.groupby("Instrument")["NACE Classification"].ffill()
        history["GICS Sector Name"] = history.groupby("Instrument")["GICS Sector Name"].ffill()
        
        history = history[history["Year"].astype(int) >= 2012]
        
        history = history.drop_duplicates(["Instrument", "Year"], keep = 'last')
        
        history["Scope12"] = history["CO2 Equivalent Emissions Direct, Scope 1"] + history["CO2 Equivalent Emissions Indirect, Scope 2"]
        history["Scope123"] = history["Scope12"] + history["CO2 Equivalent Emissions Indirect, Scope 3"]
        history['Scope12'] = pd.to_numeric(history['Scope12'], errors='coerce')
        history['Scope123'] = pd.to_numeric(history['Scope123'], errors='coerce')
        
        self.history = history
        
        #emission_cols = ['CO2 Equivalent Emissions Direct, Scope 1',	'CO2 Equivalent Emissions Indirect, Scope 2',
        #                 'CO2 Equivalent Emissions Indirect, Scope 3'	, 'Scope12',	'Scope123']
        
        # Last year emissions
        emission_cols = ['CO2 Equivalent Emissions Direct, Scope 1',	'CO2 Equivalent Emissions Indirect, Scope 2', 'Scope12']
        
        keep_cols = ['GICS Sector Name'] + emission_cols
        
        last = history[history["Year"] == '2023'][keep_cols]
        
        last = last.groupby('GICS Sector Name').sum().sort_values('Scope12', ascending = False)
        
        self.last = last
        self.n = len(last)
        
        
        # Scenario data
        
        rates = pd.read_excel(scenarios_path, index_col = "Scenario")
        rates.columns = rates.columns.astype(str)
        
        self.scenar_dict = {i: index for i, index in enumerate(rates.index)}
        
        grouped = (
            history.groupby(["GICS Sector Name", "Year"])["Scope12"]
            .sum()
            .reset_index()
        )

        pivot = grouped.pivot(index="GICS Sector Name", columns="Year", values="Scope12")

        pivot = pivot.loc[:, [y for y in pivot.columns if int(y) <= 2023]]

        pivot.columns = pivot.columns.astype(str)

        pivot = pivot.reindex(last.index)

        pivot.index = last.index
        
        self.sectors = pivot

        # Begin at 2021
        mus = rates.drop(columns = '2020')

        self.mus = mus
    
        base = self.last[["Scope12"]].copy()
        base.rename(columns = {"Scope12": '2023'}, inplace = True) 
        self.base = base               
        sces = []
        for scenar in mus.index:
            sce = pivot.copy().loc[:, [y for y in pivot.columns if int(y) <= 2020]]
            
            #ind = sce.index.str.replace(r"\s*\(n=\d+\)", "", regex=True)
            #sce.index = ind
            #nus = nus.reindex(ind)
            last_year = int(sce.columns[-1])
            max_year = f"{last_year+1}"
            while last_year < 2050:  
                max_year = f"{last_year+1}"
                sce[max_year] = mus.loc[scenar, max_year] * sce[f"{last_year}"] / 100 + sce[f"{last_year}"]
                sce = sce.copy()
                last_year += 1
            sces.append((scenar, sce))
            
        scenario_emissions = np.stack([df.to_numpy() for (_,df) in sces], axis=0)

        # K x n x T [scenar x company x time]
        # Time 2021 to 2050

        self.scenario_emissions = scenario_emissions
        
        self.realized = pivot.copy()
        
        self.ref_year = 2023
        self.history_count = self.start_year
        
    def initialize_parameters(self, central_std, beta, nus):
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

        Returns
        -------
        None.

        '''
        
        # Do not keep the last nu to avoid a constraint
        self.theta = np.concatenate([np.array([central_std, beta]), nus])
        self.theta = self.theta.flatten()
        # 0 : Central std
        # 1 : Beta
        # 2:2+n (excluded) : Nus     
        
#%%

    def get_density(self, theta, dt, scenario = 0, t = 0):
        #Log
        
        # PLACEHOLDER
        #log_det = 0
        #log_coeff = -0.5 * (n * np.log(2 * np.pi) + log_det)
        
        # Substract nus
        #dt = np.ones(self.n)
        
        shift = theta[2:]
        if t+self.start_year >= self.ref_year:
            a_t = np.prod(1+self.mus.loc[:, '2022':str(t+self.start_year)]/100, axis = 1)
            a_t = a_t @ self.probas
        
            shift = shift * a_t
            
        # Np array 
        mean = self.scenario_emissions[scenario, :, t] + shift
        return(multivariate_normal.logpdf(dt, mean = mean, cov = theta[0] * np.eye(self.n) + theta[0] * np.ones((self.n,self.n))))
      
    def full_density(self, theta, intensities, t):
        # \bold(f)_{t|t-1}
        density = np.zeros(self.K)
        for j in range(self.K):
            # Scenario j
            density[j] = self.get_density(theta, intensities, j, t)
        self.density = density
        
        # Put the densities as a column matrix
        return(self.density[:, np.newaxis])
    
#%% Filter

    def filter_step(self, intensities, get_probas=False):
        '''
        Perform a step of the Hamilton filter to evaluate the new conditional probabilities.
    
        Parameters
        ----------
        intensities : Series
            Carbon rates at year t for the companies.

        get_probas : Bool, optional
            Whether to return the updated probabilities.
    
        Returns
        -------
        np.ndarray or None
            Updated probabilities if get_probas is True.
        '''
    
        # Compute scenario densities (S x 1)
        # Log-densities
        density_val = self.full_density(self.theta, intensities, self.history_count - self.start_year)
    
        # Update probabilities via Bayes rule
        num = self.probas * density_val  # Element-wise multiplication (S x 1)
        marginal = np.sum(num)
    
        # Save previous probabilities
        self.history_pi.loc[:, str(self.history_count)] = self.probas.flatten()
        self.history_count += 1
        self.history_marginal.loc[str(self.history_count)] = marginal
    
        self.probas = num / marginal
    
        if get_probas:
            return self.probas

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
        # At t=0 use a previous mean rate of 0
        for col in full_intensities.columns:
            t = int(col) - self.start_year
            # Parameters central std, betas, nus are used in the density here
            q1 -= np.dot(self.full_density(theta, full_intensities.loc[:,col], t).T, self.probas).item()
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
    
    def constraint_eq(self, theta):
        '''Constraint: theta[n+1] == -sum(theta[2:n+1]). aka sum(nu_i) = 0'''
        
        return self.base["2023"].iloc[-1] * theta[self.n+1] + self.base["2023"].iloc[:-1] @ theta[2:self.n+1]
    
    def constraint_eq_pi(self, pi):
        '''Constraint: sum(pi) == 1'''
        return 1-np.sum(pi)
    
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
        
        epsilon = 1e-9
        bounds = [
        (epsilon, None),                 # theta[0] > 0
        (epsilon, 1 - epsilon),    # 0 < theta[1] < 1
    ] + [(None, None)] * n

        def check_bounds(theta, bounds):
            for i, (value, (lower, upper)) in enumerate(zip(theta, bounds)):
                if lower is not None and value < lower:
                    print(f"theta[{i}] = {value} is lower than {lower}")
                    return False
                if upper is not None and value > upper:
                    print(f"theta[{i}] = {value} is higher than {upper}")
                    return False
            return True
        
#        if not check_bounds(self.theta, bounds):
#            raise ValueError("Initial theta values are not in the bounds")
        
        value = self.q1(self.theta, full_intensities)
        
        cons = ({'type':'eq', 'fun':self.constraint_eq})

        result = minimize(self.q1, self.theta, args = (full_intensities), 
                          bounds = bounds, constraints = cons, method = 'SLSQP', options={'disp': True})
        tol = 1e-6
        if result.fun < (value - tol):
            self.theta = result.x
            if not result.success:
                print("Warning : Optimization did not converge")
            print("Theta", self.theta)
            #wait = input("Press Enter")
        else:
            print("Failure optimization Q1")
               
        # Flatten probabilities to optimize
        print(self.probas)
        
        self.pi = self.probas

      
    def EM(self, full_intensities, n_iter, get_all_probas = False, reset_probas = True):
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
        expected_loglk = [0]
        loglk = [self.hist_log_lk()]
        #self.emissions_by_sectors()
        for l in range(n_iter):
            if reset_probas:
                self.probas = np.ones(len(self.mus))/len(self.mus)
                self.probas = self.probas[:, np.newaxis]
            # E step
            if l > 0:
                expected_loglk.append(self.log_lk(self.theta, self.pi, full_intensities))
            #print("Q1 + Q2 =", expected_loglk[l])
            
            # Reset time, which is updated at every filter step 
            
            #all_probas = np.zeros((self.K, full_intensities.shape[1]))
            all_probas = pd.DataFrame(index = self.mus.index)
                
            self.history_count = self.start_year
            
            for col in full_intensities.columns:
                #print("t = ", t)
                # Update probabilities thanks to the filter

                all_probas.loc[:,col] = self.filter_step(full_intensities.loc[:,col], get_probas = True).flatten()

                #print("Probas",  all_probas[:, t])
                
            # M step: Update theta and pi
            self.M_step(full_intensities)
            #self.probas = self.pi
            loglk.append(self.hist_log_lk())

        #print("Final Q1+Q2 =",self.log_lk(self.theta, self.pi, full_intensities))
        print("Q1 + Q2 history :", expected_loglk)
        if get_all_probas:
            return expected_loglk, loglk, all_probas
        return expected_loglk, loglk        
#%%
a = Detection(np.ones(7)/7)

#central_var = 1e9 * np.random.rand()
central_var = 1e12
beta = np.random.rand()
nus = 2e5 * np.random.dirichlet(np.ones(11)) - 2e5 / 11
a.initialize_parameters(central_var, beta, nus)

#%%

es = a.scenario_emissions.copy()
es0 = es[0,:,:]
es0 = pd.DataFrame(es0)

es0.columns = [str(i) for i in range(2012,2051)]
#for col in [str(i) for i in range(2012,2033)]:
#    a.filter_step(es0[col] + 2e6 * np.random.dirichlet(np.ones(11)) - 2e6 / 11)
    
    
#%%
realized = a.realized.copy()
elk, llk, allp= a.EM(realized, n_iter = 3, get_all_probas = True)