# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:43:21 2024

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

class Modelnu():  
    
#%% Initialization
    def __init__(self, Time, initial_law = np.array([0.35, 0.15, 0.25, 0.25]), data_file = "Data/Stoxx_data_Scope12.xlsx", history = True):
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
        
        self.start_year = 2009
        
        for i in range(14, -1, -1):
            df["Scope12 Y-{i}".format(i = i)] = df["Scope 1 Y-{i}".format(i = i)] + df["Scope 2 Y-{i}".format(i = i)]
       
        indicators = df[["Instrument","GICS Sector Name"]].copy()
        
        for i in range(13, -1, -1):
            # Actual year / Former year from Y-13 to Y-0
            #indicators["Rate Y-{i}".format(i = i)] = 100 * df["Total Y-{i}".format(i = i)] / df["Total Y-{j}".format(j = i+1)]
            # Keep as a percentage, with absolute percentage increase to account for negative values

            indicators["Rate Y-{i}".format(i = i)] = 100 * (df["Scope12 Y-{i}".format(i = i)] - df["Scope12 Y-{j}".format(j = i+1)]) / abs(df["Scope12 Y-{j}".format(j = i+1)])
        
        # Reduce number of rates and drop NaN
        indicators.replace([np.inf, -np.inf], np.nan, inplace = True)
        
        #ind = indicators.dropna()
        #ind = ind.iloc[:6]
        
        # Use sector decarbonation rates
        sectors = indicators.copy()
        #sectors.drop(sectors.columns[0], axis = 1, inplace = True)
        
        df_merged = sectors.merge(df, on=["Instrument", "GICS Sector Name"])

        # Real Estate decarbonation rates have many outliers
        df_merged = df_merged[df_merged["GICS Sector Name"] != "Real Estate"]

        decarbonation_rates = {}
        
        for i in range(13, -1, -1):
            numerator = df_merged.groupby("GICS Sector Name").apply(
                lambda x: np.nansum(x[f"Rate Y-{i}"] * x[f"Scope12 Y-{i+1}"])
            )
            denominator = df_merged.groupby("GICS Sector Name")[f"Scope12 Y-{i+1}"].sum()
            
            decarbonation_rates[f"Decarbonation Rate Y-{i}"] = numerator / denominator
        
        decarbonation_df = pd.DataFrame(decarbonation_rates)
        
        # Real Estate decarbonation rates have many outliers
        #decarbonation_df.drop("Real Estate", inplace = True)
        
        self.indicators = decarbonation_df
        #print(sectors.columns)
        
        # Central carbon rates
        self.mus = pd.DataFrame(np.zeros((len(initial_law),Time)))
        
        # Historical central carbon rate
        
        annual_mean_rates = {}

        for i in range(13, -1, -1):
            numerator = np.nansum(df_merged[f"Rate Y-{i}"] * df_merged[f"Scope12 Y-{i+1}"])
            denominator = np.nansum(df_merged[f"Scope12 Y-{i+1}"])
            
            if denominator != 0:
                annual_mean_rates[f"Decarbonation Rate Y-{i}"] = numerator / denominator
            else:
                annual_mean_rates[f"Decarbonation Rate Y-{i}"] = np.nan
        
        annual_mean_df = pd.DataFrame(annual_mean_rates, index=[0]).T
        annual_mean_df.columns = ["Annual Mean Decarbonation Rate"]
        
        self.rates = annual_mean_df
        
        total_numerator = np.nansum([
            annual_mean_df["Annual Mean Decarbonation Rate"].iloc[i] * np.nansum(df_merged[f"Scope12 Y-{i+1}"]) 
            for i in range(13, -1, -1)
        ])
        total_denominator = np.nansum([
            np.nansum(df_merged[f"Scope12 Y-{i+1}"]) 
            for i in range(13, -1, -1)
        ])
        
        overall_mean_decarbonation_rate = total_numerator / total_denominator if total_denominator != 0 else np.nan
            
        # Duration of the historical data
        self.T0 = self.indicators.shape[1] 
  
        for t in range(self.T0):
            self.mus.iloc[:, t] = overall_mean_decarbonation_rate * np.ones(len(initial_law)) 
            
        self.emissions_by_sectors()
    
    def compute_mean_rates(self, rates,emissions):
        dt = np.sum(rates*emissions/np.sum(emissions))
        return(dt)
    
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
        
        # Do not keep the last nu to avoid a constraint
        self.theta = np.concatenate([np.array([central_std, beta]), nus[:-1], sigmas])
        self.theta = self.theta.flatten()
        assert len(nus) == len(sigmas), "Mismatch in dimensions of nus and sigmas"
        # 0 : Central std
        # 1 : Beta
        # 2:1+n (excluded) : Nus
        # 1+n:1+2n : Sigmas^2

    def rename_rates(self):
        start_year = 2010
        
        for i in range(0, 14):
            self.indicators.rename(columns = {f"Decarbonation Rate Y-{i}": 2023-i}, inplace = True)
        #self.mus.columns = [str(start_year + i) for i in range(self.mus.shape[1])]
        self.mus.columns = [start_year + i for i in range(self.mus.shape[1])]

    def get_scenario_data(self, path = "Data/scenarios.xlsx", date_max = 2051):
        rates = pd.read_excel(path)
        
        scenar_dict = {i: index for i, index in enumerate(rates["Scenario"])}
        
        # Begin at 2021
        self.mus.loc[:, rates.columns[2:date_max]] = rates 
        return scenar_dict
    
    def get_simul_data(self, path = "Data/simul.xlsx", sheet = 0, complete_emissions = False):
        simul = pd.read_excel(path, sheet_name = sheet)

        # First columns are an index and NA values
        simul = simul[simul.columns[2:]]
        simul_sorted = simul.sort_values(by=2021, ascending=False).reset_index(drop=True)
        
        # The sector with the highest historical rate gets the highest rate 
        # from the simulation, and so on
        new_year = self.indicators.columns[-1]
        
        # Not weighted by emissions here since we attribute rates
        histo_means = self.indicators.mean(axis = 1)
        histo_order = histo_means.sort_values(ascending=False).index
        simul_sorted.index = histo_order
        
        # Start at future data keeping rates up to 2023
        num_dates = simul.columns[-1] - new_year
        for i in range(new_year +1, new_year +1 + num_dates):
            self.indicators[i] = simul_sorted[i]
            
    def get_future_data_only(self, path = "Data/fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx", sheet = 0):
        # DEPRECATED
        
        simul = pd.read_excel(path, index_col = 0, sheet_name = sheet)
        
        start_year = 2023
        
        # Emissions start year
        self.start_year = start_year-1
    
        simul_sorted = simul.sort_values(by=start_year, ascending=False)
        
        self.get_scenario_data(scenar_path)
        #self.mus.columns = [start_year + i for i in range(self.mus.shape[1])]
        
        self.mus = self.mus.loc[:,start_year:]
        
        # The sector with the highest historical rate gets the highest rate 
        # from the simulation, and so on
        histo_means = self.indicators.mean(axis = 1)

        histo_order = histo_means.sort_values(ascending=False).index
        self.index_mapping = dict(zip(simul_sorted.index, histo_order))

        # Reindex
        simul_sorted.index = histo_order

        # Get rid of old columns
        self.indicators = self.indicators.iloc[:, :0] 
        
        self.indicators = pd.concat([self.indicators, simul_sorted.loc[:, simul_sorted.columns[1:]]], axis=1)
        
       
    def future_data_df(self, simul_df, scenar_df, date_max=2051):
        start_year = 2023
        self.start_year = start_year - 1
    
        self.mus = scenar_df.copy()
        self.mus = self.mus.loc[:, scenar_df.columns[2:date_max]]

        simul_aligned = simul_df.reindex(self.indicators.index)
    
        self.indicators = self.indicators.iloc[:, :0]
        self.indicators = pd.concat([self.indicators, simul_aligned], axis=1)   
       
    def emissions_by_sectors(self):
        self.emissions = self.df[[f"Scope12 Y-{i}" for i in range(14, -1, -1)] + ["GICS Sector Name"]].groupby(by = "GICS Sector Name").sum()
        for i in range(14, -1, -1):
            self.emissions.rename(columns = {f"Scope12 Y-{i}": 2023-i}, inplace = True)
        #for i in range(2024, 2024+len(self.indicators)):
        #    self.emissions[i] = self.indicators[i] * self.emissions[i-1] / 100 + self.emissions[i-1]
            
    def update_emissions(self):
        max_year = self.emissions.columns[-1] +1
        self.emissions[max_year] = self.indicators[max_year] * self.emissions[max_year-1] / 100 + self.emissions[max_year-1]
        self.emissions = self.emissions.copy()

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
        diago = np.diag( np.array([cov]) / theta[1+n:])
        #denom = (1 + np.ones(n).dot(diago).dot(np.ones(n)))
        # Simplified formula
        denom = 1 + cov * np.sum(1/theta[1+n:])
            
        inverse = 1/ (cov) * (diago - diago.dot(np.ones((n, n))).dot(diago) / denom)
    
        # determinant
        # cov**n * det(D_{j,t}) * denom
        
        det = np.prod(theta[1+n:]) * denom
        #det = np.exp(np.sum(np.log(theta[1+n:])))
        #print("det1", det)
        #print("det", np.exp(np.sum(np.log(theta[1+n:]))))
        
        # Computing the density
        coeff = 1/ np.sqrt( (2* np.pi) **n * det) 
        #print("denom", denom)
        #print("det", det)
        #print("Coeff", coeff)
        #wait = input("Enter")
        
        # self.mus.iloc[scenar, t] =  mu is a single value
        # Add back nu_n as the opposite of the sum of the (nu_i)i
        
        vector = (intensities - (self.mus.iloc[scenar, t] + np.concatenate([theta[2:1+n], [-np.sum(theta[2:1+n])]])))
       # print("vector", vector)

        inside = -1/2 * vector.dot(inverse).dot(vector)
        #print("inside", inside)
       # print(inverse)
        
        if is_log:
            #print("Logcoeff", np.log(coeff))
            return(np.log(coeff) + inside)
        else:
            #print("coeff", coeff)
            #print("fac", np.exp(inside))
            #print(np.log(coeff) + inside)
            return(coeff * np.exp(inside))
 
    def explicit_density_2(self, theta, intensities, previous_intensity, scenar, t, is_log=False):
        intensities = np.asarray(intensities)
        mu = self.mus.iloc[scenar, t]
    
        n = len(intensities)
    
        cov = theta[0] + theta[1] * (previous_intensity - mu)**2
    
        # Weights (diagonal elements of D)
        d = theta[1 + n:]
        d_inv = 1 / d
        denom = 1 + cov * np.sum(d_inv)
    
        nu = np.concatenate([theta[2:1 + n], [-np.sum(theta[2:1 + n])]])
    
        vector = intensities - (mu + nu)
    
        # Compute inverse efficiently using Sherman-Morrison-like identity
        weighted = vector * d_inv
        weighted_sum = np.sum(weighted)
        quad_form = (1 / cov) * (np.dot(weighted, vector) - (cov * weighted_sum**2) / denom)
    
        # Determinant
        det = np.prod(d) * denom
        log_det = np.sum(np.log(d)) + np.log(denom)
    
        # Log density
        log_coeff = -0.5 * (n * np.log(2 * np.pi) + log_det)
        log_density = log_coeff - 0.5 * quad_form
    
        return log_density if is_log else np.exp(log_density)
    
    def explicit_density_vec(self, theta, intensities, previous_intensity, scenars, t, is_log):
        n = len(intensities)
        theta_shifts = theta[2:1+n]
        last_component = -np.sum(theta_shifts)
        theta_total = np.concatenate([theta_shifts, [last_component]])
        theta_cov_terms = theta[1+n:]
        sum_inv_cov = np.sum(1 / theta_cov_terms)
    
        results = []
    
        for scenar in scenars:
            mu = self.mus.iloc[scenar, t]
            cov = theta[0] + theta[1] * (previous_intensity - mu) ** 2
    
            diago = np.diag(np.array([cov]) / theta_cov_terms)
            denom = 1 + cov * sum_inv_cov
    
            inverse = (1 / cov) * (diago - diago.dot(np.ones((n, n))).dot(diago) / denom)
    
            det = np.prod(theta_cov_terms) * denom
            coeff = 1 / np.sqrt((2 * np.pi) ** n * det)
    
            vector = intensities - (mu + theta_total)
            inside = -0.5 * vector.dot(inverse).dot(vector)
    
            if is_log:
                results.append(np.log(coeff) + inside)
            else:
                results.append(coeff * np.exp(inside))
    
        return np.array(results)
 
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
        
        # Put the densities as a column matrix
        return(self.density[:, np.newaxis])
    
    def full_density_2(self, theta, intensities, previous_intensity, t, is_log=False):
        '''
        Return a vector with the density values for each scenario in a vectorized way.
    
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
        is_log : Bool, optional
            If True, return log-densities.
    
        Returns
        -------
        np.ndarray
            Densities (or log-densities) as a column vector (S x 1).
        '''
        intensities = np.asarray(intensities)
        n = len(intensities)
        S = len(self.mus)
    
        mu_vec = self.mus.iloc[:, t].values          # Shape: (S,)
        cov = theta[0] + theta[1] * (previous_intensity - mu_vec) ** 2  # Shape: (S,)
        
        d = theta[1 + n:]
        d_inv = 1 / d
        denom = 1 + cov * np.sum(d_inv)              # Shape: (S,)
        
        nu = np.concatenate([theta[2:1 + n], [-np.sum(theta[2:1 + n])]])  # Shape: (n,)
    
        # Shape: (S, n)
        vector = intensities[None, :] - (mu_vec[:, None] + nu[None, :])
    
        weighted = vector * d_inv[None, :]           # Shape: (S, n)
        weighted_sum = np.sum(weighted, axis=1)      # Shape: (S,)
        quad_form = (1 / cov) * (
            np.einsum("ij,ij->i", weighted, vector) - 
            (cov * weighted_sum ** 2) / denom
        )                                            # Shape: (S,)
        
        log_det = np.sum(np.log(d)) + np.log(denom)  # Shape: (S,)
        log_coeff = -0.5 * (n * np.log(2 * np.pi) + log_det)
        log_density = log_coeff - 0.5 * quad_form    # Shape: (S,)
    
        densities = log_density if is_log else np.exp(log_density)
        return densities[:, np.newaxis]
    
    def full_density(self, theta, intensities, previous_intensity, t, is_log = False):
        js = np.arange(len(self.mus))
        self.density = self.explicit_density_vec(theta, intensities, previous_intensity, js, t, is_log)
        return self.density[:, np.newaxis]
                

#%% Filter
    
    def filter_step(self, intensities, previous_intensity, get_probas = False):
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
        #print(intensities)
        #print("Prev")
        #print(previous_intensity)
        #print()
        #wait = input("Enter")
        num = np.multiply(self.probas, density_val)
        
        marginal = np.ones(self.probas.shape).T.dot(num).item() # since it is a matrix
        
        #print("marginal", marginal)
        
        
        self.history_pi[:, self.history_count] = self.probas.flatten()
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal

        self.probas = num/marginal
        if self.history_count + self.start_year > 2023 and self.emissions.columns[-1] +1 < self.indicators.columns[-1] +1:
            self.update_emissions()
        #print(self.probas)
        if get_probas:
            return self.probas
        #print(self.probas)
        #wait = input("Enter")
        
    def filter_step(self, intensities, previous_intensity, get_probas=False):
        '''
        Perform a step of the Hamilton filter to evaluate the new conditional probabilities.
    
        Parameters
        ----------
        intensities : Series
            Carbon rates at year t for the companies.
        previous_intensity : Float
            Mean carbon rate for the previous year.
        get_probas : Bool, optional
            Whether to return the updated probabilities.
    
        Returns
        -------
        np.ndarray or None
            Updated probabilities if get_probas is True.
        '''
    
        # Compute scenario densities (S x 1)
        density_val = self.full_density(self.theta, intensities, previous_intensity, self.history_count)
    
        # Update probabilities via Bayes rule: p(j|t) ∝ p(j|t−1) * f(y_t | j)
        num = self.probas * density_val  # Element-wise multiplication (S x 1)
        marginal = np.sum(num)           # Scalar
    
        # Save previous probabilities
        self.history_pi[:, self.history_count] = self.probas.flatten()
        self.history_count += 1
        self.history_marginal[self.history_count] = marginal
    
        # Normalize
        self.probas = num / marginal
    
        # Optionally update emissions
        if (self.history_count + self.start_year > 2023 and
            self.emissions.columns[-1] + 1 < self.indicators.columns[-1] + 1):
            self.update_emissions()
    
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
        q1 = - np.dot(self.full_density(theta, full_intensities.iloc[:,0], 0 * full_intensities.iloc[:,0].mean(axis = 0), 0, is_log = True).T, self.probas).item()
        for t in range(1,full_intensities.shape[1]):
            # Parameters central std, betas, nus are used in the density here
            #q1 -= np.dot(self.full_density(theta, full_intensities.iloc[:,t+1], full_intensities.iloc[:,t].mean(axis = 0), t, is_log = True).T, self.probas).item()
            q1 -= np.dot(self.full_density(theta, full_intensities.iloc[:,t], self.compute_mean_rates(full_intensities.iloc[:,t], self.emissions[self.start_year + t]), t, is_log = True).T, self.probas).item()
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
        
        epsilon = 1e-9
        bounds = [
        (epsilon, None),                 # theta[0] > 0
        (epsilon, 1 - epsilon),    # 0 < theta[1] < 1
    ] + [(None, None)] * (n-1) + [(epsilon, None)] * n  # Positivité pour theta[2+n : 2+2n]

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

        result = minimize(self.q1, self.theta, args = (full_intensities), 
                          bounds = bounds, method = 'SLSQP', options={'disp': True})
        tol = 1e-6
        if result.fun < (value - tol):
            self.theta = result.x
            if not result.success:
                print("Warning : Optimization did not converge")
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
        self.emissions_by_sectors()
        for l in range(n_iter):
            if reset_probas:
                self.probas = np.ones(len(self.mus))/len(self.mus)
                self.probas = self.probas[:, np.newaxis]
            # E step
            if l > 0:
                expected_loglk.append(self.log_lk(self.theta, self.pi, full_intensities))
            #print("Q1 + Q2 =", expected_loglk[l])
            
            # Reset time, which is updated at every filter step 
            
            all_probas = np.zeros((len(self.mus), full_intensities.shape[1]))
                
            self.history_count = 0
            for t in range(full_intensities.shape[1]):
                #print("t = ", t)
                #print("emissions t = ", self.start_year + t)
                # Update probabilities thanks to the filter

                all_probas[:,t] = self.filter_step(full_intensities.iloc[:,t], self.compute_mean_rates(full_intensities.iloc[:,t], self.emissions[self.start_year + t]), get_probas = True).flatten()
                #print("Probas",  all_probas[:, t])
                
                #print("Indicators time ", full_intensities.columns[t])
                #print("Decarbo rate ", self.compute_mean_rates(full_intensities.iloc[:,t], self.emissions[self.start_year + t]))
            
            # M step: Update theta and pi
            self.M_step(full_intensities)
            #self.probas = self.pi
            loglk.append(self.hist_log_lk())

        #print("Final Q1+Q2 =",self.log_lk(self.theta, self.pi, full_intensities))
        print("Q1 + Q2 history :", expected_loglk)
        if get_all_probas:
            return expected_loglk, loglk, all_probas
        return expected_loglk, loglk