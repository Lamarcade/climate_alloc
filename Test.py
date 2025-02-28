# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:22:53 2025

@author: LoïcMARCADET
"""
from Modelnu import Modelnu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from scipy.stats import multivariate_normal, norm

import Config

# Below 2, CurPo, Delayed, Fragmented, Low Dem, NDC, NZ
#%% 

def full_process(initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]), sheet = 0):
    mm = Modelnu(41, initial_law = initial_law)
    mm.rename_rates()
    fi = mm.indicators
    p,q = fi.shape
    
    #nn = 0.05 * np.ones(p)
    #nn[p//2:] -= 0.05 * np.ones(p - p//2)
    
    #nn = [0.2, -0.3, 0.1, -0.05, 0.15, -0.1, -0.15, 0.3, -0.10, -0.05]
    
    central_std = np.random.rand()
    beta = np.random.rand()
    
    nus = np.random.dirichlet(np.ones(p))

    nus -= 1/ p
    sigmas = np.random.rand(p)
    
    mm.initialize_parameters(central_std, beta, nus, sigmas)
   
    dicti = mm.get_scenario_data()
    mm.get_simul_data(sheet = sheet)
    
    elk, lk = mm.EM(mm.indicators, n_iter = 5)
    
    return mm, elk, lk, dicti


#%%
def comparison(initial_law = np.ones(7)/7):
    mm = Modelnu(14, initial_law = initial_law)
    mm.rename_rates()
    fi = mm.indicators
    p,q = fi.shape
    
    #nn = 0.05 * np.ones(p)
    #nn[p//2:] -= 0.05 * np.ones(p - p//2)
    
    #nn = [0.2, -0.3, 0.1, -0.05, 0.15, -0.1, -0.15, 0.3, -0.10, -0.05]
    
    central_std = np.random.rand()
    beta = np.random.rand()
    
    nus = np.random.dirichlet(np.ones(p))

    nus -= 1/ p
    sigmas = np.random.rand(p)
    
    mm.initialize_parameters(central_std, beta, nus, sigmas)
    
    dicti = mm.get_scenario_data(date_max = 2024)
    #mm.get_simul_data(sheet = sheet)
    
    elk, lk = mm.EM(mm.indicators, n_iter = 2)
    
    
    
    return mm, elk, lk, dicti

#%%  

def calibrate_future_data(len_simul = 27, initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]), 
                          future_path = "Data/fake_simul.xlsx", scenar_path = "Data/fake_scenarios.xlsx", 
                          sheet = 0, n_iter = 5):
    simul = Modelnu(len_simul, initial_law = initial_law)
    simul.get_future_data_only(future_path, 
                                    scenar_path = "Data/fake_scenarios.xlsx", 
                                    sheet = sheet)
    fi = simul.indicators
    p,q = fi.shape
    central_std = np.random.rand()
    beta = np.random.rand()
    
    nus = np.random.dirichlet(np.ones(p))

    nus -= 1/ p
    sigmas = np.random.rand(p)
    
    simul.initialize_parameters(central_std, beta, nus, sigmas)
    simul.EM(simul.indicators, n_iter = n_iter) 
    return simul

#%% Theoretical values 
central_std = Config.CENTRAL_STD
beta = Config.BETA
nus = Config.NUS
sigmas = Config.SIGMAS


#%%
def plot_params(model, theoretical):
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(theoretical, model.theta)
    #plt.yscale('log')
    #plt.xscale('log')
    
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    
    plt.plot()
    
#plot_params(simul, theoretical_params)

#%% Filter with ideal parameters

def verify_filter():
    fake_simul = Modelnu(len(Config.MUS_NZ), initial_law = np.array([0.5, 0.3, 0.2]))
    
    # Sheets
    # 1 : NZ 2:CurPo 3: FW
    fake_simul.get_future_data_only("Data/fake_simul.xlsx", 
                                    scenar_path = "Data/fake_scenarios.xlsx", 
                                    sheet = 0)
    fi = fake_simul.indicators
    p,q = fi.shape
    
    fake_simul.initialize_parameters(central_std, beta, nus, sigmas)
    
    full_probas = pd.DataFrame(np.zeros((3, fi.shape[1] - 1)))
    for t in range(fi.shape[1] - 1):
        # Update probabilities thanks to the filter
        probas = fake_simul.filter_step(fi.iloc[:,t+1], fi.iloc[:,t].mean(axis = 0), get_probas = True)
        full_probas.iloc[:, t] = np.array(probas).reshape(-1)  
    return fake_simul, full_probas

#%% Tests

# =============================================================================
# simul = calibrate_future_data(len(Config.MUS_NZ), initial_law = np.array([0.5, 0.3, 0.2]))
# 
# # See which theoretical nu corresponds to which sector
# mapping = simul.index_mapping
# 
# nus_df = pd.DataFrame({"nus": nus, "sigmas":sigmas})
# 
# # See which sector corresponds to which theta
# theta_mapping = {sec:i for i, sec in enumerate(simul.indicators.index)}
# 
# # Map from theoretical nu to optimized nu in theta
# nus_df.index = nus_df.index.map(mapping).map(theta_mapping)
# nus_df.sort_index(inplace = True)
# nus_sector_order = nus_df["nus"].values[:-1]
# sigmas_sector_order = nus_df["sigmas"].values
# 
# theoretical_params = np.concatenate([np.array([central_std, beta]), nus_sector_order, sigmas_sector_order])
# 
# params_index = pd.Index(["Central_std", "Beta"]).union(simul.indicators.index[:-1], sort = False).union(pd.Index([val +"_std" for val in simul.indicators.index.values]), sort = False)
# params_df = pd.DataFrame(theoretical_params, index = params_index)
# =============================================================================

mm, elk, lk, dicti = comparison()