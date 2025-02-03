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
mm = Modelnu(41, initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]))
mm.rename_rates()
fi = mm.indicators
p,q = fi.shape

nn = 0.05 * np.ones(p)
nn[p//2:] -= 0.05 * np.ones(p - p//2)

#nn = [0.2, -0.3, 0.1, -0.05, 0.15, -0.1, -0.15, 0.3, -0.10, -0.05]

mm.initialize_parameters(0.0001, 0.5, nn, 0.1 * np.ones(p))
#mm.EM(mm.indicators, n_iter = 5)    
dicti = mm.get_scenario_data()
mm.get_simul_data(sheet = 0)

#elk, lk = mm.EM(mm.indicators, n_iter = 5)


#%%   
simul = Modelnu(27, initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]))
simul.get_future_data_only()
fi = simul.indicators
p,q = fi.shape
nn = 0.05 * np.ones(p)
nn[p//2:] -= 0.05 * np.ones(p - p//2)

simul.initialize_parameters(0.0001, 0.1, nn, 0.01 * np.ones(p))
#simul.EM(simul.indicators, n_iter = 5) 

#%%
# Theoretical values 
central_std = Config.CENTRAL_STD
beta = Config.BETA
nus = Config.NUS
sigmas = Config.SIGMAS

theoretical_params = np.concatenate([np.array([central_std, beta]), nus[:-1], sigmas])

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
fake_simul = Modelnu(27, initial_law = np.array([0.5, 0.3, 0.2]))

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
