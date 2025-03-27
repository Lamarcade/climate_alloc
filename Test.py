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
from matplotlib.backends.backend_pdf import PdfPages

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


#%% Compare only up to 2024

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

def calibrate_future_data(len_simul = 28, initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]), 
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
def no_calibration(len_simul = 28, initial_law = np.ones(7)/7, 
                   future_path = "Data/fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx",
                   sheet = 0):
    simul = Modelnu(len_simul, initial_law = initial_law)
    simul.get_future_data_only(future_path, 
                                    scenar_path = "Data/scenarios.xlsx", 
                                    sheet = sheet)
    fi = simul.indicators
    p,q = fi.shape
    
    simul.initialize_parameters(central_std, beta, nus, sigmas)
    simul.history_count = 0
    
    history_probas = np.zeros((7, fi.shape[1] - 1))
    for t in range(fi.shape[1] - 1):
        history_probas[:, t] = simul.filter_step(fi.iloc[:,t+1], fi.iloc[:,t].mean(axis = 0), get_probas = True).flatten()

    return history_probas, simul

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

#%% Assess probabilities with no calibration
###### OBSOLETE

def all_probas(future_path = "Data/fixed_params.xlsx"):
    no_calib = pd.DataFrame()
    future = pd.ExcelFile(future_path)
    
    params_scenars = future.sheet_names
    
    for sheet in params_scenars:
        probas, simul = no_calibration(sheet = sheet, future_path = future_path)
        no_calib[sheet] = probas.flatten()
        
    # Scenario names
    no_calib.index = [Config.INDEX2SCENAR[i] for i in no_calib.index]
    return(no_calib)

#%%

def all_probas_history(future_path = "Data/full_fixed_params.xlsx", output = "Data/history_nocalib.xlsx"):
    future = pd.ExcelFile(future_path)
    
    params_scenars = future.sheet_names
    
    for sheet in params_scenars:
        all_probas, simul = no_calibration(sheet = sheet, future_path = future_path)
  
        scenario_df = pd.DataFrame(all_probas, columns = range(2024, 2051))
    
        # Scenario names
        scenario_df.index = [Config.INDEX2SCENAR[i] for i in scenario_df.index]
    
        with pd.ExcelWriter(output, mode='a', if_sheet_exists = "overlay") as writer:  
            scenario_df.to_excel(writer, sheet_name = sheet)

#%% Randomly initialize different models and keep the best one
#### OBSOLETE

def best_model_future_data(len_simul = 28, initial_law = np.ones(7)/7, 
                          future_path = "Data/fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx", 
                          sheet = 0, n_iter = 3, n_models = 5):
    
    best_lk = 0
    for i in range(n_models):
        simul = Modelnu(len_simul, initial_law = initial_law)
        simul.get_future_data_only(future_path, 
                                        scenar_path = "Data/scenarios.xlsx", 
                                        sheet = sheet)
        fi = simul.indicators
        p,q = fi.shape
        central_std = np.random.rand()
        beta = np.random.rand()
        
        nus = np.random.dirichlet(np.ones(p))
    
        nus -= 1/ p
        sigmas = np.random.rand(p)
        
        simul.initialize_parameters(central_std, beta, nus, sigmas)
        elk, lk = simul.EM(simul.indicators, n_iter = n_iter) 
        
        if lk[-1] > best_lk:
            best_lk = lk[-1]
            best_model = simul
            
    return best_model, elk, lk

#%%

def all_probas_calibration(len_simul = 28, initial_law = np.ones(7)/7, 
                          future_path = "Data/full_fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx", 
                          n_iter = 3, n_models = 5):
    calib = pd.DataFrame()
    future = pd.ExcelFile(future_path)
    
    params_scenars = future.sheet_names
    
    for sheet in params_scenars:
        print("Now calibrating with scenario ", sheet)
        model, elk, lk = best_model_future_data(len_simul, initial_law, future_path, scenar_path, sheet = sheet, n_iter = n_iter, n_models = n_models)
        calib[sheet] = model.probas.flatten()
        
    # Scenario names
    calib.index = [Config.INDEX2SCENAR[i] for i in calib.index]
    return(calib)

#%% All probabilities histories with calibration

def best_model_probas(len_simul = 28, initial_law = np.ones(7)/7, 
                          future_path = "Data/full_fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx", 
                          sheet = 0, n_iter = 3, n_models = 5):
    
    best_lk = 0
    for i in range(n_models):
        simul = Modelnu(len_simul, initial_law = initial_law)
        simul.get_future_data_only(future_path, 
                                        scenar_path = "Data/scenarios.xlsx", 
                                        sheet = sheet)
        fi = simul.indicators
        p,q = fi.shape
        central_std = np.random.rand()
        beta = np.random.rand()
        
        nus = np.random.dirichlet(np.ones(p))
    
        nus -= 1/ p
        sigmas = np.random.rand(p)
        
        simul.initialize_parameters(central_std, beta, nus, sigmas)
        elk, lk, all_probas = simul.EM(simul.indicators, n_iter = n_iter, get_all_probas = True) 
        
        if lk[-1] > best_lk:
            best_lk = lk[-1]
            best_model = simul
            best_probas = all_probas
            
    return best_model, best_probas

def all_probas_history_calib(future_path = "Data/full_fixed_params.xlsx", output = "Data/history_calib.xlsx"):
    future = pd.ExcelFile(future_path)
    
    params_scenars = future.sheet_names
    
    for sheet in params_scenars:
        print("Now calibrating with scenario ", sheet)
        best_model, all_probas = best_model_probas(sheet = sheet, future_path = future_path)
  
        scenario_df = pd.DataFrame(all_probas, columns = range(2024, 2051))
    
        # Scenario names
        scenario_df.index = [Config.INDEX2SCENAR[i] for i in scenario_df.index]
    
        with pd.ExcelWriter(output, mode='a', if_sheet_exists = "overlay") as writer:  
            scenario_df.to_excel(writer, sheet_name = sheet)
#%% Stackplot

dict_abbrev = {"Below 2°C": "B2°", "Current Policies" : "CurPo", "Delayed transition" : "Delay",
               "Fragmented World" : "Frag", "Low demand": "LowD", 
               "Nationally Determined Contributions (NDCs)": "NDCs", 
               "Net Zero 2050":"NZ", "Optimistic":"Opti", "Pessimistic":"Pessi", "Middle": "Mid"}

colors = {
    "B2°": "#1f77b4",
    "CurPo": "#ff7f0e",
    "Delay": "#2ca02c",
    "Frag": "#d62728",
    "LowD": "#9467bd",
    "NDCs": "#8c564b",
    "NZ": "#e377c2",
}

def probas_plot(path = "Data/history_nocalib.xlsx", output = "Figs/stackplots_nocalib.pdf"):
    future = pd.ExcelFile(path)
    
    scenars = future.sheet_names
    
    with PdfPages(output) as pdf:
        for sheet in scenars:
            probas = future.parse(sheet, index_col = 0)
            plot_probas = probas.transpose()

            plot_probas.rename(columns = dict_abbrev, inplace = True)
            year_sort = plot_probas.index[-1]
            plot_probas = plot_probas[plot_probas.columns[plot_probas.loc[year_sort].argsort()[::-1]]]
            
            color_list = [colors[col] for col in plot_probas.columns if col in colors]
            
            plot_probas.plot.area(title = sheet, color = color_list)
            plt.legend(loc = 'upper left')
            
            pdf.savefig()
            plt.close()

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

#mm, elk, lk, dicti = comparison()

#probas, mm = no_calibration()

#%%
#no_calib = all_probas(future_path = "Data/full_fixed_params.xlsx")

#no_calib.to_excel("Data/probas_comparison.xlsx", sheet_name = "No calibration")

#with pd.ExcelWriter('Data/full_probas_comparison.xlsx', mode='a', if_sheet_exists = "overlay") as writer:  
#    no_calib.to_excel(writer, sheet_name = "No calibration")

#calib = all_probas_calibration(future_path = "Data/full_fixed_params.xlsx")

#with pd.ExcelWriter('Data/full_probas_comparison.xlsx', mode='a', if_sheet_exists = "overlay") as writer: 
#    calib.to_excel(writer, sheet_name = "Calibration")

#%% Evolution of probas

#all_probas_history(future_path = "Data/full_fixed_params.xlsx")
#probas_plot()

#all_probas_history_calib(output= "Data/history_calib2.xlsx")
#probas_plot(path = "Data/history_calib2.xlsx", output = "Figs/stackplots_calib.pdf")

# With intermediate scenarios

#no_calib = all_probas_history(future_path = "Data/intermediate.xlsx", output = "Data/history_intermediate.xlsx")

#probas_plot(path = "Data/history_intermediate.xlsx", output = "Figs/stackplots_intermediate.pdf")

