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
from math import inf
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from tqdm import tqdm

import Config

# Below 2, CurPo, Delayed, Fragmented, Low Dem, NDC, NZ

#%%

def input_params(simul):
    
    fi = simul.indicators
    p,q = fi.shape
    central_var = 25 * np.random.rand()
    beta = np.random.rand()
    nus = 20 * np.random.dirichlet(np.ones(p)) - 20 / p
    sigmas = 225 * np.random.rand(p)
    simul.initialize_parameters(central_var, beta, nus, sigmas)
    return simul

#%% 

def full_process(initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15]), sheet = 0):
    mm = Modelnu(42, initial_law = initial_law)
    mm.rename_rates()
    
    mm = input_params(mm)
    
   
    dicti = mm.get_scenario_data()
    mm.get_simul_data(sheet = sheet)
    
    elk, lk = mm.EM(mm.indicators, n_iter = 2)
    
    return mm, elk, lk, dicti


#%% Compare only up to 2024

def comparison(initial_law = np.ones(7)/7, n_iter = 2):
    mm = Modelnu(15, initial_law = initial_law)
    mm.rename_rates()
    
    mm = input_params(mm)
    
    dicti = mm.get_scenario_data(date_max = 2024)
    
    elk, lk, probas = mm.EM(mm.indicators, n_iter = n_iter, get_all_probas = True)
    
    
    return mm, elk, lk, dicti, probas

def best_past(n_iter = 3, n_models = 2): 
    best_lk = -inf
    best_model = None
    
    for i in range(n_models):
        simul, elk, lk, dicti, probas = comparison(n_iter = n_iter)
        
        print("elk ", elk)
        print("lk ", lk)
        if lk[-1] > best_lk:
            best_lk = lk[-1]
            best_model = simul
            best_probas = probas
            
    col_years = range(Config.START_YEAR, 2023)
    scenario_df = pd.DataFrame(best_probas)
    scenario_df.columns = col_years

    scenario_df.index = [Config.INDEX2SCENAR[i] for i in scenario_df.index]
    return(best_model, scenario_df)        

#%% Theoretical values 
central_std = Config.CENTRAL_STD
beta = Config.BETA
nus = Config.NUS
sigmas = Config.SIGMAS

future = Config.FUTURE_START_YEAR

#%% Verify filter on true scenarios
def no_calibration(future_df, scenar_df,len_simul = 29, 
                   initial_law = np.ones(7)/7, sheet = 0):
    simul = Modelnu(len_simul, initial_law = initial_law)
    
    simul.future_data_df(future_df, scenar_df)

    fi = simul.indicators
    p,q = fi.shape
    
    simul.initialize_parameters(central_std, beta, nus, sigmas)
    simul.history_count = 0
    
    history_probas = np.zeros((7, fi.shape[1]))
    for t in range(fi.shape[1]):

        history_probas[:, t] = simul.filter_step(fi.iloc[:,t], simul.compute_mean_rates(fi.iloc[:,t], simul.emissions[simul.start_year + t]), get_probas = True).flatten()

    return history_probas, simul

#%% Filter with ideal parameters

def verify_filter(fake_path = "Data/fake_simul.xlsx", scenar_path = "Data/fake_scenarios.xlsx", sheet = 0):
    fake_simul = Modelnu(len(Config.MUS_NZ) +1, initial_law = np.array([0.34, 0.33, 0.33]))
    
    # Sheets
    # 1 : NZ 2:CurPo 3: FW
    fake_simul.get_future_data_only(fake_path, 
                                    scenar_path = scenar_path, 
                                    sheet = sheet)
    fi = fake_simul.indicators
    p,q = fi.shape
    
    fake_simul.initialize_parameters(central_std, beta, nus, sigmas)
    
    full_probas = pd.DataFrame(np.zeros((3, fi.shape[1])))
    for t in range(fi.shape[1]):
        # Update probabilities thanks to the filter
        
        probas = fake_simul.filter_step(fi.iloc[:,t], fake_simul.compute_mean_rates(fi.iloc[:,t], fake_simul.emissions[future + t-1]), get_probas = True)
        full_probas.iloc[:, t] = np.array(probas).reshape(-1)  
    return full_probas, fake_simul

#%% History of filter probabilities with ideal parameters

def all_probas_history(future_path = "Data/full_fixed_params.xlsx", output = "Data/history_nocalib.xlsx", fake = False):
    future = pd.ExcelFile(future_path)
    
    params_scenars = future.sheet_names
    models = []
    
    future_file = pd.ExcelFile(future_path)

    future_data_dict = {sheet: future_file.parse(sheet, index_col=0) 
                        for sheet in future_file.sheet_names}
    params_scenars = list(future_data_dict.keys())
    
    for sheet in params_scenars:
        future_df = future_data_dict[sheet]
        if fake:
            scenar_df = pd.read_excel("Data/fake_scenarios.xlsx")
            all_probas, simul = verify_filter(fake_path = future_path, scenar_path = "Data/fake_scenarios.xlsx", sheet = sheet)
            col_years = range(Config.FUTURE_START_YEAR, Config.FUTURE_START_YEAR + all_probas.shape[1])
        else:
            scenar_df = pd.read_excel("Data/scenarios.xlsx")
            all_probas, simul = no_calibration(future_df = future_df, 
                                               scenar_df = scenar_df, sheet = sheet)
            col_years = range(Config.FUTURE_START_YEAR, 2051)
  
        scenario_df = pd.DataFrame(all_probas)
        scenario_df.columns = col_years
    
        # Scenario names
        if fake:
            scenario_df.index = [Config.INDEX3[i] for i in scenario_df.index]
        else:
            scenario_df.index = [Config.INDEX2SCENAR[i] for i in scenario_df.index]
    
        with pd.ExcelWriter(output, mode='a', if_sheet_exists = "overlay") as writer:  
            scenario_df.to_excel(writer, sheet_name = sheet)
            
        models.append(simul)
            
    return params_scenars, models

#%% All probabilities histories with calibration

def single_model_run(i, len_simul, initial_law, future_df, scenar_df, n_iter):
    from Modelnu import Modelnu
    import numpy as np
    import tempfile, joblib

    simul = Modelnu(len_simul, initial_law=initial_law)
    simul.future_data_df(future_df, scenar_df)
    fi = simul.indicators
    p, q = fi.shape

    simul = input_params(simul)
    elk, lk, all_probas = simul.EM(simul.indicators, n_iter=n_iter, get_all_probas=True)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    joblib.dump(simul, tmp_file.name)

    return lk[-1], tmp_file.name, all_probas

def best_model_parallel(len_simul=29, initial_law=np.ones(7)/7,
                      future_df=None, scenar_df=None,
                      sheet=None, n_iter=3, n_models=16, n_jobs=16):

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(single_model_run, i, len_simul, initial_law,
                            future_df, scenar_df, n_iter)
            for i in range(n_models)
        ]
        
        for f in tqdm(futures, desc=f"Running models for '{sheet}'", unit="model", dynamic_ncols=True):
            try:
                lk, model, probas = f.result()
                results.append((lk, model, probas))
            except Exception as e:
                print("Error in model run:", e)

    if not results:
        raise RuntimeError("No valid model was returned")

    best_lk, best_model, best_probas = max(results, key=lambda x: x[0])
    return best_model, best_probas, results

def best_model_probas(len_simul = 29, initial_law = np.ones(7)/7, 
                          future_path = "Data/full_fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx", 
                          sheet = 0, n_iter = 3, n_models = 2):
    
    best_lk = -inf
    best_model = None
    for i in range(n_models):
        simul = Modelnu(len_simul, initial_law = initial_law)
        simul.get_future_data_only(future_path, 
                                        scenar_path = scenar_path, 
                                        sheet = sheet)

        simul = input_params(simul)
        elk, lk, all_probas = simul.EM(simul.indicators, n_iter = n_iter, get_all_probas = True) 
        
        print("elk ", elk)
        print("lk ", lk)
        if lk[-1] > best_lk:
            best_lk = lk[-1]
            best_model = simul
            best_probas = all_probas
            
    return best_model, best_probas

def all_probas_history_calib(future_path = "Data/full_fixed_params.xlsx", scenar_path = "Data/scenarios.xlsx",
                             output = "Data/parallel_calib.xlsx",
                             len_simul = 29, initial_law = np.ones(7)/7,
                             n_iter = 3, n_models = 16, fake = False):
    future_file = pd.ExcelFile(future_path)
    scenar_df = pd.read_excel(scenar_path)

    future_data_dict = {sheet: future_file.parse(sheet, index_col=0) 
                        for sheet in future_file.sheet_names}
    params_scenars = list(future_data_dict.keys())
    
    models = []
    
    for sheet in tqdm(params_scenars, desc="Calibrating all scenarios"):
        print("Now calibrating with scenario ", sheet)
        future_df = future_data_dict[sheet]
    
        best_model, all_probas, results = best_model_parallel(
            len_simul=len_simul, initial_law=initial_law,
            future_df=future_df, scenar_df=scenar_df, sheet = sheet,
            n_iter=n_iter, n_models=n_models
        )
        if fake:
            col_years = range(Config.FUTURE_START_YEAR, Config.FUTURE_START_YEAR + all_probas.shape[1])
        else:
            col_years = range(Config.FUTURE_START_YEAR, 2051)
        
        scenario_df = pd.DataFrame(all_probas)
        scenario_df.columns = col_years
    
        # Scenario names
        if fake:
            scenario_df.index = [Config.INDEX3[i] for i in scenario_df.index]
        else:
            scenario_df.index = [Config.INDEX2SCENAR[i] for i in scenario_df.index]
    
        with pd.ExcelWriter(output, mode='a', if_sheet_exists = "overlay") as writer:  
            scenario_df.to_excel(writer, sheet_name = sheet)
            
        models.append(best_model)
        
    return params_scenars, models, results
        
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

def probas_plot(path = "Data/history_nocalib.xlsx", output = "Figs/stackplots_nocalib.pdf", focus = None):
    future = pd.ExcelFile(path)
    
    scenars = future.sheet_names
    
    with PdfPages(output) as pdf:
        for sheet in scenars:
            probas = future.parse(sheet, index_col = 0)
            plot_probas = probas.transpose()

            plot_probas.rename(columns = dict_abbrev, inplace = True)
            if focus is not None:
                year_sort = plot_probas.index[focus]
            else:
                year_sort = plot_probas.index[-1]
            plot_probas = plot_probas[plot_probas.columns[plot_probas.loc[year_sort].argsort()[::-1]]]
            
            color_list = [colors[col] for col in plot_probas.columns if col in colors]
            if focus:
                plot_probas.loc[:year_sort, :].plot.area(title = sheet, color = color_list)
            else:
                plot_probas.plot.area(title = sheet, color = color_list)
            plt.legend(loc = 'upper left')
            
            pdf.savefig()
            plt.close()

def average_probas_plot(paths, output="Figs/Simul/Test3/stackplots_avg.pdf", focus=None):
    future_data_dicts = []
    for path in paths:
        excel_file = pd.ExcelFile(path)
        future_data_dicts.append({
            sheet: excel_file.parse(sheet, index_col=0)
            for sheet in excel_file.sheet_names
        })

    scenars = list(future_data_dicts[0].keys())

    with PdfPages(output) as pdf:
        for sheet in scenars:
            dfs = [d[sheet] for d in future_data_dicts]
            stacked = pd.concat(dfs).groupby(level=0).mean()  

            plot_probas = stacked.transpose()
            plot_probas.rename(columns=dict_abbrev, inplace=True)

            year_sort = focus if focus is not None else plot_probas.index[-1]
            plot_probas = plot_probas[plot_probas.columns[plot_probas.loc[year_sort].argsort()[::-1]]]

            color_list = [colors[col] for col in plot_probas.columns if col in colors]
            if focus:
                plot_probas.loc[:year_sort, :].plot.area(title=sheet, color=color_list)
            else:
                plot_probas.plot.area(title=sheet, color=color_list)
            plt.legend(loc='upper left')

            pdf.savefig()
            plt.close()
#%% Models evaluation

def compare(simul):
    #See which theoretical nu corresponds to which sector
    mapping = simul.index_mapping
    
    nus_df = pd.DataFrame({"nus": nus, "sigmas":sigmas})
    
    # See which sector corresponds to which theta
    theta_mapping = {sec:i for i, sec in enumerate(simul.indicators.index)}
    
    # Map from theoretical nu to optimized nu in theta
    nus_df.index = nus_df.index.map(mapping).map(theta_mapping)
    nus_df.sort_index(inplace = True)
    nus_sector_order = nus_df["nus"].values[:-1]
    sigmas_sector_order = nus_df["sigmas"].values
    
    theoretical_params = np.concatenate([np.array([central_std, beta]), nus_sector_order, sigmas_sector_order])
    
    params_index = pd.Index(["Central_std", "Beta"]).union(simul.indicators.index[:-1], sort = False).union(pd.Index([val +"_std" for val in simul.indicators.index.values]), sort = False)
    params_df = pd.DataFrame(theoretical_params, columns = ["Theoretical"], index = params_index)
    params_df["Simulation"] = simul.theta
    return params_df

def likelihoods(models, params_scenars):
    lks = []
    for model in models:
        lks.append(model.hist_log_lk())
    return(pd.DataFrame(lks, columns = ["LogLK"], index = params_scenars))

def calibration_effect(calib_path="Data/history_calib.xlsx", nocalib_path="Data/history_nocalib.xlsx"):
    calib = pd.ExcelFile(calib_path)
    no_calib = pd.ExcelFile(nocalib_path)
    scenars = calib.sheet_names

    kl_dict = {}


    for sheet in scenars:
        cp = calib.parse(sheet, index_col=0)
        ncp = no_calib.parse(sheet, index_col=0)

        kld_values = {}
        for col in cp.columns:
            p = cp[col].values
            q = ncp[col].values
            kld_values[col] = entropy(p, q) 

        kl_dict[sheet] = kld_values

    kl_df = pd.DataFrame(kl_dict).T
    kl_df = kl_df.T

    plt.figure(figsize=(12, 6))
    for scenar in kl_df.columns:
        plt.plot(kl_df.index, kl_df[scenar], label=scenar)

    plt.title("KL-divergence between calibration and no calibration", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("KL Divergence")
    plt.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()  

def average_calibration_effect(calib_template="Data/Simul/Test3/Calib_{}.xlsx",
                                nocalib_template="Data/Simul/Test3/Nocalib_{}.xlsx",
                                n_simulations=10, mini = False):
    kl_dict = {}
    
    all_calibs = [pd.ExcelFile(calib_template.format(i)) for i in range(1, n_simulations + 1)]
    all_nocalibs = [pd.ExcelFile(nocalib_template.format(i)) for i in range(1, n_simulations + 1)]

    scenars = all_calibs[0].sheet_names

    clusters = {
    "Cluster 1 (CP)": [
        "Current Policies",
        "Nationally Determined Contributions (NDCs)"
    ],
    "Cluster 2 (FW)": [
        "Below 2°C",
        "Fragmented World"
    ],
    "Cluster 3 (NZ)": [
        "Delayed transition",
        "Low demand",
        "Net Zero 2050"
    ]
}
    for sheet in scenars:
        all_kls = []

        for calib_file, nocalib_file in zip(all_calibs, all_nocalibs):
            cp = calib_file.parse(sheet, index_col=0)
            ncp = nocalib_file.parse(sheet, index_col=0)
            
                        
            cp_clusters = pd.DataFrame({
            cluster: cp.loc[scenarios].sum()
            for cluster, scenarios in clusters.items()
        }).T
            ncp_clusters = pd.DataFrame({
            cluster: ncp.loc[scenarios].sum()
            for cluster, scenarios in clusters.items()
        }).T

            kl_values = {}
            for col in cp.columns:
                p = cp_clusters[col].values
                q = ncp_clusters[col].values

                #p = p + 1e-12
                #q = q + 1e-12
                #p /= p.sum()
                #q /= q.sum()

                #kl_values[col] = entropy(p, q)
                kl_values[col] = sum(abs(p-q))/2

            all_kls.append(pd.Series(kl_values))

        if mini:
            avg_kl = min(all_kls, key=lambda s: s.loc[2050])
        else:
            avg_kl = pd.concat(all_kls, axis=1).mean(axis=1)
        kl_dict[sheet] = avg_kl

    kl_df = pd.DataFrame(kl_dict)

    plt.figure(figsize=(12, 6))
    for scenar in kl_df.columns:
        plt.plot(kl_df.index, kl_df[scenar], label=scenar)

    if mini:
        plt.title("Evolution of the lowest-in-2050 Total Variation Distance between calibration and no calibration", fontsize=14)
    else:
        plt.title("Average Total Variation Distance between calibration and no calibration", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("TVD")
    plt.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return kl_df

def final_calibration_effect(calib_template="Data/Simul/Test3/Calib_{}.xlsx",
                                nocalib_template="Data/Simul/Test3/Nocalib_{}.xlsx",
                                n_simulations=10):
    kl_dict = {}
    
    all_calibs = [pd.ExcelFile(calib_template.format(i)) for i in range(1, n_simulations + 1)]
    all_nocalibs = [pd.ExcelFile(nocalib_template.format(i)) for i in range(1, n_simulations + 1)]

    scenars = all_calibs[0].sheet_names

    clusters = {
    "Cluster 1 (CP)": [
        "Current Policies",
        "Nationally Determined Contributions (NDCs)"
    ],
    "Cluster 2 (FW)": [
        "Below 2°C",
        "Fragmented World"
    ],
    "Cluster 3 (NZ)": [
        "Delayed transition",
        "Low demand",
        "Net Zero 2050"
    ]
}

    for sheet in scenars:
        all_kls = []

        for calib_file, nocalib_file in zip(all_calibs, all_nocalibs):
            cp = calib_file.parse(sheet, index_col=0)
            ncp = nocalib_file.parse(sheet, index_col=0)
            
            cp_clusters = pd.DataFrame({
            cluster: cp.loc[scenarios].sum()
            for cluster, scenarios in clusters.items()
        }).T
            ncp_clusters = pd.DataFrame({
            cluster: ncp.loc[scenarios].sum()
            for cluster, scenarios in clusters.items()
        }).T

            p = cp_clusters[cp_clusters.columns[-1]].values
            q = ncp_clusters[ncp_clusters.columns[-1]].values

            #p = p + 1e-12
            #q = q + 1e-12
            #p /= p.sum()
            #q /= q.sum()

            #kl_value = entropy(p, q)
            
            kl_value = sum(abs(p-q)/2)

            all_kls.append(kl_value)

        #avg_kl = pd.concat(all_kls, axis=1).mean(axis=1)
        kl_dict[sheet] = all_kls

    kl_df = pd.DataFrame(kl_dict)

    plt.figure(figsize=(12, 6))
    x_labels = kl_df.columns
    x_positions = range(len(x_labels))
    x_offset = 0.15
    
    for i, scenario in enumerate(x_labels):
        y_values = kl_df[scenario]
        x_values = [i] * len(y_values)  
        plt.scatter(x_values, y_values, label=scenario)
        
        y_min = y_values.min()
        y_max = y_values.max()
        

        plt.text(i - x_offset, y_min, f"{y_min:.3f}", color="blue", fontsize=8, ha='center', va='bottom')
        plt.text(i - x_offset, y_max, f"{y_max:.3f}", color="red", fontsize=8, ha='center', va='top')

    plt.title("Total Variation Distance between calibration and no calibration", fontsize=14)
    plt.xlabel("Scenario")
    plt.ylabel("TVD")
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return kl_df

def verif_probas_plot(path = "Data/history_calib.xlsx", nopath = "Data/history_nocalib.xlsx", scenar = "Fragmented World", title = None, titleno = None):
    future = pd.ExcelFile(path)
    nofuture = pd.ExcelFile(nopath)

    probas = future.parse(scenar, index_col = 0)
    noprobas = nofuture.parse(scenar, index_col = 0)
    plot_probas = probas.transpose()
    plot_no = noprobas.transpose()

    plot_probas.rename(columns = dict_abbrev, inplace = True)
    plot_no.rename(columns = dict_abbrev, inplace = True)

    year_sort = plot_probas.index[-1]
    sorted_cols = plot_probas.loc[year_sort].sort_values(ascending=False).index.tolist()
    
    
    common_cols = [col for col in sorted_cols if col in plot_no.columns]
    plot_probas = plot_probas[common_cols]
    plot_no = plot_no[common_cols]
    
    color_list = [colors[col] if col in colors else 'grey' for col in common_cols]
    
    plot_probas.plot.area(title = scenar, color = color_list)
    plt.legend(loc = 'upper left')
    plt.title(title)

    plot_no.plot.area(title = scenar, color = color_list)
    plt.legend(loc = 'upper left')
    plt.title(titleno)
    plt.show()

#%%
# =============================================================================
# a, b, c, d, e = comparison()
# theta = a.theta
# params = ["central_var", "beta"]
# params_nu = [f'nu_{i}' for i in range(1, 10)]
# params_sigma = [f'var_{i}' for i in range(1, 11)]
# indexed = params + params_nu + params_sigma
# 
# theta_df = pd.DataFrame(theta)
# theta_df.index = indexed
# =============================================================================

#%% Test 1

#params_scenars, models = all_probas_history(future_path = "Data/fake_simul.xlsx", output = "Data/fake_nocalib.xlsx", fake = True)
#probas_plot(path = "Data/fake_nocalib.xlsx", output = "Figs/fake_stackplots_nocalib.pdf", focus = 30)

#%% Test 2

# =============================================================================
# if __name__ == '__main__':
#     freeze_support()
#     params_scenars, models, res = all_probas_history_calib(future_path = "Data/fake_ordered.xlsx", scenar_path = "Data/fake_scenarios.xlsx",
#                              output = "Data/fake_calib_new.xlsx",
#                              len_simul = len(Config.MUS_NZ) + 1, initial_law = np.ones(3)/3,
#                              n_iter = 3, n_models = 16, fake = True)
#     probas_plot(path = "Data/fake_calib_new.xlsx", output = "Figs/fake_stackplots_calib_new.pdf", focus = 30)
# =============================================================================

#%% Test 3

#params_scenars, models = all_probas_history(future_path = "Data/full_fixed_params.xlsx")
#probas_plot()

# =============================================================================
# for i in tqdm(range(1, 11), desc="Simulation number"):
#      input_path = f"Data/Simul/Test3/All/Test3_{i}.xlsx"
#      output_path = f"Data/Simul/Test3/All/Nocalib_{i}.xlsx"
#      all_probas_history(future_path=input_path, output=output_path)
# =============================================================================
    
# paths = [f"Data/Simul/Test3/Nocalib_{i}.xlsx" for i in range(1, 11)]
# average_probas_plot(paths, output="Figs/stackplots_avg_test3nocalib.pdf")

# =============================================================================
# def run_one(i):
#     input_path = f"Data/Simul/Test3/All/Test3_{i}.xlsx"
#     output_path = f"Data/Simul/Test3/All/Nocalib_{i}.xlsx"
#     all_probas_history(future_path=input_path, output=output_path)
#     return i  
# 
# if __name__ == "__main__":
#     freeze_support()  
# 
#     N = 1000
#     max_workers = 12   
#     
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(run_one, i) for i in range(1, N+1)]
#         
#         for f in tqdm(as_completed(futures), total=N, desc="Parallel models"):
#             pass
# =============================================================================


#%% Test 4

#params_scenars, models = all_probas_history_calib(n_iter = 3, n_models = 2)
#probas_plot(path = "Data/history_calib.xlsx", output = "Figs/stackplots_calib.pdf")

#if __name__ == '__main__':
#    freeze_support()
#    params_scenars, models, res_last_scenar = all_probas_history_calib(n_iter = 3, n_models = 8)
#    probas_plot(path = "Data/parallel_calib.xlsx", output = "Figs/parallel_stackplots_calib.pdf")

# =============================================================================
# if __name__ == '__main__':
#     freeze_support()
#     for i in tqdm(range(1, 11), desc="Running calibrations"):
#         input_path = f"Data/Simul/Test3/Test3_{i}.xlsx"
#         output_path = f"Data/Simul/Test3/Calib_{i}.xlsx"
#         
#         
#         all_probas_history_calib(
#             future_path=input_path, scenar_path="Data/scenarios.xlsx",
#             output=output_path, len_simul=29, initial_law=np.ones(7)/7,
#             n_iter=3,n_models=16, fake=False)
# 
# 
#     paths = [f"Data/Simul/Test3/Calib_{i}.xlsx" for i in range(1, 11)]
#     average_probas_plot(paths, output="Figs/stackplots_avg_test3.pdf")
# =============================================================================

#average_calibration_effect()

#%% Test 5 
#scen, models = all_probas_history(future_path = "Data/intermediate.xlsx", output = "Data/intermediate_nocalib.xlsx")
#probas_plot(path = "Data/intermediate_nocalib.xlsx", output = "Figs/intermediate_stackplots_nocalib.pdf")

#params_scenars, models_c = all_probas_history_calib(future_path = "Data/intermediate.xlsx", output = "Data/intermediate_calib.xlsx")
#probas_plot(path = "Data/intermediate_calib.xlsx", output = "Figs/intermediate_stackplots_calib.pdf")

#params_middle = compare(models_c[2])
#lks = likelihoods(models_c, params_scenars)

#calibration_effect(calib_path="Data/intermediate_calib.xlsx", nocalib_path="Data/intermediate_nocalib.xlsx")

#%% Test 7

#model, calib = best_past(n_iter = 3, n_models = 16)

#%% Verifications

#kl_df = final_calibration_effect()
#kl_df = average_calibration_effect(mini = True)
verif_probas_plot(path = "Data/Simul/Test3/Calib_4.xlsx", nopath = "Data/Simul/Test3/Nocalib_4.xlsx", scenar = "Fragmented World", title = "DelT (Calibration)", titleno = "DelT (No calibration)")


