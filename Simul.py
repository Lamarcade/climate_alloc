# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:25:40 2025

@author: LoïcMARCADET
"""

import pandas as pd
import numpy as np, numpy.random
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

import Config

#%%

final_df = pd.read_excel("Data/scenarios.xlsx", index_col = "Scenario")
final_df = final_df.drop(columns = 2020)

#%%

n = len(Config.GICS)
s2i = Config.SCENAR2INDEX
i2s = Config.INDEX2SCENAR
i2a = Config.INDEX2ABB
ts = Config.THREE_SCENAR
i3 = Config.INDEX3

all_scenar = list(s2i.keys())

central_std = Config.CENTRAL_STD
beta = Config.BETA
nus = Config.NUS
sigmas = Config.SIGMAS


nus_df = Config.NUS_ORDER
sigmas_df = Config.SIGMAS_ORDER

last_em = Config.LAST_EM

scenar_index = 6
scenar_name = "Net Zero 2050"

N = 100
#%%

def simul_order(central_std, beta, nus_df, scenar_index, scenar_name = None, mus = final_df):
    n = len(sigmas)
    matrix = central_std * np.ones((n,n)) + central_std * np.eye(n)

    if scenar_name is not None:
        locmu = scenar_name
    else:
        locmu = i2s[scenar_index]
    
    mu_old = mus.loc[locmu, 2024]
    
    emissions = last_em["Scope12"]
    
    shift = (nus_df["Spreads"].values * mus.loc[locmu, 2023]/100)
    
    mean = (mu_old * np.abs(emissions) / 100) + emissions
    
    new_em = multivariate_normal(mean = mean , cov = matrix).rvs() 
    
    dis = pd.DataFrame({2024: new_em + shift}, index = Config.HISTO_ORDER)
    
    new_dis = {}
    for col in mus.columns[mus.columns > 2024].values:
        mu_new = mus.loc[locmu, col]
        
        shift = nus_df["Spreads"].values

        a_t = np.prod(1+mus.loc[locmu, 2023:col]/100)
            
        shift = shift * a_t
                
        # Np array 
        mean = (mu_new * np.abs(new_em) / 100) + new_em

        
        # Old cov matrix
        #normalized_weights = emissions / emissions.sum()
        #dt = np.average(di, weights=normalized_weights)
        
        # matrix = (central_std + beta * (dt - mu_old)**2) * np.ones((n,n))
        # matrix += np.diag(sigmas_df["Variances"].values)
        
        new_em = multivariate_normal(mean = mean, cov = matrix).rvs()
        
        new_dis[col] = new_em + shift
        
        mu_old = mu_new
        
    new_dis_df = pd.DataFrame(new_dis, index=Config.HISTO_ORDER)
    dis = pd.concat([dis, new_dis_df], axis=1)
            
    return(dis)

def compare_simul_mus(dis, scenar_index, scenar_name = None, mus = final_df):
    years = dis.columns.astype(int)
    sectors = dis.index
    
    if scenar_name is not None:
        locmu = scenar_name
    else:
        locmu = i2s[scenar_index]
    
    mus_selected = mus.loc[locmu]   
    a_t = {}
    
    for t in years:

        mus_slice = mus_selected.loc[2023:t] / 100
        a_t[t] = np.prod(1 + mus_slice)
        
    # Shifts
    spreads = nus_df["Spreads"]

    shift = pd.DataFrame(index=sectors, columns=years)
    
    for t in years:
        shift[t] = spreads * a_t[t]
        
    # Mus simulated
    mussimul = pd.DataFrame(index=sectors, columns=years)

    for t in years[1:]:
        t_prev = years[years.get_loc(t) - 1]
    
        num = (dis[t] - shift[t]) - (dis[t_prev] - shift[t_prev])
        denom = (dis[t_prev] - shift[t_prev]).abs()
    
        mussimul[t] = num / denom * 100
    
    mussimul[years[0]] = np.nan
    
    return mussimul

def simulate_N_dis_and_mus(N,
                           central_std,
                           beta,
                           nus_df,
                           scenar_index,
                           scenar_name=None,
                           mus=final_df):
    
    simulations_dis = {}
    simulations_mus = {}
    
    for k in range(N):
        dis_k = simul_order(
            central_std=central_std,
            beta=beta,
            nus_df=nus_df,
            scenar_index=scenar_index,
            scenar_name=scenar_name,
            mus=mus
        )
        
        simulations_dis[k] = dis_k
        
        mus_k = compare_simul_mus(
            dis=dis_k,
            scenar_index=scenar_index,
            scenar_name=scenar_name,
            mus=mus
        )
        
        simulations_mus[k] = mus_k
    
    # {sim_id: DataFrame}
    return simulations_dis, simulations_mus

def plot_mus_and_simul(sim_mus, mus, scenar_name, alpha=0.95):

    df_list = []
    for k, df in sim_mus.items():
        tmp = df.copy()
        tmp["sim_id"] = k
        df_list.append(tmp.reset_index().melt(
            id_vars=["index", "sim_id"],
            var_name="year",
            value_name="mussimul"
        ))
    df_all = pd.concat(df_list, ignore_index=True)
    df_all.rename(columns={"index": "sector"}, inplace=True)
    df_all["year"] = df_all["year"].astype(int)


    stats = df_all.groupby(["sector", "year"])["mussimul"].agg(
        mean="mean",
        std="std",
        count="count"
    )

    z = 1.96
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["lower"] = stats["mean"] - z * stats["se"]
    stats["upper"] = stats["mean"] + z * stats["se"]

    mus_scenar = mus.loc[scenar_name]


    sectors = stats.index.get_level_values(0).unique()
    years = sorted(stats.index.get_level_values(1).unique())

    for sector in sectors:
        data = stats.loc[sector]

        plt.figure(figsize=(9, 5))

        plt.fill_between(
            years,
            data["lower"],
            data["upper"],
            alpha=0.2
        )

        plt.plot(years, data["mean"], label="Mean of simulated mus")

        mus_values = mus_scenar.loc[years]
        plt.plot(years, mus_values, linestyle="--", label=f"Mus ({scenar_name})")

        plt.title(f"{sector} — Comparaison mus vs mussimul")
        plt.xlabel("Year")
        plt.ylabel("Rate %")
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
#%% Tests

#sim_dis, sim_mus = simulate_N_dis_and_mus(N,central_std,beta,nus_df,scenar_index,scenar_name=None, mus=final_df)
#plot_mus_and_simul(sim_mus = sim_mus, mus = final_df, scenar_name = scenar_name, alpha=0.95)