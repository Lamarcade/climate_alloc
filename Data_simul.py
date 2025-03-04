# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:24:07 2025

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np, numpy.random
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from scipy.stats import multivariate_normal, norm

import Config


#%%

sectors = ["Communication Services", "Consumer Discretionary",
           "Consumer Staples", "Energy", "Financials",
           "Health Care", "Industrials", 
           "Information Technology", "Materials",
           "Utilities"] 

def scenario_means():
    #data = pd.DataFrame({"Sector": sectors})
    data = pd.DataFrame(100, index=sectors, columns=[f"Y-{i}" for i in range(10, -1, -1)])
    
    folder = "Data/NGFS5/"
    file = "IAM_data.xlsx"
    
    ngfs = pd.read_excel(folder + file)
    
    # Approximation EU-15, lacks Norway, Switzerland and Poland for CAC40
    filtered = ngfs[ngfs["Region"] == "GCAM 6.0 NGFS|EU-15"]
    
    indic = "CO2"
    filtered = filtered[filtered["Variable"].str.contains(indic, case=False, na=False)]
    
    #%%
    scenarios_means = filtered[filtered["Variable"] == "Emissions|CO2"]
    
    scenar = scenarios_means.melt(
        id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
        var_name="Year",
        value_name="Value"
    )
    
    scenar["Year"] = scenar["Year"].astype(int)
    scenar["Value"] = pd.to_numeric(scenar["Value"], errors="coerce")
    scenar = scenar.infer_objects() # Avoid warning
    
    def interpolate_group(group):
        # Group on first 5 columns
        group_metadata = group.iloc[0, :5] 
        
        # Interpolate based on the grouped data
        group = group.set_index("Year").reindex(range(2020, 2051))
        group["Value"] = group["Value"].interpolate(method="linear")
        group.reset_index(inplace=True)
        
        # Put back the grouping columns
        for col in group_metadata.index:
            group[col] = group_metadata[col]
        return group
    
    df_int = (
        scenar.groupby(["Model", "Scenario", "Region", "Variable", "Unit"], group_keys=False)
        .apply(interpolate_group)
    )
    
    df_int = df_int[
        ["Model", "Scenario", "Region", "Variable", "Unit", "Year", "Value"]
    ]
    #%%
    
    #%%
    scenario_data = df_int.copy()
    scenario_data["Ratio"] = scenario_data.groupby(['Model', 'Scenario', 'Region', 'Variable'])['Value']\
                     .transform(lambda x: x / abs(x.shift(1)))
    scenario_data.drop({"Model", "Region", "Unit", "Value", "Variable"}, axis = 1, inplace = True)
    
    
    final_df = scenario_data.pivot(index=["Scenario"], columns="Year", values="Ratio")
    
    #final_df.to_excel("Data/scenarios.xlsx")

#%%
final_df = pd.read_excel("Data/scenarios.xlsx", index_col = "Scenario")
#%%
df_plot = final_df.reset_index().melt(id_vars="Scenario", var_name="Year", value_name="Ratio")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x="Year", y="Ratio", hue="Scenario", marker="o")

plt.title("Carbonation rate for each scenario", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("CR", fontsize=14)
plt.legend(title="Scenario", fontsize=12)
plt.grid(True)
plt.show()

#%% Simulate data for sectors

n_sectors = 10 

scenar2index = {"Below 2°C": 0, "Current Policies" : 1, "Delayed transition" : 2,
               "Fragmented World" : 3, "Low demand": 4, 
               "Nationally Determined Contributions (NDCs)":5, 
               "Net Zero 2050":6}

index2scenar = {0 : "Below 2°C", 1: "Current Policies", 2: "Delayed transition",
               3:"Fragmented World", 4: "Low demand", 
               5: "Nationally Determined Contributions (NDCs)", 
               6:"Net Zero 2050"}


def scenario_simul(data = final_df, scenar_index = 6, n_sectors = n_sectors):
    # Number of years
    etas = np.random.dirichlet(np.ones(n_sectors),size=len(data.columns)).transpose()

    # Sum to 0 instead of 1
    etas -= 1/ n_sectors
    
    # Retrieve average scenario path
    base_data = data.loc[index2scenar[scenar_index]]
    
    paths = pd.DataFrame(np.ones((n_sectors, len(data.columns))), columns = data.columns)
    for i in range(len(paths)):
        paths.loc[i] = base_data
        paths.loc[i] += etas[i,:]
    return paths

def simul_constant(data = final_df, scenar_index = 6, n_sectors = n_sectors):
    nus = np.random.dirichlet(np.ones(n_sectors))
    
    # Sum to 0 instead of 1
    nus -= 1/ n_sectors
    
    # Retrieve average scenario path
    base_data = data.loc[index2scenar[scenar_index]]
    
    paths = pd.DataFrame(np.ones((n_sectors, len(data.columns))), columns = data.columns)
    for i in range(len(paths)):
        paths.loc[i] = base_data
        paths.loc[i] += nus[i]
    return paths

index_used = 1
paths = simul_constant(scenar_index = index_used)

summed, base = paths.mean(axis = 0), final_df.loc[index2scenar[index_used]]
print((summed - base).sum())

#with pd.ExcelWriter('Data/simul.xlsx', mode='a', if_sheet_exists = "overlay") as writer:  
#    paths.to_excel(writer, sheet_name = index2scenar[index_used])
    
#%%
three_scenar = ["Current Policies", "Fragmented World", "Net Zero 2050"]
#%%

central_std = Config.CENTRAL_STD
beta = Config.BETA
nus = Config.NUS
sigmas = Config.SIGMAS

def simul_parameters(central_std, beta, nus, sigmas, scenar_index = index_used, scenar_name = None, mus = final_df):
    n = len(sigmas)
    matrix = central_std * np.ones((n,n))
    matrix += np.diag(sigmas)
    if scenar_name is not None:
        locmu = scenar_name
    else:
        locmu = index2scenar[index_used]
    
    mu_old = mus.loc[locmu, 2023]
    di = multivariate_normal(mean = nus + mu_old, cov = matrix).rvs()
    
    dis = pd.DataFrame({2023: di})
    
    new_dis = {}
    for col in mus.columns[mus.columns > 2023].values:
        mu_new = mus.loc[locmu, col]
        dt = np.mean(di)
        
        matrix = (central_std + beta * (dt - mu_old)**2) * np.ones((n,n))
        matrix += np.diag(sigmas)
        di = multivariate_normal(mean = nus + mu_new, cov = matrix).rvs()
        
        new_dis[col] = di
        
        mu_old = mu_new
    new_dis_df = pd.DataFrame(new_dis)
    dis = pd.concat([dis, new_dis_df], axis=1)
        
    return(dis)

for scenar_used in three_scenar:
    
    dis = simul_parameters(central_std, beta, nus, sigmas, scenar_name = scenar_used)

    with pd.ExcelWriter('Data/fixed_params.xlsx', mode='a', if_sheet_exists = "overlay") as params_writer:  
        dis.to_excel(params_writer, sheet_name = scenar_used)
    
#%% Simulate fake scenarios

fake_mus = final_df.loc[three_scenar].copy()

end = 2021 + len(Config.MUS_NZ)

new_columns = pd.DataFrame(0.0, index=fake_mus.index, columns=range(2051, end))

fake_mus = pd.concat([fake_mus, new_columns], axis=1)

    
fake_mus.loc["Current Policies", 2021:] = Config.MUS_CURPO
fake_mus.loc["Fragmented World", 2021:] = Config.MUS_FW
fake_mus.loc["Net Zero 2050", 2021:] = Config.MUS_NZ

# =============================================================================
# with pd.ExcelWriter('Data/fake_scenarios.xlsx', mode='w') as fake_writer:  
#     fake_mus.to_excel(fake_writer)
#     
# #scenar_used = "Net Zero 2050"
# for scenar_used in three_scenar:
#     fake_dis = simul_parameters(central_std, beta, nus, sigmas, scenar_name = scenar_used, mus = fake_mus)
#     
#     with pd.ExcelWriter('Data/fake_simul.xlsx', mode='a', if_sheet_exists = "overlay") as params_writer:  
#         fake_dis.to_excel(params_writer, sheet_name = scenar_used)
# =============================================================================