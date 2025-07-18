# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:10:13 2025

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import Config

#%%

data_file = "Data/Stoxx_data_Scope12.xlsx"

df = pd.read_excel(data_file)
df = df.drop(df.columns[0], axis = 1)

start_year = 2009

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

emissions = df[[f"Scope12 Y-{i}" for i in range(14, -1, -1)] + ["GICS Sector Name"]].groupby(by = "GICS Sector Name").sum()
for i in range(14, -1, -1):
    emissions.rename(columns = {f"Scope12 Y-{i}": 2023-i}, inplace = True)
    
path = "Data/scenarios.xlsx"
date_max = 2051
rates = pd.read_excel(path)
    
scenar_dict = {i: index for i, index in enumerate(rates["Scenario"])}

Time = date_max - start_year

mus = pd.DataFrame(np.zeros((7,Time)))

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

murates = annual_mean_df

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
T0 = indicators.shape[1] 
mus.columns = [start_year + i for i in range(mus.shape[1])]

for t in range(T0):
    mus.iloc[:, t] = overall_mean_decarbonation_rate * np.ones(7) 

# Begin at 2021
mus.loc[:, rates.columns[2:date_max]] = rates 

#%%
#dict_emissions = {d: emissions.copy() for d in Config.SCENAR2INDEX.keys()}

nem = np.array(emissions)
nem = nem[np.newaxis, :]
nem = np.repeat(nem, repeats=7, axis=0)

# indice 14 ~ 2023

for k in range(7):
    for j, t in enumerate(range(2024, date_max)):
        nem[k,:,j+14] = nem[k,:,j+13] * mus[t] / 100 + nem[k,:,j+13]
    # Avoid fragmentation
    nem = nem.copy()