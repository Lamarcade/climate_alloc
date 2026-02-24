# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:49:49 2026

@author: LoïcMARCADET
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import re
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
sns.set(style="darkgrid")
#%%

folder = "Data/NGFS5/"
file = "IAM_data.xlsx"

ngfs = pd.read_excel(folder + file)

# Approximation EU-15, lacks Norway, Switzerland and Poland for CAC40
filtered = ngfs[ngfs["Region"] == "GCAM 6.0 NGFS|EU-15"]

indic = "Kyoto"
filtered = filtered[filtered["Variable"].str.contains(indic, case=False, na=False)]

def reshape(df):

    df["Sector"] = df["Variable"].apply(lambda x: x.split("|")[-1] if "|" in x else "Global")
    
    year_cols = [col for col in df.columns if col.isdigit()]
    cols_to_keep = ["Model", "Scenario", "Sector"] + year_cols
    
    df_filtered = df[cols_to_keep].copy()
    
    return df_filtered

sectors = reshape(filtered)

def interpolate_years(df, start=2020, end=2050, context_cols = ["Scenario", "Sector"]):
    
    year_cols = sorted([col for col in df.columns if col.isdigit() and start <= int(col) <= end])
    target_years = [str(y) for y in range(start, end + 1)]
    
    def interpolate_row(row):
        series = row[year_cols].astype(float)
        interpolated = pd.Series(index=target_years, dtype=float)
        interpolated.loc[year_cols] = series.values
        interpolated = interpolated.interpolate(method='linear')
        return interpolated
    
    interpolated_df = df.apply(interpolate_row, axis=1)
    
    final_df = pd.concat([df[context_cols].reset_index(drop=True), interpolated_df.reset_index(drop=True)], axis=1)
    return final_df

kyoto = interpolate_years(sectors)

#kyoto = kyoto[kyoto["Scenario"] != 'Low demand']

restricted = kyoto[kyoto["Sector"] == "Kyoto Gases"]

#%%

year_cols = [c for c in restricted.columns if str(c).isdigit()]
year_cols = sorted(year_cols, key=int)

log_df = restricted.copy()
log_df[year_cols] = np.log(log_df[year_cols])

dlog_df = restricted[["Scenario", "Sector"]].copy()

dlog = restricted.copy()
dlog = dlog.drop(columns = "Sector")
dlog[year_cols] = np.log(dlog[year_cols]).diff(axis=1)

#dlog.to_excel("Data/scenarios.xlsx")

scelog = dlog.drop(columns = "2020")
scelog = scelog.set_index("Scenario")

#scelog.to_excel("Data/logscenarios.xlsx")

#%%
ngfs_costs = pd.read_csv("../Data/ngfs_cc.csv", index_col = "Scenario")
ngfs_costs = ngfs_costs.dropna()

y_cols = [col for col in ngfs_costs.columns if col.isdigit()]

costs = ngfs_costs.copy()
costs = costs.rename(index={"Below 2?C": "Below 2°C"})

# $1 in 2010 is worth $1.44 in 2024 ??
#costs[year_cols] *= 1.44

costs = costs[y_cols]

costs = costs.reset_index()
costs = interpolate_years(costs, context_cols = "Scenario", start = 2025)

#%%

kyoto_corr = restricted.set_index("Scenario")
kyoto_corr = kyoto_corr.loc[:,kyoto_corr.columns.isin(year_cols)]
kyoto_corr = kyoto_corr.loc[:, kyoto_corr.columns >= "2025"]

costs_corr = costs.set_index("Scenario")

# Scenario-dependent correl
k, c = kyoto_corr.align(costs_corr, join="inner", axis=None)

corr_by_scenario = k.apply(lambda row: row.corr(c.loc[row.name]), axis=1)
corr_by_scenario.name = "corr"
corr_by_scenario

# Correl by year

corr_by_year = pd.Series(
    {year: k[year].corr(c[year]) for year in k.columns},
    name="corr"
)
corr_by_year

#
x = k.stack(dropna=True)   # MultiIndex (Scenario, Year)
y = c.stack(dropna=True)

xy = x.to_frame("kyoto").join(y.to_frame("costs"), how="inner")

corr_global = xy["kyoto"].corr(xy["costs"], method="pearson")
corr_global

