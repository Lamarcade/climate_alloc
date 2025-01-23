# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:24:07 2025

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

sectors = ["Communication Services", "Consumer Discretionary",
           "Consumer Staples", "Energy", "Financials",
           "Health Care", "Industrials", 
           "Information Technology", "Materials",
           "Utilities"] 

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
                 .transform(lambda x: x / x.shift(1))
scenario_data.drop({"Model", "Region", "Unit", "Value", "Variable"}, axis = 1, inplace = True)


sc = scenario_data.pivot(index=["Scenario"], columns="Year", values="Ratio")

#%%
df_plot = sc.reset_index().melt(id_vars="Scenario", var_name="Year", value_name="Ratio")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x="Year", y="Ratio", hue="Scenario", marker="o")

plt.title("Carbonation rate for each scenario", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("CR", fontsize=14)
plt.legend(title="Scenario", fontsize=12)
plt.grid(True)
plt.show()

#%% Simulate data for sectors

