# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:24:07 2025

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np

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
df_long = filtered.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
    var_name="Year",
    value_name="Value"
)


df_long["Year"] = df_long["Year"].astype(int)
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long = df_long.infer_objects()  # Avoid warning

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

df_interp = (
    df_long.groupby(["Model", "Scenario", "Region", "Variable", "Unit"], group_keys=False)
    .apply(interpolate_group)
)

df_interp = df_interp[
    ["Model", "Scenario", "Region", "Variable", "Unit", "Year", "Value"]
]

#print(df_interp.head(20))

#%%
scenario_data = df_interp.copy()
scenario_data["Ratio"] = scenario_data.groupby(['Model', 'Scenario', 'Region', 'Variable'])['Value']\
                 .transform(lambda x: x / x.shift(1))
scenario_data.drop({"Model", "Region", "Unit", "Value"}, axis = 1, inplace = True)