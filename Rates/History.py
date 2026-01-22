# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:03:41 2025

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd

data_file = "Data/stoxx_1311.xlsx"

history = pd.read_excel(data_file, index_col = 0)

df_valid = history[history["Financial Period Absolute"].notna()]

instruments = df_valid["Instrument"].unique()

history = history[history["Instrument"].isin(instruments)]

history['Year'] = history['Financial Period Absolute'].str.extract(r'(\d+)').astype(str)

history["NACE Classification"] = history.groupby("Instrument")["NACE Classification"].ffill()
history["GICS Sector Name"] = history.groupby("Instrument")["GICS Sector Name"].ffill()

history = history[history["Year"].astype(int) >= 2012]

history = history.drop_duplicates(["Instrument", "Year"], keep = 'last')

def filter_jumps(history):
    
    cols = ['CO2 Equivalent Emissions Direct, Scope 1',	'CO2 Equivalent Emissions Indirect, Scope 2']

    epsilon = 1e1 
    threshold = 1e3  
    
    df = history.copy()

    df["Year"] = df["Year"].astype(int)
    
    df = df.sort_values(["Instrument", "Year"]).reset_index(drop=True)
    
    df_before = df.copy()

    def detect_double_jump(group):
    
        for c in cols:
            values = group[c].values.astype(float)
    
            ratios = []
            for i in range(len(values) - 1):
                v1, v2 = values[i], values[i+1]
                if pd.isna(v1) or pd.isna(v2):
                    ratios.append(np.nan)
                    continue
                ratios.append((max(v1, v2) + epsilon) / (min(v1, v2) + epsilon))
            ratios = np.array(ratios)
    
            for i in range(1, len(values) - 1):
                jump_before = ratios[i-1] >= threshold
                jump_after = ratios[i] >= threshold
                
                if jump_before and jump_after:
                    values[i] = np.nan
    
            group[c] = values
    
        return group


    dfa = df.groupby("Instrument", group_keys=False).apply(detect_double_jump)
    
    def jump_last_year(group):
        if group["Year"].nunique() <= 2:
            return group
        
        years_sorted = np.sort(group["Year"].unique())
    
        last_year = years_sorted[-1]
        prev_year = years_sorted[-2]
    
        for c in cols:
            v_prev = group.loc[group["Year"] == prev_year, c].values[0]
            v_last = group.loc[group["Year"] == last_year, c].values[0]
    
            if pd.isna(v_prev) or pd.isna(v_last):
                continue
    
            ratio = (max(v_prev, v_last) + epsilon) / (min(v_prev, v_last) + epsilon)
    
            if ratio >= threshold:
                group.loc[group["Year"] == last_year, c] = np.nan
    
        return group


    df = dfa.groupby("Instrument", group_keys=False).apply(jump_last_year)
    
    def changed(a, b):
        return (
            (~a.isna() & b.isna())
            | (a.isna() & ~b.isna())
            | (~a.isna() & ~b.isna() & (a != b))
        )
    
    mask_changes = pd.DataFrame({
        c: changed(df_before[c], df[c])
        for c in cols
    })
    
    df["Year"] = df["Year"].astype(str)
    
    modified = df.loc[mask_changes.any(axis=1), "Instrument"].unique()
    
    return(df, modified)

history, modified = filter_jumps(history)

history["Scope12"] = history["CO2 Equivalent Emissions Direct, Scope 1"] + history["CO2 Equivalent Emissions Indirect, Scope 2"]
history["Scope123"] = history["Scope12"] + history["CO2 Equivalent Emissions Indirect, Scope 3"]
history['Scope12'] = pd.to_numeric(history['Scope12'], errors='coerce')
history['Scope123'] = pd.to_numeric(history['Scope123'], errors='coerce')

history.to_excel("Data/history_processed.xlsx")

emission_cols = ['CO2 Equivalent Emissions Direct, Scope 1',	'CO2 Equivalent Emissions Indirect, Scope 2', 'Scope12']

keep_cols = ['GICS Sector Name'] + emission_cols

last = history[history["Year"] == '2023'][keep_cols]

last = last.groupby('GICS Sector Name').sum().sort_values('Scope12', ascending = False)

grouped = (
    history.groupby(["GICS Sector Name", "Year"])["Scope12"]
    .sum()
    .reset_index()
)

pivot = grouped.pivot(index="GICS Sector Name", columns="Year", values="Scope12")

pivot = pivot.loc[:, [y for y in pivot.columns if int(y) <= 2023]]

pivot.columns = pivot.columns.astype(str)

pivot = pivot.reindex(last.index)

pivot.index = last.index

pivot.to_excel("Data/history_sums.xlsx")

last.to_excel("Data/history_last.xlsx")