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

history["Scope12"] = history["CO2 Equivalent Emissions Direct, Scope 1"] + history["CO2 Equivalent Emissions Indirect, Scope 2"]
history["Scope123"] = history["Scope12"] + history["CO2 Equivalent Emissions Indirect, Scope 3"]
history['Scope12'] = pd.to_numeric(history['Scope12'], errors='coerce')
history['Scope123'] = pd.to_numeric(history['Scope123'], errors='coerce')

emission_cols = ['CO2 Equivalent Emissions Direct, Scope 1',	'CO2 Equivalent Emissions Indirect, Scope 2', 'Scope12']

keep_cols = ['GICS Sector Name'] + emission_cols

last = history[history["Year"] == '2023'][keep_cols]

last = last.groupby('GICS Sector Name').sum().sort_values('Scope12', ascending = False)

last = np.log(last)

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

pivot = np.log(pivot)

pivot.to_excel("Data/history_sums.xlsx")

last.to_excel("Data/history_last.xlsx")