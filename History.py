# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:17:14 2025

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np

history = pd.read_excel("Data/Stoxx_revenuedate.xlsx", index_col = 0)

history["CDPScope12"] = history["CDP CO2 Equivalent Emissions Direct Scope 1"] + history["CDP CO2 Equivalent Emissions Indirect Scope 2"]
history["CDPScope123"] = history["CDP CO2 Equivalent Emissions Direct Scope 1"] + history["CDP CO2 Equivalent Emissions Indirect Scope 2"] + history["CDP CO2 Equivalent Emissions Indirect Scope 3"]

history["Scope12"] = history["CO2 Equivalent Emissions Direct, Scope 1"] + history["CO2 Equivalent Emissions Indirect, Scope 2"]
history["Scope123"] = history["CO2 Equivalent Emissions Direct, Scope 1"] + history["CO2 Equivalent Emissions Indirect, Scope 2"] + history["CO2 Equivalent Emissions Indirect, Scope 3"]

history.dropna(subset=["Financial Period Absolute"], inplace = True)
history['Year'] = history['Financial Period Absolute'].str.extract(r'(\d+)').astype(int)

history['GICS Sector Name'] = history.sort_values(['Instrument', 'Year']) \
                           .groupby('Instrument')['GICS Sector Name'] \
                           .ffill()


history['GICS Sector Name'] = history['GICS Sector Name'].fillna("Unknown")

def average_emissions(df, col = "Scope12", sector = 'GICS Sector Name', intensity = False):

    sector_counts = df.groupby(sector)['Instrument'].nunique()


    renamed_sector = {s: f"{s} (n={sector_counts[s]})" for s in sector_counts.index}
    
    
    df = df.copy()
    df[sector] = df[sector].map(renamed_sector)
    df = df.drop_duplicates(subset= ['Instrument', 'Scope12', 'CDPScope12', 'Year'])

    avg_scope12_by_sector_year = df.groupby(['Year', sector])[col].mean().unstack()
    sum_total_emissions_by_sector_year = df.groupby(['Year', sector])[col].sum(min_count=1).unstack()

    return avg_scope12_by_sector_year, sum_total_emissions_by_sector_year

averaged, summed = average_emissions(history, col = "CDPScope12")

summed.T.to_excel("Data/history_sums.xlsx")