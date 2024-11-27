# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:42:21 2024

@author: LoïcMARCADET
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme()

stoxx = pd.read_excel("Data/Stoxx_data.xlsx")

#%% Company data

indicators = stoxx[["Instrument", "GICS Sector Name"]].copy()
for i in range(13, -1, -1):
    # Actual year / Former year from Y-13 to Y-0
    indicators["Rate Y-{i}".format(i = i)] = 100 * stoxx["Total Y-{i}".format(i = i)] / stoxx["Total Y-{j}".format(j = i+1)]
    
for i in range(13, -1, -1):
    # Total emissions / Revenue 
    indicators["Intensity Y-{i}".format(i = i)] =  stoxx["Total Y-{i}".format(i = i)] / stoxx["Revenue Y-{i}".format(i = i)]

#%% Sector aggregation
sectors = indicators.copy()
sectors.drop(sectors.columns[0], axis = 1, inplace = True)
sectors = sectors.groupby(by= "GICS Sector Name").mean()

sectors_rate = sectors[sectors.columns[:14]]
sectors_rate = sectors_rate.reset_index()

#%% Dense format to plot
def dense_format(df):
    df_dense = df.melt(id_vars="GICS Sector Name", 
                  var_name="Year", 
                  value_name="Intensity")

    df_dense["Year"] = df_dense["Year"].str.extract(r'Y-(\d+)').astype(int)
    df_dense = df_dense.sort_values(by="Year", ascending = False)
    df_dense["Year"] *= -1
    return df_dense

dense_rate = dense_format(sectors_rate)

sectors_intensity = sectors[sectors.columns[14:]]
sectors_intensity = sectors_intensity.reset_index()
dense_intensity = dense_format(sectors_intensity)

#%% Plots
def sector_rate(save = False):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=dense_rate, 
                 x="Year", 
                 y="Intensity", 
                 hue="GICS Sector Name", 
                 marker="o")
    
    plt.title("Sector mean carbon rate evolution", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Rate (%)", fontsize=12)
    plt.legend(title="GICS sector", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figs/rate.png")

def sector_intensity(save = False):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=dense_intensity, 
                 x="Year", 
                 y="Intensity", 
                 hue="GICS Sector Name", 
                 marker="o")
    
    plt.title("Sector mean carbon intensity evolution", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.legend(title="GICS sector", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figs/intensity.png")

def count_sectors():
    sector_counts = indicators["GICS Sector Name"].value_counts().reset_index()
    sector_counts.columns = ["GICS Sector Name", "Number of Companies"]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sector_counts, 
                x="Number of Companies", 
                y="GICS Sector Name", 
                palette="viridis")
    
    plt.title("#Companies in each GICS sector", fontsize=14)
    plt.xlabel("#Companies", fontsize=12)
    plt.ylabel("GICS Sector", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.show()
    #plt.savefig("Figs/sectors.png")