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
    indicators["Rate Y-{i}".format(i = i)] = stoxx["Total Y-{i}".format(i = i)] / stoxx["Total Y-{j}".format(j = i+1)]
    
for i in range(13, -1, -1):
    # Total emissions / Revenue 
    indicators["Intensity Y-{i}".format(i = i)] =  stoxx["Total Y-{i}".format(i = i)] / stoxx["Revenue Y-{i}".format(i = i)]

#%% Sector aggregation
sectors = indicators.copy()
sectors.drop(sectors.columns[0], axis = 1, inplace = True)
sectors = sectors.groupby(by= "GICS Sector Name").mean()


#%% Dense format to plot
def dense_format(df, value_name, is_sector = True):
    
    id_vars = "GICS Sector Name" if is_sector else "Instrument"
    df_dense = df.melt(id_vars=id_vars, 
                  var_name="Year", 
                  value_name=value_name)

    df_dense["Year"] = df_dense["Year"].str.extract(r'Y-(\d+)').astype(int)
    df_dense = df_dense.sort_values(by="Year", ascending = False)
    df_dense["Year"] *= -1
    return df_dense


sectors_rate = sectors[sectors.columns[:14]]
sectors_rate = sectors_rate.reset_index()
dense_rate = dense_format(sectors_rate, "Rate")

sectors_intensity = sectors[sectors.columns[14:]]
sectors_intensity = sectors_intensity.reset_index()
dense_intensity = dense_format(sectors_intensity, "Intensity")

#%% Sample sector and companies
def select_sectors(df, sector, outliers = None):
    new_df = df.copy()
    new_df = new_df[new_df["GICS Sector Name"] == sector]
    if outliers is not None:
        for outlier in outliers:
            new_df = new_df[new_df["Instrument"] != outlier]

    mean_row = new_df[new_df.columns[2:]].mean(axis = 0)
    mean_row['Instrument'] = 'Mean'
    mean_row['GICS Sector Name'] = 'Consumer Staples'
    new_df = pd.concat([new_df, mean_row.to_frame().T], ignore_index=True)
    return new_df

def rate_or_int(df, rate = True):
    if rate:
        new_df = pd.concat([df["Instrument"], df[df.columns[2:16]]], axis = 1)
        #new_df = new_df.reset_index()
        dense_df = dense_format(new_df, "Rate", is_sector = False)
    else:
        new_df = pd.concat([df["Instrument"], df[df.columns[16:]]], axis = 1)
        #new_df = new_df.reset_index()
        dense_df = dense_format(new_df, "Intensity", is_sector = False)
    return dense_df

df_cs = select_sectors(indicators, "Consumer Staples")
df_en = select_sectors(indicators, "Energy")
df_fi = select_sectors(indicators, "Financials", outliers = ["MWDP.PA"])


rate_cs = rate_or_int(df_cs, True)
int_cs = rate_or_int(df_cs, False)

rate_en = rate_or_int(df_en, True)
int_en = rate_or_int(df_en, False)

rate_fi = rate_or_int(df_fi, True)
int_fi = rate_or_int(df_fi, False)

#%% Plots

def history_plot(df, y, is_sector = False, title = "", save = False, filename = None):
    plt.figure(figsize=(12, 6))
    
    if is_sector:
        sns.lineplot(data=df, x="Year", y=y, hue="GICS Sector Name", marker="o", legend = False)
    else:
        df_no_mean = df[df['Instrument'] != 'Mean']  
        df_mean = df[df['Instrument'] == 'Mean']   
        sns.lineplot(data=df_mean, x='Year', y=y, color='red', marker='*', linestyle = "", label='Mean', markersize=15)
        sns.lineplot(data=df_no_mean, x='Year', y=y, color='blue', hue = "Instrument", 
                     legend = False,linewidth=0.8, alpha=0.4)

    plt.title(title, fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(y, fontsize=12)
    if is_sector:
        plt.legend(title="GICS sector", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"Figs/{filename}.png")

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
    if save:
        plt.savefig("Figs/intensity.png")

def count_sectors(save):
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
    if save:
        plt.savefig("Figs/sectors.png")
        
#%% 
#history_plot(rate_cs, "Rate", is_sector = False, title = "Consumer staples companies carbon rate evolution", save = True, filename = "ratecs")
#history_plot(int_cs, "Intensity", is_sector = False, title = "Consumer staples companies carbon intensity evolution", save = True, filename = "intcs")

#history_plot(rate_en, "Rate", is_sector = False, title = "Energy companies carbon rate evolution", save = True, filename = "rateen")
#history_plot(int_en, "Intensity", is_sector = False, title = "Energy companies carbon intensity evolution", save = True, filename = "inten")

history_plot(rate_fi, "Rate", is_sector = False, title = "Financial companies carbon rate evolution", save = True, filename = "ratefinooutlier")
history_plot(int_fi, "Intensity", is_sector = False, title = "Financial companies carbon intensity evolution", save = True, filename = "intfinooutlier")