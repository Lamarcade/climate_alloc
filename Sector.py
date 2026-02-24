# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:07:44 2026

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import re

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

def interpolate_years(df, start=2020, end=2050):
    context_cols = ["Scenario", "Sector"]
    
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

#%%

# --- 1) reshape en long ---
year_cols = [c for c in log_df.columns if str(c).isdigit()]
df_long = log_df.melt(
    id_vars=["Scenario", "Sector"],
    value_vars=year_cols,
    var_name="Year",
    value_name="log_emissions"
)
df_long["Year"] = df_long["Year"].astype(int)

# (optionnel) filtrer un secteur si besoin
# sector = "Kyoto Gases"
# df_long = df_long[df_long["Sector"] == sector].copy()

# --- 2) plot des trajectoires ---
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=df_long.sort_values("Year"),
    x="Year",
    y="log_emissions",
    hue="Scenario",
    marker="o",
    linewidth=2,
    ax=ax
)

ax.set_title("Évolution des log-émissions par scénario")
ax.set_xlabel("")
ax.set_ylabel("log(émissions)")
ax.legend(title="Scénario", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
sns.despine()
plt.tight_layout()
plt.show()

# --- 3) matrice Y (années x scénarios) ---
Y = df_long.pivot_table(
    index="Year",
    columns="Scenario",
    values="log_emissions",
    aggfunc="mean"  # au cas où doublons
).sort_index()

# --- 4) covariance empirique des logs entre scénarios ---
SigmaY = Y.cov()   # covariance (n_scenarios x n_scenarios)
CorrY  = Y.corr()  # corrélation (souvent plus comparable)

# --- 5) heatmap covariance ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    SigmaY,
    cmap="rocket_r",
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Covariance empirique des log-émissions"},
    ax=ax
)
ax.set_title("Heatmap — Covariance des log-émissions (estimée sur les années)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- (bonus) heatmap corrélation ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    CorrY,
    vmin=0.3, vmax=1,
    cmap="vlag",
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Corrélation empirique des log-émissions"},
    ax=ax
)
ax.set_title("Heatmap — Corrélation des log-émissions (estimée sur les années)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%%

sec_long = kyoto.melt(
    id_vars=["Scenario", "Sector"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Emissions"
)
sec_long["Year"] = sec_long["Year"].astype(int)

for scen in sec_long["Scenario"].unique():
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=sec_long[sec_long["Scenario"] == scen].sort_values("Year"),
        x="Year",
        y="Emissions",
        hue="Sector",
        marker="o",
        linewidth=2,
        ax=ax
    )
    
    ax.set_title(f"Emissions sectorielles projetées {scen}")
    ax.set_xlabel("")
    ax.set_ylabel("Emissions tCO2eq")
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    sns.despine()
    plt.tight_layout()
    plt.show()
