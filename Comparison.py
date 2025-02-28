# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:01:45 2025

@author: LoïcMARCADET
"""


#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Modelnu import Modelnu

sns.set_theme()

stoxx = pd.read_excel("Data/Stoxx_data.xlsx")

scenarios = pd.read_excel("Data/scenarios.xlsx")


#%% 
initial_law = np.array([0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.15])
mm = Modelnu(18, initial_law = initial_law)
mm.rename_rates()

fi = mm.indicators

#%%

scenar_plot = scenarios.set_index("Scenario")
years = [2021, 2022, 2023]
fi_plot = fi[years]
scenar_plot = scenar_plot[years]

# --- Tracer les évolutions ---
plt.figure(figsize=(10, 6))

# Palette de couleurs pour les secteurs
palette = sns.color_palette("tab10", n_colors=len(fi_plot))

# Tracer les secteurs
for i, (sector, values) in enumerate(fi_plot.iterrows()):
    plt.plot(values.index, values.values, marker="o", label=sector, color=palette[i])


# --- Mise en forme ---
plt.title("Évolution des valeurs par secteur (2021-2023)", fontsize=14)
plt.xlabel("Année", fontsize=12)
plt.ylabel("Valeur", fontsize=12)
plt.xticks([2021, 2022, 2023])
plt.legend(loc="best", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# --- Afficher le graphique ---
plt.show()
    
#%%
plt.figure(figsize=(10, 6))

# Palette de couleurs pour les secteurs
palette = sns.color_palette("tab10", n_colors=len(fi_plot))
# Tracer les scénarios avec un style différent
scenario_styles = [(0, (1, 10)), (0, (1, 5)), (0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1))]
for i, (scenario, values) in enumerate(scenar_plot.iterrows()):
    plt.plot(values.index, values.values, linestyle = scenario_styles[i], linewidth=2, label=scenario)
    
    
# --- Mise en forme ---
plt.title("Évolution des valeurs par scénario (2021-2023)", fontsize=14)
plt.xlabel("Année", fontsize=12)
plt.ylabel("Valeur", fontsize=12)
plt.xticks([2021, 2022, 2023])
plt.legend(loc="best", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# --- Afficher le graphique ---
plt.show()