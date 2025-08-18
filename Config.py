# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:23:29 2025

@author: LoïcMARCADET
"""
import numpy as np
import pandas as pd

CENTRAL_STD = 10
BETA = 0.1
NUS = np.array([2, -3, 1, -0.5, 1.5, -1.0, -1.5, 3.0, -1.0, -0.5])*1
SIGMAS = 100 * np.ones(len(NUS))

N_YEARS = 500
FUTURE_START_YEAR = 2023
START_YEAR = 2009

MUS_CURPO = np.linspace(0, 500, N_YEARS)
MUS_FW = np.linspace(0, 0, N_YEARS)
MUS_NZ = np.linspace(0, -500, N_YEARS)

SCENAR2INDEX = {"Below 2°C": 0, "Current Policies" : 1, "Delayed transition" : 2,
               "Fragmented World" : 3, "Low demand": 4, 
               "Nationally Determined Contributions (NDCs)":5, 
               "Net Zero 2050":6}

INDEX2SCENAR = {0 : "Below 2°C", 1: "Current Policies", 2: "Delayed transition",
               3:"Fragmented World", 4: "Low demand", 
               5: "Nationally Determined Contributions (NDCs)", 
               6:"Net Zero 2050"}

INDEX2ABB= {0 : "B2C", 1: "CP", 2: "DelT",
               3:"FW", 4: "LowD", 
               5: "NDC", 
               6:"NZ"}

THREE_SCENAR = ["Current Policies", "Fragmented World", "Net Zero 2050"]
INDEX3 = {0 : "Current Policies", 1: "Fragmented World", 2: "Net Zero 2050"}

#EMI = [12142172.555,45419671.249,35427299.17,252341275.3,
#4476767.77,14422595.09,153641770.68, 4024171.59512,
#522426997.0,1887437.35,251314998.23]

GICS = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples',
       'Energy', 'Financials', 'Health Care', 'Industrials',
       'Information Technology', 'Materials', 'Real Estate', 'Utilities']

#EM_LAST = pd.DataFrame(EMI, index = GICS)

HISTORY_EM = pd.read_excel("Data/history_sums.xlsx", index_col = 0)
LAST_EM = pd.DataFrame(HISTORY_EM.loc[:,2022])

DF_ORDER = LAST_EM.copy()
DF_ORDER["Sector"] = DF_ORDER.index.str.replace(r"\s*\(n=\d+\)", "", regex=True)
DF_ORDER = DF_ORDER[DF_ORDER["Sector"] != "Real Estate"]

# Historical rates ordered 
HISTO_ORDER = DF_ORDER.sort_values(2022, ascending=False)["Sector"].tolist()

#OLD_HISTO_ORDER = ['Materials','Consumer Discretionary','Utilities','Communication Services',
# 'Industrials','Information Technology','Consumer Staples','Financials','Energy','Health Care']

# Sort by descending order
NUS_2 = np.array([2.5, -3, 1, 0, 1.5, -1.5, -2, 3.0, -1.0, -0.5])*1
NUS_FIXED = -np.sort(-NUS_2)
NUS_ORDER = pd.DataFrame(NUS_FIXED, index = HISTO_ORDER, columns = ["Spreads"])

SIGMAS_ORDER = pd.DataFrame(SIGMAS, index = HISTO_ORDER, columns = ["Variances"])
