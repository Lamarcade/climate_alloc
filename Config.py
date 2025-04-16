# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:23:29 2025

@author: LoïcMARCADET
"""
import numpy as np

CENTRAL_STD = 10
BETA = 0.1
NUS = np.array([2, -3, 1, -0.5, 1.5, -1.0, -1.5, 3.0, -1.0, -0.5])*1
SIGMAS = 100 * np.ones(len(NUS))

N_YEARS = 500
FUTURE_START_YEAR = 2023
START_YEAR = 2009

MUS_CURPO = 100 * np.linspace(0.97, 5.0, N_YEARS)
MUS_FW = 100 * np.linspace(0.97, 0, N_YEARS)
MUS_NZ = 100 * np.linspace(0.97, -5.0, N_YEARS)

SCENAR2INDEX = {"Below 2°C": 0, "Current Policies" : 1, "Delayed transition" : 2,
               "Fragmented World" : 3, "Low demand": 4, 
               "Nationally Determined Contributions (NDCs)":5, 
               "Net Zero 2050":6}

INDEX2SCENAR = {0 : "Below 2°C", 1: "Current Policies", 2: "Delayed transition",
               3:"Fragmented World", 4: "Low demand", 
               5: "Nationally Determined Contributions (NDCs)", 
               6:"Net Zero 2050"}

THREE_SCENAR = ["Current Policies", "Fragmented World", "Net Zero 2050"]
INDEX3 = {0 : "Current Policies", 1: "Fragmented World", 2: "Net Zero 2050"}


 