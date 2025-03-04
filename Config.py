# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:23:29 2025

@author: LoïcMARCADET
"""
import numpy as np

CENTRAL_STD = 0.05
BETA = 0.0001
NUS = np.array([0.2, -0.3, 0.1, -0.05, 0.15, -0.1, -0.15, 0.3, -0.10, -0.05])
SIGMAS = 0.001 * np.ones(len(NUS))

N_YEARS = 500

MUS_CURPO = np.linspace(0.97, 5.0, N_YEARS)
MUS_FW = np.linspace(0.97, 0, N_YEARS)
MUS_NZ = np.linspace(0.97, -5.0, N_YEARS)

SCENAR2INDEX = {"Below 2°C": 0, "Current Policies" : 1, "Delayed transition" : 2,
               "Fragmented World" : 3, "Low demand": 4, 
               "Nationally Determined Contributions (NDCs)":5, 
               "Net Zero 2050":6}

INDEX2SCENAR = {0 : "Below 2°C", 1: "Current Policies", 2: "Delayed transition",
               3:"Fragmented World", 4: "Low demand", 
               5: "Nationally Determined Contributions (NDCs)", 
               6:"Net Zero 2050"}

THREE_SCENAR = ["Current Policies", "Fragmented World", "Net Zero 2050"]

 