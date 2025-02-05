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


 