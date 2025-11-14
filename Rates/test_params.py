# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:02:03 2025

@author: LoïcMARCADET
"""

mm = Model(14)


fi = mm.indicators
p,q = fi.shape

nn = np.ones(6)
nn[3:] -= 2 * np.ones(3)
np.sum(nn)
mm.initialize_parameters(0.1, 0.5, nn, 1 * np.ones(6)

)
mm.EM(mm.indicators.loc[:,mm.indicators.columns[2:]], n_iter = 2)