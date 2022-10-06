import os
import sys
import fitsio
import numpy as np
from rho_stats_v4 import plot_overall_rho

name = 'y6a2_piff_v3_allres_v3'
work = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/rho_stats_output'
if not os.path.isdir(work):
    os.mkdir(work)
bands = 'griz'

plot_overall_rho(work, name, bands=bands)
