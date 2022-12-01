"""
Runs tau statistics on given PIFF reserve star catalog and galaxy shear catalog 
on each band separately or combined.
To calculate on combined band data, use string with multiple bands.
That is: bands = ['griz'] will calculate the tau stats over the combined griz catalog.
         bands = ['g', 'r', 'i', 'z'] will calculate tau stats over each band individually.
"""

import fitsio
import numpy as np
import os, sys
# from tqdm import tqdm
from matplotlib import pyplot as plt
from rho_stats import (measure_tau_mpi, write_stats,
                       write_stats_tau, plot_overall_tau)

bands = ['riz'] # for plot_overall_tau function
band = bands[0] # for specifying file name
name = 'y6a2_piff_v3_allres_v3'
ver = 2

cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'

piff_fn = os.path.join(cat_dir, 'y6a2_piff_v3_allres_v3_taustat_input_v2.fits')
# mdet_fn = '/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/metadetection_v2.fits'
mdet_fn = os.path.join(cat_dir, 'y6a2_mdet_v2_response_corrected.fits')
patch_fn = ('/global/cfs/cdirs/des/y6-shear-catalogs/'
            'patches-centers-altrem-npatch200-seed9999.fits')

# piff_cols = ['RA', 'DEC', 'T_DATA', 'T_MODEL', 'DELTA_T', 
#              'G1_DATA', 'G1_MODEL', 'G2_DATA', 'G2_MODEL',
#              'DELTA_G1', 'DELTA_G2', 'G1_X_DELTAT', 'G2_X_DELTAT']


#load metadetect catalog and correct for global shear response
# FIXME: correct for response in mdet catalog

# get patch_centers file

max_sep = 250
work_dir = ('/global/cfs/cdirs/des/schutt20/y6_psf/'
            'y6a2_piff_testing/tau_stats_output')
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

print('Computing tau statistics...')
stats = measure_tau_mpi(piff_fn, mdet_fn, patch_fn, max_sep=max_sep,
                        output_dir=work_dir, version=ver)
print('Computation complete.')
# If doing subset of taus, write_stats_tau and plot_overall tau won't
# work!
# TODO: write a write_stats_tau that can take the fits files and make
# the .json file from those.

stat_file = os.path.join(work_dir, "tau_%s_%s_%i.json"%(name,band,ver))
print('Computation complete. Writing stats to file: %s'%stat_file)
write_stats_tau(stat_file,*stats)
print('Wrote stats to file.')

 print('Plotting tau statistics and writing to file...')
 plot_overall_tau(work_dir, name, bands, ver)
 print('Plotting complete.')
