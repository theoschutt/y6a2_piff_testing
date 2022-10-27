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
from rho_stats import measure_tau, write_stats, write_stats_tau, plot_overall_tau

bands = ['griz', 'riz']
name = 'y6a2_piff_v3_allres_v3'
ver = 1

cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'
cattype = 'hsm'
piff_fn = os.path.join(cat_dir, '%s_collated.fits'%name)
piff_cols = ['RA', 'DEC', 'T_DATA', 'T_MODEL', 'G1_DATA', 'G1_MODEL', 'G2_DATA', 'G2_MODEL']


#load metadetect catalog and correct for global shear responsez
mdet_fn = '/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/metadetection_v2.fits'
print('Loading metadetect catalog: %s'%mdet_fn)
mdet_input_flat = fitsio.read(mdet_fn)
mdet_input_flat['g1'] *= 1/mdet_input_flat['R_all']
mdet_input_flat['g2'] *= 1/mdet_input_flat['R_all']
print('Load complete. Total rows: %s'%len(mdet_input_flat))

#limit catalogs for testing purposes
# piff_cat = piff_cat[:100000]
# mdet_input_flat = mdet_input_flat[:100000]

max_sep = 250
max_mag = -1
work_dir = '/global/cfs/cdirs/des/schutt20/y6_psf/y6a2_piff_testing/tau_stats_output'
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)
    
band = 'riz'
# load in rows for given bands with PSF_FWHM < 1.5"
row_idx = os.path.join(cat_dir, '%s_fwhm_lt1.5_%s_10M_idx.txt'%(name,band))
print('Loading PSF catalog indices for band(s): %s'%band)
print('Using index file: %s'%row_idx)
piff_rows = np.fromfile(row_idx, int)
print('Load complete. Total rows: %i'%len(piff_rows))

print('Loading PSF catalog with above indices: %s'%piff_fn)
piff_cat = fitsio.read(piff_fn, rows=piff_rows, columns=piff_cols)
print('Load complete.')

print('Computing tau statistics...')
stats = measure_tau(piff_cat, mdet_input_flat, max_sep, max_mag, 
                    work_dir, cattype=cattype, subtract_mean=True)
stat_file = os.path.join(work_dir, "tau_%s_%s_%i.json"%(name,band,ver))
print('Computation complete. Writing stats to file: %s'%stat_file)
write_stats_tau(stat_file,*stats)
print('Wrote stats to file.')

print('Plotting tau statistics and writing to file...')
plot_overall_tau(work_dir, name, bands, ver)
print('Plotting complete.')
