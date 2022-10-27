"""
Runs rho statistics on given PIFF reserve star catalog on each band separately or combined.
To calculate on combined band data, set `bands` as a multi-letter string.
That is: bands = ['griz'] will calculate the rho stats over the combined griz catalog.
         bands = ['g', 'r', 'i', 'z'] will calculate rho stats over each band individually.
"""
import os
import sys
import fitsio
import numpy as np
from rho_stats import measure_rho, write_stats, plot_overall_rho

filename = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/y6a2_piff_v3_allres_v3_collated.fits'
cat = fitsio.read(filename,
                  columns=['BAND', 'PSF_FWHM', 'RA', 'DEC', 'FLUX', 'T_DATA', 'T_MODEL',
                           'G1_DATA', 'G1_MODEL', 'G2_DATA', 'G2_MODEL'])

cat = cat[cat['PSF_FWHM'] < 1.5]

max_sep = 250
max_mag = -1
cattype='hsm'
name = 'y6a2_piff_v3_allres_v3'
work = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/rho_stats_output'
if not os.path.isdir(work):
    os.mkdir(work)
bands = ['griz', 'riz']

for band in bands:
    print('Band: ', band)
    if len(band) > 1:
        data = cat[cat['BAND'] in band]
    else:
        data = cat[cat['BAND'] == band]
    stats = measure_rho(data, max_sep, max_mag, cattype=cattype, subtract_mean=True, do_rho0=True)
    stat_file = os.path.join(work, "rho_%s_%s.json"%(name,band))
    write_stats(stat_file,*stats)

plot_overall_rho(work, name, bands=bands)
