"""
Runs rho statistics on given PIFF reserve star catalog on each band separately or combined.
To calculate on combined band data, set `bands` as a multi-letter string.
That is: bands = ['griz', 'riz'] will calculate the rho stats over the combined griz catalog and then the combined riz catalog.
         bands = ['g', 'r', 'i', 'z'] will calculate rho stats over each band individually.
"""
import os
import sys
import fitsio
import numpy as np
from rho_stats import measure_rho, write_stats, plot_overall_rho

filename = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/y6a2_piff_v3_allres_v3_collated.fits'
idx_file = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/y6a2_piff_v3_allres_v3_griz_incoadd_idx.npy'
print('Using PIFF catalog: ', filename)
print('Using catalog entries file: ', idx_file)
cat_idx = np.load(idx_file)
cat = fitsio.read(filename, rows=cat_idx,
                  columns=['BAND', 'PSF_FWHM', 'RA', 'DEC', 'FLUX', 'T_DATA', 'T_MODEL',
                           'G1_DATA', 'G1_MODEL', 'G2_DATA', 'G2_MODEL'])
print('Catalog loaded with %i entries.'%len(cat))
max_sep = 250
max_mag = -1
cattype='hsm'
name = 'y6a2_piff_v3_allres_v3'
work = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/rho_stats_output'
version = 2
if not os.path.isdir(work):
    os.mkdir(work)
bands = ['griz', 'riz'] # ['g','r','i','z']

for band in bands:
    print('Band: ', band)
    if len(band) > 1:
        data = cat[np.isin(cat['BAND'], [b for b in band])]
        print('Number of entries: ',len(data))
    else:
        data = cat[cat['BAND'] == band]
        print('Number of entries: ',len(data))
    stats = measure_rho(data, max_sep, max_mag, cattype=cattype, subtract_mean=True, do_rho0=True)
    stat_file = os.path.join(work, "rho_%s_%s_%i.json"%(name,band,version))
    write_stats(stat_file,*stats)

plot_overall_rho(work, name, bands=bands, version=version)
