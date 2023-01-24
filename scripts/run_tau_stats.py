"""
Runs tau statistics on given PIFF reserve star catalog and galaxy shear catalog 
on each band separately or combined.
To calculate on combined band data, use string with multiple bands.
That is: bands = ['griz'] will calculate the tau stats over the combined griz catalog.
         bands = ['g', 'r', 'i', 'z'] will calculate tau stats over each band individually.
"""

import fitsio
import os, sys
from rho_stats import (measure_tau_mpi, write_stats,
                       write_stats_tau, write_tau_from_fits,
                       plot_overall_tau)

bands = ['riz'] # for plot_overall_tau function
band = bands[0] # for specifying file name
name = 'y6a2_piff_v3_allres_v3'
ver = 4

cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'

piff_fn = os.path.join(cat_dir, 'y6a2_piff_v3_allres_v3_taustat_input_v5.fits')
# mdet_fn = '/global/cscratch1/sd/myamamot/des-y6-analysis/y6_measurement/metadetection_v2.fits'
mdet_fn = os.path.join(cat_dir, 'y6a2_mdet_v2_response_corrected.fits')
patch_fn = ('/global/cfs/cdirs/des/y6-shear-catalogs/'
            'patches-centers-altrem-npatch200-seed9999.fits')

print('Piff catalog: ',piff_fn)
print('Shear catalog: ',mdet_fn)
print('Shear catalog patches: ',patch_fn)

max_sep = 250
work_dir = ('/global/cfs/cdirs/des/schutt20/y6_psf/'
            'y6a2_piff_testing/tau_stats_output')
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

# set which corr functions to calculate
tau0 = True
tau2 = True
tau5 = True
write_json = True # set to False if all tau stats won't be written by
                  # end of run.

print('Computing tau statistics...')
stats = measure_tau_mpi(piff_fn, mdet_fn, patch_fn, max_sep=max_sep,
                        tau0=tau0, tau2=tau2, tau5=tau5,
                        output_dir=work_dir, version=ver)

print('Computation complete. Writing stats to file: %s'%stat_file)
stat_file = os.path.join(work_dir, "tau_%s_%s_%i.json"%(name,band,ver))

if write_json:
    
    if (tau0 and tau2 and tau5):
        write_stats_tau(stat_file,*stats)
    else:
        tau0_fn = os.path.join(work_dir, 'tau0_stats_%s.fits'%(ver))
        tau2_fn = os.path.join(work_dir, 'tau2_stats_%s.fits'%(ver))
        tau5_fn = os.path.join(work_dir, 'tau5_stats_%s.fits'%(ver))
        write_tau_from_fits(stat_file, tau0_fn, tau2_fn, tau5_fn)
    print('Wrote stats to file.')

print('Plotting tau statistics and writing to file...')
plot_overall_tau(work_dir, name, bands, ver)
print('Plotting complete.')
