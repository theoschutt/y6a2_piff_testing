"""
Plots tau statistics from FITS files (written using GGCorrelation.write).
"""
import os
from rho_stats import write_tau_from_fits, plot_overall_tau

bands = ['riz'] # for plot_overall_tau function
band = bands[0] # for specifying file name
name = 'y6a2_piff_v3_allres_v3'
ver = 4

max_sep=250
work_dir = ('/global/cfs/cdirs/des/schutt20/y6_psf/'
            'y6a2_piff_testing/tau_stats_output')
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

stat_file = os.path.join(work_dir, "tau_%s_%s_%i.json"%(name,band,ver))

tau0_fn = os.path.join(work_dir, 'tau0_stats_%s.fits'%(ver))
tau2_fn = os.path.join(work_dir, 'tau2_stats_%s.fits'%(ver))
tau5_fn = os.path.join(work_dir, 'tau5_stats_%s.fits'%(ver))
write_tau_from_fits(stat_file, tau0_fn, tau2_fn, tau5_fn)
print('Wrote stats to file.')

print('Plotting tau statistics and writing to file...')
plot_overall_tau(work_dir, name, bands, ver)
print('Plotting complete.')
