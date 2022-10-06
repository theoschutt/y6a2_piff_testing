import os
from rho_stats_v3_res import measure_rho, write_stats, plot_overall_rho
import fitsio
# TODO: not done
max_sep = 250
max_mag = 0
name = 'piff_v3_res'
work = '/global/homes/s/schutt20/gband_proj/plotting_scripts/output'
bands = 'griz'
tag = ''.join(band)
piff_stars = fitsio.read('../ea_queries/.fits')

stats = measure_rho(piff_stars, max_sep, max_mag, subtract_mean=True, do_rho0=False)
stat_file = os.path.join(work, "rho_%s_%s.json"%(name,tag))
write_stats(stat_file,*stats)
plot_overall_rho(work, name)
