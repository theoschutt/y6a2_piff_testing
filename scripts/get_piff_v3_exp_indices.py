import fitsio
import numpy as np
import os

"""
EACH USE: Change cat_file AND outfile (and possibly bands).
"""
cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'
cat_file = 'y6a2_piff_v2_hsm_allres_collated.fits'
cat_path = os.path.join(cat_dir,cat_file)

exp_file = 'y6a2_piff_v3_test_01_allres_v3_collated.fits'
exp_path = os.path.join(cat_dir,exp_file)

outfile = 'y6a2_piff_v2_hsm_allres_x_v3_test_01_idx.txt'
outpath = os.path.join(cat_dir, outfile)

print('Loading file: ', cat_path)

fits = fitsio.FITS(cat_path)
fits.upper = True

cat = fits[1]['EXPNUM'][:]

print(len(cat), cat[:5])

print('Finished loading catalog.')

print('Loading exposure list: ', exp_path)

expfits = fitsio.FITS(exp_path)

expos = expfits[1]['EXPNUM'][:]

print(len(expos), expos[:5])

print('Matching exposures...')

mask = np.isin(cat, expos)
idx_arr = np.linspace(0, len(mask)-1, len(mask), dtype=int)
band_idx = idx_arr[mask]

# write indices to file, one idx per line

print('Writing index file: ', outpath)
print('Number of entries: ', len(band_idx))
with open(outpath, 'w') as f:
    for i in band_idx:
        f.write(str(i)+'\n')
print('File written.')