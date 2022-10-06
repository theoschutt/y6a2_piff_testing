import fitsio
import numpy as np
import os

"""
EACH USE: Change cat_file AND outfile (and possibly bands).
"""
cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'
cat_file = 'y6a2_piff_v3_allres_v3_collated.fits'
cat_path = os.path.join(cat_dir,cat_file)
bands = 'griz'

print('Loading file: ', cat_path)

fits = fitsio.FITS(cat_path)
fits.upper = True

cat = fits[1]['BAND'][:]

print('Finished loading catalog.')

for band in bands:
    print('Getting entry indices for band: ', band)
    band_mask = (cat['BAND'] == band)
    idx_arr = np.linspace(0, len(band_mask)-1, len(band_mask), dtype=int)
    band_idx = idx_arr[band_mask]
    
    # write indices to file, one idx per line
    outfile = 'y6a2_piff_v3_allres_v3_%s_idx.txt'%band
    outpath = os.path.join(cat_dir, outfile)
    print('Writing index file: ', outpath)
    print('Number of entries: ', len(band_idx))
    with open(outpath, 'w') as f:
        for i in band_idx:
            f.write(str(i)+'\n')
    print('File written.')