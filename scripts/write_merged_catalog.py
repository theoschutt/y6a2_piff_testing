import os
from catalog_utils import matchExpDataToCatalog

path = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs'
cat = os.path.join(path, 'piff_sample_cat_v2.fits')
exp = os.path.join(path, 'y6a2_piff_v3_expinfo_v1.fits')
outfile = os.path.join(path, 'piff_sample_cat_v2_merged.fits')
# cat = os.path.join(path, 'y6a2_piff_v3_allres_v1_collated.fits')
# exp = os.path.join(path, 'y6a2_piff_v3_expinfo_v1.fits')
# outfile = os.path.join(path, 'y6a2_piff_v3_allres_v1_merged.fits')

matchExpDataToCatalog(cat, exp, 'EXPNUM', outfile)