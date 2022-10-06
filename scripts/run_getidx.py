#!/usr/bin/env python

import os
import fitsio
from catalog_utils import mkCutCatalog

cat_dir = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs'
catalog = 'y6a2_piff_v3_allres_v2_collated.fits'
outfile = 'y6a2_piff_v3_allres_v2_survey.fits'

cat_path = os.path.join(cat_dir, catalog)
cat_fits = fitsio.FITS(cat_path, vstorage='object')

mkCutCatalog('PROGRAM', 'survey', cat_fits, cat_dir, outfile)
