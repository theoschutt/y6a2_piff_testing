import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

starpath = '/global/homes/s/schutt20/summer2021/Useful/psf_y3a1-v29-all.fits'
goldpath = '/global/homes/s/schutt20/gband_proj/catalogs/y6gold_stars_merged.fits'
outfile = 'catalogs/psf-y3a1-v29-all_x_y6gold20_collated.fits'

# starpath = 'star_cat_test.fits'
# goldpath = 'gold_cat_test.fits'
# outfile = 'output_cat_test.fits'

star_cat = fitsio.read(starpath)
gold_cat = fitsio.read(goldpath)

# get RA/DEC for catalogs
starra = star_cat['ra']
starra[starra > 180.] -= 360.
stardec = star_cat['dec']
goldra = gold_cat['RA']
goldra[goldra > 180.] -= 360.
golddec = gold_cat['DEC']

star_coord = SkyCoord(starra*u.degree, stardec*u.degree)
gold_coord = SkyCoord(goldra*u.degree, golddec*u.degree)

# do catalog matching
print('Starting matching.')
idx, d2d, d3d = star_coord.match_to_catalog_sky(gold_coord,
                                                nthneighbor=1)             
print('Matching complete.')
# limit distance separation for good matches
max_sep = d2d < 0.5 * u.arcsec

# pull matches from Gold catalog
gold_match = gold_cat[idx[max_sep]]
print('Total PSF stars: ', len(star_cat), '\n',
      'Number matched: ', len(gold_match), '\n',
      'Percentage: ', len(gold_match)/len(star_cat))

# print('Writing testing catalogs.')
# star_test_cat = fitsio.FITS('star_cat_test.fits', 'rw')
# star_test_cat.write(star_cat[max_sep][:10])
# gold_test_cat = fitsio.FITS('gold_cat_test.fits', 'rw')
# gold_test_cat.write(gold_match[:10])
# star_test_cat.close()
# gold_test_cat.close()

# create new fits file for star catalog with added columns
print('Writing new catalog: ', outfile)
match_fits = fitsio.FITS(outfile, 'rw', clobber=True)
match_fits.write(star_cat)
# add magnitudes to objects with Gold catalog match
for band in 'GRIZ':
    flux_name = 'BDF_MAG_{}_CORRECTED'.format(band.upper())
    print('Calculating mag column: ', flux_name)
    flux_col = -1 * np.ones((len(star_cat),)) #-1 for no gold match                         
#    flux_name = 'PSF_FLUX_APER_8_{}'.format(band.upper())
    flux_col[max_sep] = gold_match[flux_name]
#    print(flux_col[match_idx])
    print('len(flux_col): ', len(flux_col),
          'len(flux_col[max_sep]): ', len(flux_col[max_sep]))
    print('Writing mag column: ', flux_name)
    match_fits[1].insert_column(name=flux_name, data=flux_col)
    print('Completed writing mag column.')

# add G-I and I-Z color columns for all objects
gmag = gold_match['BDF_MAG_G_CORRECTED']
imag = gold_match['BDF_MAG_I_CORRECTED']
zmag = gold_match['BDF_MAG_Z_CORRECTED']
for color in ('GI_COLOR', 'IZ_COLOR'):
    print('Calculating color colunn: ', color)
    if color == 'GI_COLOR':
        default_color = 1.6 # default g-i star color
        color_data = gmag - imag
    else:
        default_color = 0.25  # default i-z star color
        color_data = imag - zmag

    # color_col = default_color * np.ones((len(star_cat),))
    color_col = -1. * np.ones((len(star_cat),))
    color_col[max_sep] = color_data
    
    print('Writing color column: ', color)
    match_fits[1].insert_column(name=color, data=color_col)
    print('Completed writing color column.')

match_fits.close()
