import fitsio

catpath = '/global/cscratch1/sd/schutt20/y6a2_piff_testing/y6a2_piff-model_paths_noY_collated.fits'
urlprefix = 'https://desar2.cosmology.illinois.edu/DESFiles/desarchive/'

filename = 'y6a2_piff-model_urls.dat'

fits = fitsio.FITS(catpath)
cat = fits[1][:]
print('Total URLs: ',len(cat))
with open(filename, 'w') as f:
    for i, row in enumerate(cat):
        if i%100000 == 0:
            print('Number URLs written: ', i)
        line = urlprefix+row[0]+'/'+row[1]
        f.write(line + '\n')
