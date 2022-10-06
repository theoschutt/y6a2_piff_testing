import numpy as np
import fitsio
import glob

modellist = 'y6a2_r1_piffmodels.csv'
print('Opening %s'%modellist)
with open(modellist, 'r') as f:
    files = f.readlines()

psffiles = [] 
for file in files:
    psfpath = file.strip('\n')
    psffiles.append(psfpath)
psffiles = psffiles[1:] #skip column name

file_arr = np.asarray(psffiles)
print(file_arr[:10])

# cat = fitsio.read('/global/cscratch1/sd/schutt20/y6a2_piff_testing/y6a2_piff_v2_hsm_allres_w_psffile_collated.fits')

filename = 'y6a2_piff_v2_hsm_allres_incoadds.fits'

# catfiles = cat['MODEL_FILENAME'][:]

# print('Starting match.')
# match_idx = np.isin(catfiles, file_arr)

# newcat = cat[match_idx]

# print('Writing catalog.')
# fitsio.write(outfile, newcat)

catlist = glob.glob('/global/cscratch1/sd/schutt20/y6a2_piff_testing/y6a2_piff_v2_hsm_allres_w_psffile_0*.fits')
# goldcat = fitsio.read('/global/homes/s/schutt20/gband_proj/piffify_files/y6gold_stars_merged.fits')
print('Number of catalogs to match: ', len(catlist))
star_tot = 0
for i in range(len(catlist)):
    print(i, catlist[i])
    cat = fitsio.read(catlist[i])    
    
    catfiles = np.asarray(cat['MODEL_FILENAME'][:])
    print(catfiles[:10])

    print('Starting match.')
    match_idx = np.isin(catfiles, file_arr)

    newcat = cat[match_idx]
    star_tot += len(newcat)

    print('Writing catalog.')
    if (i!=0):
        with fitsio.FITS(filename, 'rw') as f:
            f[-1].append(newcat)
    else:
        fitsio.write(filename, newcat)
    print(star_tot)
    print('%s complete.'%catlist[i])