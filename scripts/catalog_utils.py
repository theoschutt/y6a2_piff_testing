"""
Various useful convenience functions for FITS catalog editing and/or manipulation.
"""

import fitsio
import numpy as np
import os
from astropy.table import Table
from pandas import DataFrame
import tqdm
import glob

def loadColumn(catalog_fitstable, column):
    fname = catalog_fitstable
    fits = fitsio.FITS(fname)
    fits.upper = True
    print('Loading catalog column: ', column)
    print('from catalog: ', catalog_fitstable)
    cat = fits[1][column][:]
    print('Finished loading catalog.')

    return cat

def collateCatalog(file_stem,
                   fits_dir='/global/cscratch1/sd/schutt20/y6a2_piff_testing/ea_queries/',
                   out_dir='/global/cscratch1/sd/schutt20/y6a2_piff_testing/catalogs/'):
    """
    Collates the FITS files produced by an easyaccess SQL query into one file.
    """
    
    catlist = glob.glob(os.path.join(fits_dir, '%s_0*.fits'%file_stem))
    print('Number of catalogs to combine: ', len(catlist))

    outfile = os.path.join(out_dir, '%s_collated.fits'%file_stem)

    star_tot = 0
    print('Starting catalog creation.')
    for i in range(len(catlist)):
        if (i%10 == 0):
            print('{} catalogs complete.'.format(i))

        current_ = fitsio.FITS(catlist[i])    
        hsm_arr = hsm[1][:]
        star_tot += len(hsm_arr)

        if (i!=0):
            with fitsio.FITS(outfile, 'rw') as f:
                f[-1].append(hsm_arr)
        else:
            fitsio.write(outfile, hsm_arr)
        print('Total cumulative entries: ', star_tot)
    print('Catalog creation complete.')


def getBandIndices(bands, catalog_fitstable, catalog_directory, outfile_stem):
    """
    Gets the indices of all rows with a band entry matching a given (case-sensitive) 
    band string (e.g. 'griz' or 'Y'). Writes the indices to a .txt file. Useful for
    avoiding repeated searching over large catalogs.
    """
    print('Using bands: ', bands)

    cat = loadColumn(catalog_fitstable, 'BAND')

    for band in bands:
        print('Getting entry indices for band: ', band)
        band_idx = np.where(cat == band)[0]

        # write indices to file formatted such that we can use 
        # np.fromfile() to retrieve band_idx
        outfile = '%s_%s_idx.txt'%(outfile_stem, band)
        outpath = os.path.join(catalog_directory, outfile)
        print('Writing index file: ', outpath)
        print('Number of entries: ', len(band_idx))
        with open(outpath, 'w') as f:
            band_idx.tofile(f)
        print('File written.')


def mkIndexFiles(column, value, catalog_fitstable, catalog_directory, outfile_stem):
    """
    Gets the indices of all rows with an entry matching the given (case-sensitive) 
    `value' in `column'. Writes the indices to a .txt file. Useful for
    avoiding repeated searching over large catalogs.
    """
    cat = loadColumn(catalog_fitstable, column)

    print('Getting entry indices for value: ', value)
    val_idx = np.where(cat == value)[0]

    # write indices to file formatted such that we can use 
    # np.fromfile() to retrieve val_idx
    outfile = '%s_%s_idx.txt'%(outfile_stem, value)
    outpath = os.path.join(catalog_directory, outfile)
    print('Writing index file: ', outpath)
    print('Number of entries: ', len(val_idx))
    with open(outpath, 'w') as f:
        val_idx.tofile(f)
    print('File written.')
    
    return val_idx

def mkCutCatalog(column, value, catalog_fitstable, catalog_directory, outfile):
    cat_path = os.path.join(catalog_directory, catalog)
    
    print('Matching value: ', value, 'on column: ', column)
    print('Loading file: ', cat_path)

    cat = loadColumn(catalog_fitstable, column)

    print('Getting entry indices for value: ', value)
    val_idx = np.where(cat == value)[0]
    
    print('Number of entries: ', len(val_idx))
    
    new_cat = catalog_fitstable[1][val_idx]
    
    outpath = os.path.join(catalog_directory, outfile)
    fitsio.write(outpath, new_cat)

def matchExpDataToCatalog(cat_fits, exp_fits, match_column, outfile):
    """
    Matches the entries in a recarray of exposure data, exp_arr, to a more fine-grained
    recarray catalog, cat_arr (e.g. a star catalog). Matches using the match_column string,
    which should be a column name in both catalogs (usually 'EXPNUM').
    
    return_columns should be a list of column names in exp_arr, which will be matched to
    cat_arr.
    
    Creates array of left join of cat_arr and exp_arr, i.e. all data from cat_arr, and all
    matching exposure data concatenated along axis=1.
    
    Writes array to FITS file.
    
    TODO: implement tqdm progress bar. Might need Dask instead of pandas?
    """
    print(cat_fits)
    print(exp_fits)
    
    # using astropy Table to prevent big/little-endedness errors
    cat_arr = Table.read(cat_fits)
    exp_arr = Table.read(exp_fits)

    cat_df = DataFrame.from_records(cat_arr, columns=list(cat_arr.columns))
    exp_df = DataFrame.from_records(exp_arr, columns=list(exp_arr.columns))
        
    join_df = cat_df.merge(exp_df, 'left', on=match_column, copy=False, validate='m:1')
    
    fitsio.write(outfile, join_df.to_records())
    print('Wrote file: ', outfile)
